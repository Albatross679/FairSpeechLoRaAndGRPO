# Maximize-VRAM Skill — Gap Research

**Researched:** 2026-04-11
**Purpose:** Find techniques NOT already documented in the `maximize-vram` skill and project knowledge doc. Feeds a future skill update.
**Sources read in full:** `SKILL.md`, `advanced-techniques.md`, `common-pitfalls.md`, `variable-length-data.md`, `knowledge/maximizing-vram-usage-ml.md`.

---

## Already Covered (do not duplicate)

These are in the existing skill + project doc. Anything in this list is out of scope for the "new" section below.

**Phase 1 — diagnosis:**
- Three bottleneck types (I/O, CPU, VRAM) + decision tree
- `nvidia-smi`, `nvitop`, `watch -n 0.5 nvidia-smi`
- `torch.cuda.memory_allocated/reserved/max_memory_allocated`
- PyTorch Profiler with `profile_memory=True`, `schedule()`, tensorboard
- `torch.cuda.memory._record_memory_history()` → `_dump_snapshot()` → pytorch.org/memory_viz flame graph

**Phase 2 — core VRAM techniques (Techniques 1–9):**
- Mixed precision fp16/bf16 + GPU-arch matching (T4/V100 → fp16; Ampere+ → bf16)
- Gradient checkpointing (`use_reentrant=False`), O(sqrt(n_layers)) activation savings
- Maximize `per_device_train_batch_size` first, then `gradient_accumulation_steps`
- 8-bit optimizers (`adamw_bnb_8bit`, `bnb.optim.AdamW8bit`)
- SDPA / FlashAttention-2 via `attn_implementation="flash_attention_2"`
- QLoRA / NF4 / double quant / `prepare_model_for_kbit_training` / `paged_adamw_8bit`
- Liger Kernel (`use_liger_kernel=True`, RMSNorm/RoPE/SwiGLU/FusedLinearCrossEntropy) — including FLCE memory savings for large vocab
- Unsloth (single-GPU, pre-quantized 4bit, Llama/Mistral/Qwen/Gemma/Phi)
- GaLore (gradient low-rank projection, `rank`, `update_proj_gap`, `proj_type`)
- LOMO / AdaLomo
- AdaLoRA
- FSDP (FULL_SHARD/SHARD_GRAD_OP/NO_SHARD/HYBRID_SHARD, `fsdp_use_orig_params`, `fsdp_cpu_ram_efficient_loading`, `fsdp_offload_params`)
- DeepSpeed ZeRO stages 1/2/3 + CPU offload
- `optim="adamw_torch_fused"`

**Phase 3 — data pipeline:**
- `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `drop_last`
- `non_blocking=True` transfers
- Pre-tokenize / pre-extract features offline
- Dynamic padding, `pad_to_multiple_of=8`
- Length bucketing / `group_by_length=True` / custom `LengthBucketSampler`
- TRL `SFTTrainer(packing=True)` + flash_attn varlen + masking/position ID gotchas
- Frame-budget batching with custom `BatchSampler` (ASR/video)

**Phase 4 — verification / common pitfalls:**
- 80–90% VRAM utilization target
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `torch.cuda.memory_summary()`
- `torch_empty_cache_steps` periodic cleanup
- `save_total_limit` to cap checkpoints
- Retained computation graphs (`.item()` vs appending loss tensors)
- `torch.no_grad()` during eval
- `bf16` on Turing/Volta pitfall
- Separate `per_device_eval_batch_size`
- `num_workers=0` pitfall
- `torch.compile` for runs >1000 steps (basic usage)
- Mixed precision via `autocast` + `GradScaler`

---

## New Techniques to Add

### A. Parallelism beyond FSDP/ZeRO-3

#### A1. FSDP2 (per-parameter sharding via `fully_shard`) — HIGH confidence

**One-liner:** PyTorch-native successor to FSDP1; per-parameter DTensor sharding with ~7% lower peak memory and cleaner composition with TP/PP/CP.

**What it does:** Instead of flattening and concatenating groups of tensors before chunking (FSDP1), FSDP2 chunks each parameter individually on dim-0 across the DP workers. Each parameter becomes its own DTensor. [CITED: pytorch.org/docs/stable/distributed.fsdp.fully_shard]

**Memory/throughput impact:**
- **~7% lower peak GPU memory** than FSDP1 on Llama 2 7B at same loss curve. [CITED: huggingface.co/docs/accelerate FSDP1 vs FSDP2]
- **~1.5% MFU gain** on Llama 7B / 8×H100. [CITED: arxiv.org/abs/2410.06511 — TorchTitan]
- Deterministic memory usage without FSDP1's `limit_all_gathers=True` blocking pattern.

**When to reach for it:** Any new multi-GPU training run. FSDP1 should be considered legacy for new code.

**Recipe:**
```python
from torch.distributed.fsdp import fully_shard, FSDPModule

# Per-layer sharding
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)  # shard top-level too
```
With HF Accelerate: set `fsdp_version: 2` in the FSDP plugin config (available in `accelerate >= 1.0`).

**Compatibility:**
- QLoRA: ✅ (use `fsdp_use_orig_params=True` equivalent — FSDP2 uses orig params by default).
- torch.compile: ✅ (much cleaner than FSDP1 + compile).
- Liger Kernel: ✅.
- TP / PP / CP composition: ✅ (this is FSDP2's main architectural win — DTensor-native).

**Gotchas:**
- Different state-dict shape than FSDP1 → migration care needed for existing checkpoints.
- Opacus support landed only recently; third-party libraries may still lag.

**Confidence:** HIGH — multiple official sources and production use in torchtitan.

---

#### A2. Tensor Parallelism (TP) via `torch.distributed.tensor.parallel` — HIGH confidence

**One-liner:** Shard individual matmul ops (attention heads, MLP hidden dim) across GPUs so each GPU holds a slice of the weights *and* does a slice of the compute.

**What it does:** Uses `ColwiseParallel` / `RowwiseParallel` to partition Linear-layer weights along columns or rows. Each GPU runs its shard of the matmul; an all-reduce combines results. Unlike FSDP (which all-gathers full weights for each op), TP keeps weights sharded during compute — lower peak memory for *individual* layers. [CITED: docs.pytorch.org/tutorials/intermediate/TP_tutorial.html]

**Memory/throughput impact:**
- For a layer with hidden dim H: peak parameter memory per GPU drops from H² to H²/TP_degree during forward/backward — unlike FSDP where you temporarily all-gather to full size.
- Lets you fit a larger single layer (e.g., a 100k+ vocab lm_head) that doesn't fit on one GPU even after FSDP all-gather.
- In torchtitan 3D runs: **12.59% speedup on Llama 3.1 70B / 256 GPUs (2D) and 30% on Llama 3.1 405B / 512 GPUs (3D)** over optimized baselines. [CITED: arxiv.org/abs/2410.06511]

**When to reach for it:** (1) A single layer doesn't fit on one GPU (common for huge lm_heads or MoE experts). (2) You're scaling past ~64 GPUs and FSDP alone has communication bottlenecks.

**Recipe:**
```python
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel)
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh["tp"]

for layer in model.layers:
    parallelize_module(layer, tp_mesh, {
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
    })
```

**Compatibility:**
- FSDP2: ✅ compose via 2D mesh (DP × TP).
- torch.compile: ✅.
- QLoRA: ⚠️ — works but less tested. bitsandbytes 4bit layers need care around sharding dims.
- Liger: ✅.

**Gotchas:**
- TP requires **high-bandwidth intra-node interconnect (NVLink)** because it all-reduces on every matmul. On PCIe-only systems it's a throughput disaster.
- Per-op activation memory often goes *up* in TP if not paired with Sequence Parallelism (see A3).
- Each TP shard sees a different slice of attention heads → `num_heads % tp_size == 0` required.

**Confidence:** HIGH.

---

#### A3. Sequence Parallelism (Megatron-style, paired with TP) — HIGH confidence

**One-liner:** Shards LayerNorm/Dropout/residual activations along the sequence dimension across TP ranks. Only meaningful as an add-on to TP.

**What it does:** In pure TP, attention and MLP weights are sharded, but LayerNorm and Dropout operate on the full activation tensor — which gets replicated across TP ranks, wasting memory. Sequence Parallelism splits these "replicated" regions along the sequence dim so each rank only holds `seq_len / tp_size` tokens of the norm/dropout activations. Conversions happen via `all-gather` and `reduce-scatter` at the TP region boundaries. [CITED: arxiv.org/abs/2205.05198 Reducing Activation Recomputation in Large Transformer Models]

**Memory/throughput impact:**
- **~half the activation memory of plain TP** for layernorms/dropouts (the paper's figure 5).
- Combined with selective activation recomputation (see B1): 5× total activation reduction while still running >90% of compute without recomputation.

**When to reach for it:** You're already using TP and LayerNorm activations have become a measurable fraction of VRAM.

**Recipe:** In PyTorch's `torch.distributed.tensor.parallel`, pass `SequenceParallel` to `parallelize_module`:
```python
from torch.distributed.tensor.parallel import SequenceParallel

parallelize_module(layer, tp_mesh, {
    "attention_norm": SequenceParallel(),
    "ffn_norm": SequenceParallel(),
    # ... plus TP on the linear layers as in A2
})
```

**Compatibility:** Requires TP. Works with FSDP2, torch.compile, flash_attention. [CITED: torchtitan docs]

**Gotchas:** Only meaningful combined with TP — on its own it does nothing for DP training.

**Confidence:** HIGH.

---

#### A4. Context Parallelism / Ring Attention (`torch.distributed.tensor.experimental.context_parallel`) — HIGH confidence

**One-liner:** Shards the sequence dimension across GPUs so each GPU holds only `seq_len / CP_degree` tokens, with ring-shuffled KV communication to compute full attention.

**What it does:** For sequences longer than one GPU's memory, splits the sequence across devices. Each GPU computes local attention against its Q shard, then receives remote KV shards in a ring pattern, updating the running softmax (same as FlashAttention tiling but across GPUs). All-to-all variants (Ulysses) shuffle KV by heads instead of sequence. [CITED: docs.pytorch.org/tutorials/unstable/context_parallel.html]

**Memory/throughput impact:**
- Enables **sequence lengths up to 1M tokens on Llama3-8B / 32× H100** in torchtitan. [CITED: discuss.pytorch.org/t/distributed-w-torchtitan Breaking Barriers]
- DeepSpeed Ulysses reports **2.5× speedup at 4× longer sequences** than the prior SOTA SP baseline. [CITED: arxiv.org/abs/2309.14509]

**When to reach for it:** Sequence length is so long that per-token activation × seq_len doesn't fit on one GPU even with FlashAttention.

**Recipe (PyTorch native):**
```python
from torch.distributed.tensor.experimental import context_parallel

cp_mesh = mesh["cp"]
with context_parallel(cp_mesh, buffers=[q, k, v], buffer_seq_dims=[2, 2, 2]):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Compatibility:**
- FSDP2 + TP + PP: ✅ (core torchtitan 4D compose story).
- torch.compile: ✅.
- flash_attention: needed as the inner kernel. Flash varlen works.
- QLoRA: ⚠️ not yet a documented pairing.

**Gotchas:**
- Requires `is_causal` handling — non-causal attention has different ring schedule.
- Load balance: naive ring is unbalanced for causal masks. Pass-KV variants fix this.
- Only worth the complexity past ~32k tokens.

**Confidence:** HIGH.

---

#### A5. Pipeline Parallelism via `torch.distributed.pipelining` + Zero-Bubble schedules — HIGH confidence

**One-liner:** Splits model by *layers* across GPUs; Zero-Bubble 1F1B schedule eliminates the pipeline-bubble idle time that plagued earlier PP.

**What it does:** Partitions layers across GPUs (stage 1 = layers 0–7, stage 2 = 8–15, etc.). Microbatches flow through the pipeline. The naive "GPipe" schedule has bubbles where GPUs idle. 1F1B interleaves one forward and one backward per microbatch; Zero-Bubble splits backward into "grad-input" and "grad-weight" so the optimizer step bypasses sync barriers. [CITED: arxiv.org/abs/2401.10241]

**Memory/throughput impact:**
- **Up to 23% throughput gain over 1F1B** at the same memory budget; **31% with relaxed memory.** [CITED: arxiv.org/abs/2401.10241]
- Memory-wise, PP reduces per-GPU param/grad/optim by `1/PP_degree` — complementary to DP sharding.

**When to reach for it:** You're training beyond what TP+FSDP can do on a single node; you have cross-node GPU connectivity with high latency where TP all-reduce would kill you.

**Recipe:**
```python
from torch.distributed.pipelining import Schedule1F1B, pipeline, SplitPoint

pipe = pipeline(model, mb_args=sample_input, split_spec={
    "layers.8": SplitPoint.BEGINNING,
    "layers.16": SplitPoint.BEGINNING,
    "layers.24": SplitPoint.BEGINNING,
})
stage = pipe.build_stage(stage_idx, device, pp_group)
schedule = Schedule1F1B(stage, n_microbatches=8, loss_fn=loss_fn)
# Alternative: ScheduleInterleaved1F1B, ScheduleZBVZeroBubble
```

**Compatibility:**
- FSDP2 + TP + CP (torchtitan 4D): ✅.
- Gradient checkpointing: ✅.
- Liger / torch.compile: ✅.
- QLoRA: ⚠️ less tested; FSDP2+QLoRA is the mature path.

**Gotchas:**
- Requires `n_microbatches ≥ pp_size × 2` or bubble reappears.
- Model must be splittable into clean stage boundaries.
- Zero-Bubble schedules need more memory for the split backward than plain 1F1B.

**Confidence:** HIGH.

---

#### A6. HSDP (Hybrid Sharded Data Parallel) — HIGH confidence

**One-liner:** Shard intra-node (NVLink-fast), replicate inter-node (Ethernet-slow). Middle ground between FSDP and DDP for multi-node.

**What it does:** Uses a 2D device mesh. Within a node, does full FSDP (all-gather weights for each op — cheap over NVLink at 600–900 GB/s). Across nodes, does DDP (all-reduce gradients once per step — fits in slow Ethernet budget at 25–50 GB/s). [CITED: apxml.com/courses/distributed-training-pytorch-fsdp hybrid-sharding-strategies]

**Memory/throughput impact:**
- Pure FSDP throughput **degrades on low-bandwidth inter-node links** as node count grows. HSDP **maintains near-linear scaling** by localizing the heavy traffic. [CITED: huggingface.co/blog/accelerate-nd-parallel]
- Memory per GPU: same as FSDP within a node (`params / intra_node_size`), worse than global FSDP (`params / total_gpus`).

**When to reach for it:** Multi-node training where inter-node bandwidth is the bottleneck (most non-Infiniband clusters). You can afford the memory cost of not sharding across nodes.

**Recipe — FSDP2:**
```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# 2D mesh: (replicate, shard)
mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                        mesh_dim_names=("replicate", "shard"))

for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)
```
With accelerate: `fsdp_sharding_strategy: HYBRID_SHARD` in config.

**Compatibility:** Everything FSDP2 supports.

**Gotchas:**
- Only helps if model+grads+optim fit in `total_params / gpus_per_node`.
- Use when inter-node bandwidth < ~100 Gbps. On Infiniband clusters, pure FSDP often wins.

**Confidence:** HIGH.

---

### B. Activation memory reduction

#### B1. Selective Activation Recomputation (Megatron-style) — HIGH confidence

**One-liner:** Instead of recomputing *all* checkpointed activations on backward (plain grad checkpointing), recompute *only* the low-compute / high-memory ops — primarily core attention.

**What it does:** Profiles which ops have a bad operation-per-byte ratio. Attention's softmax/dropout/QK-dot have memory cost O(seq²) but modest compute → recomputing them is cheap. The linear projections are expensive to recompute but small. Selective recomputation checkpoints only the inputs to attention blocks. [CITED: arxiv.org/abs/2205.05198]

**Memory/throughput impact:**
- **5× activation-memory reduction** on 530B GPT-style model. [CITED: arxiv.org/abs/2205.05198]
- **>90% reduction** in recomputation overhead vs full grad checkpointing (because you only redo ~5–10% of the compute).
- **+29% training throughput** over traditional full recomputation, reaching 54.2% MFU on 530B GPT-3 style model.

**When to reach for it:** You're using gradient checkpointing (~20% compute overhead) and that overhead is measurably hurting wall clock. Selective lets you keep most of the memory savings for ~2% overhead instead.

**Recipe — PyTorch native (FSDP2 / torchtitan):**
```python
# torchtitan uses "selective op" and "selective layer" modes:
# selective_op: checkpoint only matmul ops (leave elementwise/softmax alone)
# selective_layer: checkpoint every Nth transformer layer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing)

# "full" everywhere is the existing technique. Selective is the new gap.
# torchtitan: --activation_checkpoint.mode=selective_op (or selective_layer)
```

Via Megatron-LM: `--recompute-granularity selective`.

**Compatibility:**
- FSDP2: ✅ (this is how torchtitan uses it).
- torch.compile: ✅ with `CheckpointImpl.NO_REENTRANT`.
- QLoRA + HF Trainer: ❌ direct — HF Trainer only exposes full checkpointing. Manual patching required.
- Liger: ✅.

**Gotchas:**
- `use_reentrant=False` is mandatory.
- Picking *what* to checkpoint is model-specific. Default torchtitan policy is a reasonable starting point.

**Confidence:** HIGH.

---

#### B2. Activation CPU Offload (via checkpoint wrapper `offload_to_cpu=True`) — HIGH confidence

**One-liner:** Move checkpointed activations from GPU to CPU pinned memory between forward and backward; prefetch async.

**What it does:** Pairs with gradient checkpointing. After the forward pass stashes a checkpoint's input, that input is copied to CPU pinned memory on a dedicated CUDA stream so the GPU compute continues. Before backward, the activation is prefetched back. [CITED: docs.axolotl.ai/docs/gradient_checkpointing.html, docs.pytorch.org/docs/stable/fsdp.html]

**Memory/throughput impact:**
- Combined with activation checkpointing + parameter offload, FSDP can train models up to **20× larger** than plain DDP on the same hardware. [CITED: pytorch.org/blog/efficient-large-scale-training-with-pytorch]
- Throughput cost: modest if async overlap is on, substantial if not (~1.3–1.5× slower is typical).

**When to reach for it:** After gradient checkpointing, you still can't fit the batch size you need, and CPU memory is plentiful. Cheaper than ZeRO-Infinity NVMe offload.

**Recipe:**
```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl)

def wrap(module):
    return checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        offload_to_cpu=True,   # <-- the key flag
    )
# Apply to transformer layers before FSDP wrap
```

Axolotl: `activation_offloading: true`.

**Compatibility:** FSDP2 ✅, torch.compile ✅, QLoRA ✅ (used in common QLoRA+FSDP recipes), Liger ✅.

**Gotchas:**
- Requires `pin_memory=True` CPU allocation (big RSS usage).
- Async overlap depends on PCIe bandwidth; on PCIe gen3 you may see stalls.
- Does not stack with activation recomputation on the *same* layer (pick one per layer).

**Confidence:** HIGH.

---

### C. Optimizer memory beyond 8-bit AdamW / GaLore / LOMO

#### C1. Muon optimizer (newton-schulz orthogonalized momentum) — MEDIUM-HIGH confidence

**One-liner:** SGD-momentum + Newton-Schulz orthogonalization of the update. No second moment → half the optimizer state of AdamW, ~1.35× faster convergence.

**What it does:** After SGD-momentum, applies ~5 iterations of Newton-Schulz to each 2D parameter's update matrix, extracting its orthogonal component. This is a cheap approximation of the polar decomposition. The orthogonal update is equivalent in direction to a spectral normalization. [CITED: kellerjordan.github.io/posts/muon, arxiv.org/abs/2502.16982 Muon is Scalable]

**Memory/throughput impact:**
- **~33% optimizer state memory savings** vs AdamW (only momentum, no `v`). [CITED: medium.com/@md.abir1203 Muon Optimizer Complete Guide]
- **~35% fewer wall-clock hours** than AdamW to match loss at 1.5B scale (10h vs 13.3h on 8×H100). [CITED: kellerjordan.github.io/posts/muon]
- Moonshot's Kimi team reported ~2× throughput gain at 16B scale with MuonClip.

**When to reach for it:** Pre-training runs where optimizer memory matters and you can afford to validate convergence. For LoRA it's irrelevant (optimizer state is already tiny).

**Recipe:**
```python
# Reference implementation: github.com/KellerJordan/Muon
from muon import Muon
# Muon only handles 2D hidden params; use AdamW for embeddings/norms/lm_head
hidden_params = [p for n, p in model.named_parameters()
                 if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
other_params  = [p for n, p in model.named_parameters() if p not in hidden_params]

optimizers = [
    Muon(hidden_params, lr=0.02, momentum=0.95),
    torch.optim.AdamW(other_params, lr=3e-4),
]
```

**Compatibility:**
- FSDP / FSDP2: ⚠️ Newton-Schulz needs the full parameter, so Muon + FSDP requires special handling (all-gather then apply). Some community forks exist.
- torch.compile: ✅.
- QLoRA / LoRA: ❌ — you'd apply Muon to the LoRA `A`/`B` matrices but the benefit is small; AdamW is fine.
- Liger: independent.

**Gotchas:**
- **Hyperparameter sensitivity**: LR must be retuned (Muon's effective scale differs from AdamW).
- **Only for hidden 2D params**: apply AdamW to embeddings/norms/lm_head.
- **FSDP composition is not plug-and-play** — community implementations vary. Confirm before using in production multi-GPU.
- Newton-Schulz iterations are the added compute cost (~5% step overhead typical).

**Confidence:** MEDIUM-HIGH. Technique is real, production-proven at Kimi/Moonshot, but FSDP composition is still evolving.

---

#### C2. APOLLO / APOLLO-Mini — HIGH confidence

**One-liner:** SGD-like memory cost with AdamW-level quality. MLSys'25 Outstanding Paper Honorable Mention.

**What it does:** Observes that AdamW's learning-rate adaptation can be approximated as a *structured* (channel-wise) scaling. Maintains an auxiliary low-rank optimizer state via pure random projection (no SVD, unlike GaLore). APOLLO-Mini goes further with rank-1 tensor-wise scaling. [CITED: arxiv.org/abs/2412.05270]

**Memory/throughput impact:**
- **3× throughput on 8×A100-80GB** vs AdamW by allowing 4× larger batch sizes.
- **LLaMA-13B pretraining with naive DDP on A100-80GB** — previously required ZeRO-2+.
- **LLaMA-7B pretraining on a single GPU with <12 GB** (with weight quantization).

**When to reach for it:** Full-parameter pretraining or full FT where AdamW optimizer state is the limiting factor. Directly competes with GaLore and often beats it.

**Recipe:**
```bash
pip install apollo-torch
```
```python
from apollo_torch import APOLLOAdamW
optimizer = APOLLOAdamW(
    model.parameters(),
    lr=1e-4,
    scale_type="channel",   # "tensor" for APOLLO-Mini
    rank=256,               # 1 for Mini
    update_proj_gap=200,
    scale=1.0,
    proj_type="std",
)
```
HF Trainer: `optim="apollo_adamw"` (availability varies by transformers version — check release notes).

**Compatibility:**
- FSDP: ⚠️ similar story to GaLore — random projection needs the full grad; pair with `fsdp_use_orig_params=True` and check the implementation.
- torch.compile: ✅.
- LoRA/QLoRA: ❌ — not useful, optimizer state already small.
- Gradient checkpointing: ✅.

**Gotchas:**
- Random-projection rank vs convergence tradeoff. Start with the paper's defaults.
- Not yet as battle-tested as GaLore in public forks.

**Confidence:** HIGH (peer-reviewed MLSys'25, available on GitHub).

---

#### C3. Q-GaLore (INT4 projection + INT8 weights) — HIGH confidence

**One-liner:** GaLore + aggressive quantization of projection matrix (INT4) and weights (INT8) + adaptive subspace updates.

**What it does:** Extends GaLore by (1) quantizing the projection matrices to INT4, (2) quantizing the entire model to INT8 with stochastic rounding for gradient accumulation, (3) skipping SVD updates in layers whose subspace is stable. [CITED: arxiv.org/abs/2407.08296]

**Memory/throughput impact:**
- **Pretrain LLaMA-7B from scratch on a single 16 GB RTX 4060 Ti.**
- **50% memory reduction vs GaLore and LoRA** at fine-tuning.
- **Outperforms QLoRA at the same memory cost.**
- **~36% of the SVD operations** of plain GaLore (most are skipped adaptively), giving significant time savings.

**When to reach for it:** You want full-parameter (not LoRA) training and can't afford GaLore's memory. Especially good for consumer GPUs.

**Recipe:**
```bash
pip install q-galore-torch
```
```python
from q_galore_torch import QGaLoreAdamW8bit

optimizer = QGaLoreAdamW8bit(
    [{"params": galore_params, "rank": 128, "update_proj_gap": 200,
      "scale": 0.25, "proj_type": "std", "proj_quant": True, "proj_bits": 4}],
    lr=1e-4,
)
```

**Compatibility:**
- FSDP: ⚠️ same caveat as GaLore.
- QLoRA: ❌ (orthogonal — Q-GaLore is a full-training technique).
- torch.compile: verify per version.
- Liger: ✅.

**Gotchas:**
- INT8 stochastic-rounding weight updates are numerically delicate; follow reference LR carefully.
- Startup slower than GaLore due to quantization setup.

**Confidence:** HIGH.

---

#### C4. Lion optimizer — MEDIUM confidence for gap value

**One-liner:** Sign-momentum optimizer. Half AdamW's optimizer state. Discovered via symbolic AutoML at Google.

**What it does:** Keeps only first moment (momentum). Update is `sign(momentum)` scaled by LR. [CITED: github.com/lucidrains/lion-pytorch]

**Memory impact:** **~33% optimizer state savings** (momentum only, no variance). Same as Muon but without the Newton-Schulz step. [CITED: blog.sotaaz.com AdamW vs Lion]

**When to reach for it:** Pretraining where you want lower optimizer memory than AdamW but don't want Muon's FSDP complexity. **On fine-tuning / LoRA, Lion is not a VRAM win — optimizer state is already tiny** — so only recommend for full-param training.

**Recipe:**
```python
from lion_pytorch import Lion
# Lion LR = AdamW_lr / 3 to 10, weight_decay × 3 to 10
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1.0)
```
HF Trainer: some variants expose `optim="lion_8bit"` via bitsandbytes.

**Compatibility:** FSDP ✅, torch.compile ✅, LoRA/QLoRA (works but no VRAM benefit).

**Gotchas:**
- **Hyperparameter retuning mandatory.** LR 3–10× smaller than AdamW, WD 3–10× larger.
- Sign-based updates are noisier — warmup more critical.

**Confidence:** MEDIUM for *gap value*. Technique is HIGH confidence; its place in the skill is as "consider if you need optimizer memory savings and Muon/APOLLO feel too bleeding-edge."

---

#### C5. BAdam (block coordinate descent full-FT) — HIGH confidence

**One-liner:** Runs Adam on **one transformer layer at a time**, keeping optimizer state for only that block. Full-parameter fine-tuning in near-LoRA memory.

**What it does:** Partitions parameters into `D` blocks (typically = number of transformer layers). Each "block epoch" updates only one block's parameters and holds Adam state only for that block. Over many block epochs, every parameter is trained. [CITED: arxiv.org/abs/2404.02827]

**Memory/throughput impact:**
- **Full-parameter Llama-3-8B fine-tuning on a single RTX 3090 (24 GB).** [CITED: github.com/Ledzy/BAdam]
- **Full-parameter Llama-3-70B on 4×A100-80GB.**
- Outperforms LoRA on MT-bench and math benchmarks at similar memory.
- Per-step is comparable to LoRA; convergence in total wall clock is slower than full Adam but much better than LoRA quality-wise.

**When to reach for it:** You want full-FT quality on consumer or mid-tier GPUs, and LoRA quality isn't enough. Direct competitor to GaLore / APOLLO / Q-GaLore in the "full FT in LoRA memory" niche.

**Recipe:**
```bash
pip install badam
```
```python
from badam import BlockOptimizer

base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer = BlockOptimizer(
    base_optimizer=base_optimizer,
    named_parameters_list=list(model.named_parameters()),
    switch_block_every=100,    # steps per block
    switch_mode="ascending",   # layer 0, 1, 2, ...
    active_modules=[],         # block boundary = transformer layer
)
```

**Compatibility:**
- FSDP: ⚠️ non-trivial (BCD interacts with FSDP's grad hooks). Check reference repo.
- torch.compile: ✅.
- LoRA: pointless combination.
- Gradient checkpointing: ✅.

**Gotchas:**
- Each block update only touches one layer's params — LR and #steps-per-block need tuning.
- "switch_mode" picks which block to train next. Ascending / random / descending all behave differently.

**Confidence:** HIGH (NeurIPS 2024 poster).

---

#### C6. Schedule-Free AdamW — MEDIUM confidence for VRAM relevance

**One-liner:** Eliminates the need for LR scheduling + warmup by using iterate averaging. Small memory bonus (no scheduler state) but the real value is robustness.

**What it does:** Replaces momentum with an interpolation between gradient evaluation point and averaging point. No schedule needed; LR stays constant. [CITED: arxiv.org/abs/2405.15682]

**Memory impact:** Minor — saves the schedule state and any EMA/averaging buffers. **Not a primary VRAM technique**, but an "also free" when it works.

**When to reach for it:** Long pretraining runs where picking a schedule length is awkward.

**Recipe:**
```bash
pip install schedulefree
```
```python
from schedulefree import AdamWScheduleFree
optimizer = AdamWScheduleFree(model.parameters(), lr=1e-3, warmup_steps=1000)
# IMPORTANT: call optimizer.train() before training steps, optimizer.eval() before validation
```

**Compatibility:** FSDP ✅, torch.compile ✅.

**Gotchas:**
- **Must call `optimizer.train()` / `optimizer.eval()`** — forgetting breaks the iterate-averaging. This is the #1 footgun.
- Not a VRAM win per se; include as "low-cost upgrade to AdamW" in a later section, not as primary.

**Confidence:** MEDIUM for gap value (small VRAM impact). HIGH for technique correctness.

---

### D. PEFT variants beyond LoRA / QLoRA / AdaLoRA

#### D1. DoRA (Weight-Decomposed Low-Rank Adaptation) — HIGH confidence

**One-liner:** Decomposes each weight into magnitude + direction; applies LoRA only to direction, trains magnitude separately. Better quality than LoRA at same rank; **costs more VRAM during training**.

**What it does:** `W = m · (W₀ + BA) / ‖W₀ + BA‖`. The magnitude vector `m` and LoRA matrices are trained separately. The paper shows DoRA can achieve LoRA-r=8 quality with LoRA-r=4, potentially letting you use lower rank. [CITED: arxiv.org/abs/2402.09353]

**Memory/throughput impact:**
- **DoRA costs more training memory than plain LoRA** — this is key to know. The "efficient DoRA" variant via detached normalization reduces this by **~24.4% for LLaMA** and **~12.4% for VL-BART**. [CITED: arxiv.org/abs/2402.09353]
- Inference cost: **zero** after merging — decomposed components merge back into base weights.

**When to reach for it:** LoRA quality is insufficient AND you have VRAM headroom OR you're willing to use lower rank to offset DoRA's overhead. Don't default to DoRA expecting a VRAM win.

**Recipe — HF PEFT:**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", ...],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    use_dora=True,    # <-- the flag
)
model = get_peft_model(model, config)
```

**Compatibility:**
- QLoRA: ✅ (use `use_dora=True` with a 4bit base).
- FSDP: ✅ with `use_orig_params=True`.
- Unsloth: ⚠️ version-dependent; Unsloth historically had partial DoRA support.
- torch.compile: ✅.

**Gotchas:**
- **Higher VRAM, not lower** — framing matters. The skill should list DoRA as a "quality upgrade that costs VRAM" not a VRAM-saving technique.
- Slower per step than LoRA.

**Confidence:** HIGH.

---

#### D2. rsLoRA (Rank-Stabilized) — HIGH confidence

**One-liner:** Changes the LoRA scaling factor from `α/r` to `α/sqrt(r)`. Free stability improvement at high rank.

**What it does:** The default LoRA scaling `α/r` causes the effective learning rate to shrink as rank grows — meaning high-rank LoRA (r=64, 128) doesn't benefit proportionally. rsLoRA fixes this by scaling `α/sqrt(r)`. Trivial change. [CITED: peft docs]

**Memory impact:** **Zero.** Same rank = same memory. But lets you effectively use higher ranks for better quality.

**When to reach for it:** You're trying LoRA at r=64 or higher and seeing disappointing results.

**Recipe:**
```python
config = LoraConfig(r=64, lora_alpha=16, use_rslora=True, ...)
```

**Compatibility:** Everything LoRA compatible. ✅.

**Confidence:** HIGH. Note: this is a **quality**, not memory, technique — include as an "also worth knowing" footnote.

---

#### D3. PiSSA (Principal Singular values and Singular vectors Adaptation) — HIGH confidence

**One-liner:** Initialize LoRA `A`, `B` with the principal singular vectors of the base weight instead of random. Freeze the residual.

**What it does:** `W₀ = U Σ Vᵀ`. Split into top-`r` components (goes into `BA`, trainable) and residual (stays frozen). Converges faster than random LoRA; reduces quantization error when combined with QLoRA. [CITED: github.com/GraphPKU/PiSSA, peft docs]

**Memory impact:** Same as LoRA at inference / training. **Reduces QLoRA quantization error** because the residual (with small singular values) quantizes with less relative error.

**When to reach for it:** QLoRA runs where the 4-bit quantization is hurting quality more than you can afford.

**Recipe:**
```python
config = LoraConfig(
    r=16,
    init_lora_weights="pissa",     # or "pissa_niter_4" for fast SVD
    ...
)
```

**Compatibility:** QLoRA ✅, FSDP ✅, Unsloth ⚠️ version-dependent.

**Gotchas:**
- Requires converting the base model after init (LoRA matrices were initialized from the base). Reference repo has a `save_pretrained` wrapper.
- `pissa_niter_N` uses truncated SVD — fast but approximate.

**Confidence:** HIGH. NeurIPS 2024 Spotlight.

---

#### D4. LoftQ (Quantization-aware LoRA init) — HIGH confidence

**One-liner:** Jointly optimizes quantization + LoRA init so that `quant(W) + BA ≈ W`. Reduces QLoRA accuracy drop.

**What it does:** At init, alternates between quantizing `W - BA` and SVDing to improve `A, B` — so the LoRA adapters absorb the quantization error. [CITED: peft docs]

**Memory impact:** Same as LoRA. **The value is closing the gap between QLoRA and 16-bit LoRA** — purely quality.

**When to reach for it:** QLoRA runs where you're hitting quality ceilings. Pair with PiSSA or use instead of it.

**Recipe:**
```python
config = LoraConfig(r=16, init_lora_weights="loftq", ...)
```

**Confidence:** HIGH. Note: quality technique, not memory — keep it as "also relevant" note in a QLoRA section.

---

### E. Low-precision training beyond fp16/bf16

#### E1. fp8 Training via `torchao.float8` + FSDP2 + torch.compile — HIGH confidence

**One-liner:** Replace all Linear layers with Float8Linear; train in fp8 on H100+. The PyTorch-native alternative to NVIDIA Transformer Engine.

**What it does:** `torchao.float8.convert_to_float8_training` recursively replaces `nn.Linear` with `Float8Linear`. Matmul weights, inputs, and grad outputs are quantized to E4M3 / E5M2 per step with rowwise scaling; attention and layernorm stay in bf16. Pairs with FSDP2's fp8 all-gather to also speed cross-GPU comms. [CITED: docs.pytorch.org/ao/stable/pretraining.html, pytorch.org/blog/training-using-float8-fsdp2]

**Memory/throughput impact:**
- **Up to 50% throughput speedup** for 70B / 405B models on H100/H200 vs bf16 FSDP1 baseline. [CITED: pytorch.org/blog/training-using-float8-fsdp2]
- **1.34–1.43×** on 2K-H200 clusters with rowwise recipe. [CITED: github.com/pytorch/ao float8 README]
- **Activation memory reduced** since intermediate linear outputs are stored fp8 (1 byte/element).
- Reported: **20% of the speedup from float8 all_gathers** alone (comms savings).

**When to reach for it:** H100 (Hopper, CC 9.0) or Blackwell training with model size ≥ ~1B. Not applicable to A100 or consumer Ampere (no fp8 Tensor Cores). Probably not worth it for small models.

**Recipe:**
```python
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

config = Float8LinearConfig.from_recipe_name("rowwise")  # or "tensorwise"
convert_to_float8_training(model, config=config,
                           module_filter_fn=lambda mod, fqn: "lm_head" not in fqn)

# Must use torch.compile for performance:
model = torch.compile(model)
# And FSDP2 for all-gather speedup:
for layer in model.layers:
    fully_shard(layer, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
fully_shard(model)
```

**Compatibility:**
- **torch.compile: MANDATORY** — without it, performance regresses vs bf16. [CITED: torchao pretraining docs]
- FSDP2: ✅ first-class.
- FSDP1: ⚠️ limited; no fp8 all-gather.
- Liger: ⚠️ verify per release — fp8 kernels may conflict with Liger's fused matmul.
- QLoRA: ❌ conceptually incompatible (QLoRA already quantizes).
- H100 / Hopper: ✅ required.
- A100 and earlier: ❌ no hardware fp8 Tensor Cores.

**Gotchas:**
- **Exclude lm_head / embeddings** from fp8 — huge accuracy drop otherwise.
- Delayed-scaling variants need amax history buffers (small extra memory).
- Rowwise recipe is numerically stabler than tensorwise; start there.
- Loss curves should match bf16 within noise; if not, your scaling is wrong.

**Confidence:** HIGH.

---

#### E2. MXFP8 / MX Formats (Blackwell) — MEDIUM confidence (newer hardware)

**One-liner:** Block-scaled FP8 / FP6 / FP4 formats native to Blackwell (B100 / B200). Smaller scaling granularity than H100 fp8.

**What it does:** MX formats use a shared 8-bit exponent per 32-element block. Finer-grained than Hopper's tensor-level or rowwise scaling → better numerics and simpler recipes. `torchtitan/docs/mxfp8.md` documents the current PyTorch recipe. [CITED: github.com/pytorch/torchtitan/blob/main/docs/mxfp8.md]

**Memory/throughput impact:**
- Better numerics → can drop to **lower bit widths (MXFP6, MXFP4)** for inference and possibly training.
- Throughput gains over H100 fp8 depend entirely on Blackwell silicon.

**When to reach for it:** Blackwell (B100, B200, RTX 50xx) only. If you're on Hopper, use E1 (Hopper fp8) instead.

**Compatibility:** FSDP2 ✅, torch.compile required.

**Gotchas:**
- **Hardware limited** — no fallback path on Hopper.
- Still maturing; training recipes evolve monthly.

**Confidence:** MEDIUM — hardware is new and adoption is early.

---

### F. Attention kernels beyond Flash Attention 2 + SDPA

#### F1. FlashAttention-3 (H100-native, async, fp8) — HIGH confidence

**One-liner:** Hopper-optimized rewrite of FlashAttention using warp specialization, TMA, and fp8 support. ~1.5–2× faster than FA2 on H100.

**What it does:** Uses (1) warp-specialization to overlap matmul and softmax, (2) TMA (Tensor Memory Accelerator) for async memory movement, (3) hardware fp8 matmul with block quantization + incoherent processing for better accuracy. [CITED: arxiv.org/abs/2407.08608]

**Memory/throughput impact:**
- **1.5–2.0× speedup over FA2** in forward pass (up to 740 TFLOPs/s fp16 on H100).
- **1.5–1.75× in backward.**
- **1.3 PFLOPs/s in fp8** with **2.6× better RMSE** than naive fp8 attention (9.1e-3 vs 2.4e-2).
- H100 utilization: **35% with FA2 → 75% with FA3**.

**When to reach for it:** H100 training. On Ampere (A100), FA2 is still the right choice (FA3 requires Hopper TMA).

**Recipe:**
```bash
pip install "flash-attn>=3.0.0"   # or build from Dao-AILab/flash-attention
```
Transformers routes via `attn_implementation="flash_attention_2"` — needs a recent transformers + flash-attn 3 to actually use FA3. Check release notes.

**Compatibility:**
- H100 only (SM90). Ampere → FA2.
- fp8 via torchao: pair with E1.
- FSDP / TP / CP: ✅.
- torch.compile: ✅ as opaque custom op.

**Gotchas:**
- Still version-churning (as of early 2026). API + HF routing may lag.
- Backward pass gains are smaller than forward.

**Confidence:** HIGH.

---

#### F2. PyTorch FlexAttention (`torch.nn.attention.flex_attention`) — HIGH confidence

**One-liner:** Write custom attention masks and score modifications as Python functions; PyTorch JITs them into FlashAttention-quality kernels. Programmable attention without writing Triton.

**What it does:** Takes `score_mod` and `mask_mod` callables plus Q/K/V tensors. Under the hood, `torch.compile` generates a block-sparse FlashAttention kernel that skips fully-masked blocks via `BlockMask`. Enables sliding window, ALiBi, document masks, tanh-soft-cap, etc. as one-line changes — and now supports **trainable parameters** in `score_mod`. [CITED: docs.pytorch.org/docs/stable/nn.attention.flex_attention.html, pytorch.org/blog/flexattention-for-inference]

**Memory/throughput impact:**
- **2.4× faster training** vs manual-mask attention in torchtune. [CITED: pytorch.org/blog/flexattention-for-inference]
- Fully masked blocks are *skipped* — for sliding-window attention with window << seq_len, memory and compute scale by the mask sparsity ratio.
- Matches FlashAttention performance for dense causal attention.

**When to reach for it:** (1) Any non-standard attention mask (sliding window, document packing masks, prefix-LM, custom block-diagonal for sequence packing). (2) You'd otherwise reach for xFormers or write Triton.

**Recipe:**
```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def sliding_window_mask(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= 512

block_mask = create_block_mask(sliding_window_mask, B=None, H=None,
                                Q_LEN=seq_len, KV_LEN=seq_len)

out = flex_attention(q, k, v, block_mask=block_mask)
```

**Compatibility:**
- Requires PyTorch ≥ 2.5. Requires `torch.compile`.
- FSDP2 ✅. TP ✅.
- fp8 / low-precision: check per version (there's a known degradation issue in some precisions).

**Gotchas:**
- Performance is good only under `torch.compile`. Eager path is slow.
- Compile time is not free — amortize over a real training run.
- Document-mask block granularity is coarser than token-level; verify packing correctness.

**Confidence:** HIGH.

---

#### F3. Sequence packing via `flash_attn_varlen_func` (already named but worth elevating) — HIGH confidence

Already mentioned in the existing skill's packing section ("FlashAttention variable-length is the performance-critical piece"). **No gap** — covered.

---

### G. Compiler / kernel fusion beyond torch.compile defaults

#### G1. `torch.compile(mode="reduce-overhead")` with CUDA Graphs — HIGH confidence

**One-liner:** Trades a small memory bump for CUDA-graph capture of the compiled training step — eliminates per-step kernel launch overhead.

**What it does:** "reduce-overhead" mode captures the compiled graph into a CUDA graph and replays it instead of re-launching each kernel individually. Launch overhead savings are biggest for small batch size / short steps. [CITED: docs.pytorch.org/docs/stable/generated/torch.compile.html]

**Memory/throughput impact:**
- **Biggest wins at small batch / short seq_len**, where kernel launch overhead dominates.
- **~1.3–2.0× speedups** on mid-to-large models (compile + CUDA graphs combined). [CITED: medium.com/@bhagyarana80 PyTorch Compile Moves]
- **Small memory cost** — CUDA graphs hold their own tensor workspace.

**When to reach for it:** Small-batch training (common for LoRA + memory-limited GPUs), or inference-like fine-tuning workflows.

**Recipe:**
```python
model = torch.compile(model, mode="reduce-overhead")
# Or: training_args.torch_compile_mode = "reduce-overhead"
```

**Modes comparison:**
| Mode | Use case | Speedup | Memory | Compile time |
|---|---|---|---|---|
| `default` | General training | 1.2–1.5× | base | fast |
| `reduce-overhead` | Small-batch / launch-bound | 1.3–1.8× | slightly more | moderate |
| `max-autotune` | Inference / very long runs | 1.5–2.0×+ | base | slow (minutes) |

**Compatibility:**
- FSDP2 ✅. FSDP1: known compile issues. Prefer FSDP2.
- QLoRA: ⚠️ `torch.compile` on bitsandbytes 4-bit matmul is still version-sensitive.
- Liger: ✅.
- Gradient checkpointing: ✅ (requires `use_reentrant=False`).

**Gotchas:**
- CUDA graph capture fails on dynamic shapes — use `pad_to_multiple_of` + fixed seq_len (or frame-budget with padding to a max).
- Recompile storms on shape changes. Pin shapes.
- Not helpful if you're already compute-bound with large batches.

**Confidence:** HIGH. Gap-worthy because existing skill only mentions `torch.compile` in passing.

---

#### G2. Thunder (Lightning AI) — LOW-MEDIUM confidence for gap value

**One-liner:** Source-to-source PyTorch compiler that composes nvFuser + torch.compile + cuDNN + TransformerEngine into one pipeline. Alternative fusion layer.

**What it does:** `thunder.jit(model)` — traces a Python program and dispatches ops across multiple executors. Reports ~40% speedup vs eager, and up to **3.42× faster text generation, 1.69× faster fwd+bwd** on DeepSeek-R1-Distill-Llama-1.5B. [CITED: github.com/Lightning-AI/lightning-thunder]

**When to reach for it:** You want to experiment with multiple fusion backends (including TE fp8) without rewriting. Niche — most users can stay on torch.compile.

**Compatibility:** Interops with torch.compile (can be used *with* it, not only instead).

**Gotchas:**
- Project maturity: active but smaller ecosystem than torch.compile.
- Model coverage is narrower — verify your model works before adopting.

**Confidence:** LOW-MEDIUM for gap value. Technique is real but probably not essential for a general skill. Include as "if curious, try Thunder" note only.

---

### H. Memory allocator / driver-level tricks

#### H1. Full `PYTORCH_CUDA_ALLOC_CONF` cheat sheet — HIGH confidence

**One-liner:** The skill mentions `expandable_segments:True` only. Other options help different failure modes.

**What they do:** [CITED: docs.pytorch.org/docs/stable/notes/cuda.html]

| Option | Effect | When to use |
|---|---|---|
| `expandable_segments:True` | (Already covered) Reduces fragmentation | OOM after many steps |
| `max_split_size_mb:N` | Native allocator won't split blocks larger than N MB | Large contiguous allocations failing even with free memory |
| `garbage_collection_threshold:0.8` | Allocator proactively frees old unused blocks when >80% used | Memory creep without clear leak |
| `backend:cudaMallocAsync` | Use CUDA's async allocator instead of native | Rare — experimental, different fragmentation profile |
| `roundup_power2_divisions:[N]` | Round allocation sizes to power-of-2 / N | Reduce fragmentation in allocator-heavy workloads |
| `pinned_use_background_threads:True` | Async pinned CPU mem allocation | Large pinned CPU allocations blocking main thread |

**Gotcha:** **`garbage_collection_threshold` is IGNORED when `backend:cudaMallocAsync` is set.** Pick one backend. [CITED: pytorch CUDA semantics docs]

**Recommended default for training:**
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
```

**Confidence:** HIGH.

---

#### H2. NCCL tuning: `NCCL_BUFFSIZE` / `NCCL_P2P_CHUNKSIZE` — MEDIUM confidence

**One-liner:** Per-GPU NCCL comm buffers consume VRAM. Defaults are conservative; tuning can save a few hundred MB per rank.

**What it does:** `NCCL_BUFFSIZE` (default **4 MiB**) controls the buffer NCCL uses between GPU pairs. On memory-constrained training, reducing it saves VRAM. On throughput-critical training with fast fabrics, increasing can overlap better. [CITED: docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html]

**When to reach for it:** Multi-GPU training where FSDP/TP is close to OOM and nvidia-smi shows a few hundred MB of "NCCL overhead" per rank.

**Recipe:**
```bash
export NCCL_BUFFSIZE=2097152   # 2 MiB, half default
```

**Gotchas:**
- Very workload-dependent — benchmark before committing.
- Too small can tank comms throughput.
- Only matters when VRAM is truly the bottleneck.

**Confidence:** MEDIUM. Include as "if VRAM is tight on multi-GPU, check NCCL_BUFFSIZE" footnote.

---

### I. Data pipeline gaps

#### I1. NVIDIA DALI — HIGH confidence

**One-liner:** GPU-accelerated data loading and preprocessing library. Offloads JPEG decode, audio decode, augmentation to GPU.

**What it does:** Replaces CPU-based DataLoader pipelines with GPU-executed pipelines (image decode via nvJPEG, audio decode, resize, normalize, crop, augment). Async execution with prefetch. [CITED: developer.nvidia.com/dali]

**Memory/throughput impact:**
- **72% / 37% / 43% training-time improvement** for ResNet18 / 50 / 152 for constant augmentation load vs DataLoader. [CITED: developer.nvidia.com/blog/rapid-data-pre-processing-with-nvidia-dali]
- Eliminates CPU bottleneck entirely for image-heavy training.
- Pushes some VRAM cost (decoding buffers) but frees CPU.

**When to reach for it:** GPU utilization is low because CPU is saturated (common for image training with heavy augmentation). Less essential for text / pre-tokenized pipelines.

**Recipe:**
```python
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def image_pipeline():
    images, labels = fn.readers.file(file_root=data_dir, name="Reader")
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, device="gpu", size=(224, 224))
    return images, labels

pipe = image_pipeline(batch_size=64, num_threads=4, device_id=0)
loader = DALIGenericIterator(pipe, ["images", "labels"], reader_name="Reader")
```

**Compatibility:** PyTorch ✅, HF Trainer ⚠️ needs custom dataset adapter.

**Gotchas:**
- Learning curve — pipeline DSL differs from PyTorch Dataset.
- Not useful for text tokenization (already cheap).
- For **audio**, DALI has audio ops, but torchcodec (I2) is becoming the PyTorch-native choice.

**Confidence:** HIGH.

---

#### I2. torchcodec (GPU video/audio decoding, PyTorch-native) — HIGH confidence

**One-liner:** PyTorch's new official video/audio decoder. **Deprecates torchaudio.load / torchaudio.save in 2.9.** CUDA decoding path.

**What it does:** `torchcodec.decoders.VideoDecoder` / `AudioDecoder` produces PyTorch tensors directly from compressed media, with optional CUDA decode on supported formats. [CITED: pytorch.org/blog/torchcodec, github.com/meta-pytorch/torchcodec]

**Memory/throughput impact:**
- **Consistently the best-performing PyTorch library for decoding many videos at once** (per PyTorch blog).
- CUDA decode path cuts CPU load dramatically for video; for audio, the main win is the unified API.

**When to reach for it:** Any audio/video training (including ASR) where the decoder is a hot path. **This is directly relevant to the project** — the ASR pipeline currently goes through torchaudio.load, which is now deprecated.

**Recipe:**
```python
from torchcodec.decoders import AudioDecoder
decoder = AudioDecoder("audio.wav")
samples = decoder.get_samples_played_in_range(start_seconds=0, stop_seconds=30)
# samples.data: tensor, samples.sample_rate: int

# Via torchaudio (backward-compat wrapper, still calls torchcodec):
import torchaudio
waveform, sr = torchaudio.load_with_torchcodec("audio.wav")
```

**Compatibility:** torchaudio 2.8+ delegates to it. torchaudio 2.9 removes the old backend.

**Gotchas:**
- **Project-specific:** the existing ASR scripts using `torchaudio.load` will keep working via the compat shim until torchaudio 2.9. **Plan migration now.**
- CUDA decode support depends on codec / container; WAV typically stays CPU, lossy formats benefit most.

**Confidence:** HIGH.

---

#### I3. FFCV — MEDIUM confidence

**One-liner:** Custom `.beton` file format + data loader that outperforms DataLoader, WebDataset, and DALI on ImageNet-style pipelines.

**What it does:** Dataset pre-packed into page-structured `.beton` files; loader uses random-page access for better randomness and constant-memory-footprint shuffling. [CITED: github.com/libffcv/ffcv, arxiv.org/abs/2306.12517]

**Throughput impact:**
- Outperforms DataLoader, WebDataset, MXNet, and DALI in the paper's benchmarks.
- Constant memory footprint with quasi-random loading (vs WebDataset's linear-in-workers footprint).

**When to reach for it:** You've outgrown DataLoader, can pre-convert your dataset, and DALI integration is too heavy. Strong for pure vision training; less mature for modern multi-modal data.

**Gotchas:**
- Requires dataset conversion step (one-time but nontrivial).
- Newer framework churn is slower than PyTorch-native tools.
- Audio/video not a focus.

**Confidence:** MEDIUM. Include as an option for I/O-bound vision pipelines.

---

#### I4. Megatron Energon — MEDIUM confidence

**One-liner:** Multimodal data loader built on WebDataset, distributed-aware, used by NVIDIA NeMo.

**What it does:** Shards large-scale multimodal data into compressed WebDataset shards, with distributed-training-aware sampling, deterministic resumption, and blend-weighted source mixing. [CITED: NVIDIA NeMo framework docs]

**When to reach for it:** Multimodal (image+text+audio) pretraining at scale. Overkill for single-modality training.

**Confidence:** MEDIUM — niche but real.

---

### J. Framework-specific patterns

#### J1. Axolotl defaults worth copying — HIGH confidence

Axolotl's default recipe contains several things the skill doesn't call out explicitly:

- **`sample_packing: true` + `pad_to_sequence_len: true`** — Axolotl intentionally pads packed sequences to a fixed length. Reason: *varying packed lengths cause VRAM spikes even if each batch is smaller than fully padded*. [CITED: github.com/axolotl-ai-cloud/axolotl discussion 1101]
- **`flash_attention: true`** with packing → uses FA2 varlen API automatically.
- **`gradient_checkpointing: true`** paired with `activation_offloading: true` for tight VRAM.
- **`eval_sample_packing: false`** — packing during eval is often broken, disable explicitly.
- **Sequence parallelism (beta)** — Axolotl exposes Ulysses-style SP as `sequence_parallel_degree: N`. [CITED: docs.axolotl.ai/docs/sequence_parallelism.html]

**Gap value:** The skill's packing section should note the "pad-to-fixed-length for packed sequences" subtlety. This contradicts the usual "dynamic padding wins" advice because packing's whole point is that everything is already the same length.

**Confidence:** HIGH.

---

#### J2. torchtitan recipe — HIGH confidence

The canonical modern recipe for scale-out PyTorch training, maintained by Meta/PyTorch. Combines:

- FSDP2 (per-parameter `fully_shard`)
- TP + SP + CP composable via DeviceMesh
- Pipeline parallelism with Zero-Bubble schedules
- Selective Activation Checkpointing (`selective_op` mode recommended default)
- torchao float8 training on H100
- torch.compile mandatory
- Native sharded checkpointing

Full config in `torchtitan/train_configs/llama3_8b.toml`. [CITED: github.com/pytorch/torchtitan]

**When to reach for it:** You're pretraining or doing full-param fine-tuning at ≥16 GPUs and want the PyTorch-native path (not DeepSpeed).

**Confidence:** HIGH.

---

### K. Monitoring / diagnosis tools

#### K1. NVIDIA Nsight Systems (`nsys`) — HIGH confidence

**One-liner:** System-wide profiler that captures CPU, GPU, Python, and CUDA activity on a shared timeline. Finds GPU idleness that PyTorch Profiler misses.

**What it does:** Traces kernel launches, memcpys, NVTX ranges, Python call stack sampling, NCCL comms — all on one timeline. Shows periodic gaps in the CUDA row that indicate GPU starvation. [CITED: arikpoz.github.io 2025 guide]

**When to reach for it:** GPU-Util oscillating but PyTorch Profiler's DataLoader breakdown says "nothing obvious." nsys reveals whether the gap is launch overhead, Python GIL, H2D transfer, or NCCL comm.

**Recipe:**
```bash
nsys profile \
  --pytorch=autograd-shapes-nvtx \
  --python-sampling=true \
  --backtrace=none \
  -o training_profile \
  python train.py --max-steps 20
```
Then open `training_profile.nsys-rep` in the Nsight Systems GUI.

**Published results:** Recent 2025 case study: **3.2× training speedup** from nsys-driven optimizations (training job that took a week now finishes in 2 days). [CITED: arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x]

**Compatibility:** Any CUDA workload. Requires local install of Nsight Systems (free from NVIDIA).

**Gotchas:**
- Profile a *short* run (20–50 steps) — profile files balloon fast.
- `.nsys-rep` files are large; open on a workstation, not over SSH.

**Confidence:** HIGH.

---

#### K2. `nvidia-smi dmon` / `gpustat` — MEDIUM confidence

Lightweight alternatives to nvitop:
- **`nvidia-smi dmon -s pucm`** — one-line-per-second power, utilization, clocks, memory streaming. Good for headless monitoring.
- **`gpustat`** — one-line-per-GPU snapshot; cleaner than `nvidia-smi` output for quick checks.

**Gap value:** LOW — nvitop already covers this. Include only as footnote.

**Confidence:** MEDIUM (tools are real, but marginal value over nvitop).

---

### L. Honorable mentions: 2024–2026 techniques the skill predates

#### L1. Stochastic rounding for bf16 optimizer states — MEDIUM confidence

**What it does:** When updating bf16 weights in-place (no fp32 master copy), nearest-rounding kills small updates. Stochastic rounding makes the update unbiased in expectation; Kahan summation carries the rounding error as a separate 16-bit accumulator. [CITED: arxiv.org/abs/2010.06192, optimi.benjaminwarner.dev/kahan_summation]

**Memory impact:** Drops the fp32 master copy of weights → **saves 4P bytes** (big for full FT). Kahan summation costs 2P bytes back, net **-2P bytes**.

**When to reach for it:** Full-parameter training where fp32 master weights are a significant fraction of VRAM and you're already on bf16. Not useful for LoRA.

**Recipe:** Use `optimi` library (MIT, pure PyTorch), or Adafactor with Kahan summation (some impls). HF Trainer doesn't expose this directly.
```python
# Example via optimi
from optimi import AdamW
optimizer = AdamW(model.parameters(), lr=1e-4, kahan_summation=True)
```

**Compatibility:** FSDP ✅, torch.compile ✅, LoRA: no benefit.

**Gotchas:**
- Only worth it for full-param training ≥1B.
- Convergence verification is essential — numerical subtleties.

**Confidence:** MEDIUM — technique is solid, adoption is narrow.

---

#### L2. ZeRO-Infinity NVMe offload — HIGH confidence (but narrow value)

**What it does:** DeepSpeed extension that offloads parameters, gradients, and optimizer states to **NVMe SSDs** instead of just CPU RAM. Async prefetching overlaps NVMe→CPU→GPU movement with compute. [CITED: deepspeed.ai/tutorials/zero, microsoft.com/research/blog zero-infinity]

**Memory impact:** Enables **trillion-parameter training on a single DGX-2 node**, **30+ trillion on 32 nodes**. Bandwidth-limited by NVMe — much slower per step than in-GPU training.

**When to reach for it:** Only when the model (params + grads + optim) exceeds **total CPU RAM**, which is rare outside pretraining research. For 99% of users, FSDP2 + CPU offload is sufficient.

**Compatibility:** DeepSpeed ecosystem only (FSDP2 doesn't have an NVMe-tier offload yet as of early 2026).

**Gotchas:**
- Requires fast NVMe (per-GPU NVMe is ideal).
- Throughput collapse if NVMe saturates.
- Use `io_device_path` tuning.

**Confidence:** HIGH technique-wise; LOW in terms of how often it's the right answer. Mention in the "last resort" decision row.

---

#### L3. torchao `CPUOffloadOptimizer` — HIGH confidence

**One-liner:** PyTorch-native, torch.compiled replacement for bitsandbytes' paged_adamw. Moves optimizer state **and** gradients to CPU; runs the optimizer step on CPU.

**What it does:** Wraps any existing optimizer; stashes state on CPU pinned memory, moves gradients to CPU for the optimizer step, copies updated weights back. Pair with `torch.optim.AdamW(fused=True)` as the inner optimizer. [CITED: github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim]

**Memory/throughput impact:**
- **~60% VRAM reduction** on single-GPU training (claimed in torchao docs).
- Benchmark: **39.1 GB peak vs 51.6 GB fused AdamW vs 39.3 GB bitsandbytes 8bit** — so it matches bitsandbytes 8bit in memory while being pure PyTorch. [CITED: github.com/pytorch/ao]
- Expect ~1.5–2× slower per step than in-GPU optimizer.

**When to reach for it:** Single-GPU full-param FT where optimizer doesn't fit. Alternative to `paged_adamw_8bit`. PyTorch-native instead of bitsandbytes-dependent.

**Recipe:**
```python
from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

optimizer = CPUOffloadOptimizer(
    model.parameters(),
    optimizer_class=torch.optim.AdamW,
    fused=True,        # CPU-side fused AdamW
    offload_gradients=True,
)
```

**Compatibility:**
- Single GPU only.
- QLoRA: redundant (QLoRA optimizer state is already small).
- FSDP: ❌ (FSDP has its own offload).
- torch.compile: ✅.

**Gotchas:**
- Requires ~2× model size of CPU RAM on top of optimizer state.
- Fast CPU matters — on old Xeons the bottleneck moves to the CPU optimizer step.

**Confidence:** HIGH.

---

#### L4. torchao low-bit optimizers (`AdamW8bit`, `AdamW4bit`) — HIGH confidence

**One-liner:** Pure-PyTorch 8-bit and 4-bit AdamW. `torch.compile`-friendly competitor to `bitsandbytes.AdamW8bit`.

**What it does:** Quantizes optimizer state to int8 or int4. Written in pure PyTorch + torch.compile — no CUDA C kernels like bitsandbytes. Benchmarks match bitsandbytes at 8-bit; 4-bit halves again. [CITED: twitter.com/marksaroufim status 1809398186198593566, github.com/pytorch/ao low_bit_optim]

**When to reach for it:** You want the speed/memory of bitsandbytes 8bit but prefer a pure-PyTorch dependency (easier install, avoids bnb version drama). Or you want **4-bit** optimizer state — bitsandbytes doesn't ship 4bit AdamW.

**Recipe:**
```python
from torchao.prototype.low_bit_optim import AdamW8bit, AdamW4bit
optimizer = AdamW4bit(model.parameters(), lr=1e-4)  # ~8× savings vs fp32 state
```

**Compatibility:** FSDP ✅, torch.compile ✅, LoRA (tiny benefit), QLoRA ✅.

**Gotchas:** 4-bit version is newer — verify convergence.

**Confidence:** HIGH.

---

## Rejected Candidates

Techniques investigated but not included as new gaps:

- **VeRA (shared random projections)** — Smaller LoRA state but quality tradeoff is significant and FSDP support is weak. Niche.
- **LoHA / LoKr (Hadamard/Kronecker PEFT)** — Real and in PEFT, but memory vs LoRA is similar and community use is small. Quality-focused, not memory-focused.
- **LoRA-GA, MoRA, DoubleLoRA, LoRA+** — Mostly quality improvements, not memory. Already have the "LoRA variants" category covered; listing all would bloat the skill.
- **xFormers memory-efficient attention** — Superseded by SDPA/FA2/FA3 in PyTorch 2.x. Mentioned only as a fallback for very old GPUs; not a gap.
- **PagedAttention (vLLM)** — Inference-only. No training uptake. Not a gap.
- **Ring Attention (standalone library)** — PyTorch now ships native context parallelism (A4). Third-party Ring Attention libs are legacy.
- **nvFuser** — Folded into torch.compile / TorchInductor. No longer a standalone choice.
- **NCCL persistent all-reduce** — Too niche, benefits only very specific workloads. NCCL tuning H2 covers the accessible gains.
- **MIG (Multi-Instance GPU) / MPS** — These *reduce* per-process VRAM by partitioning, which is the opposite of what the skill is for. Only relevant in multi-tenant contexts.
- **Transparent Huge Pages** — Marginal effect, difficult to verify, not a training-specific technique.
- **`nvitop` replacements (py-spy, memray, gpustat)** — Mentioned as K2 footnote; skill already names nvitop.
- **Axolotl + LlamaFactory detailed configs** — Framework-specific recipes. J1 captures the principle; full configs would duplicate their docs.
- **Lion optimizer as primary VRAM win** — Included as C4 but flagged MEDIUM — its main value is speed, not memory (it's -33% state vs AdamW, same as Muon). Include but don't headline.
- **Schedule-free AdamW** — Included as C6 but flagged MEDIUM; memory impact is incidental.
- **Prodigy / D-adapt / CAME / SOAP** — Real but minor VRAM differences vs AdamW. Quality-oriented optimizers; not gap-worthy for a VRAM skill.
- **AdamW-mini** — Real paper but not yet in mainstream frameworks; low adoption maturity.

---

## Priority: Top 10 to add first

Ranked by biggest single-technique VRAM/throughput win for typical readers:

1. **A1. FSDP2 (`fully_shard`)** — Every new multi-GPU training project should use this. 7% memory win for free, unlocks the rest of the torchtitan recipe (A2/A3/A4/A5/B1/E1). The single most important update to the skill.
2. **B1. Selective Activation Recomputation** — Keeps most of grad-checkpointing's memory win at 2% compute overhead instead of 20%. Massive throughput unlock for anyone currently using full checkpointing.
3. **E1. torchao fp8 training** — Up to 50% throughput on H100. The modern precision story; existing skill stops at bf16.
4. **F1. FlashAttention-3** — 1.5–2× faster attention on H100. Automatic win for Hopper users.
5. **F2. FlexAttention** — Enables efficient sliding window, document-packing, and custom masks without writing kernels. Relevant even on A100.
6. **B2. Activation CPU Offload** — Enables 20× larger models than DDP when paired with checkpointing. A natural extension of the skill's existing checkpointing section.
7. **I2. torchcodec** — Directly relevant to the project's ASR pipeline (torchaudio.load deprecation). Immediate concrete action item.
8. **A4. Context Parallelism** — The long-context story PyTorch now ships natively. Essential for any >32k-token training.
9. **C2. APOLLO / APOLLO-Mini** — Best current alternative to GaLore for full-param FT on consumer GPUs. MLSys'25 pedigree.
10. **G1. `torch.compile(mode="reduce-overhead")` + CUDA Graphs** — The existing skill mentions torch.compile only as a footnote. Deserves its own section with mode comparison table.

**Runner-ups (include in secondary update):**
- H1. Full `PYTORCH_CUDA_ALLOC_CONF` cheat sheet (quick fragmentation diagnosis)
- A6. HSDP (multi-node specific)
- C1. Muon (pretraining specific)
- J1. Axolotl packing-with-fixed-length gotcha (subtle but bites people)
- L3. torchao `CPUOffloadOptimizer` as modern paged_adamw replacement
- K1. `nsys` as deeper profiler when PyTorch Profiler is inconclusive

---

## Sources

### High confidence (primary)
- [PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [TorchTitan paper (arxiv 2410.06511)](https://arxiv.org/abs/2410.06511)
- [TorchTitan GitHub](https://github.com/pytorch/torchtitan)
- [Huggingface Accelerate FSDP1 vs FSDP2 guide](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2)
- [PyTorch tensor parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [Reducing Activation Recomputation paper (arxiv 2205.05198)](https://arxiv.org/abs/2205.05198)
- [PyTorch context parallel tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [DeepSpeed-Ulysses paper (arxiv 2309.14509)](https://arxiv.org/abs/2309.14509)
- [Zero Bubble Pipeline Parallelism paper (arxiv 2401.10241)](https://arxiv.org/abs/2401.10241)
- [PyTorch pipelining docs](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [torchao float8 pretraining docs](https://docs.pytorch.org/ao/stable/pretraining.html)
- [PyTorch blog — float8 + FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
- [FlashAttention-3 paper (arxiv 2407.08608)](https://arxiv.org/abs/2407.08608)
- [PyTorch FlexAttention docs](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [PyTorch FlexAttention blog](https://pytorch.org/blog/flexattention-for-inference/)
- [APOLLO paper (arxiv 2412.05270)](https://arxiv.org/abs/2412.05270)
- [APOLLO GitHub](https://github.com/zhuhanqing/APOLLO)
- [Q-GaLore paper (arxiv 2407.08296)](https://arxiv.org/abs/2407.08296)
- [BAdam paper (arxiv 2404.02827)](https://arxiv.org/abs/2404.02827)
- [BAdam GitHub](https://github.com/Ledzy/BAdam)
- [DoRA paper (arxiv 2402.09353)](https://arxiv.org/abs/2402.09353)
- [PiSSA GitHub](https://github.com/GraphPKU/PiSSA)
- [Muon blog — Keller Jordan](https://kellerjordan.github.io/posts/muon/)
- [Muon is Scalable paper (arxiv 2502.16982)](https://arxiv.org/abs/2502.16982)
- [Schedule-Free optimizer paper (arxiv 2405.15682)](https://arxiv.org/abs/2405.15682)
- [PyTorch CUDA semantics (alloc config)](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA DALI docs](https://developer.nvidia.com/dali)
- [torchcodec PyTorch blog](https://pytorch.org/blog/torchcodec/)
- [torchao low-bit optimizers](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim)
- [Axolotl multipack docs](https://docs.axolotl.ai/docs/multipack.html)
- [Axolotl gradient checkpointing / activation offloading](https://docs.axolotl.ai/docs/gradient_checkpointing.html)
- [Liger Kernel paper (arxiv 2410.10989)](https://arxiv.org/pdf/2410.10989)
- [Revisiting BFloat16 Training paper (arxiv 2010.06192)](https://arxiv.org/abs/2010.06192)
- [optimi Kahan summation docs](https://optimi.benjaminwarner.dev/kahan_summation/)
- [DeepSpeed ZeRO docs](https://www.deepspeed.ai/tutorials/zero/)
- [Lightning Thunder GitHub](https://github.com/Lightning-AI/lightning-thunder)
- [Nsight Systems 3× training speedup case study](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
- [FFCV paper (arxiv 2306.12517)](https://arxiv.org/abs/2306.12517)
- [NVIDIA NCCL tuning guide](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)
- [Huggingface Accelerate ND-parallel guide](https://huggingface.co/blog/accelerate-nd-parallel)

### Medium confidence
- Lion / AdamW-vs-Lion comparison blogs (technique real; gap value marginal for LoRA users)
- Thunder benchmark reports (HIGH technique / LOW gap value)
- MXFP8 torchtitan docs (newer hardware, limited validation data)

---

## Metadata

- **Research date:** 2026-04-11
- **Valid until:** ~2026-10 (fast-moving areas: FSDP2, torchao, FlexAttention, fp8 — re-check in 6 months)
- **Total new techniques identified:** 30 across 12 categories
- **Highest priority single item:** FSDP2 (A1) — unlocks everything else in the torchtitan recipe
- **Highest project-local impact:** torchcodec (I2) — torchaudio.load deprecation affects this project's ASR pipeline directly
