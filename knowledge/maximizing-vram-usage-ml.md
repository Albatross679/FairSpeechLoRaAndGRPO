---
fileClass: Knowledge
name: Maximizing VRAM Usage in ML Training
description: Comprehensive guide to making VRAM the bottleneck in ML training — profiling, optimization techniques, and common practices
created: 2026-04-07
updated: 2026-04-07
tags: [gpu, vram, optimization, training, profiling, mixed-precision, deepspeed]
aliases: [VRAM optimization, GPU memory optimization, ML training bottleneck]
---

# Maximizing VRAM Usage in ML Training

## The Core Question

"How do I make sure VRAM is the bottleneck?" is really two questions:

1. **Eliminate non-VRAM bottlenecks** — ensure CPU preprocessing, data loading, and host-to-device transfers are NOT what's limiting your throughput.
2. **Maximize what fits in VRAM** — once VRAM-bound, pack more useful work (larger batches, longer sequences) into the available memory.

If your GPU utilization is low (say 30-50%), you are likely NOT VRAM-bound — you are CPU-bound or I/O-bound, and adding more VRAM techniques will not help.

---

## Part 1: Diagnosing Your Actual Bottleneck

### The Three Bottleneck Types

| Bottleneck | Symptom | GPU-Util (nvidia-smi) | GPU Memory | CPU | Fix Category |
|---|---|---|---|---|---|
| **I/O-bound** | GPU idles between batches waiting for data | Low (10-40%) spiky | Low-moderate | Often low (waiting on disk) | DataLoader, storage, prefetch |
| **CPU-bound** | GPU idles between batches waiting for preprocessing | Low (10-40%) spiky | Low-moderate | High (100% on cores) | More workers, faster transforms, pre-tokenize |
| **VRAM-bound** | GPU runs continuously, OOM if batch increases | High (80-100%) steady | Near capacity | Moderate | Mixed precision, grad checkpoint, etc. |

### Step-by-Step Diagnosis

**Step 1: Check GPU utilization pattern**

```bash
# Watch GPU utilization over time (refresh every 0.5s)
watch -n 0.5 nvidia-smi

# Better: use nvitop for real-time per-process stats
pip install nvitop
nvitop
```

What to look for:
- **GPU-Util column stays 90-100%**: You are compute-bound or VRAM-bound. Good.
- **GPU-Util oscillates (e.g., 0% -> 100% -> 0%)**: GPU is starving for data. You are I/O or CPU bound.
- **GPU Memory-Usage near max BUT GPU-Util low**: Your data pipeline cannot keep up. Fix the pipeline first.

**Step 2: Profile with PyTorch Profiler**

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        # Your training step here
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prof.step()
        if step >= 6:  # wait(1) + warmup(1) + active(3) + buffer
            break

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Then launch TensorBoard:
```bash
tensorboard --logdir=./profiler_logs
```

Navigate to the **PyTorch Profiler** tab. Key views:
- **Overview**: Shows "Step Time Breakdown" — if "DataLoader" or "Other" dominates, you're I/O/CPU-bound.
- **Trace View**: Timeline showing CPU vs CUDA activity. Gaps in CUDA = GPU starvation.
- **Memory View**: Shows allocation patterns per operator.

**Step 3: Quick VRAM snapshot**

```python
import torch

# After loading model and running one training step:
print(torch.cuda.memory_summary(abbreviated=True))

# Key numbers:
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3
total = torch.cuda.get_device_properties(0).total_mem / 1024**3

print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved:  {reserved:.2f} GB")
print(f"Peak:      {peak:.2f} GB")
print(f"Total:     {total:.2f} GB")
print(f"Headroom:  {total - peak:.2f} GB")
```

**Decision Tree:**

```
Is GPU-Util consistently > 90%?
├── YES → Is peak VRAM > 85% of total?
│   ├── YES → You are VRAM-bound. Go to Part 3.
│   └── NO  → You are compute-bound (kernels are the limit).
│             Try torch.compile, FlashAttention, or larger batch.
└── NO  → You are starved for data. Go to Part 2.
    ├── Is CPU near 100%? → CPU-bound. Add workers, optimize transforms.
    └── Is CPU low too?   → I/O-bound. Fix storage, prefetching, data format.
```

---

## Part 2: Eliminating Non-VRAM Bottlenecks

Fix these FIRST. No amount of VRAM optimization helps if your GPU is idle 60% of the time.

### DataLoader Configuration

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=per_device_batch_size,
    num_workers=4,              # Start here, increase until GPU-Util stabilizes
    pin_memory=True,            # Pre-allocates page-locked CPU memory for fast H2D transfer
    persistent_workers=True,    # Avoids worker respawn overhead between epochs
    prefetch_factor=2,          # Each worker prefetches 2 batches (total = 2 * num_workers)
    drop_last=True,             # Avoids variable-size last batch
)
```

**For HuggingFace Trainer:**

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=2,
    # ... other args
)
```

### Tuning num_workers

Rule of thumb: start with `num_workers = 4`, increase until GPU-Util stops improving.

```python
import time
import torch

# Benchmark different num_workers values
for nw in [0, 2, 4, 8]:
    loader = DataLoader(dataset, batch_size=8, num_workers=nw, pin_memory=True)
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 50:
            break
        # Simulate moving to GPU
        if isinstance(batch, dict):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    elapsed = time.time() - start
    print(f"num_workers={nw}: {elapsed:.2f}s for 50 batches")
```

### Non-blocking Transfers

When using `pin_memory=True`, transfer tensors with `non_blocking=True`:

```python
batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
```

This overlaps the H2D copy with GPU compute from the previous batch.

### Pre-tokenize / Pre-process Offline

If CPU time is dominated by tokenization or audio feature extraction, pre-process your dataset offline and save to disk:

```python
# BEFORE: Tokenize on-the-fly (slow, CPU-bound)
class SlowDataset:
    def __getitem__(self, idx):
        text = self.texts[idx]
        return self.tokenizer(text, padding="max_length", truncation=True)

# AFTER: Pre-tokenize and save as Arrow/memmap
from datasets import Dataset
dataset = Dataset.from_dict({"text": texts})
dataset = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
dataset.save_to_disk("preprocessed/")
```

For audio data specifically: pre-compute mel spectrograms or feature extractions and save as `.npy` or in an Arrow dataset rather than decoding WAV files on every epoch.

---

## Part 3: Maximizing What Fits in VRAM

### Understanding Where VRAM Goes

For a model with P parameters trained with AdamW in mixed precision:

| Component | FP32 Training | Mixed Precision (AMP) | With Grad Checkpointing |
|---|---|---|---|
| Model weights | 4P bytes | 2P (fp16) + 4P (fp32 master copy) | Same |
| Gradients | 4P bytes | 2P bytes (fp16) | Same |
| Optimizer states (AdamW) | 8P bytes (momentum + variance) | 8P bytes (fp32) | Same |
| Activations | Variable (dominates for large batch/seq) | ~Half (fp16 activations) | Dramatically reduced |
| **Total (excluding activations)** | **16P bytes** | **16P bytes** | **16P bytes** |

Key insight: mixed precision does NOT reduce optimizer state memory (AdamW states stay fp32). Its main savings come from **activations** and **faster compute**. To reduce optimizer memory, use 8-bit optimizers.

For a 1.7B parameter model (like Qwen3-ASR):
- Model weights (fp16): ~3.4 GB
- FP32 master copy: ~6.8 GB
- Gradients (fp16): ~3.4 GB
- Optimizer states (AdamW fp32): ~13.6 GB
- **Total before activations: ~27 GB** (does not fit on T4 15GB)

This is why **LoRA is essential** on a T4 — you only train ~0.5-2% of parameters, so optimizer states shrink proportionally. With LoRA (say 10M trainable params):
- Frozen model (fp16): ~3.4 GB
- LoRA gradients: ~20 MB
- LoRA optimizer states: ~80 MB
- Activations: 1-8 GB depending on batch/sequence length

### Technique 1: Mixed Precision (AMP)

**Use fp16 on T4** (T4 is Turing architecture, compute capability 7.5 — no native bf16 support).

```python
# For HuggingFace Trainer:
training_args = TrainingArguments(
    fp16=True,  # Use fp16 on T4 (NOT bf16)
    # bf16=True,  # Only for Ampere+ (A100, A10, H100, RTX 3090+)
)

# For manual training loop:
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast(device_type="cuda", dtype=torch.float16):
        output = model(**batch)
        loss = output.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**Impact**: Activations stored in fp16 (half the memory). Tensor Core acceleration on T4. ~1.3-1.5x speedup.

**T4-specific note**: The existing project code uses `bf16=True` which the T4 may handle via software emulation (slow) or may error. Switch to `fp16=True` for T4 hardware.

### Technique 2: Gradient Checkpointing

Recomputes intermediate activations during backward pass instead of storing them all. Trades ~20% more compute time for massive activation memory savings.

```python
# For HuggingFace Trainer:
training_args = TrainingArguments(
    gradient_checkpointing=True,
)

# For manual models:
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**Impact**: Activation memory drops from O(n_layers) to O(sqrt(n_layers)). On a 24-layer model, this can reduce activation memory by 4-6x.

**Always use `use_reentrant=False`** — the reentrant variant is deprecated and incompatible with torch.compile.

### Technique 3: Gradient Accumulation

Simulate larger effective batch sizes without increasing per-step VRAM.

```python
training_args = TrainingArguments(
    per_device_train_batch_size=2,     # What fits in VRAM
    gradient_accumulation_steps=16,    # Effective batch = 2 * 16 = 32
)
```

**Critical**: Keep `per_device_train_batch_size` as large as VRAM allows. Then use `gradient_accumulation_steps` to reach your target effective batch size. Do NOT reduce batch size just to increase accumulation steps — that wastes GPU parallelism.

| Config | Effective Batch | GPU Utilization | Recommendation |
|---|---|---|---|
| batch=1, accum=32 | 32 | Poor (GPU underutilized per step) | Avoid |
| batch=4, accum=8 | 32 | Good | Preferred |
| batch=8, accum=4 | 32 | Better | Best if it fits |

### Technique 4: 8-bit Optimizers (bitsandbytes)

Quantizes optimizer states from fp32 to int8, reducing optimizer memory by ~75%.

```bash
pip install bitsandbytes
```

```python
# With HuggingFace Trainer:
training_args = TrainingArguments(
    optim="adamw_bnb_8bit",  # Built-in support
)

# Manual usage:
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
```

**Impact**: AdamW states go from 8 bytes/param to ~2 bytes/param. For full fine-tuning of large models this is significant. For LoRA with few trainable params, the savings are small (optimizer states are already tiny).

### Technique 5: Memory-Efficient Attention (SDPA / FlashAttention)

Fused attention kernels that avoid materializing the full N x N attention matrix.

```python
# SDPA is automatic in PyTorch 2.1+ and HuggingFace Transformers
# Just ensure your model loads with the right attention implementation:
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    attn_implementation="sdpa",  # or "flash_attention_2"
)
```

**Impact**: Attention memory goes from O(N^2) to O(N) where N is sequence length. Critical for long sequences.

FlashAttention-2 requires Ampere+ GPU. On T4, SDPA will use the memory-efficient attention backend from xFormers automatically.

### Technique 6: torch.compile

Fuses operations into optimized kernels, reducing overhead and sometimes memory.

```python
model = torch.compile(model, mode="reduce-overhead")

# With HuggingFace Trainer:
training_args = TrainingArguments(
    torch_compile=True,
    torch_compile_backend="inductor",
)
```

**Note**: Has a startup cost (first few steps are slow due to compilation). Best for long training runs.

### Technique 7: DeepSpeed ZeRO (Multi-GPU or CPU Offload)

For when the model truly does not fit on a single GPU, even with all the above techniques.

| ZeRO Stage | What it Shards | Memory Reduction | Overhead |
|---|---|---|---|
| Stage 1 | Optimizer states | Moderate | Minimal |
| Stage 2 | Optimizer states + gradients | Significant | Low |
| Stage 3 | Optimizer states + gradients + parameters | Maximum | Higher communication |
| + CPU Offload | Moves sharded states to CPU RAM | Even more | CPU-GPU bandwidth becomes bottleneck |

**Single-GPU use case**: ZeRO-2 or ZeRO-3 with CPU offload can help fit models that otherwise OOM, at the cost of throughput.

```python
# deepspeed_config.json for single GPU with CPU offload
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "bf16": {"enabled": false},
    "fp16": {"enabled": true}
}
```

```python
training_args = TrainingArguments(
    deepspeed="deepspeed_config.json",
)
```

**When to use DeepSpeed**: When gradient checkpointing + mixed precision + 8-bit optimizer still OOMs. For LoRA on a 1.7B model on T4, DeepSpeed is usually overkill.

---

## Part 4: The Practitioner Workflow

Follow this sequence to go from "underutilizing GPU" to "VRAM is the bottleneck":

### Phase A: Establish Baseline

```bash
# 1. Check what you're working with
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_properties(0))"

# 2. Run training for a few steps, observe
watch -n 1 nvidia-smi  # In a separate terminal
python train.py  # Your training script
```

Record: GPU-Util%, Memory-Usage, steps/second.

### Phase B: Fix the Data Pipeline (if GPU-Util < 80%)

1. Set `num_workers=4`, `pin_memory=True`, `persistent_workers=True`
2. Re-run, check GPU-Util
3. If still low, increase `num_workers` to 8
4. If still low, pre-process data offline (tokenize, extract features, save to disk)
5. Profile with PyTorch Profiler to find the exact bottleneck

### Phase C: Enable Memory Optimizations

Apply in this order (each one is independent, cumulative):

1. **Mixed precision**: `fp16=True` (T4) or `bf16=True` (Ampere+)
2. **Gradient checkpointing**: `gradient_checkpointing=True`
3. **Increase batch size**: Double `per_device_train_batch_size` until OOM, then back off one step
4. **Gradient accumulation**: Set `gradient_accumulation_steps` to reach desired effective batch size
5. **8-bit optimizer**: `optim="adamw_bnb_8bit"` (if optimizer states are a significant fraction of VRAM)

### Phase D: Verify VRAM-Bound

After applying optimizations:

```python
# Run a few training steps, then:
peak = torch.cuda.max_memory_allocated() / 1024**3
total = torch.cuda.get_device_properties(0).total_mem / 1024**3
utilization = peak / total * 100
print(f"Peak VRAM: {peak:.2f} GB / {total:.2f} GB ({utilization:.1f}%)")
```

**Target**: 80-90% VRAM utilization with stable GPU-Util at 90-100%.

- Below 80% VRAM: You can increase batch size more.
- Above 95% VRAM: Risk of OOM from fragmentation or variable-length inputs. Back off slightly.
- 80-90% VRAM with 90%+ GPU-Util: You are VRAM-bound. Mission accomplished.

### Phase E: Fine-Tune (Optional)

- Try `torch.compile` for additional speedup
- Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- Try Liger Kernel for fused operations (supported models only)

---

## Part 5: Common Pitfalls and VRAM Wasters

### Pitfall 1: bf16 on Non-Ampere GPUs

**What goes wrong**: Setting `bf16=True` on T4/V100 (Turing/Volta). These GPUs lack native bf16 Tensor Cores. PyTorch may silently fall back to fp32 or use slow software emulation.

**Fix**: Use `fp16=True` on T4 and V100. Use `bf16=True` only on A100, A10, H100, RTX 30xx+.

```python
# Detect at runtime:
if torch.cuda.get_device_capability()[0] >= 8:  # Ampere+
    training_args.bf16 = True
else:
    training_args.fp16 = True
```

### Pitfall 2: Retaining Computation Graphs

**What goes wrong**: Storing loss tensors in a list for logging without calling `.item()`. The entire computation graph stays in VRAM.

```python
# BAD: Retains computation graph in VRAM
all_losses = []
for batch in dataloader:
    loss = model(**batch).loss
    all_losses.append(loss)  # Graph retained!

# GOOD: Detach the scalar
all_losses = []
for batch in dataloader:
    loss = model(**batch).loss
    all_losses.append(loss.item())  # Scalar, graph freed
    loss.backward()
```

### Pitfall 3: Not Clearing Cache Between Experiments

**What goes wrong**: Running multiple training trials in the same process without clearing GPU memory.

```python
# Between trials:
del model, optimizer, trainer
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()  # Reset peak tracking too
```

### Pitfall 4: Unnecessary eval-mode memory

**What goes wrong**: Running evaluation with gradients enabled.

```python
# GOOD: No gradients during eval
model.eval()
with torch.no_grad():
    outputs = model(**batch)
```

HuggingFace Trainer handles this automatically, but custom loops often forget.

### Pitfall 5: DataLoader pin_memory=False (Default)

**What goes wrong**: Without pinned memory, every batch transfer to GPU goes through pageable memory, adding latency and preventing overlap.

**Fix**: Always set `pin_memory=True` unless you have a specific reason not to (e.g., custom collator returning non-tensors that break pinning).

### Pitfall 6: Variable-Length Sequences Without Padding Strategy

**What goes wrong**: Padding all sequences to `max_length` wastes compute and VRAM. But variable-length batches cause unpredictable VRAM spikes that cause OOM.

**Fix**: Use dynamic padding (pad to longest in batch) + `max_length` cap:

```python
# With HuggingFace tokenizer:
tokenizer(texts, padding="longest", truncation=True, max_length=512)
```

This gives you smaller average batch VRAM while capping the worst case.

### Pitfall 7: Saving All Checkpoints

**What goes wrong**: Each checkpoint saves a copy of model weights + optimizer states. Five checkpoints of a 3GB model = 15GB of disk I/O that may also cause VRAM spikes during serialization.

**Fix**: Limit checkpoints:

```python
training_args = TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # Keep only last 2 checkpoints
)
```

### Pitfall 8: CUDA Memory Fragmentation

**What goes wrong**: After many allocations/deallocations, VRAM becomes fragmented. You have 2GB free but cannot allocate a 1GB contiguous block.

**Fix**:

```bash
# Set before launching training:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This uses expandable segments (PyTorch 2.4+) which dramatically reduces fragmentation.

### Pitfall 9: Forgetting That Evaluation Doubles Memory Temporarily

**What goes wrong**: Training uses 90% VRAM, then evaluation (with different batch composition) causes OOM.

**Fix**: Use a smaller `per_device_eval_batch_size`, or set `torch_empty_cache_steps` to periodically free cached memory.

---

## Part 6: Quick Reference — Profiling Commands

### nvidia-smi (basic monitoring)
```bash
# One-shot
nvidia-smi

# Continuous monitoring (every 1 second)
nvidia-smi --loop=1

# Query specific metrics
nvidia-smi --query-gpu=gpu_name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

### nvitop (better monitoring)
```bash
pip install nvitop
nvitop  # Interactive TUI with per-process breakdown
```

### PyTorch memory functions
```python
# Current state
torch.cuda.memory_allocated()      # Currently allocated by tensors
torch.cuda.memory_reserved()       # Reserved by caching allocator (>= allocated)
torch.cuda.max_memory_allocated()  # Peak allocated since last reset

# Detailed breakdown
print(torch.cuda.memory_summary())

# Reset peak tracking (useful between experiments)
torch.cuda.reset_peak_memory_stats()

# Force free cached memory
torch.cuda.empty_cache()
```

### PyTorch Profiler with memory
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    # Run 1 training step
    pass

# Table view sorted by CUDA memory
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
```

### Memory snapshot (PyTorch 2.1+)
```python
# Start recording
torch.cuda.memory._record_memory_history()

# Run your training step(s)
train_step()

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)  # Stop recording

# Visualize at: https://pytorch.org/memory_viz
# Upload the .pickle file for interactive flame graph
```

---

## Part 7: Project-Specific Recommendations

For this project (Qwen3-ASR-1.7B LoRA fine-tuning on T4 15GB):

### Current Training Config Issues

The existing `train_standard_lora.py` uses `bf16=True` — this should be changed to `fp16=True` for T4 hardware. T4 is compute capability 7.5 (Turing), which has fp16 Tensor Cores but no native bf16 support.

### Recommended Training Arguments

```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,     # Try 4 first (up from current 2)
    gradient_accumulation_steps=4,     # Effective batch = 16
    gradient_checkpointing=True,       # Already enabled, keep it
    fp16=True,                         # Changed from bf16 for T4
    dataloader_num_workers=4,          # Currently 0 (default)
    dataloader_pin_memory=True,        # Currently False
    dataloader_persistent_workers=True,
    optim="adamw_torch_fused",         # Faster than default adamw_torch
    torch_empty_cache_steps=100,       # Periodic cache cleanup
    save_total_limit=2,                # Limit checkpoint disk usage
    remove_unused_columns=False,       # Already set
)
```

### VRAM Budget Estimation for T4 (15 GB)

With LoRA on Qwen3-ASR-1.7B:
- Frozen model weights (fp16): ~3.4 GB
- LoRA weights + gradients + optimizer: < 0.5 GB (tiny fraction of params)
- Audio encoder activations: ~1-3 GB (depends on audio length)
- LM activations (with grad checkpointing): ~2-4 GB (depends on batch/seq length)
- CUDA overhead + fragmentation: ~1-2 GB
- **Estimated total**: ~8-12 GB

This means batch_size=4 should fit comfortably on T4 with these optimizations. Test by increasing until peak VRAM reaches ~12-13 GB (80-85% of 15 GB).

---

## Sources

- [HuggingFace: Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/main/perf_train_gpu_one) — primary reference for Trainer optimization arguments
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — official PyTorch optimization techniques
- [PyTorch Lightning: Speed Up Training](https://lightning.ai/docs/pytorch/stable/advanced/speed.html) — DataLoader and training loop optimization
- [DeepSpeed ZeRO Documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html) — memory sharding stages
- [HuggingFace: Visualize and Understand GPU Memory](https://huggingface.co/blog/train_memory) — VRAM breakdown formulas
- [PyTorch Memory Snapshot Visualizer](https://pytorch.org/memory_viz) — interactive memory profiling
- [Lyceum: Predict PyTorch VRAM Usage](https://lyceum.technology/magazine/predict-vram-usage-pytorch-model/) — memory estimation formulas
- [NVIDIA Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html) — Tensor Core alignment requirements
