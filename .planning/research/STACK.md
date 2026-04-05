# Technology Stack: GRPO Fairness-Aware ASR Fine-Tuning

**Project:** GRPO Fairness-Aware ASR Fine-Tuning for Qwen3-ASR-1.7B
**Researched:** 2026-04-05
**Hardware Constraint:** Single NVIDIA RTX A4000 (16 GB VRAM)

## Executive Decision

**Use a custom GRPO training loop built on PyTorch + PEFT, not TRL's GRPOTrainer.** TRL's GRPOTrainer is designed for text-only LLMs (prompt-in, text-out). Qwen3-ASR-1.7B is an encoder-decoder audio model (audio-in, text-out) with a non-standard `generate()` interface requiring audio features as input. Forcing it into TRL's text-centric API would require extensive monkey-patching of generation, tokenization, and data collation. A custom ~200-line GRPO loop using the well-documented algorithm (generate completions, compute rewards, normalize advantages, clipped surrogate loss) is cleaner, more debuggable, and fully under your control for the fairness reward.

**Confidence: HIGH** — TRL docs confirm text/conversation format assumptions; Qwen3-ASR uses audio feature input that does not fit this API.

---

## Recommended Stack

### Core Training Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch | 2.2+ (match existing) | Training loop, autograd, GPU compute | Already in stack; custom GRPO loop needs raw autograd control | HIGH |
| PEFT | 0.18.x (latest stable) | LoRA adapter injection and management | HuggingFace standard for parameter-efficient fine-tuning; seamless integration with any `nn.Module`; manages adapter save/load/merge | HIGH |
| bitsandbytes | 0.45+ | 4-bit NF4 quantization of frozen base weights (QLoRA) | Cuts base model memory from ~3.4 GB (FP16) to ~0.85 GB, critical for 16 GB VRAM budget | HIGH |
| Transformers | 4.48+ (match existing) | Model loading, tokenizer, `from_pretrained()` | Already in stack; PEFT and bitsandbytes hook into Transformers models | HIGH |
| Accelerate | 1.3+ | Mixed-precision context, gradient accumulation helpers | Lightweight; avoids manual `torch.cuda.amp` boilerplate | MEDIUM |

### LoRA Configuration (PEFT)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (rank) | 16 | Standard for 1-2B models; 8 is too constrained for fairness signal, 32 adds memory for marginal gain |
| `lora_alpha` | 32 | Alpha = 2r is the standard scaling convention |
| `lora_dropout` | 0.05 | Mild regularization; matches ASR LoRA literature (CoVoGER EMNLP 2025) |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` on LLM decoder only | Freeze entire AuT encoder (300M params) + projector; train only decoder attention. This is where linguistic/fairness bias lives |
| `task_type` | `CAUSAL_LM` | Qwen3-1.7B decoder is causal LM |
| `modules_to_save` | `[]` (none) | Keep projector frozen; LoRA on decoder attention is sufficient for fairness intervention |

**Trainable parameters estimate:** ~3.5M out of ~2B total (~0.17%), well within memory budget.

**Confidence: HIGH** for PEFT/LoRA approach. **MEDIUM** for exact target modules — the official Qwen3-ASR-Finetuning repo provides an SFT script but no LoRA config. The decoder attention targets follow Qwen3 LoRA conventions from Qwen documentation and the CoVoGER paper (r=8, alpha=16 on Qwen2.5).

### Quantization (QLoRA Path)

| Technology | Config | Purpose |
|------------|--------|---------|
| bitsandbytes | `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16` | Quantize frozen LLM decoder weights to 4-bit NF4. Compute remains in BF16 for numerical stability |

### Reward Computation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| jiwer | existing | WER computation per utterance | Already in stack; needed for accuracy component of reward |
| NumPy | existing | Group-level fairness metric aggregation | Already in stack; fast vectorized mean/std across demographic groups |

No new libraries needed. The composite reward `R = (1-lambda)(1-WER) + lambda(-|WER_g - WER_mean|)` is computed from jiwer WER scores grouped by demographic labels already in the CSV manifests.

### Data Pipeline

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| datasets (HF) | existing | Load Fair-Speech and Common Voice 24 | Already in stack; handles audio column, demographic metadata |
| torchaudio | existing | Audio loading, resampling to 16kHz | Already in stack |
| soundfile | existing | Audio I/O fallback | Already in stack |

### Experiment Tracking and Baselines

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| wandb | 0.19+ | Training curves, Pareto frontier logging, hyperparameter tracking | Standard for ML experiment tracking; free for academic use; better than TensorBoard for comparing lambda sweep runs | MEDIUM |
| CSV logging | built-in | Fallback metric logging | Matches existing pipeline pattern (CSV manifests as data contracts) |

### Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Matplotlib | existing | Pareto frontier plots, training curves | Already in stack |
| Seaborn | existing | Fairness metric heatmaps | Already in stack |

---

## VRAM Budget Analysis (16 GB RTX A4000)

### Option A: QLoRA (Recommended)

| Component | VRAM (GB) | Notes |
|-----------|-----------|-------|
| AuT encoder (300M, FP16, frozen) | ~0.6 | Frozen, no gradients stored |
| Projector (frozen, FP16) | ~0.05 | Small linear layer |
| LLM decoder (1.7B, NF4 quantized, frozen) | ~0.85 | 4-bit NF4 via bitsandbytes |
| LoRA adapters (3.5M params, BF16) | ~0.007 | Negligible |
| LoRA gradients + optimizer states (AdamW 8-bit) | ~0.03 | 8-bit AdamW via bitsandbytes cuts optimizer memory ~2x |
| Activations / gradient checkpointing | ~2-4 | Depends on sequence length; gradient checkpointing keeps this bounded |
| KV cache for G=4 completions per prompt | ~1-3 | 4 completions x ~200 tokens each; this is the GRPO-specific cost |
| Audio features (batch of Fbank) | ~0.2 | 128-dim Fbank at 12.5 Hz, small |
| PyTorch CUDA overhead | ~1.0 | Allocator fragmentation, kernels |
| **Total estimated** | **~6-10 GB** | **Fits comfortably in 16 GB** |

**Headroom: ~6-10 GB free.** This allows G=4 completions per prompt (the GRPO group size), batch_size=2-4, and max_completion_length up to ~300 tokens.

### Option B: FP16 LoRA (Without Quantization)

| Component | VRAM (GB) | Notes |
|-----------|-----------|-------|
| AuT encoder (300M, FP16, frozen) | ~0.6 | Same |
| Projector (frozen, FP16) | ~0.05 | Same |
| LLM decoder (1.7B, FP16, frozen) | ~3.4 | Full precision frozen weights |
| LoRA adapters + gradients + optimizer | ~0.07 | Still small |
| Activations / gradient checkpointing | ~2-4 | Same |
| KV cache for G=4 completions | ~1-3 | Same |
| Audio features + CUDA overhead | ~1.2 | Same |
| **Total estimated** | **~8-12 GB** | **Fits but tighter** |

**Headroom: ~4-8 GB free.** Still feasible. May need G=2 or smaller batch if sequences are long.

### Recommendation

**Start with Option B (FP16 LoRA)** for simplicity — avoids bitsandbytes quantization complexity and potential numerical issues. The 1.7B decoder in FP16 is only 3.4 GB. Fall back to Option A (QLoRA) only if you hit OOM with G=4 and batch_size=2.

**Confidence: MEDIUM** — These are calculated estimates, not measured. Actual VRAM depends on Qwen3-ASR's specific memory allocation patterns, which are not documented. The estimates are conservative (high-side) to account for unknowns.

### Critical GRPO Memory Parameters

| Parameter | Recommended | Impact |
|-----------|-------------|--------|
| `num_generations` (G) | 4 | Each generation multiplies KV cache. Start at 4, reduce to 2 if OOM |
| `max_completion_length` | 200 tokens | ASR transcriptions are short; 200 tokens covers most utterances |
| `per_device_batch_size` | 2 | With G=4, effective batch = 8 completions per step |
| `gradient_accumulation_steps` | 8 | Effective batch = 16 prompts = 64 completions. Sufficient for stable GRPO advantage estimation |
| `gradient_checkpointing` | True | Essential — trades ~30% speed for ~50% activation memory reduction |

---

## What NOT to Use

### TRL GRPOTrainer — Do NOT use for this project
**Why not:** TRL's GRPOTrainer expects text prompts and text completions. It calls `model.generate()` with tokenized text input. Qwen3-ASR requires audio features (Fbank) as encoder input and produces text output. The data collation, generation, and log-probability computation all assume text-in/text-out. Adapting it would require subclassing and overriding `_generate_and_score_completions`, `_get_per_token_logps`, and the data collator — essentially rewriting the trainer. A clean custom loop is less code and more maintainable.

**What to do instead:** Write a custom GRPO training loop (~200 lines) that:
1. Loads audio, runs encoder to get features
2. Generates G completions per audio sample using the decoder
3. Computes per-completion WER and fairness reward
4. Normalizes advantages within each group
5. Computes clipped surrogate policy gradient loss on decoder log-probs
6. Updates only LoRA parameters

### OpenRLHF — Do NOT use
**Why not:** Designed for large-scale multi-GPU RLHF with Ray. Massive overkill for single-GPU LoRA training of a 1.7B model. Heavy dependency footprint, complex setup, no ASR support.

### Unsloth — Do NOT use (despite memory benefits)
**Why not:** Unsloth provides excellent memory optimization for GRPO (up to 8x reduction), but it only supports standard text LLMs. Qwen3-ASR is not in Unsloth's supported model list. The audio encoder + projector + decoder architecture is incompatible with Unsloth's kernel optimizations.

### loralib — Do NOT use
**Why not:** The original LoRA library by Microsoft. PEFT supersedes it with better HuggingFace integration, more methods, active maintenance (0.18.x vs loralib's last update in 2023), and automatic `target_modules` resolution.

### DeepSpeed — Do NOT use (single GPU)
**Why not:** DeepSpeed ZeRO is for multi-GPU memory sharding. On a single GPU, it adds overhead without benefit. Standard PyTorch + gradient checkpointing is sufficient.

### Full Fine-Tuning — Do NOT use
**Why not:** 1.7B decoder + 300M encoder = 2B parameters. Full fine-tuning requires ~24 GB VRAM minimum (model + gradients + optimizer states in FP16). Exceeds 16 GB A4000. LoRA reduces trainable params to ~3.5M.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| GRPO framework | Custom PyTorch loop | TRL GRPOTrainer | Text-only API; Qwen3-ASR is audio-in model |
| LoRA library | PEFT 0.18.x | loralib | Unmaintained; worse HF integration |
| RL framework | Custom GRPO | OpenRLHF | Overkill for single-GPU; no ASR support |
| Memory optimization | bitsandbytes QLoRA (fallback) | Unsloth | No Qwen3-ASR support |
| Experiment tracking | wandb | TensorBoard | wandb better for parallel lambda sweeps |
| Optimizer | AdamW (torch or 8-bit bnb) | SGD | AdamW is standard for transformer LoRA; adaptive LR crucial for RL stability |

---

## Installation

```bash
# New dependencies (add to existing environment)
pip install peft>=0.18.0
pip install bitsandbytes>=0.45.0  # only if using QLoRA path
pip install wandb>=0.19.0         # optional, for experiment tracking

# Already in environment (verify versions)
# torch >= 2.2.0
# transformers >= 4.48.0
# torchaudio
# jiwer
# datasets
# numpy, pandas, matplotlib, seaborn
```

No new heavy frameworks. Three pip packages maximum (peft, bitsandbytes, wandb). The GRPO algorithm is implemented in project code, not imported from a library.

---

## Key Implementation Notes

### GRPO Algorithm (Custom Implementation)

The GRPO algorithm is well-documented and straightforward to implement:

```python
# Pseudocode for one training step
for batch in dataloader:
    audio, labels, demographics = batch
    
    # 1. Generate G completions per audio sample
    with torch.no_grad():
        encoder_out = model.encoder(audio)  # frozen
        completions = [model.decoder.generate(encoder_out) for _ in range(G)]
    
    # 2. Compute rewards
    wers = [compute_wer(comp, ref) for comp in completions]
    fairness = compute_fairness_penalty(wers, demographics)
    rewards = (1-lam) * (1 - wers) + lam * fairness
    
    # 3. Normalize advantages within group
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # 4. Compute policy gradient with clipping
    logprobs = model.decoder.forward(encoder_out, completions)  # with LoRA
    old_logprobs = logprobs.detach()  # or from reference model
    ratio = (logprobs - old_logprobs).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    
    # 5. Update LoRA parameters only
    loss.backward()
    optimizer.step()
```

### Qwen3-ASR Architecture Specifics

- **AuT encoder** (300M params): Processes 128-dim Fbank features at 12.5 Hz with 8x downsampling. Uses FlashAttention with dynamic 1-8s windows.
- **Projector**: Linear layer aligning encoder output to decoder input space.
- **LLM decoder** (Qwen3-1.7B): Standard causal LM with 1024 hidden dim, 16 attention heads.
- **LoRA targets**: Apply LoRA to decoder attention only (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Encoder and projector stay frozen — the fairness intervention targets the decoder's language modeling bias, not acoustic features.

### Reference Policy for KL

For GRPO on a 1.7B model with LoRA, the reference policy is the frozen base decoder (without LoRA adapters). PEFT makes this trivial: `model.disable_adapter_layers()` gives the reference policy, `model.enable_adapter_layers()` gives the current policy. No need to load a second copy of the model.

**Confidence: HIGH** — PEFT documentation confirms `disable_adapter_layers()` API.

---

## Sources

### Official Documentation (HIGH confidence)
- [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) — confirmed text-only API, custom reward function interface
- [TRL v1.0.0 release](https://pypi.org/project/trl/) — latest version, March 2026
- [PEFT GitHub](https://github.com/huggingface/peft) — v0.18.1, January 2026
- [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1) — architecture details (AuT 300M + Qwen3-1.7B decoder)
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR) — finetuning directory with SFT script
- [Qwen LoRA Fine-tuning docs](https://www.mintlify.com/QwenLM/Qwen/finetuning/lora) — LoRA configuration for Qwen models
- [bitsandbytes GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes) — 4-bit quantization

### Research Papers (HIGH confidence)
- [GRPO for Speech Recognition (ASRU 2025)](https://arxiv.org/abs/2509.01939) — up to 18.4% relative WER improvement with GRPO on ASR
- [Advancing Speech Understanding with GRPO](https://arxiv.org/abs/2509.16990) — GRPO applied to speech-aware LMs
- [CoVoGER (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.320.pdf) — LoRA config for Qwen2.5 ASR (r=8, alpha=16)
- [RL for LLM-based ASR and TTS](https://arxiv.org/pdf/2509.18569) — 5.3% relative WER improvement

### Community/Blog Sources (MEDIUM confidence)
- [GRPO VRAM Requirements article](https://ghost.oxen.ai/grpo-vram-requirements-for-the-gpu-poor/) — memory estimation for GRPO training
- [Modal VRAM estimation](https://modal.com/blog/how-much-vram-need-fine-tuning) — general VRAM calculation methodology
- [Qwen3-ASR fine-tuning issue #68](https://github.com/QwenLM/Qwen3-ASR/issues/68) — community discussion on fine-tuning challenges
