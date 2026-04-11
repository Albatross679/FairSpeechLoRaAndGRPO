# Technology Stack — v2.0 Attention Head Surgery

**Project:** LLM-ASR Fairness v2.0 — Hallucination Mitigation via Attention Head Surgery
**Target model:** Whisper-large-v3 (1.5B params; 32 decoder layers × 20 self-attention heads = 640 head positions)
**Reference:** Calm-Whisper (Wang et al., Interspeech 2025, arXiv 2505.12969)
**Researched:** 2026-04-11
**Overall confidence:** HIGH

## Executive Decision

**Use native PyTorch forward hooks on `WhisperAttention.out_proj` for per-head masking** (stable across transformers 4.x/5.x), OR pin `transformers<5` to use the built-in `decoder_head_mask` API. **Prefer hooks** — they survive future transformers upgrades and also support the per-head fine-tuning phase via `register_hook` on `.weight` gradients.

## Critical Finding: `decoder_head_mask` removed in transformers v5

Verified by reading `modeling_whisper.py` directly across 6 transformers releases:

| transformers | `layer_head_mask` refs | `decoder_head_mask` refs |
|---|---|---|
| v4.44.0 | 26 | 6 |
| v4.46.0 | 26 | 6 |
| v4.48.0 | 26 | 6 |
| v4.50.0 | 26 | 6 |
| v4.55.0 | 17 | 5 |
| v4.56.0 | 17 | 5 |
| main (~v5.5.3) | **0** | **0** |

In v5, `WhisperAttention.forward` was refactored to `**kwargs: Unpack[FlashAttentionKwargs]` and `layer_head_mask` was dropped. `WhisperDecoder.forward` no longer threads `head_mask` through. A fresh `pip install transformers` today would **silently ignore** any `decoder_head_mask` kwarg passed to `generate()`.

**`pyproject.toml` currently has `transformers>=4.44` with no upper bound** — this is a latent bug for this milestone.

## Two Viable Implementation Paths

### Path A — Pin `transformers<5` and use `decoder_head_mask`

```python
# Whisper-large-v3: 32 layers × 20 heads; mask head 7 in layer 3
mask = torch.ones(model.config.decoder_layers, model.config.decoder_attention_heads)
mask[3, 7] = 0.0
pred_ids = model.generate(features, decoder_head_mask=mask.to(device), ...)
```

**How it works (v4.44 source, line ≈538):** `attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights` — multiplies post-softmax attention weights by mask. Canonical "mask a head" operation and matches Calm-Whisper methodology.

**Caveat:** Providing `decoder_head_mask` auto-disables SDPA and falls back to eager attention:
> "WhisperModel is using WhisperSdpaAttention, but torch.nn.functional.scaled_dot_product_attention does not support output_attentions=True or layer_head_mask not None. Falling back to the manual attention"

For the 511-utterance diagnosis sweep this is fine. Must explicitly set `attn_implementation="eager"` in `from_pretrained` to avoid silent fallback bugs.

### Path B — Forward hooks on `out_proj` (RECOMMENDED)

Architecture-stable across transformers 4.x and 5.x. Zero version pin needed.

```python
def pre_hook(head_idx, num_heads):
    def hook(module, args):
        x = args[0]  # (bsz, tgt, embed)
        bsz, tgt, embed = x.shape
        head_dim = embed // num_heads
        x = x.clone().view(bsz, tgt, num_heads, head_dim)
        x[:, :, head_idx, :] = 0.0
        return (x.view(bsz, tgt, embed),)
    return hook

layer = model.model.decoder.layers[layer_idx]
handle = layer.self_attn.out_proj.register_forward_pre_hook(
    pre_hook(head_idx, layer.self_attn.num_heads)
)
```

**Why this is algebraically correct:** `out_proj` is `Linear(embed_dim, embed_dim)` equivalent to `sum_h (W_h @ head_h)`. Zeroing `head_h` at `out_proj` input removes its contribution exactly. Hooking *after* `self_attn` is wrong because `out_proj` already mixed the heads.

**Stability:** `self.num_heads`, `self.head_dim`, and `self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)` are identical in transformers v4.44 and main (v5). Hooks survive upgrades.

## Per-Head Fine-Tuning

**No library** (PEFT, TransformerLens, nnsight, transformers itself) provides "train only head K of layer L" as a first-class API. Calm-Whisper's description ("freeze everything except heads 1, 6, 11") is a manual PyTorch operation.

**No public Calm-Whisper code:** Exhaustive search of arXiv 2505.12969, ISCA Interspeech 2025 (`wang25b_interspeech`), author affiliations, and Semantic Scholar turned up zero code repositories. Re-implement; ~200 lines total.

### Recommended Technique — Gradient Masking Hooks

`q_proj`, `k_proj`, `v_proj` rows `[h*head_dim : (h+1)*head_dim]` are head `h`. `out_proj` columns are the same slices. `requires_grad` cannot be set per-slice, so mask gradients:

```python
def grad_mask_hook(trainable_head_idxs, num_heads, head_dim, is_row_slice):
    def hook(grad):
        mask = torch.zeros_like(grad)
        for h in trainable_head_idxs:
            if is_row_slice:  # q/k/v_proj: heads are rows
                mask[h*head_dim:(h+1)*head_dim, :] = 1.0
            else:  # out_proj: heads are columns
                mask[:, h*head_dim:(h+1)*head_dim] = 1.0
        return grad * mask
    return hook

for layer_idx, heads in trainable_heads_per_layer.items():
    sa = model.model.decoder.layers[layer_idx].self_attn
    sa.q_proj.weight.register_hook(grad_mask_hook(heads, sa.num_heads, sa.head_dim, True))
    sa.k_proj.weight.register_hook(grad_mask_hook(heads, sa.num_heads, sa.head_dim, True))
    sa.v_proj.weight.register_hook(grad_mask_hook(heads, sa.num_heads, sa.head_dim, True))
    sa.out_proj.weight.register_hook(grad_mask_hook(heads, sa.num_heads, sa.head_dim, False))

# Freeze everything else, unfreeze only the relevant projection weights
for p in model.parameters(): p.requires_grad = False
for layer_idx in trainable_heads_per_layer:
    for proj in ["q_proj","k_proj","v_proj","out_proj"]:
        getattr(model.model.decoder.layers[layer_idx].self_attn, proj).weight.requires_grad = True
```

**Validation contract:** after one optimizer step, assert `||Δw||==0` for all non-trained head slices. This MUST be a unit test.

**Note on LoRA per-head:** PEFT's `target_modules` applies LoRA to the entire `nn.Linear`, not per-head. No per-head rank allocation in any released PEFT version.

## Interpretability Libraries — None Usable

| Library | Version | Whisper support | Verdict |
|---|---|---|---|
| **TransformerLens** | 2.18.0 | NOT supported (verified: 237 models in `OFFICIAL_MODEL_NAMES`, zero whisper). Hard-coded for GPT-style decoder-only. | Do not use |
| **nnsight** | 0.6.3 | Wraps any `nn.Module` but adds trace/envoy abstraction over raw hooks. | Not needed |
| **captum** | 0.8.0 | Feature attribution (IG, LRP), not head ablation. | Wrong tool |

**Recommendation:** Raw `torch.nn.Module.register_forward_pre_hook` + `register_hook` on `.weight`. Zero new dependencies for per-head analysis and fine-tuning.

## VAD Library Selection

| Library | Version | Approach | Accuracy (TPR @ 5% FPR) | Verdict |
|---|---|---|---|---|
| **silero-vad** | 6.2.1 | PyTorch DNN | **87.7%** | **Recommended** |
| **webrtcvad-wheels** | 2.0.14 | GMM (C) | 50% | Too noisy for silence-injection use case |
| **pyannote-audio** | — | PyTorch DNN | Highest, but GPU-required, slow on short files | Overkill |
| **speechbrain VAD** | — | PyTorch DNN | Heavier dep, worse DX | Not chosen |

**Pick: silero-vad 6.2.1.** webrtcvad's 50% TPR would silently pass through injected silence in the perturbation experiments, defeating the VAD ablation. Install via `pip install silero-vad==6.2.1` (preferred over `torch.hub.load` for reproducibility).

## Recommended `pyproject.toml` Changes

```toml
dependencies = [
    # --- CHANGED ---
    "transformers>=4.48,<5.0",  # v5 removed layer_head_mask; pin <5 OR rely on hook-based Path B
    # --- NEW ---
    "silero-vad>=6.2,<7.0",     # DNN VAD for silence-injection preprocessing ablation
]
```

**Total v2.0 stack delta:** one new package (silero-vad) + one version pin change. Everything else uses `torch`, `transformers`, and `peft` already in the project.

## Rejected Additions (Explicit)

| Package | Reason |
|---|---|
| TransformerLens | No Whisper support |
| nnsight | Adds abstraction layer over raw hooks for no benefit |
| captum | Feature attribution, not head ablation |
| whisper-flash-attention | FA2 disables head_mask path entirely |
| pyannote-audio (for VAD) | GPU-hungry, slow on short utterances, overkill |
| webrtcvad | 50% TPR too low for silence-injection regime |
| bitsandbytes 4-bit | Whisper-large-v3 (1.5B) fits in FP16 with 49 GB available |
| TRL | v2.0 is supervised FT of head slices, no RL |

## Integration with Existing `run_inference.py`

Three minimal modifications to `scripts/run_inference.py`:

1. **`load_whisper` (~line 307):** Force eager attention:
   ```python
   model = WhisperForConditionalGeneration.from_pretrained(
       hf_id,
       torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
       attn_implementation="eager",   # REQUIRED for layer_head_mask; SDPA silently ignores it
   ).to(args.device).eval()
   ```

2. **`infer_whisper` (~line 324):** Accept optional `decoder_head_mask` kwarg:
   ```python
   gen_kwargs = dict(max_new_tokens=440, language="en", task="transcribe")
   if loaded.get("decoder_head_mask") is not None:
       gen_kwargs["decoder_head_mask"] = loaded["decoder_head_mask"]
   pred_ids = model.generate(features, **gen_kwargs)
   ```

3. **New file `scripts/head_analysis.py`** (or `scripts/run_head_masking_sweep.py` per ARCHITECTURE.md): Thin wrapper that sweeps masks over 20×32=640 positions, calls `infer_whisper` once per mask, writes results CSV compatible with existing `compute_*_metrics.py`.

Per-head fine-tuning needs a **separate new script** `scripts/training/finetune_heads.py` (not supported by existing training scripts).

## Gaps and Open Questions

1. **Whisper-large-v3 layer×head geometry:** 32 layers × 20 heads = 640 positions. PROJECT.md's "20 heads" is per-layer. Calm-Whisper masks "heads 1, 6, 11" in each layer → 3 × 32 = 96 trainable slices. Diagnosis sweep of all 640 is ~5 min/sweep × 511 utterances on a 49 GB GPU — feasible but ARCHITECTURE phase should decide if per-layer or flat sweep.
2. **`generate()` threading `decoder_head_mask` through cached KV steps:** Not explicitly documented in any transformers version. **Smoke test required at phase start** — pass a zero-mask on one head and verify output differs from unmasked at decoded position > 1 (not just position 1).
3. **VAD threshold tuning:** silero-vad default threshold is 0.5. Per-severity tuning (25/50/75% injection) may be needed.
4. **Accent-diverse fine-tuning data:** PROJECT.md says "accent-diverse audio" but constraint is "cannot collect new labeled data." Presumably reuses CV24 training split with accent labels. Data-prep concern, not stack.

## Confidence Assessment

| Area | Level | Reason |
|---|---|---|
| `decoder_head_mask` in v4.x | HIGH | Source-verified across 6 versions |
| `decoder_head_mask` removed in v5 | HIGH | Grep on main branch: zero hits |
| SDPA fallback with head_mask | HIGH | Direct quote from v4.44 source |
| `out_proj` pre-hook correctness | HIGH | Algebraic; `num_heads`/`head_dim`/`out_proj` stable |
| Gradient masking technique | HIGH | Standard PyTorch; validation test covers it |
| silero-vad pick | HIGH | Picovoice 2025, pyannote #604, py-webrtcvad #68 corroborate |
| No TransformerLens Whisper | HIGH | Source-verified model list |
| Calm-Whisper no public code | HIGH | Exhaustive search returned nothing |
| PEFT lacks per-head | HIGH | `target_modules` operates on whole `nn.Linear` |

## Sources

- Calm-Whisper: https://arxiv.org/abs/2505.12969
- Calm-Whisper ISCA: https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.html
- HuggingFace Whisper docs: https://huggingface.co/docs/transformers/model_doc/whisper
- transformers v4.44 Whisper source: https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/whisper/modeling_whisper.py
- transformers main Whisper source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
- TransformerLens model list: https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py
- nnsight 0.6 docs: https://nnsight.net/notebooks/tutorials/get_started/walkthrough/
- silero-vad: https://github.com/snakers4/silero-vad
- Picovoice VAD benchmark 2025: https://picovoice.ai/blog/best-voice-activity-detection-vad-2025/
- HF transformers #27044 (head pruning): https://github.com/huggingface/transformers/issues/27044
