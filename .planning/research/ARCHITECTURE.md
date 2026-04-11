# Architecture Research — v2.0 Attention Head Surgery

**Domain:** Hallucination mitigation on Whisper-large-v3 via per-head attention analysis, decode ablation, VAD preprocessing, and selective head fine-tuning
**Researched:** 2026-04-11
**Confidence:** HIGH (grounded in existing pipeline contract + Calm-Whisper methodology)
**Scope note:** This replaces v1.0 (GRPO) architecture. v1.0 LoRA training infrastructure at `scripts/training/` is NOT reused verbatim (target model changed from Qwen3-ASR to Whisper-large-v3), but its patterns (manifest-driven config, `outputs/<run>/adapter/`, evaluation bridge) ARE reused.

---

## 1. Integration Principle

The existing pipeline is a **linear CSV-manifest data-flow pipeline** with five stages:

```
prepare_*.py → generate_perturbations.py → run_inference.py →
compute_*_metrics.py → generate_*_plots.py
```

Every stage is a standalone script. Stages communicate only through files on disk (CSV manifests, prediction CSVs, analysis JSONs, PNG figures). The only "registry" coupling is the `MODEL_REGISTRY` dict inside `run_inference.py`.

**v2.0 rule:** every new capability is a **new script** that slots into this pipeline, OR a **minimal extension** of `run_inference.py` via a new model key in `MODEL_REGISTRY`. We do NOT add config flags that reshape `run_inference.py`'s control flow. We do NOT invent new data contracts — we extend CSV schemas additively.

**Why not a `--head_mask` flag on `run_inference.py`?**
- Blows up argument matrix: `--head_mask`, `--beam_size`, `--repetition_penalty`, `--length_penalty`, `--no_repeat_ngram`, `--vad` etc. would all need to multiplex through one entry point.
- Couples diagnostic experiments (20 sweeps over heads) with routine benchmarking.
- Breaks the "one script = one purpose" convention of the existing 28 scripts.

**Why not a single monolithic `run_head_experiments.py`?**
- Couples diagnosis (per-head sweep), ablation (decode params), preprocessing (VAD), and training (head fine-tune) into one script — fights the stage-oriented contract.

**Adopted approach:** three new inference-side scripts, one new training script, zero changes to `compute_*_metrics.py`, and one additive column on prediction CSVs (`experiment_tag`).

---

## 2. System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                      EXISTING PIPELINE (unchanged)                      │
│  prepare_dataset.py  generate_perturbations.py  compute_fairness_*.py   │
│  error_decomposition.py  whisper_hallucination_analysis.py              │
│  generate_all_plots.py                                                  │
└──────┬─────────────────────────┬──────────────────────────┬─────────────┘
       │ reuse manifests          │ reuse prediction schema  │ reuse metrics
       ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         v2.0 NEW SUBSYSTEM                               │
│                                                                          │
│  ┌───────────────────────────┐   ┌──────────────────────────────────┐  │
│  │ INFERENCE SIDE            │   │ TRAINING SIDE                     │  │
│  │                           │   │                                   │  │
│  │ whisper_head_hooks.py     │   │ train_whisper_heads.py            │  │
│  │  (shared lib — masking +  │   │  (selective head fine-tune,       │  │
│  │   attention capture hooks)│   │   reuses LoRA/PEFT pattern from   │  │
│  │         │                 │   │   v1.0, but trains subset of      │  │
│  │         ▼                 │   │   decoder.layers[i].self_attn)    │  │
│  │ run_head_masking_sweep.py │   │         │                         │  │
│  │  (diagnosis loop over     │   │         ▼                         │  │
│  │   all 20 heads, top-k     │   │  outputs/head-finetune/           │  │
│  │   identification)         │   │   run_<tag>/                      │  │
│  │                           │   │     adapter/ or state_dict.pt     │  │
│  │ run_decode_ablation.py    │   │     config.json                   │  │
│  │  (beam / rep pen /        │   │     wandb/                        │  │
│  │   length pen / ngram)     │   │                                   │  │
│  │                           │   │         │                         │  │
│  │ apply_energy_vad.py       │   │         ▼                         │  │
│  │  (preprocessing — writes  │   │  Register fine-tuned checkpoints  │  │
│  │   new manifest + audio    │   │  as new MODEL_REGISTRY entries    │  │
│  │   clips)                  │   │  → evaluated via run_inference.py │  │
│  └────────────┬──────────────┘   └──────────────────────────────────┘  │
│               │                                                          │
│               ▼ (all four emit the same prediction CSV schema)           │
│  outputs/head-surgery/<experiment_tag>/predictions_*.csv                 │
│               │                                                          │
└───────────────┼──────────────────────────────────────────────────────────┘
                │
                ▼  (existing metrics code, UNCHANGED)
        compute_fairness_metrics.py  error_decomposition.py
        whisper_hallucination_analysis.py → figures
```

---

## 3. Component Responsibilities

| Component | File | Responsibility | Integration Point |
|-----------|------|----------------|-------------------|
| Head hook library | `scripts/whisper_head_hooks.py` | Shared module: `attach_head_mask_hooks(model, layer_idx, head_idx)`, `capture_attention_weights(model)`, unhook helpers. Imported by all three inference scripts. | Consumed by `run_head_masking_sweep.py`, `run_decode_ablation.py`, `train_whisper_heads.py`. NOT a script. |
| Per-head sweep | `scripts/run_head_masking_sweep.py` | Loads Whisper-large-v3 once, loops over all 20 decoder self-attention heads, masks one head at a time, runs inference on the 511-utterance Indian CV subset, writes one prediction CSV per mask condition. Emits `top_k_heads.json` summary. | Reads manifest from `outputs/manifests/cv_indian_511.csv`. Writes to `outputs/head-surgery/diagnosis/`. |
| Decode ablation | `scripts/run_decode_ablation.py` | Loads Whisper-large-v3 once, loops over a grid of `{beam_size, repetition_penalty, length_penalty, no_repeat_ngram_size}`, writes one prediction CSV per grid cell. No head masking. | Same manifest input. Writes to `outputs/head-surgery/decode-ablation/`. |
| Energy VAD preprocessing | `scripts/apply_energy_vad.py` | Reads an input audio manifest, computes frame-level RMS energy, trims low-energy regions (or splits into segments), writes new audio files + new manifest pointing to trimmed clips. | Slots **before** `run_inference.py` (not after). Writes `outputs/vad/<manifest_name>/` with new `audio_path` column. Composable with perturbation pipeline. |
| Selective head fine-tune | `scripts/train_whisper_heads.py` | Loads Whisper-large-v3, freezes all params, unfreezes only the `q_proj`/`k_proj`/`v_proj`/`o_proj` slices of the identified top-k heads (or wraps them in a LoRA adapter restricted to those modules), trains on accent-diverse manifest, saves checkpoint + W&B logs. | Reads `top_k_heads.json` from diagnosis step. Writes to `outputs/head-finetune/run_<tag>/`. |
| Fine-tuned model registration | `scripts/run_inference.py` (minimal edit) | Add entries like `"whisper-large-v3-hf-tuned": {..., "type": "whisper", "checkpoint": "outputs/head-finetune/run_<tag>"}`. `load_whisper()` gains an optional checkpoint path branch. | THE ONLY EDIT to existing code. ≤15 lines. |

---

## 4. Project Structure (new files only)

```
scripts/
├── whisper_head_hooks.py              # NEW — shared hook library (no __main__)
├── run_head_masking_sweep.py          # NEW — diagnosis driver
├── run_decode_ablation.py             # NEW — decoding hyperparameter grid
├── apply_energy_vad.py                # NEW — preprocessing, manifest-in/manifest-out
├── train_whisper_heads.py             # NEW — selective head fine-tune
└── run_inference.py                   # MODIFIED — add 1-3 MODEL_REGISTRY entries
                                       #           and a checkpoint-path branch in
                                       #           load_whisper()

outputs/
├── manifests/
│   └── cv_indian_511.csv              # NEW — pre-filtered diagnosis subset
├── head-surgery/
│   ├── diagnosis/
│   │   ├── predictions_whisper_large_v3_mask_L3H7.csv
│   │   ├── predictions_whisper_large_v3_mask_L3H0.csv
│   │   ├── ... (20 files)
│   │   ├── predictions_whisper_large_v3_baseline.csv
│   │   └── top_k_heads.json           # {"top_k": [{"layer":3,"head":7,"ins_reduction":0.62},...]}
│   ├── decode-ablation/
│   │   ├── predictions_whisper_large_v3_beam5_rep1.2_len1.0_ngram3.csv
│   │   └── ablation_summary.json
│   └── vad-eval/
│       └── predictions_whisper_large_v3_vad_silence25pct.csv
├── vad/
│   └── cv_indian_511_vad/             # trimmed clips + new manifest
│       ├── audio/*.wav
│       └── manifest.csv
└── head-finetune/
    └── run_<tag>/
        ├── adapter/                    # if LoRA-wrapped
        ├── state_dict.pt               # if direct unfreeze
        ├── config.json                 # {top_k_heads, lr, epochs, train_data, ...}
        ├── wandb/
        └── eval/                       # predictions CSVs under the fine-tuned model key
```

### Rationale

- **`whisper_head_hooks.py` is a library, not a script.** Every piece of v2.0 code that touches decoder attention (masking, capture, fine-tune) needs the same hook mechanics. Centralizing avoids three copies of fragile `register_forward_pre_hook` logic.
- **`outputs/head-surgery/` is a new subtree** disjoint from `outputs/commonvoice/` and `outputs/fairspeech/`. This keeps diagnosis artifacts from polluting the canonical benchmark results directory.
- **Fine-tuned checkpoints live under `outputs/head-finetune/run_<tag>/`**, mirroring v1.0's `outputs/standard-lora/` convention so the evaluation bridge pattern transfers.
- **The 511-utterance Indian-accent manifest is built once** (`outputs/manifests/cv_indian_511.csv`) and reused across all diagnosis + ablation + fine-tune-eval runs — single source of truth for "the hallucination probe set".

---

## 5. Key Architectural Patterns

### Pattern 1: Hook library, not a model subclass

**What:** `whisper_head_hooks.py` exposes `attach_head_mask_hooks(model, masks)` and `detach_hooks(handles)`. It registers `forward_pre_hook`s on `model.model.decoder.layers[i].self_attn` that zero out the corresponding head's `q_proj`/`o_proj` slice (or multiply attention weights by a mask in the `scaled_dot_product_attention` path).

**When:** Always, for both diagnosis and fine-tune. Never subclass `WhisperForConditionalGeneration`.

**Why:** HuggingFace model classes are brittle to subclass (many generation-path methods). Hooks are idempotent, removable, and do not require re-`to("cuda")`. Matches Calm-Whisper's reference implementation style.

**Shape:**
```python
# whisper_head_hooks.py
def attach_head_mask_hooks(model, masks):
    """masks: list of (layer_idx, head_idx) to zero out in decoder self-attention."""
    handles = []
    for layer_idx, head_idx in masks:
        attn = model.model.decoder.layers[layer_idx].self_attn
        handles.append(attn.register_forward_pre_hook(_make_mask_hook(head_idx)))
    return handles

def detach_hooks(handles):
    for h in handles:
        h.remove()
```

**Trade-offs:**
- Pro: Zero changes to Whisper class, removable per-utterance, composable with LoRA later.
- Con: Tied to HF's current decoder attention module layout — breaks if `transformers` internal names change. Pin version in `pyproject.toml`.

### Pattern 2: Load once, sweep many

**What:** Each new inference script loads Whisper-large-v3 **once** in a single Python process, then loops over the experimental axis (head index, decode config, etc.), attaching/detaching hooks between runs. One prediction CSV per experiment cell.

**When:** Diagnosis sweep (20 runs × 511 utts), decode ablation (~30-60 grid cells × 511 utts).

**Why:** Whisper-large-v3 cold-load is ~30s; loading 20 times wastes ~10 minutes and VRAM fragmentation risks. Single-process matches how `run_inference.py` currently handles multi-perturbation runs via `--perturbations` comma-list.

**Shape:**
```python
# run_head_masking_sweep.py
loaded = load_whisper(args)  # from run_inference import load_whisper
writer = None
for layer_idx in range(N_DECODER_LAYERS):
    for head_idx in range(N_HEADS_PER_LAYER):
        tag = f"mask_L{layer_idx}H{head_idx}"
        handles = attach_head_mask_hooks(loaded["model"], [(layer_idx, head_idx)])
        writer = IncrementalCSVWriter(output_path(tag), df, ...)
        infer_whisper(df, args, loaded, writer=writer)
        detach_hooks(handles)
# final baseline pass with no hooks
infer_whisper(df, args, loaded, writer=baseline_writer)
```

**Reuse:** `load_whisper()` and `infer_whisper()` are already factored out of `run_inference.py` (lines 307-378) — we call them directly. No copy-paste.

**Trade-offs:**
- Pro: 20× faster than relaunching a subprocess per head.
- Con: A crash mid-sweep loses the in-memory loop state but prediction CSVs written so far are intact (`IncrementalCSVWriter` appends on every batch).

### Pattern 3: Additive CSV schema — `experiment_tag` column

**What:** Existing `IncrementalCSVWriter` emits columns: `utterance_id, reference, hypothesis, hypothesis_raw, wer, num_hyp_words, num_ref_words, perturbation, gender, accent, age, model, generation, architecture`. We add **one** column: `experiment_tag` (e.g., `"mask_L3H7"`, `"beam5_rep1.2"`, `"vad_on"`, `"baseline"`).

**When:** All v2.0 inference scripts set `experiment_tag`. Existing `run_inference.py` sets it to `"baseline"` (default value) or omits — both backwards compatible.

**Why:** Downstream metrics scripts (`compute_fairness_metrics.py`, `error_decomposition.py`, `whisper_hallucination_analysis.py`) group by `model` and `perturbation`. Adding `experiment_tag` as an optional groupby key means:
- Old prediction CSVs (without the column) still work (pandas treats missing column as NaN).
- New scripts can slice by `(model, experiment_tag)` for diagnosis analysis.
- `compute_*_metrics.py` need ZERO edits if we invoke them per-tag by filtering the CSV up front, OR a trivial edit to add an optional `--experiment_tag` filter flag.

**Shape:**
```python
# whisper_head_hooks.py or extended IncrementalCSVWriter
result_row["experiment_tag"] = self.experiment_tag  # e.g., "mask_L3H7"
```

**Trade-offs:**
- Pro: Zero data-contract breakage, backwards compatible.
- Con: Requires one-line edit to `IncrementalCSVWriter.__init__` and `_build_row`. Logged as a modification.

### Pattern 4: VAD as manifest-transformer, not inference flag

**What:** `apply_energy_vad.py` reads a manifest (e.g., `cv_indian_511.csv`), processes each audio file (trim silence via frame RMS threshold), writes processed clips to `outputs/vad/<name>/audio/`, writes a new manifest pointing to the processed paths, preserving all demographic columns.

**When:** Whenever we want VAD-on evaluation. To interact with silence-injection: run VAD on the silence-injected perturbation output (the manifest that points into `perturbed_audio/silence_25pct/`).

**Why:** VAD is a property of the **input audio**, not the model call. Putting it before inference means:
- It composes with the existing perturbation matrix (any `perturbation × vad_state` cell is just one manifest).
- `run_inference.py` needs zero knowledge of VAD — it reads the manifest and runs.
- Testing VAD against silence-injection is trivial: `apply_energy_vad.py --manifest silence_25pct_manifest.csv → vad_silence25_manifest.csv → run_inference.py`.

**Pipeline integration:**
```
prepare_dataset.py      → cv_indian_511.csv
generate_perturbations.py → perturbed_audio/silence_25pct/*.wav (existing)
apply_energy_vad.py     → outputs/vad/cv_indian_511_silence25_vad/manifest.csv  [NEW]
run_inference.py --manifest outputs/vad/cv_indian_511_silence25_vad/manifest.csv
                --model whisper-large-v3
```

**Trade-offs:**
- Pro: Single-responsibility; VAD and inference stay decoupled; silence-injection × VAD matrix is a flat file product.
- Con: Disk cost — N copies of the audio. Mitigated: diagnosis probe set is only 511 utts, trivial.

### Pattern 5: Selective fine-tuning via module targeting, not weight surgery

**What:** `train_whisper_heads.py` reads `top_k_heads.json`, constructs a set of target module names (e.g., `["model.decoder.layers.3.self_attn.q_proj", "model.decoder.layers.3.self_attn.o_proj", ...]`), then either:
- (a) **Direct unfreeze:** `param.requires_grad = True` only for parameter indices corresponding to the target heads, OR
- (b) **LoRA restricted:** use `peft.LoraConfig(target_modules=[...])` to wrap only the target projection matrices. The PEFT path is simpler but updates the entire projection, not just one head's slice. Option (a) is the faithful Calm-Whisper-style replication.

**Recommended:** **Option (a) direct unfreeze with masked gradients** — zero out gradients on the non-targeted head slices in a hook. This matches the Calm-Whisper method literally.

**Why reuse the v1.0 LoRA infra pattern (not the code):** v1.0's `scripts/training/train_standard_lora.py` is written for Qwen3-ASR-1.7B and PEFT target modules specific to that model. The **patterns** to reuse:
- HuggingFace `Trainer` + `TrainingArguments` wrapper
- Dataset/collator pattern (`data_loader.py`, `data_collator.py`)
- W&B logging hooks
- Checkpoint directory layout (`outputs/<name>/adapter/`)
- VRAM budgeting helpers (`tune_vram.py`)

The **code** should not be copy-pasted — it is model-specific and would drag Qwen3-ASR imports into Whisper training. Write a fresh `train_whisper_heads.py` that mimics the file layout and reuses pure helpers (W&B init, seed handling, `outputs/` convention).

**Shape:**
```python
# train_whisper_heads.py (sketch)
model = WhisperForConditionalGeneration.from_pretrained(...)
for p in model.parameters():
    p.requires_grad = False
target_params = []
for (layer_idx, head_idx) in top_k_heads:
    attn = model.model.decoder.layers[layer_idx].self_attn
    for proj in [attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj]:
        proj.weight.requires_grad = True
        target_params.append(proj.weight)
# register grad hook that zeros non-head slices
register_head_grad_mask(target_params, head_idx)
Trainer(model, args, train_dataset, ...).train()
```

**Trade-offs:**
- Pro: Trains only ~0.5% of params, tiny checkpoint, fast.
- Con: Grad-masking hook must be correct or the whole projection drifts. Add a unit test in `validate_head_training.py` that checks non-target slices have ||Δw|| = 0 after one step.

### Pattern 6: Fine-tuned models register as new MODEL_REGISTRY entries

**What:** After a fine-tune run completes at `outputs/head-finetune/run_20260420_heads3/`, we add an entry to `MODEL_REGISTRY`:
```python
"whisper-large-v3-hf-tuned-v1": {
    "hf_id": "openai/whisper-large-v3",
    "checkpoint": "outputs/head-finetune/run_20260420_heads3",
    "generation": 2,
    "architecture": "Encoder-Decoder (head fine-tuned)",
    "params": "1.5B",
    "type": "whisper",
},
```

`load_whisper()` gets one new branch: if `"checkpoint"` is set, load the base model then `model.load_state_dict(torch.load(ckpt))` (or PEFT adapter merge). Everything else — the entire existing evaluation, fairness metrics, and figure pipeline — Just Works.

**Why:** This is the cheapest possible integration with the existing benchmark matrix. Fine-tuned models get evaluated across the same 12 perturbations × 2 datasets as baselines for direct comparison tables.

**Trade-offs:**
- Pro: Zero changes to `compute_fairness_metrics.py`, `error_decomposition.py`, `generate_all_plots.py`.
- Con: `MODEL_REGISTRY` grows by 1-3 entries per fine-tune experiment. Acceptable — it already has 9 entries.

---

## 6. Data Flow

### Diagnosis flow (per-head identification)

```
prepare_dataset.py                         [existing]
    │
    ▼
cv_test_manifest.csv                       [existing]
    │
    ▼ (filter accent == "Indian English")
outputs/manifests/cv_indian_511.csv        [NEW — one-time build]
    │
    ▼
run_head_masking_sweep.py                  [NEW]
  - loads whisper-large-v3 once
  - for each (layer, head) in 20 heads:
      attach mask hook → infer → detach
  - writes 21 prediction CSVs + top_k_heads.json
    │
    ▼
outputs/head-surgery/diagnosis/
  ├── predictions_whisper_large_v3_baseline.csv
  ├── predictions_whisper_large_v3_mask_L{0..N}H{0..M}.csv
  └── top_k_heads.json
    │
    ▼
compute_fairness_metrics.py / error_decomposition.py   [existing, unchanged]
  - invoked per-tag, filtered via experiment_tag column
  - produces insertion rate per mask condition
    │
    ▼
generate_all_plots.py                      [existing, possibly one new figure]
```

### Decode ablation flow

```
outputs/manifests/cv_indian_511.csv
    │
    ▼
run_decode_ablation.py                     [NEW]
  - load whisper-large-v3 once
  - for each (beam, rep_pen, len_pen, ngram) in grid:
      override generation_config → infer
  - writes N prediction CSVs
    │
    ▼
outputs/head-surgery/decode-ablation/
    │
    ▼
error_decomposition.py → ablation_summary.json  [existing]
```

### VAD + silence-injection flow

```
perturbed_audio/silence_25pct/*.wav         [existing]
    │
    ▼ (build manifest pointing to these)
outputs/manifests/cv_indian_silence25.csv
    │
    ▼
apply_energy_vad.py                         [NEW]
    │
    ▼
outputs/vad/cv_indian_silence25_vad/
  ├── audio/*.wav
  └── manifest.csv
    │
    ▼
run_inference.py --manifest outputs/vad/...   [existing]
    │
    ▼
standard prediction CSV (experiment_tag="vad_silence25")
    │
    ▼
compute_fairness_metrics.py                 [existing, unchanged]
```

### Fine-tune flow

```
top_k_heads.json  +  accent-diverse training manifest
    │                           │
    └───────────┬───────────────┘
                ▼
train_whisper_heads.py                      [NEW]
  - freeze all, unfreeze + grad-mask target head slices
  - HuggingFace Trainer loop
  - W&B logging
    │
    ▼
outputs/head-finetune/run_<tag>/
  ├── state_dict.pt
  ├── config.json (includes top_k_heads)
  └── wandb/
    │
    ▼ (add entry to MODEL_REGISTRY in run_inference.py)
    │
    ▼
run_inference.py --model whisper-large-v3-hf-tuned-v1   [existing, with one new branch]
  - evaluated across the full 12 perturbations × 2 datasets
    │
    ▼
compute_fairness_metrics.py                 [existing, unchanged]
generate_all_plots.py                       [existing, unchanged]
```

---

## 7. Integration Contract Summary

| Existing stage | Role in v2.0 | Modified? |
|----------------|--------------|-----------|
| `prepare_dataset.py` | Source of CV/FS manifests | No |
| `generate_perturbations.py` | Produces silence-injection audio for VAD interaction tests | No |
| `run_inference.py` | Houses `load_whisper()`, `infer_whisper()`, `IncrementalCSVWriter`. Evaluates fine-tuned checkpoints. | **YES — 2 small edits**: (a) `IncrementalCSVWriter` adds `experiment_tag` column; (b) `load_whisper()` gains optional checkpoint-load branch; (c) `MODEL_REGISTRY` gains 1-3 fine-tuned entries. |
| `compute_fairness_metrics.py` | Consumes prediction CSVs from any new script | No — but optionally gains `--experiment_tag` filter arg (trivial, additive) |
| `compute_fairness_metrics_fs.py` | Same | No |
| `error_decomposition.py` | Consumes prediction CSVs | No |
| `whisper_hallucination_analysis.py` | Core analysis of insertion categorization — already does 95% of what we need for head-mask evaluation | No |
| `generate_all_plots.py` | Produces figures | No (or adds one new figure func for head-mask bar chart) |
| `validate_*.py` | Pre-flight validators | Add `validate_head_sweep.py` for the new script |
| `scripts/training/*` | v1.0 GRPO infra (archived) | **Do NOT import or modify.** Reuse patterns by re-writing. |

**Files touched in the existing codebase:** `run_inference.py` only. All other modifications are additive (new files).

---

## 8. Suggested Build Order

The build order enforces the dependency graph: diagnosis must succeed before ablation matters; ablation must complete before fine-tuning targets are locked; fine-tuning must complete before evaluation comparison tables can be rendered.

```
Phase A — Infrastructure (pre-diagnosis)
├── A1  whisper_head_hooks.py + unit tests
│       (mask one head, verify attention output differs; detach, verify identity)
├── A2  Build outputs/manifests/cv_indian_511.csv (one-off script or notebook)
├── A3  Extend IncrementalCSVWriter with experiment_tag column (run_inference.py edit)
└── A4  Smoke test: run_head_masking_sweep.py on 5 utterances × 2 heads

Phase B — Diagnosis sweep
├── B1  Full 20-head sweep on 511 Indian CV utterances (+ baseline)
├── B2  Post-hoc analysis: insertion rate per mask → top_k_heads.json
│       (reuse error_decomposition.py + a thin aggregation script)
└── B3  Validation: Calm-Whisper-comparable "3/20 heads = 75% hallucinations" check

Phase C — Decode ablation
├── C1  run_decode_ablation.py grid over {beam, rep_pen, len_pen, ngram}
├── C2  Analyze: does any decode config match head-masking's insertion reduction?
└── C3  Decision point: is decode tuning sufficient, or does fine-tune add value?

Phase D — VAD preprocessing
├── D1  apply_energy_vad.py + unit test (RMS threshold sanity)
├── D2  VAD on clean Indian-CV subset → does it affect WER?
├── D3  VAD × silence-injection matrix (25/50/75%) — does VAD rescue silence-injection failures?

Phase E — Selective head fine-tuning
├── E1  train_whisper_heads.py skeleton + grad-masking unit test
├── E2  VRAM tuning (reuse maximize-vram skill / tune_vram.py pattern)
├── E3  Smoke train: 1 head, 100 steps, verify loss decreases
├── E4  Full train with top_k_heads from Phase B
└── E5  Register fine-tuned checkpoint → MODEL_REGISTRY entry

Phase F — Full evaluation
├── F1  Run fine-tuned checkpoint across 12 perturbations × 2 datasets
│       (existing run_inference.py, no code changes needed here)
├── F2  compute_fairness_metrics.py → comparison tables
├── F3  whisper_hallucination_analysis.py → before/after insertion categorization
└── F4  generate_all_plots.py → figures
```

**Critical dependency gates:**
- B → C: If diagnosis doesn't identify clear culprit heads, re-examine mask implementation before running ablation.
- B → E: `top_k_heads.json` is the input to training — E cannot start until B is complete and validated.
- C ∥ D: Decode ablation and VAD are independent — can run in parallel.
- E → F: Fine-tuned model cannot be evaluated until training + checkpoint registration complete.

---

## 9. Answers to Specific Integration Questions

| Q | Answer |
|---|--------|
| **1. Per-head masking: wrap `run_inference.py`, standalone, or new CSV format?** | Standalone: `run_head_masking_sweep.py`. It IMPORTS `load_whisper` and `infer_whisper` from `run_inference.py` (they are already factored out) and wraps them with hook attach/detach. CSV format stays the same + one additive `experiment_tag` column. |
| **2. Head diagnosis loop: one W&B run per head?** | One Python process, one script invocation, one W&B run. Internal loop over heads. Log per-head metrics as a step/axis. Data contract to "top-k identification": `top_k_heads.json` with ranked `[{layer, head, insertion_rate, delta_vs_baseline}]`. |
| **3. Decoding strategy ablation: inside `run_inference.py` or new script?** | New script: `run_decode_ablation.py`. Same rationale as #1 — keeps argument surface small, single-purpose scripts. |
| **4. Energy VAD: pre- or post-inference?** | **Pre-inference.** VAD is a property of the audio input. `apply_energy_vad.py` is a manifest-in → manifest-out transformer. This makes silence-injection × VAD a flat cross-product of manifests. |
| **5. Selective head fine-tuning integration?** | New script: `train_whisper_heads.py`. Do NOT reuse v1.0 LoRA code (model-specific to Qwen3-ASR). DO reuse the v1.0 patterns: `outputs/<run>/` layout, W&B hooks, HF `Trainer`. Checkpoints at `outputs/head-finetune/run_<tag>/`. Registration back via new `MODEL_REGISTRY` entry + 1 new branch in `load_whisper()`. |
| **6. Can existing `compute_*_metrics.py` handle new predictions?** | **Yes, unchanged.** Prediction CSV schema is compatible (additive `experiment_tag` column is nullable). If we want to group by head-mask condition in the same analysis run, add an optional `--experiment_tag` filter flag to `compute_fairness_metrics.py`. Trivial edit. `whisper_hallucination_analysis.py` already does insertion categorization — directly usable on mask-condition CSVs. |

---

## 10. Anti-Patterns to Avoid

### Anti-Pattern 1: "Just add a `--head_mask` flag to `run_inference.py`"
**Why wrong:** Bloats argument surface, couples diagnosis code with routine inference, forces every future inference run to carry dead code paths.
**Do instead:** New script that imports reusable functions from `run_inference.py`.

### Anti-Pattern 2: Subclass `WhisperForConditionalGeneration`
**Why wrong:** Brittle to `transformers` library upgrades. Hooks on generation path are hard to override.
**Do instead:** `forward_pre_hook` on specific decoder attention modules.

### Anti-Pattern 3: Rebuild prediction CSV schema for head experiments
**Why wrong:** Breaks every downstream `compute_*_metrics.py` script.
**Do instead:** Additive `experiment_tag` column; backwards compatible.

### Anti-Pattern 4: Copy-paste `train_standard_lora.py` and retarget Whisper
**Why wrong:** File is 600+ lines full of Qwen3-ASR-specific imports (`qwen_asr`, `patch_outer_forward`, etc.). Adapting in place creates a zombie file.
**Do instead:** Write fresh `train_whisper_heads.py`, mirror the structure, import only pure helpers.

### Anti-Pattern 5: Run VAD inside `run_inference.py`
**Why wrong:** VAD state becomes an invisible flag on every eval; breaks "audio on disk = ground truth input" contract.
**Do instead:** VAD as a manifest transformer that writes new audio files.

### Anti-Pattern 6: Subprocess-per-head for the 20-head sweep
**Why wrong:** 20× cold-load of a 1.5B model (~10 min wasted) + VRAM fragmentation risk.
**Do instead:** Single process, load once, loop in Python, attach/detach hooks per iteration.

### Anti-Pattern 7: Fine-tune full decoder "to be safe"
**Why wrong:** Defeats the entire Calm-Whisper thesis (targeted surgery on 3/20 heads). Also risks degrading non-Indian-accent WER.
**Do instead:** Direct unfreeze + grad mask on `{layer_i, head_j}` slices only. Validate with a parameter-delta sanity check.

---

## 11. Sources

- `scripts/run_inference.py` — `MODEL_REGISTRY`, `load_whisper()`, `infer_whisper()`, `IncrementalCSVWriter` at lines 27-91, 307-378, 148-210
- `scripts/compute_fairness_metrics.py` — existing groupby-based metrics, no `experiment_tag` awareness yet
- `scripts/error_decomposition.py` — existing insertion/substitution/deletion computation via `jiwer.process_words()`
- `scripts/whisper_hallucination_analysis.py` — existing insertion extraction & categorization
- `scripts/training/train_standard_lora.py` (archived) — v1.0 pattern template for HF Trainer + LoRA + W&B
- `CLAUDE.md` — Architecture section ("Script-oriented pipeline: prepare → perturbation → inference → metrics → plots")
- `.planning/PROJECT.md` — v2.0 milestone scope, Calm-Whisper methodology reference, 9.62% Indian-accent insertion baseline
- `.planning/archive/v1.0-phases/ROADMAP.md` — archived GRPO infra not reused verbatim

---
*Architecture research for: v2.0 attention head surgery on Whisper-large-v3*
*Researched: 2026-04-11*
