# Phase 2: Standard LoRA Baseline + Evaluation Bridge - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 02-standard-lora-baseline-evaluation-bridge
**Areas discussed:** Training data & splits, HP selection → full training, Evaluation bridge design, Output artifacts & success

---

## Training data & splits

| Option | Description | Selected |
|--------|-------------|----------|
| Both combined | Fair-Speech + Common Voice combined. Maximizes training diversity. | ✓ |
| Fair-Speech only | ~26K utterances with clean ethnicity labels. | |
| Train separately, merge | Train two adapters, pick the better one. | |

**User's choice:** Both combined (Fair-Speech + Common Voice)
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Speaker-disjoint | No speaker appears in both train and eval. | ✓ |
| Random stratified | Random split stratified by demographic group. | |
| Use existing splits | Use official train/test splits as-is. | |

**User's choice:** Speaker-disjoint
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| 1K-2K samples | Enough for per-group WER. Fast iteration. ML prototyping approach. | ✓ |
| 6K manifest as-is | Use existing filtered manifests. | |
| 500 samples | Matches phase2_hp_sweep.py. Very fast but thin demographics. | |
| Full dataset (~42K) | Maximum data. 8+ hours. | |

**User's choice:** 1K-2K samples
**Notes:** User corrected dataset size assumptions — Common Voice raw is ~1.15M, much larger than Fair-Speech's ~26K. The ~6K CV manifest is a filtered subset.

| Option | Description | Selected |
|--------|-------------|----------|
| FS+CV train, all 3 eval | Train on FS+CV, evaluate fairness on FS+CV, accuracy on LibriSpeech. | ✓ |
| FS+CV only, skip LS | Train and evaluate only on FS and CV. | |
| All 3 train, FS+CV eval | Include LibriSpeech in training for diversity. | |

**User's choice:** FS+CV train, all 3 eval
**Notes:** LibriSpeech serves as held-out accuracy benchmark (no demographic labels for fairness metrics).

| Option | Description | Selected |
|--------|-------------|----------|
| Equal split | ~500-1K from each dataset. | ✓ |
| Proportional to size | More FS samples. Risk: thin CV accent groups. | |
| Demographic-minimum driven | Sample enough to guarantee MIN_GROUP_SIZE per group. | |

**User's choice:** Equal split (~500-1K from each)
**Notes:** None

---

## HP selection → full training

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 1 defaults | Use lr=2e-4, rank=16, alpha=32, dropout=0.05. Skip sweep. | |
| Run Optuna sweep first | Run phase2_hp_sweep.py (20 trials). Pick top config, then train. | ✓ |
| Small manual grid | Try 3-4 learning rates. Middle ground. | |

**User's choice:** Run Optuna sweep first
**Notes:** User asked to check the ML prototyping skill for guidance on step counts.

| Option | Description | Selected |
|--------|-------------|----------|
| Follow skill process, you decide details | 4-phase prototyping (sweep → validate → final). Claude decides step counts. | ✓ |
| Just sweep + train | Skip intermediate validation. | |
| Discuss step counts now | Specify exact step counts per stage. | |

**User's choice:** Follow skill process, Claude decides details
**Notes:** ML prototyping skill read and applied. Skill recommends 10% of full steps for sweep trials, 30-50% for validation runs.

| Option | Description | Selected |
|--------|-------------|----------|
| Use existing sweep as-is | 6 HPs: LR, rank, alpha ratio, dropout, target MLP, weight decay. | ✓ |
| Add batch size | Currently fixed at 2. Could try 1/2/4 with grad accumulation. | |
| Simplify to LR + rank | Fix everything else at Phase 1 defaults. | |
| Add warmup steps | Warmup fraction (0%, 5%, 10%). | |

**User's choice:** Use existing sweep as-is
**Notes:** None

---

## Evaluation bridge design

| Option | Description | Selected |
|--------|-------------|----------|
| New standalone script | New scripts/training/evaluate_adapter.py. Self-contained. | ✓ |
| Extend run_inference.py | Add --adapter_path to existing script. | |
| Merge adapter then use existing | Merge weights permanently, use existing pipeline. | |

**User's choice:** New standalone script
**Notes:** User asked for clarification on "evaluation bridge" terminology. Explained as the evaluation script connecting trained adapter to fairness metrics pipeline.

| Option | Description | Selected |
|--------|-------------|----------|
| W&B for training + CSV for eval | W&B for live tracking, CSV/JSON for final evaluation results. | ✓ |
| CSV/JSON only | Skip W&B. Simpler but harder to compare runs. | |
| W&B only | Everything in W&B. Breaks existing plotting compatibility. | |

**User's choice:** W&B for training + CSV for eval
**Notes:** User asked whether W&B affects format choice. Clarified that W&B and CSV serve different purposes.

| Option | Description | Selected |
|--------|-------------|----------|
| Same format | Match existing prediction CSV and analysis JSON structure. | ✓ |
| Extended with adapter metadata | Same base + adapter_name, lambda_value, training_steps. | |
| You decide | Claude picks during planning. | |

**User's choice:** Same format as existing benchmarking pipeline
**Notes:** None

---

## Output artifacts & success

**Outputs selected (multi-select):**
- ✓ Trained LoRA adapter weights (adapter_model.safetensors + adapter_config.json)
- ✓ Per-group WER table (FS + CV + LS) with fairness metrics
- ✓ W&B experiment dashboard
- ✗ HP sweep results summary (not selected)

**User's choice:** Adapter weights + per-group WER table + W&B dashboard
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Adapter produces valid WER | Lower WER than untrained. Per-group metrics on all 3 datasets. | ✓ |
| WER within range of base | Within ±5% of base Qwen3-ASR WER from midterm. | |
| You decide | Claude defines during planning. | |

**User's choice:** Adapter produces valid WER
**Notes:** None

---

## Claude's Discretion

- Exact step counts for HP sweep validation and final training
- W&B project naming and run organization
- Demographic group minimum sample sizes
- Gradient accumulation decisions

## Deferred Ideas

None — discussion stayed within phase scope
