---
fileClass: Task
name: Head-Surgery Diagnosis (v2.0 §4.2)
description: Attention-head-level diagnosis of Whisper-large-v3 Indian-accent hallucination, plus decoding-ablation and energy-VAD adjuncts. Diagnosis-only MVP — no fine-tuning.
status: planned
created: 2026-04-17
updated: 2026-04-17
tags:
  - v2.0
  - head-surgery
  - whisper-large-v3
  - hallucination
  - accent-fairness
aliases:
  - prd-head-surgery-diagnosis
  - v2.0 attention head surgery
---

# PRD — Head-Surgery Diagnosis (Midterm §4.2, v2.0 MVP)

## 1. Goal

Reduce Whisper-large-v3's accent-specific hallucination by **identifying** the decoder self-attention heads whose masking reduces the 9.62% Indian-accent insertion rate (midterm §3.2), without hurting non-Indian-accent WER. Adapts the Calm-Whisper (Interspeech 2025, arxiv 2505.12969) protocol from non-speech hallucination to accent hallucination.

**Milestone scope is diagnosis-only.** Selective head fine-tuning (Calm-Whisper's intervention half) is deferred. Calm-Whisper is cited as the methodology source and as the "what-if" for intervention.

## 2. Target dataset and baseline

- **Target subset:** the frozen set of self-reported Indian-accent utterances from Common Voice's test split. The midterm reports n=511 on the CV snapshot it used; the Common Voice 25 snapshot that ships on this machine (`datasets/cv-corpus-25.0-2026-03-09`) contains 510 pure-India-accent rows after filtering, plus 9 compound-label rows (`United States English|India and South Asia …`) that are excluded. **We use the 510 pure-India rows.** No subsampling within the chosen subset.
- **Model:** `openai/whisper-large-v3`, HuggingFace revision hash frozen in `repro_config.py`.
- **Baseline anchor:** midterm's 9.62% insertion rate. Stage A must reproduce this within ±0.5pp on the 510-utterance subset or the milestone halts to investigate config drift. The 9.62% target is robust to the one-utterance difference at this tolerance.

## 3. What ships (MVP: T1–T9)

| ID | Feature | Where |
|---|---|---|
| T1 | Per-head masking forward hook on `decoder.layers[L].self_attn` — serial and batched-condition variants | `scripts/head_surgery/head_mask_hook.py` |
| T2 | 640-cell diagnosis sweep (32 layers × 20 heads × 511 utterances) | `scripts/head_surgery/run_diagnosis_sweep.py` |
| T3 | Driving-ness metric: top-10 heads by Δ insertion rate + paired-bootstrap p<0.05 annotation; regression-guard gate | `scripts/head_surgery/score_heads.py` |
| T4 | Reuse of existing insertion classifier for repetition/syntactic/content breakdown | reuses `scripts/analysis/whisper_hallucination_analysis.py` |
| T5 | Decoding ablation grid (36 configs: beam × rep_penalty × no_repeat_ngram × temp_fallback) | `scripts/head_surgery/decoding_ablation_grid.py` |
| T6 | Evaluation report — before/after tables across all arms | `scripts/head_surgery/aggregate_report.py` → `docs/head_surgery_report.md` |
| T7 | Reuse of existing 216-run baseline CSVs as the "before" column (no re-run) | read-only dependency on existing outputs |
| T8 | Energy-based VAD preprocessing over silence-injected audio (25/50/75%) | `scripts/head_surgery/energy_vad.py` |
| T9 | Reproducibility contract: frozen utterance IDs, model revision hash, seed, full `generate()` config | `scripts/head_surgery/repro_config.py` |

### Out of scope (explicit non-goals)

Fine-tuning (D1 and variants D1a/D1b/D1c); accumulated/greedy masking (D2); Silero VAD comparison (D3); cross-accent head overlap analysis (D4); layer-selective vs all-layers fine-tuning ablation (D5); decoding × masking interaction (D6); publication heatmap (D7 — candidate for stretch if T1–T9 finish early); activation patching (A1); full-model fine-tuning (A2); additional datasets (A3); other Whisper sizes (A4); new insertion classifiers (A5); new fairness metrics (A6); LoRA on heads (A7); long-form inference experiments (A10).

Cite Calm-Whisper as the source methodology. Do not claim intervention results.

## 4. Design decisions (resolved during brainstorming)

| # | Decision | Value | Rationale |
|---|---|---|---|
| D1 | Milestone scope | Diagnosis-only | Class-project deadline; diagnosis alone is a citable contribution |
| D2 | Driving-ness metric | Top-10 heads by Δ insertion rate, with paired-bootstrap p<0.05 annotation | Top-K ranking produces a table regardless of effect-size distribution; bootstrap flag adds defensibility |
| D3 | Regression guard group | Composite mean WER across all non-Indian CV24 accents | Averages out per-group noise; uses more data than any single guard group |
| D4 | Regression budget | 0.5pp absolute | Matches Calm-Whisper's strict spirit; strict enough to be credible |
| D5 | Parallelization strategy | Batched-condition inference (per-sample head-mask hooks), staged rollout: serial pilot → batched full sweep | Biggest speedup per engineering dollar; staging supplies a correctness anchor before scale |
| D6 | Batch-size selection | Empirical inference-specific tuner (Stage A.5) | Inference has no optimizer/grad state; training VRAM-tune infra does not transfer directly; pick largest batch with VRAM <90% and tokens/sec within 5% of best |
| D7 | Reproducibility contract | Freeze: utterance IDs, model revision hash, seed, full `generate()` config; pin `transformers` version in `pyproject.toml` | `generate()` kwargs drift across `transformers` versions; n=511 is small enough that drift swamps signal |
| D8 | Code organization | New `scripts/head_surgery/` subdirectory | Keeps milestone cohesive; consistent with the project's role-based `scripts/` layout |

## 5. Pipeline

### Stage A — Baseline rerun (gate G1)

1. Load `openai/whisper-large-v3` at the pinned revision with the frozen `generate()` config.
2. Run on the 511 Indian-accent utterances, no hook active.
3. Compute insertion rate via `jiwer` S/D/I decomposition; compute composite non-Indian WER over the other CV24 accent groups' existing midterm hypotheses (T7).
4. Write `outputs/head_surgery/baseline_metrics.json`.

**Gate G1:** Indian-accent insertion rate ∈ [9.12%, 10.12%] (midterm 9.62% ± 0.5pp). If outside, investigate config drift; do not proceed.

### Stage A.5 — Batch-size tune (gate G1.5)

1. Fix the Stage-C config: one decoder layer hooked, per-sample head masks across all 20 heads.
2. Sweep `utterances_per_batch ∈ {1, 2, 4, 8, 16, 32}`.
3. For each setting: warmup 2 batches, time 10 batches. Record wall-clock `tokens/sec`, `torch.cuda.max_memory_allocated`, OOM flag.
4. Chosen batch size = largest setting where peak VRAM < 90% of device memory AND `tokens/sec` ≥ 95% of the best observed.
5. Write `outputs/head_surgery/tune_batch_size.json`. Downstream stages default their `--batch-size` from this file.

**Gate G1.5:** Tuner completes without OOM at the chosen value.

### Stage B — Pilot sweep (gate G2)

1. Use the serial variant of the hook.
2. Configuration: 1 decoder layer (choice: layer 15, the approximate mid-decoder layer; subject to revision in implementation-plan phase) × 20 heads × 50 utterances (random 50-sample subset of the 511, same seed from `repro_config.py`).
3. Produces `outputs/head_surgery/pilot_sweep.csv` with one row per (utterance × head) and `outputs/head_surgery/pilot_metrics.json` with per-head aggregates.

**Gate G2:** Among the 20 masked-head runs on the pilot layer, at least one head has Δ insertion rate > 0 and at least one has Δ ≤ 0. An all-zero result indicates the hook is a no-op (bug); all-positive or all-negative indicates either a strong layer signal or a masking bug. In either case, investigate before proceeding.

### Stage C — Full sweep (gate G3)

1. Switch to the batched-condition variant of the hook. Batch size from Stage A.5.
2. Configuration: 32 decoder layers × 20 heads × 511 utterances = 327,040 model forwards (amortized by the batched variant).
3. Before the full run, re-execute the pilot-layer and pilot-utterances on the batched variant.

**Gate G3:** Batched output on the 50 pilot utterances matches the serial pilot within 1e-4 WER and identical per-utterance insertion count. If not: fall back to serial + data-parallel sharding (revisit Q4-B option).

4. Run the full 640-cell sweep. Output: `outputs/head_surgery/sweep_L{L}_h{H}.csv` (or one consolidated long-form CSV), plus `outputs/head_surgery/sweep_metrics.json` with per-(L, h) aggregates.

**Gate G4:** All 640 conditions yield 511 non-empty hypothesis rows; no NaN in aggregate metrics.

### Stage D — Scoring (T3)

1. For each of 640 (L, h) conditions compute, on the 511 Indian-accent subset:
   - `Δ_ins` — Δ total insertion rate vs baseline.
   - `Δ_rep`, `Δ_syn`, `Δ_con` — Δ repetition / syntactic / content insertion rates (reuse T4 classifier).
   - Composite non-Indian WER (regression guard): run Whisper-large-v3 with head (L, h) masked on the other CV24 accent groups' utterances. (Cost note: this is an extra inference pass over the non-Indian audio for every qualifying condition. To keep cost bounded, only compute the regression guard for the **top-50** heads by `|Δ_ins|` from the baseline Indian-accent sweep; heads outside the top-50 inherit a `regression_checked=False` flag.)
   - `p_ins` — paired-bootstrap p-value for `Δ_ins`, 10,000 iterations, resampling the 511 utterances with shared resample indices between baseline and masked runs.
2. Apply the regression-guard filter: `regression_ok = (non_indian_WER_masked − non_indian_WER_baseline) ≤ 0.5pp absolute`.
3. Output:
   - `outputs/head_surgery/head_scores.csv` — 640 rows, columns `[layer, head, Δ_ins, Δ_rep, Δ_syn, Δ_con, p_ins, regression_checked, non_indian_WER_masked, regression_ok]`.
   - `outputs/head_surgery/top_k_heads.csv` — top 10 rows by `Δ_ins` filtered to `regression_ok=True` (or `regression_checked=False`), with all annotations.

**Gate G5:** Composite non-Indian WER baseline (no-masking) matches the T7-derived value from the midterm 216-run CSVs within 0.2pp. If not: T7 reuse is broken; remediate before trusting guard flags.

### Stage E — Decoding ablation grid (T5)

Independent of Stages A–D (can run in parallel).

1. Generate a 36-config grid: `beam ∈ {1, 5} × rep_penalty ∈ {1.0, 1.1, 1.3} × no_repeat_ngram ∈ {0, 3, 5} × temp_fallback ∈ {on, off}`.
2. Run Whisper-large-v3 on the 511 Indian-accent utterances under each config. Re-use Stage A.5's chosen batch size.
3. Output: `outputs/head_surgery/decoding_grid.csv`, `outputs/head_surgery/decoding_scores.csv` (per-config insertion rate, repetition-only rate, overall WER).

### Stage F — Energy VAD (T8)

Independent of Stages A–E.

1. Implement RMS-based frame energy VAD: 20ms window, 10ms hop. A frame is "silence" if `20·log10(RMS) < dB_floor` for `≥ min_silence_ms` consecutive frames.
2. Parameters: `dB_floor ∈ {−40, −35, −30}`, `min_silence_ms = 200`.
3. Input: existing `silence-25%`, `silence-50%`, `silence-75%` perturbation manifests (reuse the audio already generated for the midterm 216-run matrix).
4. Apply VAD; write filtered audio to `outputs/head_surgery/vad_filtered/`; rerun Whisper-large-v3 (baseline config, no hook) on the filtered audio.
5. Output: `outputs/head_surgery/vad_scores.csv` with insertion rate per `(severity, dB_floor)` cell, plus the corresponding unfiltered cells from T7.

### Stage G — Aggregate report (T6)

1. Reads: `head_scores.csv`, `top_k_heads.csv`, `decoding_scores.csv`, `vad_scores.csv`, baseline metrics from T7.
2. Writes `docs/head_surgery_report.md` containing:
   - Per-accent insertion rate for 6 CV24 groups × {baseline, top-10 masked heads} — hypothesis: only Indian-accent improves under masking.
   - Overall WER on CV24 (composite), Fair-Speech (overall), and LibriSpeech test-clean for baseline and for the single best head — the clean-speech regression guard for the writeup.
   - MMR (max/min ratio, from the existing midterm metrics pipeline) before/after for the top-3 heads — is fairness improved by masking?
   - Insertion-breakdown (repetition/syntactic/content) before/after for the top-10 heads.
   - Decoding-ablation table: 36 rows ordered by Indian-accent insertion rate; flag the best decoding-only config; compare to masking-only.
   - VAD table: 3 severity rows × 3 dB_floor columns of insertion rate.
   - Per-head driving-ness ranking table (all 640 rows, sortable on export).

## 6. Validation gates (summary)

| Gate | After stage | Check | On failure |
|---|---|---|---|
| G1 | A — baseline | Indian-accent insertion rate within 9.62 ± 0.5pp of midterm | Investigate `generate()` config / `transformers` version; do not proceed |
| G1.5 | A.5 — tune | Tuner completes without OOM at chosen batch size | Drop to next-smaller batch size; rerun |
| G2 | B — pilot | At least one head with Δ>0 and one with Δ≤0 in the pilot layer | Investigate hook; do not start Stage C |
| G3 | C — batched gate | Batched sweep matches serial pilot within 1e-4 WER | Fall back to serial + data-parallel shards |
| G4 | C — full sweep | 640 conditions × 511 hypotheses, no NaN | Rerun missing conditions |
| G5 | D — scoring | Composite non-Indian WER baseline matches T7 within 0.2pp | Investigate CSV reuse; rebuild guard column |

## 7. Testing

### Unit / smoke tests (`tests/test_head_surgery.py`)

- `test_head_mask_hook_zeros_head` — serial hook on `(L=0, h=0)` produces per-head attention output with row `[:, 0, :, :] == 0` on a synthetic forward.
- `test_head_mask_hook_batched_per_sample` — batched variant: sample 0 masks head 3, sample 1 masks head 7 → `[0, 3, :, :] == 0`, `[1, 7, :, :] == 0`, other positions non-zero.
- `test_baseline_reproduces_midterm` — Stage A insertion rate within gate-G1 tolerance.
- `test_frozen_repro_config_stable` — `repro_config.py` values match a committed JSON snapshot.
- `test_energy_vad_drops_silence_frames` — VAD applied to a synthetic "speech + 500ms silence + speech" clip removes the silence frames at `dB_floor = −35`.

### End-to-end

No end-to-end test of the 640-cell sweep (too expensive). Gates G2, G3, G4, G5 serve as the end-to-end coverage.

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Batched-condition hook silently produces wrong attention under autoregressive `generate()` with KV cache | **Medium** (top risk) | G3 gate; fall back to serial+shard if G3 fails |
| Paired bootstrap flags no heads significant at p<0.05 (small effect distributed across many heads) | Medium | Top-K ranking ships regardless of bootstrap outcome; writeup says "top-10, N of which are significant at p<0.05" |
| 9.62% fails to reproduce at G1 due to `transformers`-version `generate()` drift | Low–Medium | Pin `transformers` in `pyproject.toml`; T9 captures full `generate()` kwargs |
| Non-Indian composite WER baseline disagrees with T7 CSVs (CSV lost, or earlier run config differed) | Low | G5 gate explicit; if CSVs unavailable, rerun only the Whisper-large-v3 clean rows of the 216-matrix |
| Compute budget exceeded on full sweep (batched fails, serial+shard unavailable) | Medium | Scope-down lever: reduce to "top 8 layers × all 20 heads" (160 cells) triaged by a 32-cell layer-level pre-sweep. Document as writeup limitation. |
| Energy-VAD thresholds don't generalize across silence-injection severities | Low | Sweep 3 dB_floor values; report best per severity |
| Stage D's regression-guard inference (non-Indian audio per masked head) explodes compute | Medium | Only run the guard for the top-50 heads by `|Δ_ins|`; annotate the rest with `regression_checked=False` |

## 9. Dependencies and reuse

- **`scripts/analysis/whisper_hallucination_analysis.py`** — insertion classifier (repetition/syntactic/content). Reused for T4 scoring columns.
- **`scripts/inference/run_inference.py`** — Whisper loader, batching loop, CSV writer. Called from T2, T5, T6, T8.
- **Midterm 216-run baseline CSVs** (existing under `outputs/` per project convention) — provide the "before" column for all comparisons. Not recomputed (T7).
- **`pyproject.toml`** — pin `transformers`, `torch`, `jiwer` versions to the midterm's inference stack.

## 10. Open implementation questions (to resolve during plan-writing)

- Precise pilot-layer selection (mid-decoder layer 15 is a placeholder; may change to layer-0 or last-layer based on literature cues).
- Exact `generate()` kwargs from the midterm inference run — needs a read of the midterm inference logs to populate `repro_config.py`.
- Whether the T7 baseline CSVs are still intact on disk, or whether partial reruns are needed.
- Stage-D guard efficiency: top-50 heads is the proposed cap; implementation-plan phase picks the exact number based on Stage-B wall-clock measurements.

## 11. Sources

- `llm-asr-fairness-midterm.pdf` §3.2 (9.62% finding), §4.2 (proposed v2.0 methodology)
- Calm-Whisper (Wang et al., Interspeech 2025) — [arxiv 2505.12969](https://arxiv.org/abs/2505.12969)
- Investigation of Whisper ASR Hallucinations — [arxiv 2501.11378](https://arxiv.org/html/2501.11378v1)
- `.planning/research/FEATURES.md` (authoritative feature catalog)
- `.planning/STATE.md` (milestone state)

---

*PRD for v2.0 attention-head-surgery diagnosis milestone. Successor doc: implementation plan (to be created via superpowers:writing-plans).*
