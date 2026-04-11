# Project Research Summary — v2.0 Attention Head Surgery

**Project:** LLM-ASR Fairness — Hallucination Mitigation via Attention Head Surgery on Whisper-large-v3
**Domain:** ASR mechanistic interpretability + fairness remediation (CSE 5525 class project)
**Researched:** 2026-04-11
**Confidence:** HIGH for HuggingFace integration and diagnosis protocol; MEDIUM for accent-vs-non-speech transferability of Calm-Whisper; LOW for fine-tune training-signal choice
**Scope note:** This file supersedes the prior v1.0 GRPO SUMMARY.md at this path. v1.0 is archived under `.planning/archive/v1.0-phases/`.

## Executive Summary

v2.0 is a four-arm, mechanistic-interpretability-flavored remediation for the midterm finding that Whisper-large-v3 hallucinates on 9.62% of the 511-utterance Indian-accent Common Voice subset, with insertions (repetition 43% / syntactic 48% / content 9%) comprising 50.7% of all errors. The reference methodology is Calm-Whisper (Wang et al., Interspeech 2025), which showed that 3 of 20 decoder self-attention heads in Whisper-large-v3 drive >75% of non-speech hallucinations, and that freezing the model except those head slices and fine-tuning on empty-label noise data yields an 80% hallucination reduction with <0.1% WER degradation on LibriSpeech. v2.0 adapts the protocol from non-speech hallucination to accent-driven hallucination on an existing 511-utterance probe set, reusing the project's 216 baseline runs as the "before" column.

The four planned interventions are: **(1)** per-head masking diagnosis across all 32 × 20 = 640 decoder self-attention head positions (primary contribution — a ranked list of accent-hallucination-driving heads); **(2)** decoding-strategy ablation (beam size, repetition penalty, no-repeat n-gram, temperature fallback on/off) as the "cheap-fix reviewer baseline"; **(3)** energy-based VAD preprocessing composed with the existing silence-injection perturbations; and **(4)** selective fine-tuning of the top-k identified heads on accent-diverse audio (stretch). **The primary deliverable is Arm 1 (diagnosis). Arms 2 and 3 are independent parallel side-quests. Arm 4 is stretch, gated on Arm 1 completion, and must be cut under time pressure.** Given class-project timeline, the synthesis recommendation is explicit: **ship diagnosis (with bootstrap CIs, held-out test split, and a per-head heatmap) first; only start fine-tuning if diagnosis lands before the time-box deadline.**

The main risks are concentrated in HuggingFace/Whisper integration (three HIGH-severity silent-failure landmines — SDPA ignoring head_mask, KV-cached generation dropping head_mask after token 1, and temperature fallback confounding decoding ablation), in statistical hygiene at n=511 (split leakage, bootstrap noise), and in fine-tuning dynamics (catastrophic forgetting, head-weight collapse). Mitigations are concrete and live in the Pattern/Gate sections of ARCHITECTURE.md and PITFALLS.md: eager attention with runtime assertion, forward-hook-based masking validated via `output_attentions`, deterministic re-decode sanity check, 60/20/20 diagnose/rank/test split committed before the first sweep, LoRA-on-heads with L2-to-init regularization, and a mandatory multi-accent + LibriSpeech regression gate for every intervention.

## Key Findings

### Recommended Stack

The v2.0 stack delta is tiny — one new package (`silero-vad`) and one version adjustment. All core infra (PyTorch, HuggingFace `transformers`, existing inference pipeline) is already in place. The critical latent bug is that `pyproject.toml` carries `transformers>=4.44` with no upper bound, and `layer_head_mask` / `decoder_head_mask` were **removed entirely from Whisper in transformers v5** (source-verified by grep across 6 releases: v4.44 through main). A fresh install would silently drop any head mask passed to `generate()`. The synthesis recommends **forward hooks on `out_proj`** as the primary implementation path because they are architecturally stable across transformers 4.x and 5.x and also support the per-head fine-tuning phase via `.weight` gradient hooks.

**Core technologies:**
- **PyTorch forward hooks on `WhisperAttention.out_proj`** (Path B, recommended) — architecturally stable across transformers 4.x/5.x; one `register_forward_pre_hook` zeros the target head's slice at `out_proj` input; algebraically equivalent to zeroing `attn_weights @ V` for that head; no version pin required.
- **`transformers.WhisperForConditionalGeneration` with `attn_implementation="eager"`** — MANDATORY. SDPA and FlashAttention-2 silently ignore `head_mask`. The existing `load_whisper()` at `scripts/run_inference.py` line ≈307 must be edited to force eager attention and assert `model.config._attn_implementation == "eager"` on load.
- **`silero-vad==6.2.1`** — only new dependency. 87.7% TPR at 5% FPR vs WebRTC's 50%. WebRTC would silently pass through injected silence and defeat the VAD-on-silence-injection ablation.
- **`transformers>=4.48,<5.0`** (fallback Path A if hooks prove unstable) — pins to a version with working `decoder_head_mask` kwarg; recommended as a version guardrail even when using the hook path.
- **Raw `torch.nn.Module.register_hook` on projection weights** for selective fine-tuning. No library (PEFT, TransformerLens, nnsight, captum) supports per-head training as a first-class API. Gradient-masking hooks on `q_proj`/`k_proj`/`v_proj`/`out_proj` rows/columns are ≈200 lines total and must be unit-tested with a post-step `||Δw||==0` assertion on non-targeted head slices.
- **Rejected:** TransformerLens (no Whisper support — verified zero hits in `OFFICIAL_MODEL_NAMES`), nnsight (adds abstraction for no benefit), captum (wrong tool — feature attribution, not head ablation), `peft.LoraConfig` per-head (PEFT operates on whole `nn.Linear`, not slices), pyannote-audio (GPU-hungry, slow on short files), webrtcvad (50% TPR unusable).

### Expected Features

Detailed feature inventory lives in FEATURES.md. The critical framing is: **diagnosis alone (T1+T2+T3) is a complete, citable contribution.** Fine-tuning (D1) is a differentiator, not a table stake. The midterm's 216-run baseline matrix is reused verbatim as the "before" column; no re-running.

**Must have (table stakes — the MVP):**
- **T1** Per-head masking hook on Whisper decoder self-attention — foundational primitive.
- **T2** 640-cell diagnosis sweep (32 layers × 20 heads) on the 511 Indian-accent CV24 subset — core contribution.
- **T3** Accent-adapted "driving-ness" metric — insertion-rate delta + repetition-only delta + non-target WER regression budget. Replaces Calm-Whisper's binary non-speech metric.
- **T4** Reuse existing `classify_insertions.py` for the 43/48/9 repetition/syntactic/content breakdown — free.
- **T5** Decoding-strategy ablation (scoped: beam × rep_pen × no_repeat_ngram 2D grid, not 4D) — answers the obvious reviewer question.
- **T6** Standard evaluation report: per-accent insertion rate × {baseline, masked, fine-tuned}, WER on CV24+FairSpeech+LibriSpeech, MMR before/after.
- **T7** Reuse existing 216-run baseline CSVs — zero cost, authoritative "before" column.
- **T8** Energy-based VAD on silence-injected perturbation manifests (25/50/75%) — requested in midterm §4.2.
- **T9** Frozen seeds, checkpoint hash, utterance ID list, and committed `indian_accent_splits.json`.

**Should have (competitive — add if diagnosis ships on time):**
- **D1 + D1a** Selective fine-tuning of top-k heads with MLE on clean Indian-accent CV24 split — the "surgery" half of the Calm-Whisper story. **Differentiator, not table stakes.**
- **D3** Silero VAD comparison against the energy-based VAD from T8 — cheap if T8 exists; lets the writeup say "our simple VAD matches SOTA".
- **D6** Decoding × head-masking interaction cross-cell — cheap if T2 and T5 both exist; answers "is head surgery additive with cheap decoder tricks?"
- **D7** Per-head heatmap visualization for the writeup.

**Defer (explicit non-goals for v2.0):**
- **D1b / D1c** Contrastive or RL training signals — D1c in particular pulls v1.0 GRPO back in, which defeats the pivot.
- **D2** Accumulated / greedy head masking (second-order interactions) — expensive.
- **D4** Cross-accent head overlap analysis — novel but N_accents × 640 inference cost.
- **D5** Layer-selective vs all-layers fine-tune ablation.
- **Anti-features (avoid entirely):** activation patching (A1 — too complex for class-project scope), full-model fine-tune (A2 — defeats the surgery thesis), new datasets (A3 — scope killer), Whisper-small/-medium sweeps (A4 — answers wrong question), new insertion classifier (A5 — invalidates midterm comparison), custom fairness metrics (A6 — not a metrics paper), per-head LoRA (A7 — PEFT cannot do per-head), custom training loop (A8 — bug surface), GRPO revival (A9), long-form chunked experiments (A10 — different dynamic).

### Architecture Approach

v2.0 extends the existing CSV-manifest linear pipeline (`prepare_*.py → generate_perturbations.py → run_inference.py → compute_*_metrics.py → generate_*_plots.py`) by adding **four new scripts and one shared library**, and touching only one existing file (`run_inference.py`) with three small additive edits. Every new capability is its own script that slots into the pipeline; no new data contracts; prediction CSVs gain one additive nullable `experiment_tag` column for backwards compatibility. Fine-tuned checkpoints register back into `MODEL_REGISTRY` so the full 12-perturbation × 2-dataset evaluation matrix runs unchanged.

**Major components:**
1. **`scripts/whisper_head_hooks.py`** (new library, no `__main__`) — shared `attach_head_mask_hooks(model, masks)` + `detach_hooks(handles)`. Imported by all three inference scripts and the training script. Raw `register_forward_pre_hook` on `WhisperAttention.out_proj`; zero subclassing.
2. **`scripts/run_head_masking_sweep.py`** (new) — loads Whisper-large-v3 once, loops 640 (layer, head) positions, attaches/detaches hooks between each inference call, writes 641 prediction CSVs (640 masked + 1 baseline) + `top_k_heads.json`. Uses cached encoder outputs to achieve ~20x speedup over naive re-encoding.
3. **`scripts/run_decode_ablation.py`** (new) — same load-once-sweep-many pattern; grid over `{beam, rep_pen, no_repeat_ngram}` with temperature fallback explicitly disabled; writes per-cell prediction CSVs.
4. **`scripts/apply_energy_vad.py`** (new) — manifest-in → manifest-out transformer. Reads input manifest, processes audio (silero-vad or energy threshold), writes trimmed clips to `outputs/vad/<name>/audio/` + new manifest. Slots **before** `run_inference.py`, not inside it. Composes with existing silence-injection perturbations as a flat cross-product of manifests.
5. **`scripts/training/finetune_heads.py`** (new, separate from v1.0 training scripts) — freezes all params, unfreezes only target head projection weights, registers gradient-masking hooks on non-target slices, uses HF `Trainer` with conservative LR=1e-6 / ≤5 epochs (Calm-Whisper config). Reads `top_k_heads.json`. Saves to `outputs/head-finetune/run_<tag>/`.
6. **`scripts/run_inference.py`** (MODIFIED — the only existing-file edit) — three small additive changes: (a) `load_whisper()` forces `attn_implementation="eager"` and accepts optional `checkpoint` path; (b) `IncrementalCSVWriter` adds nullable `experiment_tag` column; (c) `MODEL_REGISTRY` gains 1-3 fine-tuned entries like `"whisper-large-v3-hf-tuned-v1"`.

**Key patterns:** (a) hook library not model subclass; (b) load once, sweep many (cold-load is ~30s — don't repeat it 640×); (c) additive CSV schema (zero breakage to `compute_*_metrics.py`); (d) VAD as manifest transformer (not an inference flag); (e) selective FT via module targeting + gradient masking (not weight surgery); (f) fine-tuned models register as `MODEL_REGISTRY` entries so the existing 12-perturbation × 2-dataset eval matrix runs unchanged.

**Build order (6 phases):** A Infrastructure → B Diagnosis sweep → C Decode ablation → D VAD → E Selective fine-tune → F Full evaluation. Critical gates: B→C gate requires mask implementation validation; B→E hard gate (top_k_heads.json is required input to E); C∥D run in parallel; E→F sequential.

### Critical Pitfalls — Watch Out For

PITFALLS.md lists 19 numbered pitfalls across three severity tiers. The three HIGH-severity silent-failure landmines MUST be prevented by Phase A infrastructure gates, not discovered at analysis time.

1. **Pitfall #1 — `head_mask` silently ignored under SDPA/FlashAttention [KNOWN, HIGH].** HuggingFace's default backend does not support `head_mask`; it is dropped at the attention-backend boundary. Every head ablation looks identical to baseline. **Avoid:** Load with `attn_implementation="eager"`; assert `model.config._attn_implementation == "eager"` after load; run a pre-sweep sanity check that masking head 0 changes outputs on ≥5 utterances. Budget 2-3x slower inference. **Phase A gate.**
2. **Pitfall #2 — `head_mask` not propagated through KV-cached generation [KNOWN/LIKELY, HIGH].** Even with eager attention, `decoder_head_mask` is threaded into the first forward call but may not be re-applied on subsequent autoregressive decode steps. Token 1 is masked; tokens 2..T come from the unmasked model. **Avoid:** Do NOT rely on `decoder_head_mask` kwarg. Use `register_forward_pre_hook` that zeros the head at every invocation; verify with `output_attentions=True` that the target head is zero at decoded position >1, not just position 1. Target self-attention only (cross-attention cache is write-once, hooks won't re-fire). **Phase A gate.**
3. **Pitfall #9 — Whisper's temperature fallback confounds decoding ablation [KNOWN, HIGH].** `WhisperGenerationMixin` has a silent temperature-fallback loop keyed on `compression_ratio_threshold=2.4` and `logprob_threshold`. Your "beam_size=5" run may actually execute at temperature 0.8 on half the utterances — and the effect is most pronounced on the failure cases you care about. **Avoid:** Phase C must set `temperature=0.0`, `compression_ratio_threshold=None`, `logprob_threshold=None`, `condition_on_prev_tokens=False`, and verify with a deterministic re-decode check. Freeze `GenerationConfig` as JSON in the phase deliverables. **Phase C gate.**
4. **Pitfall #6 — Cherry-picking via same-split head identification and evaluation [KNOWN].** n=511 is small; using all 511 for both head selection and reporting inflates the reduction number. **Avoid:** Commit `indian_accent_splits.json` (60/20/20 diagnose/rank/test) BEFORE the first sweep. `dev_diagnose` for the 640-cell sweep; `dev_rank` for tiebreaking top-k; `test_eval` touched exactly once for the final reported number. **Phase A deliverable, hard gate for Phase B.**
5. **Pitfall #12 — WebRTC VAD is unsuitable for accent speech [KNOWN].** WebRTC VAD has 50% TPR at 5% FPR — deletes ~half of soft-spoken speech. Energy-threshold VAD also fails on Common Voice's variable amplitude (crowdsourced, ±15-30 dB variance). **Avoid:** Use silero-vad (87.7% TPR). Per-utterance peak/RMS normalize before VAD. Validate on clean Indian-accent audio BEFORE testing on silence-injection. **Phase D gate.**
6. **Pitfall #15 — Catastrophic forgetting on other accents [KNOWN].** Fine-tuning on 511 Indian-accent utterances shifts the prior and regresses Canadian/England/US/LibriSpeech WER. Class-project tunnel vision reports only the target metric and misses the fairness regression. **Avoid:** Mix training data 50/50 Indian + other accents + LibriSpeech; use LoRA-on-heads (rank 4-8) with L2-to-init regularization; gate every checkpoint on a multi-accent + LibriSpeech regression check (≤0.5 pt WER budget); reject checkpoints that degrade MMR regardless of Indian improvement. **Phase E hard gate.**
7. **Pitfall #17 — [PROJECT] 4-arm scope creep.** The synthesis-level risk. Plan lists 4 arms; class-project execution favors depth. **Avoid:** Declare Arm 1 (diagnosis) as the required deliverable and Arms 2-4 as stretch in PROJECT.md and REQUIREMENTS.md explicitly. Time-box: if diagnosis is not complete by deadline N, cut arms. Prefer "rigorous diagnosis + small rigorous fine-tune" over "4 shallow arms". **Roadmap-level gate.**

Also important (not in top 7 but phase-gating): **#3** wrong masking target (unit test: three implementations give identical logits — Phase A); **#4** cross-attn vs self-attn confusion (named-target comment in code — Phase A); **#5** Calm-Whisper head IDs are a prior not a conclusion (run full sweep before citing #1/#6/#11 — Phase B); **#7** bootstrap CI noise at n=306 (1000× bootstrap — Phase B); **#8** insertion/WER confounding (always report S/D/I + WER + Pareto plot — all phases); **#14** fine-tune collapse to identity or zero (pre/post weight norm diff — Phase E); **#18** baseline reproduction gate (re-run Whisper-large-v3 on current infra before intervening; confirm 9.62% ± 0.5% — Phase A); **#19** no fairness regression check (full per-accent + LibriSpeech + FairSpeech table automatic per experiment — Phase F).

## Implications for Roadmap

Based on the synthesized research, v2.0 should be structured as **6 phases with explicit dependency gates**, where Phases A and B are blocking for everything else, Phases C and D run in parallel as independent side-quests, Phase E is conditional stretch, and Phase F is the aggregation sink. This matches the ARCHITECTURE.md build order and enforces the pitfall mitigations.

### Phase A: Infrastructure & Baseline Reproduction
**Rationale:** Every measurement in Phases B-F depends on (a) the per-head hook primitive working correctly, (b) the baseline being reproducible on current infra, and (c) the split file being committed before any selection happens. Skipping Phase A means all downstream numbers are either invalid or cherry-picked. Three HIGH-severity pitfalls (#1, #2, #18) are prevented here.
**Delivers:**
- `scripts/whisper_head_hooks.py` library with unit tests (mask three ways → identical logits; masking head 0 changes outputs at decode position >1 via `output_attentions=True`).
- `run_inference.py` edits: force `attn_implementation="eager"`; add `experiment_tag` column to `IncrementalCSVWriter`; assert `_attn_implementation == "eager"` on load.
- `outputs/manifests/cv_indian_511.csv` (one-off build).
- `indian_accent_splits.json` (60/20/20 diagnose/rank/test, committed before first sweep).
- Baseline re-run of Whisper-large-v3 on 511 Indian CV + LibriSpeech test-clean + Fair-Speech, with frozen `baseline_generation_config.json`. **Gate: 9.62% ± 0.5% insertion rate reproduced.**
- Smoke test: `run_head_masking_sweep.py` on 5 utterances × 2 heads.
**Addresses (features):** T1, T7 (partial — verification), T9.
**Avoids (pitfalls):** #1 (SDPA silent ignore), #2 (KV cache × mask), #3 (wrong module), #4 (self-vs-cross), #6 (split leakage), #18 (baseline drift).
**Research flag:** **STANDARD** — patterns exhaustively documented in ARCHITECTURE.md Patterns 1-3 and PITFALLS.md #1-4, #18. Direct to plan.

### Phase B: Per-Head Diagnosis Sweep (PRIMARY DELIVERABLE)
**Rationale:** This is the required v2.0 contribution. If only one phase ships, it must be this one. A ranked list of accent-hallucination-driving heads on Whisper-large-v3 is a complete citable result — independent replication or refutation of Calm-Whisper's #1/#6/#11 finding on a new hallucination type.
**Delivers:**
- 640 prediction CSVs (one per layer × head position) + 1 baseline, under `outputs/head-surgery/diagnosis/`, tagged `experiment_tag=mask_L{L}H{H}`.
- `top_k_heads.json` — ranked list with bootstrap 95% CIs per head's insertion-rate delta (1000× bootstrap over utterances).
- Accent-adapted driving-ness metric (insertion-rate delta + repetition-only delta + non-target WER budget; threshold TBD by REQUIREMENTS author).
- Honest publication of ranking regardless of whether it matches Calm-Whisper.
- Per-head heatmap figure (D7) for the writeup.
**Addresses (features):** T1 (validated by hook unit test in A), T2, T3, T4, T9 partial.
**Avoids (pitfalls):** #5 (Calm-Whisper transferability — full sweep before citing), #7 (bootstrap CIs at n=306), #8 (S/D/I decomposition + Pareto).
**Research flag:** **STANDARD** — Calm-Whisper methodology is well-documented. Only open question (encoder-feature caching for speedup) is well-understood from Whisper architecture. Direct to plan.

### Phase C: Decoding Strategy Ablation (parallel with D)
**Rationale:** Answers the unavoidable reviewer question "couldn't you just turn on `repetition_penalty=1.3`?" with numbers. Independent of Phases B and D. **Cut first under time pressure** (Pitfall #17) — least novel arm.
**Delivers:**
- `scripts/run_decode_ablation.py` (new).
- 2D grid: `repetition_penalty ∈ {1.0, 1.1, 1.3}` × `no_repeat_ngram_size ∈ {0, 3, 5}` + two beam-size points ({1, 5}). Tight grid per Pitfall #10.
- Frozen `GenerationConfig` JSON with `temperature=0.0`, `compression_ratio_threshold=None`, `logprob_threshold=None`, `condition_on_prev_tokens=False`. Pitfall #9 mitigation.
- Deterministic re-decode sanity check as a gate.
- `ablation_summary.json` + per-cell prediction CSVs in `outputs/head-surgery/decode-ablation/`.
**Addresses (features):** T5.
**Avoids (pitfalls):** #9 (temperature fallback confound), #10 (parameter non-additivity), #11 (auto-collapsed repetition).
**Research flag:** **STANDARD** — HF `GenerationConfig` documentation is authoritative. Direct to plan.

### Phase D: Energy/Silero VAD Preprocessing (parallel with C)
**Rationale:** Requested in midterm §4.2. Independent of Phases B and C. Composes with existing silence-injection perturbation manifests as a flat cross-product — minimal new code. **Cut second under time pressure.**
**Delivers:**
- `scripts/apply_energy_vad.py` (new — manifest-in, manifest-out).
- Clean-audio validation gate (WER ≤ baseline + 0.5 on clean Indian + LibriSpeech) BEFORE running on silence-injected data. Pitfall #12 mitigation.
- VAD applied to 25/50/75% silence-injection perturbation manifests.
- `silero-vad` dependency added (not WebRTC).
- (Optional D3) comparison table energy-VAD vs silero-VAD.
**Addresses (features):** T8, optionally D3.
**Avoids (pitfalls):** #12 (WebRTC over-trim), #13 (threshold overfit to one condition).
**Research flag:** **STANDARD** — silero-vad API is documented; energy-RMS VAD is trivial. Direct to plan.

### Phase E: Selective Head Fine-Tuning (STRETCH, gated on B)
**Rationale:** Turns the diagnosis into an intervention and matches the full Calm-Whisper loop. Cannot start before Phase B publishes `top_k_heads.json`. Must be cut if B completes late. Highest risk phase (collapse, forgetting, overfit at n=511). **Per ML Training Phase Plan Structure in CLAUDE.md, this phase MUST be split into infrastructure → VRAM maximization → launch sub-plans.**
**Delivers:**
- `scripts/training/finetune_heads.py` (new, separate from v1.0 training code).
- Frozen model + gradient-masking hooks on target head slices; post-step `||Δw||==0` unit test on non-target slices (hard gate).
- LoRA-on-heads variant (rank 4-8) with L2-to-init regularization, per Pitfall #14.
- Training config matches Calm-Whisper: LR=1e-6, ≤5 epochs, early stop on `dev_rank` insertion rate.
- Mixed training data (50/50 Indian + other accents + LibriSpeech sample, per Pitfall #15).
- Multi-accent + LibriSpeech regression gate (≤0.5 pt WER budget on every non-target group) — reject checkpoints that fail.
- Checkpoint at `outputs/head-finetune/run_<tag>/`; `MODEL_REGISTRY` entry added.
- W&B logging + monitoring per `babysit-training` skill.
- VRAM tuning via `tune_vram.py` / `maximize-vram` skill pattern.
**Addresses (features):** D1 + D1a.
**Avoids (pitfalls):** #14 (collapse), #15 (forgetting), #16 (overfit).
**Research flag:** **NEEDS RESEARCH** — the choice between LoRA-on-head-slices and direct-unfreeze-with-grad-masking, and the exact target-module surgery, are unverified at the library level (no PEFT per-head support). No public Calm-Whisper code. Training-signal choice (D1a clean MLE vs D1b contrastive) has methodological implications. **Run `/gsd-research-phase` before plan authoring for E.**

### Phase F: Full Evaluation & Writeup Tables (SINK)
**Rationale:** Aggregates all of B, C, D, E into the required reporting format. Per Pitfall #19 (fairness regression check), every intervention must be evaluated on the full 12-perturbation × 2-dataset matrix via the existing `run_inference.py` plus new fine-tuned `MODEL_REGISTRY` entries — no new evaluation code.
**Delivers:**
- Full fairness tables (6 CV accents + 7 Fair-Speech ethnicity + LibriSpeech) for baseline, head-masked, decode-ablated, VAD-on, and (if E ships) fine-tuned.
- MMR before/after.
- Per-head heatmap, Pareto plots (insertion rate vs WER), S/D/I decomposition, compression-ratio metrics.
- Writeup-ready figure set via existing `generate_all_plots.py`.
- `test_eval` split touched exactly once.
**Addresses (features):** T6, D7.
**Avoids (pitfalls):** #8 (insertion/WER confounding — full decomposition), #11 (auto-collapsed repetition — report compression ratio), #19 (fairness regression — automatic multi-accent table).
**Research flag:** **STANDARD** — reuses existing `compute_fairness_metrics.py`, `error_decomposition.py`, `whisper_hallucination_analysis.py`, `generate_all_plots.py`. Zero new metrics code.

### Phase Ordering Rationale

- **A blocks everything** because of Pitfalls #1, #2, #6, #18. The eager-attention assertion, the forward-hook sanity check, the committed split file, and the baseline reproduction are all load-bearing for the validity of downstream numbers. Cannot be skipped or reordered.
- **B blocks E** because E consumes `top_k_heads.json` as input. Hard gate — no training script can run without the ranked head list.
- **C and D are independent** of B and of each other; can run in parallel waves once A is complete. This lets the team make progress on C/D while B's 640-cell sweep is compute-bound.
- **E is stretch and conditional on B finishing on time.** The "ship diagnosis first" principle (Pitfall #17) means E must be cut under time pressure, and the plan must explicitly state this.
- **F is a sink**, not a new contribution — it's the existing metrics pipeline invoked with `experiment_tag` filters. Lightweight.
- **C is the first arm to cut** under time pressure (least novel — decoding params are well-studied). **D is the second arm to cut** (useful but composes with existing perturbations; not a headline result). This ordering matches the Pitfall #17 prioritization and the "ship diagnosis first, fine-tune if time permits" thesis.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase E (Selective fine-tune):** No public Calm-Whisper code exists (exhaustive search returned nothing). No library (PEFT, TransformerLens, nnsight, captum) provides per-head fine-tuning as a first-class API. The choice between LoRA-on-head-slices (PEFT can't do this natively) and direct-unfreeze-with-grad-masking (≈200 lines of custom code with a critical `||Δw||==0` unit test) is unverified at the library level. The Calm-Whisper recipe (LR=1e-6, 5 epochs, CE on empty strings) must be adapted to speech labels on a small Indian-accent CV split — the training-signal choice (D1a clean MLE vs D1b contrastive vs D1c RL) has methodological implications. **Run `/gsd-research-phase` before plan authoring for E.**

**Phases with standard patterns (skip research-phase):**
- **Phase A (Infrastructure):** Direct to plan — PITFALLS.md #1-4, #18 + ARCHITECTURE.md Patterns 1-3 are authoritative.
- **Phase B (Diagnosis):** Direct to plan — Calm-Whisper methodology + STACK.md Path B hook mechanics are authoritative.
- **Phase C (Decode ablation):** Direct to plan — HF `GenerationConfig` API + PITFALLS.md #9, #10 scope lock are authoritative.
- **Phase D (VAD):** Direct to plan — silero-vad API documented.
- **Phase F (Evaluation):** Direct to plan — reuses existing metrics scripts unchanged.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | Source-verified across 6 transformers releases; `layer_head_mask` removal in v5 confirmed by grep on main; SDPA fallback confirmed by direct quote from v4.44 source; hook algebraic correctness trivial; silero vs WebRTC numbers triangulated across Picovoice 2025 + pyannote #604 + py-webrtcvad #68. No public Calm-Whisper code (exhaustive search). |
| Features | **HIGH** on Calm-Whisper protocol numbers (arxiv 2505.12969 anchors the quantitative claims exactly); **MEDIUM** on accent-vs-non-speech transferability (Pitfall #5 — the same heads may or may not be responsible; Calm-Whisper evaluated on the same UrbanSound set they diagnosed on, so their numbers may be optimistically biased); **MEDIUM** on selective-fine-tuning training-signal choice (D1a clean MLE is simplest but gradient may be weak on already-correct samples; D1b contrastive is more principled but more code; D1c RL pulls v1.0 GRPO back in and is not recommended). |
| Architecture | **HIGH** | Integration with existing pipeline is well-understood (`MODEL_REGISTRY` dict, `load_whisper`/`infer_whisper` factored out, `IncrementalCSVWriter` extensible, metrics scripts unchanged). Only one file touched in existing code. All six build-order dependencies are concrete. |
| Pitfalls | **HIGH** for HuggingFace/Whisper integration pitfalls (#1-4, #9, #12 are all KNOWN — cited from transformers source, Whisper `transcribe.py`, Picovoice 2025, MAS-LoRA); **MEDIUM** for statistical and training-dynamics pitfalls (#7, #8, #14, #16 are LIKELY — inferred from methodology and not yet empirically verified on this codebase); **HIGH** for class-project scope traps (#17-19 are PROJECT-level and directly actionable). |

**Overall confidence:** **HIGH** for Phases A, B, C, D, F; **MEDIUM** for Phase E (which is also the stretch phase and can be cut if confidence concerns dominate).

### Gaps to Address — Open Questions for the REQUIREMENTS Author

These are the open questions consolidated across STACK.md, FEATURES.md, ARCHITECTURE.md, and PITFALLS.md. They belong to the REQUIREMENTS author for v2.0:

1. **Masking threshold for T3 (driving-ness metric):** What insertion-rate reduction qualifies a head as "hallucination-driving"? Absolute delta? Relative? Bootstrap-significant (95% CI lower bound < 0)? Calm-Whisper used a raw rate drop on a binary metric; accent is continuous. **Decision needed in REQUIREMENTS.**
2. **Regression budget for T3:** How much WER degradation on a non-Indian accent group is acceptable when masking a head? Calm-Whisper accepted <0.1% on LibriSpeech; v2.0 equivalent TBD. **Synthesis recommendation: 0.5 absolute WER points** (aligns with Pitfall #8 and #19 gates).
3. **Fine-tuning (D1) in or out of MVP?** **Synthesis recommendation: STRETCH** (D1 cut if diagnosis is late). REQUIREMENTS must commit to this explicitly or override with time-box justification.
4. **Parallelization of the 640-cell diagnosis sweep:** Serial (safe, ~hours with encoder caching) vs layer-parallel data-parallel (faster, more bug surface). Infra decision for Phase A/B. **Synthesis recommendation: Serial with cached encoder outputs** — simplest, plenty fast on a 49 GB GPU.
5. **Which CV24 non-Indian group is the regression guard?** England/US have largest n but England's WER quadruples under 75% silence per midterm — poor guard. **Synthesis recommendation: Canadian or American as primary guard; LibriSpeech test-clean as secondary.**
6. **Generation-time `decoder_head_mask` with KV cache [Pitfall #2]:** Not explicitly documented in any transformers version. Phase A gate must include "zero-mask on head 0; verify output differs from unmasked at decoded position > 1" (not just position 1). This is the concrete test that prevents Pitfall #2. **Must be a hard gate in the Phase A plan.**
7. **Whisper-large-v3 layer × head geometry:** 32 layers × 20 heads = 640 positions. Calm-Whisper's "#1/#6/#11" refers to head *indices* replicated across 32 layers (= 3 × 32 = 96 trainable slices). The diagnosis sweep should be flat over 640 positions (more informative) and is still feasible with encoder-output caching. **Synthesis recommendation: flat 640-cell sweep, visualized as 32×20 heatmap.**
8. **Fine-tuning data for D1a:** "Accent-diverse" per PROJECT.md but "no new labeled data" per constraints. **Synthesis recommendation: reuse a held-out CV24 training split with accent labels (NOT the 511-eval subset); mix 50/50 Indian + other accents + LibriSpeech per Pitfall #15.** Data-prep detail, not stack.
9. **VAD threshold tuning (Phase D):** silero default is 0.5; per-severity tuning (25/50/75% injection) may be needed. **Must tune on mixed clean + perturbed dev set, not overfitted to one condition** (Pitfall #13).
10. **Scope lock for decoding ablation (Phase C):** Commit to the 2D grid + beam pair (NOT the full 4D 81-cell sweep) as a Phase C gate — prevents Pitfall #10 scope blow-up. **Synthesis recommendation: `repetition_penalty ∈ {1.0, 1.1, 1.3}` × `no_repeat_ngram_size ∈ {0, 3, 5}` + beam ∈ {1, 5}.**
11. **MVP arm prioritization under time pressure:** If the class-project deadline tightens, what is the cut order? **Synthesis recommendation: C first (least novel), then D (useful but composes with existing infra), then E (stretch).** B is never cut.

## Sources

### Primary (HIGH confidence)
- **Calm-Whisper (Wang et al., Interspeech 2025):** https://arxiv.org/abs/2505.12969 · https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.html
- **HuggingFace transformers Whisper source (v4.44 + main):** https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/whisper/modeling_whisper.py
- **HuggingFace transformers attention backends documentation:** https://huggingface.co/docs/transformers/en/attention_interface
- **OpenAI Whisper transcribe.py source:** https://github.com/openai/whisper/blob/main/whisper/transcribe.py
- **Picovoice 2025 VAD benchmark:** https://picovoice.ai/blog/best-voice-activity-detection-vad-2025/
- **silero-vad 6.2.1:** https://github.com/snakers4/silero-vad
- **Midterm report** `llm-asr-fairness-midterm.pdf` §3.2, §4.2
- **`.planning/PROJECT.md`** — v2.0 milestone scope + constraints
- **`scripts/run_inference.py`** — `MODEL_REGISTRY`, `load_whisper()`, `infer_whisper()`, `IncrementalCSVWriter`

### Secondary (MEDIUM confidence)
- **Investigation of Whisper ASR Hallucinations (arxiv 2501.11378):** https://arxiv.org/html/2501.11378v1
- **Beyond Transcription: Mechanistic Interpretability in ASR (arxiv 2508.15882):** https://arxiv.org/html/2508.15882
- **MAS-LoRA (arxiv 2505.20006):** https://arxiv.org/html/2505.20006
- **TransformerLens model list:** https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py
- **HF transformers issue #27044 (head pruning):** https://github.com/huggingface/transformers/issues/27044

### Tertiary (LOW confidence)
- **Listen Like a Teacher (arxiv 2511.14219):** Calm-Whisper follow-up
- **Whisper-CD contrastive decoding (arxiv 2603.06193):** alternative decoding strategy

---
*Research completed: 2026-04-11*
*Supersedes v1.0 GRPO summary at this path*
*Ready for REQUIREMENTS author: yes*
