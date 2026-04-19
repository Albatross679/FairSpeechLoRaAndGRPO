# Head-Surgery Diagnosis — Results

Target model: **Whisper-large-v3** (HF revision `06f233fe`). Subgroup: Indian-accent test utterances from **Common Voice v25**, strict single-label filter, N = 484 on-disk. All numbers reproducible via `scripts/head_surgery/repro_config.py`.

This report is organized in **pipeline order** — each stage's methodology is followed immediately by its result.

---

## 0. TL;DR

1. **Primary result — negative.** Of 640 decoder self-attention heads, exactly **1** reaches bootstrap significance (p<0.05) for reducing hallucinations, and its effect is just **−0.08pp**. There is no surgical target.
2. **Secondary result — positive (keystone heads).** A small cluster of layer-0 and late-layer heads is *load-bearing*: masking **L=0, h=5** catastrophically breaks Whisper (insertion rate 1.27% → **101%**, non-Indian WER 6.7% → 73%). ~7 other heads produce a milder ~10% Indian-specific failure mode without hurting non-Indian audio — candidates for fine-tuning, not removal.
3. **Dataset caveat.** The baseline on CV25 is **1.27%**, not the midterm's 9.62% (CV24). The hallucination-driving hypothesis is redefined; this run is a reproducibility audit, not a midterm reproduction.

---

## 1. Dataset acquisition

### 1.1 Srishti's midterm (CV24) vs this milestone (CV25)

| Axis | Srishti (midterm) | This milestone |
|---|---|---|
| Dataset | **Common Voice v24** | **Common Voice v25** |
| Release date | **2025-12-05** (`cv-corpus-24.0-2025-12-05`) | **2026-03-09** (`cv-corpus-25.0-2026-03-09`) |
| Tarball | OSC path (not accessible from this VM) | `cv-corpus-25.0-en.tar.gz` (81.5 GB) — **truncated**, `gzip: unexpected EOF` on 2 independent B2 downloads |
| Filter | strict single-label `accents == "India and South Asia"` | same |
| Indian-accent test N | 511 | **484** (510 filtered − 26 past EOF) |
| Non-Indian test N (regression guard) | 500 | **422** (500 − 78 past EOF) |
| Baseline insertion rate | **9.62%** | **1.27%** (rep 0.00% / syn 0.41% / con 0.86%) — see §4 |

Reproducibility: [repro_config.py:51](../scripts/head_surgery/repro_config.py#L51), manifest `tests/fixtures/head_surgery/indian_accent_ids.json`. The ~8× drop between CV24 and CV25 on this subgroup is the root cause of the negative primary result — head masking cannot improve a near-floor baseline.

### 1.2 CV24 availability audit (2026-04-18)

CV24 was not obtainable through any public channel Srishti's original scripts assume:

| Channel | Status |
|---|---|
| Mozilla Data Collective (Common Voice org) | **No full `cv-corpus-24.0-en`** — only a 1.92 GB *en-AU conference subset* is listed; main listings moved to *Scripted Speech 25.0* |
| HuggingFace mirror (`mozilla-foundation/common_voice_*`) | **Not mirrored** — API probe: `17_0` ✅, `18_0`–`24_0` → HTTP 404 |
| OSC path Srishti used (`/users/PAS2030/srishti/…/cv-corpus-24.0-2025-12-05/en`) | Hard-coded in [prepare_splits.py:27](../scripts/data/prepare_splits.py#L27); no cross-mount from this VM |

Reproducing the midterm's CV24 Indian-accent test set therefore requires either direct OSC access or Mozilla/HF re-publishing a full v24 English bundle.

---

## 2. Audio & text preprocessing

### 2.1 Audio (MP3 → Whisper input)

Every audio file goes through the same pipeline before Whisper sees it — baseline, per-head sweep, decoding ablation, regression guard, and silence-injection all share this path ([run_diagnosis_sweep.py:35-49](../scripts/head_surgery/run_diagnosis_sweep.py#L35-L49)).

| Step | Operation | Tool | Why |
|---|---|---|---|
| 1 | Decode MP3 (CV25 native format) | `librosa` + `audioread` | `torchaudio` requires `torchcodec`, whose native lib fails to load on this VM; librosa handles MP3 out of the box |
| 2 | **Resample 48 kHz → 16 kHz** | `librosa.load(sr=16000)` | Whisper's log-mel expects 16 kHz; feeding 48 kHz warps the spectrogram 3× |
| 3 | Downmix to mono | `librosa.load(mono=True)` | Whisper input is single-channel |
| 4 | Log-mel feature extraction | `WhisperProcessor(sampling_rate=16000, padding=True)` | Standard Whisper front-end; pads shorter clips to 30 s |
| 5 | dtype cast | `.to(device, dtype=float16)` | fp16 inference on RTX A6000 |

**Known gotcha:** An early Stage A run skipped the resample step and passed 48 kHz audio with `sampling_rate=16000` to the processor. Result: **58% insertion rate** (hallucination loops from the warped spectrogram). Pinning resample-before-processor was the fix (commits `77ca299`, `3cecd6f`).

**Caching:** audio is decoded once per utterance and held in RAM across the 640-head sweep ([run_diagnosis_sweep.py:179](../scripts/head_surgery/run_diagnosis_sweep.py#L179)). No per-head redecode.

**Audio duration distribution** (from CV25's `clip_durations.tsv`, no decode needed):

![Audio duration histogram](audio_duration_histogram.png)

| Subset | N | Mean | Median | Min | Max | Std | P5 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Indian-accent | 484 | 6.16 s | 6.09 s | 1.87 s | 13.58 s | 1.83 s | 3.31 s | 9.28 s |
| Non-Indian | 422 | 6.25 s | 6.16 s | 2.66 s | 12.82 s | 1.65 s | 3.63 s | 9.24 s |

Both subsets are near-identical in duration (means differ by <0.1 s), so any subgroup gap in hallucination rate is not explained by clip-length confounds. All clips fit inside Whisper's 30-second context window — the `padding=True` step in step 4 pads every clip to 30 s.

### 2.2 Text normalization (for WER / insertion classifier)

Applied identically to **reference and hypothesis** before any metric is computed ([run_inference.py:128-145](../scripts/inference/run_inference.py#L128-L145)):

| Operation | Example |
|---|---|
| Case fold | `"YES"` → `"yes"` |
| Punctuation removal | `"hello."` → `"hello"` |
| Number normalization | `"six"` ↔ `"6"` |
| Contraction expansion | `"she'll"` → `"she will"` |

Implementation: Whisper's `EnglishTextNormalizer`. Matches the midterm's WER convention.

### 2.3 Insertion classifier

Each hallucinated token in the hypothesis is categorized into one of three disjoint buckets ([insertion_classifier.py](../scripts/head_surgery/insertion_classifier.py)):

| Category | Definition |
|---|---|
| `repetition` | N-gram loops (e.g., "thank you thank you thank you…") |
| `syntactic_completion` | Fillers, grammatical padding (e.g., "you know", "and so on") |
| `content_hallucination` | Fabricated content not in reference |

---

## 3. Model & inference setup

| Knob | Value |
|---|---|
| Model | `openai/whisper-large-v3` @ revision `06f233fe` |
| Precision | fp16 on CUDA |
| Seed | 20260417 |
| Generation config | `{max_new_tokens: 440, language: "en", task: "transcribe"}` + default temperature fallback |
| Batch size | 32 (tuned in Stage A.5) |

### Hardware

| Component | Spec |
|---|---|
| GPU | 1× **NVIDIA RTX A6000** (48 GB GDDR6, driver 580.95.05, bf16 ✅) |
| CPU | **AMD EPYC 9554** (64-core / 252 threads visible, single NUMA node) |
| System RAM | 32 GiB |
| Disk | 98 GB overlay fs (7.2 GB used) |
| Framework | PyTorch 2.10.0 + CUDA 12.8 |
| Peak VRAM during Stage C | ~11.4 GB @ bs=32, fp16 |

---

## 4. Stage A — Baseline inference

**Method.** Run Whisper-large-v3 unmodified on the 484 Indian-accent utterances. Compute insertion rate (total and by category) using the classifier in §2.3. Also tune batch size in a paired Stage A.5 pass (bs ∈ {8, 16, 24, 32, 40}) to find the maximum-throughput configuration that fits in 48 GB VRAM.

**Result.**

| Metric | Value |
|---|---:|
| Total insertion rate | **1.27%** |
| Repetition | 0.00% |
| Syntactic completion | 0.41% |
| Content hallucination | 0.86% |
| Chosen batch size | **32** @ 14.12 utts/s, 11.43 GB peak |
| Wall-clock | ~3 min (A + A.5) |
| Gate G1 (midterm parity) | **redefined** — CV24 target 9.62% unreachable |
| Gate G1.5 (batch size tune) | ✅ PASS |

Artifacts: `baseline_predictions.csv`, `baseline_metrics.json`, `tune_batch_size.json`.

---

## 5. Stage B — Pilot sweep (sanity check)

**Method.** Random 50-utterance subsample. Run every (L, h) masking on one or two pilot layers to verify:

1. The hook actually changes predictions (not a silent no-op).
2. Batched per-sample masking matches serial single-head masking (within fp16/RNG noise).
3. Per-head signal is measurable on this small sample.

**Result.**

| Check | Outcome |
|---|---|
| Hook activity verified | ✅ via hypothesis-diff inspection of `pilot_sweep.csv` |
| Pilot baseline insertion rate | 0.97% |
| Gate G2 (per-head signal) | ⚠️ WARN — 0.97% < 2% single-event quantization floor |
| Wall-clock | ~2 min |

Decision: proceed to full sweep despite G2 warning — quantization is expected to resolve with 484 utts instead of 50.

---

## 6. Stage C — Full head-masking sweep

### 6.1 Head-masking hook

For each of the 640 decoder self-attention (layer, head) cells, a PyTorch forward hook zeros out the chosen head's output projection ([head_mask_hook.py](../scripts/head_surgery/head_mask_hook.py)). A batched variant applies a per-sample mask so one inference pass can evaluate multiple heads simultaneously.

### 6.2 Sweep protocol — batched inference

Three nested loops over layers × heads × utterance batches ([run_diagnosis_sweep.py:349-359](../scripts/head_surgery/run_diagnosis_sweep.py#L349-L359)):

| Loop level | Range | Action |
|---|---|---|
| Outer | 32 layers | install `BatchedHeadMaskHook` on this layer |
| Middle | 20 heads | set per-sample mask `[batch, num_heads]` with zero for head `h` |
| Inner | ⌈484/32⌉ = 16 batches | call `Whisper.generate()` once per 32-utt batch |

**Batching math:**

| Level | Count | Notes |
|---|---:|---|
| Utterances per batch | **32** | chosen in Stage A.5, peak 11.4 GB VRAM |
| Batches per head | ⌈484 / 32⌉ = 16 | one head masked, all 32 utts in parallel |
| Heads per layer | 20 | serial within layer |
| Layers | 32 | hook reinstalled once per layer |
| **Total `Whisper.generate()` calls** | **~10,240** | vs ~309,760 if serial per utt — 30× fewer kernel launches |

Only one form of batching is exploited: **32 utterances per `generate()` call**, one head at a time. Multi-head batching (masking different heads on different samples in the same call) would divide call count by another ~20× but would invalidate the G3 (`batched ≈ serial`) parity proof, and the A6000 was already VRAM-saturated at bs=32.

Each call emits one row per utterance into `sweep.csv` (hypothesis + insertion-rate breakdown), for 309,760 total rows.

### 6.3 Result

| Metric | Value |
|---|---|
| Rows emitted | **309,760** (640 heads × 484 utts) |
| Wall-clock | **~6 h 50 min** (log: `[640/640, 409.5min]`) |
| Gate G3 (batched ≈ serial) | ✅ PASS — max \|Δ\| = 0.194% (1-utt RNG), mean \|Δ\| = 0.019% |
| Gate G4 (sweep completeness) | ✅ PASS — 309,760 rows |

Artifact: `sweep.csv` (53 MB).

---

## 7. Stage D — Scoring & significance

### 7.1 Δ statistic

For head (L, h):

$$\Delta_{L,h} = \text{insertion\_rate}_\text{baseline} - \text{insertion\_rate}_\text{masked}$$

Positive Δ ⇒ masking reduces hallucinations.

### 7.2 Paired bootstrap

**Problem.** We ran the experiment once on a single sample of 484 utterances and got one Δ per head. We need to tell apart "real effect" from "happened to land on a favorable subset of the 484."

**Mechanic — for each head, repeat 10,000 times:**

1. Draw 484 indices at random from `{0..483}`, **with replacement**.
2. Look up `baseline_count[idx]` and `masked_count[idx]` for those 484 indices.
3. Compute Δ_k = (baseline insertion rate on the resample) − (masked insertion rate on the resample).

**Definitions:**

| Term | Meaning |
|---|---|
| **With replacement** | Each draw is independent of the previous draws — the same utterance can appear 2 or 3 times in one resample; others may not appear at all. |
| **Paired** | The same 484 indices feed both the baseline and the masked counts. Cancels utterance-level "difficulty" noise; only the head's effect varies between the two sides. |
| **p-value** | `(#resamples where Δ_k ≤ 0) / 10,000`. Low p ⇒ Δ rarely flipped negative across the fake samples ⇒ effect is unlikely to be noise. |
| **Null hypothesis H0** | Δ ≤ 0 (masking does not help). One-sided. |

**Run config:**

| Parameter | Value |
|---|---|
| Resampling instances per head | **10,000** |
| Sample size per resample | 484 (matches the original pool) |
| α threshold | 0.05, **no multiple-testing correction** across 640 tests |
| Total Δ computations | 6.4 M |

Source: [score_heads.py:42-69](../scripts/head_surgery/score_heads.py#L42-L69). Bootstrap reuses Stage C per-utterance counts — no re-inference.

### 7.3 Result — Δ distribution across 640 heads

| Regime | Count | Interpretation |
|---|---:|---|
| Δ > 0 (masking reduces insertions) | 135 | best: L=20 h=11, −0.08pp, p=0.046 |
| Δ = 0 (no effect) | 376 | — |
| Δ < 0 (masking worsens insertions) | 129 | worst: L=0 h=5, +100.16pp (see §8) |
| p < 0.05 (bootstrap) | **1** | L=20 h=11 |

### 7.4 Result — Top-10 hallucination-driving heads (after regression guard)

| layer | head | Δ total | Δ rep | Δ syn | Δ con | p-val | reg. ok | non-Indian WER |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 20 | 11 | 0.001 | 0.000 | 0.001 | -0.000 | **0.046** | ✅ | 6.6% |
| 0 | 15 | 0.001 | 0.000 | 0.001 | 0.000 | 0.051 | ✅ | 6.6% |
| 22 | 19 | 0.001 | 0.000 | 0.001 | 0.000 | 0.051 | ✅ | 6.7% |
| 7 | 6 | 0.001 | 0.000 | 0.001 | -0.000 | 0.128 | ✅ | 6.7% |
| 10 | 8 | 0.001 | 0.000 | 0.001 | -0.000 | 0.374 | ✅ | 6.7% |
| 25 | 5 | 0.001 | 0.000 | 0.001 | -0.000 | 0.051 | ✅ | 6.6% |
| 0 | 14 | 0.001 | 0.000 | 0.001 | -0.000 | 0.239 | ✅ | 6.8% |
| 13 | 17 | 0.000 | 0.000 | 0.001 | -0.000 | 0.132 | – | – |
| 12 | 8 | 0.000 | 0.000 | 0.001 | -0.000 | 0.138 | – | – |
| 11 | 11 | 0.000 | 0.000 | 0.001 | -0.000 | 0.133 | – | – |

Heatmap of Δ across the 32×20 grid: [head_surgery_heatmap.png](head_surgery_heatmap.png). Full 50-head ranking: [Appendix A](#appendix-a--full-50-head-ranking).

**Wall-clock (Stage D scoring):** ~5 min for the 640-head bootstrap (no inference).

---

## 8. Stage D (cont.) — Regression guard & keystone-head finding

### 8.1 Method

For each candidate head with large |Δ|, rerun Whisper with that head masked on the **non-Indian** CV25 subset (N=422) and measure composite WER. A head passes if non-Indian WER degradation ≤ 0.5pp ([score_heads.py:128-203](../scripts/head_surgery/score_heads.py#L128-L203)). Scope: top-50 heads by |Δ| (compute-bound).

Purpose: ensure a proposed "surgical target" doesn't fix Indian-accent hallucinations at the cost of breaking transcription for everyone else.

### 8.2 Result — keystone (hallucination-suppressing) heads

Masking a small cluster of heads damages transcription. These are the opposite of hallucination drivers: they are *hallucination suppressors* (load-bearing circuits). All four metrics — insertion rate and WER, on Indian and non-Indian — are reported per head, vs the unmasked baseline `(Indian 1.27% / 10.93% · non-Indian 0.85% / 6.65%)`. Non-Indian columns derive from a follow-up rerun ([scripts/head_surgery/_8b2_keystone_non_indian_insertion.py](../scripts/head_surgery/_8b2_keystone_non_indian_insertion.py); the original Stage D guard stored only aggregate non-Indian WER, not per-utterance hypotheses needed for an insertion-rate breakdown).

| (L, h) | Indian ins | Indian WER | Non-Indian ins | Non-Indian WER | Regression |
|---|---:|---:|---:|---:|:---:|
| baseline | 1.27% | 10.93% | 0.85% | 6.65% | (reference) |
| **0, 5** | **101.43%** | **120.88%** | **52.26%** | **67.09%** (+60.4 pp) | **❌ FAIL** — catastrophic on both subgroups |
| 0, 13 | 10.19% | 20.20% | 0.97% | 6.86% (+0.21 pp) | ⚠️ borderline non-Indian, large Indian damage |
| 0, 18 | 10.13% | 19.86% | 0.88% | 6.70% (+0.05 pp) | ✅ non-Indian / large Indian damage |
| 0, 1  | 10.11% | 19.92% | 0.90% | 6.58% (−0.07 pp) | ✅ non-Indian / large Indian damage |
| 29, 18 | 10.09% | 19.82% | 0.88% | 6.70% (+0.05 pp) | ✅ non-Indian / large Indian damage |
| 27, 15 | 10.09% | 19.67% | 0.85% | 6.67% (+0.02 pp) | ✅ non-Indian / large Indian damage |
| 11, 9 | 10.07% | 19.67% | 0.85% | 6.65% (+0.00 pp) | ✅ non-Indian / large Indian damage |
| 13, 19 | 10.07% | 19.77% | 0.88% | 6.67% (+0.02 pp) | ✅ non-Indian / large Indian damage |

Two patterns are sharper now that all four metrics are visible:

- **L=0 h=5 is a universal keystone.** Indian metrics blow up (101 / 121 pp); non-Indian metrics also blow up (52 / 67 pp). Removing it breaks Whisper for *every* accent. Single point of failure.
- **The other 7 heads are Indian-accent-specific keystones.** Indian metrics jump ~9 pp insertion / ~9 pp WER; non-Indian metrics stay within ~0.2 pp on both insertion rate AND WER. Whisper has heads that selectively serve Indian-accent transcription — masking them transparently breaks Indian-accent audio without disturbing other accents at all. This is a stronger localization claim than the prior wording suggested: not "load-bearing in general" but "**accent-specifically load-bearing**." These are **fine-tuning candidates** (recover what they do for Indian-accent without removing them), not removal targets.

**Wall-clock:** original Stage D regression guard ~25 min for 38/50 candidates. Follow-up 8-keystone × 422-utt non-Indian rerun (Non-Indian columns above) ~6.6 min on the same A6000.

---

## 8b. Stage D (cont.) — Fixing-set analysis

### 8b.1 Method

Reframes the question from "which single head reduces hallucination the most on average?" (§7) to "for each of the 45 Indian-accent utterances with ≥1 hallucinated token, which (L, h) masks eliminate at least one token of that utterance — and what is the minimum head set whose union covers all of them?"

A head (L, h) is considered **valid** for the fixing set only if all three hold:
1. It strictly reduces the insertion count on ≥1 affected utterance.
2. It introduces no new insertions on any utterance in the pool (global no-harm).
3. It passes the non-Indian regression guard from §8 (`regression_ok=True` or `regression_checked=False`).

A binary coverage matrix `[n_affected × n_valid_heads]` is then solved two ways: **greedy** (picks the column covering the most uncovered rows, repeats) and **ILP optimal** via `scipy.optimize.milp` (for comparison). Source: [`scripts/head_surgery/fixing_set_analysis.py`](../scripts/head_surgery/fixing_set_analysis.py).

### 8b.2 Result — coverage statistics

| Metric | Value |
|---|---:|
| Affected utterances (baseline insertion count > 0) | **45** |
| Valid heads after three filters | **115** (of 640) |
| Greedy cover size | **8** |
| ILP optimum cover size | **8** |
| Unhelpable utterances (no single-head mask can fix) | **30** |
| Analysis runtime | 66.22 s (no GPU) |

### 8b.3 Result — greedy ordering

| Order | (Layer, Head) | Newly covered utterances | Cumulative coverage |
|---:|---|---:|---:|
| 1 | (L=0, h=15) | 3 | 3 |
| 2 | (L=20, h=11) | 3 | 6 |
| 3 | (L=22, h=19) | 3 | 9 |
| 4 | (L=25, h=16) | 2 | 11 |
| 5 | (L=11, h=11) | 1 | 12 |
| 6 | (L=13, h=17) | 1 | 13 |
| 7 | (L=16, h=13) | 1 | 14 |
| 8 | (L=20, h=6) | 1 | 15 |

### 8b.4 Unhelpable utterances

30 utterances have at least one hallucinated token at baseline that **no valid single-head mask** can eliminate under the three-filter criterion. Their IDs are listed in [`minimum_surgical_set.json`](../outputs/head_surgery/minimum_surgical_set.json). These represent the floor of what single-head masking can achieve on this dataset.

Per-utterance references, hypotheses, and categorised insertions for all 45 affected utterances (15 helpable + 30 unhelpable) are tabulated in [Appendix B](#appendix-b--per-utterance-baseline-hallucinations).

### 8b.5 Interpretation caveats

The numbers in §8b.2–§8b.4 are the mechanically-correct output of the three-filter + min-set-cover formulation, but five interpretive limits bound what they can be cited as supporting:

1. **The cover is a *necessary* condition, not *sufficient*.** Size 8 means no fewer than 8 single-head masks could possibly cover the 15 helpable utterances under these filters — **if** multi-head masking decomposed linearly. The sweep is single-head, so masking all 8 simultaneously is not guaranteed to fix any of them. Interaction effects may *reduce* coverage (redundant circuits cancel) or *change* it non-monotonically (two individually safe heads combined can introduce new harm). Validating the cover requires a separate GPU run that installs the entire 8-head set together.

2. **"30 unhelpable" is conditional on the three filters.** Some of these utterances have single-head masks that fix them but introduce new hallucinations elsewhere (filter ii) or fail the non-Indian regression guard (filter iii). They are unhelpable *under this criterion*, not in an absolute sense. Relaxing filter (ii) to "no new repetition-class harm" or adding a head-level damage budget would change the count.

3. **Filter (ii) has no tolerance band.** A single utterance going from baseline insertion count 0 → 1 under a masked condition — even due to fp16 + RNG jitter, which the report has documented in Gate G3 — eliminates an otherwise-safe head. With 484 utterances each rolling the "new insertion due to noise" die once per 640 conditions, the `n_valid_heads = 115` number has unquantified sensitivity to single-token variance. A statistical relaxation (e.g., reject only if harm is significant at p<0.05 across utterances) would likely raise n_valid_heads and lower the cover size.

4. **`n_valid_heads = 115` was data-derived, not predicted.** The implementation plan estimated "empirically likely ≤50"; the actual count is 2.3× larger. More heads than anticipated clear the three filters — this is a finding *from* the analysis, not a calibration *for* it.

5. **greedy = ILP = 8 is empirically optimal on *this* matrix, not provably so in general.** Greedy min-set-cover is a log-factor approximation; it coincides with ILP here because of the specific coverage structure (long tail of singleton-covering heads after the top 4). A slightly different sweep could produce a matrix where greedy overshoots ILP by 1–2 heads.

**Downstream claims to avoid:** "masking these 8 heads fixes 15 utterances" (unverified — see #1); "only 115 of 640 heads are safe to mask" (unquantified noise sensitivity — see #3); "greedy is provably optimal here" (empirical coincidence — see #5).

### 8b.6 Cross-reference to §8

The catastrophic keystone head **L=0 h=5** (§8.2, +100.16 pp) is by construction excluded from the fixing set (filter 2). The sole bootstrap-significant head from §7.4, **L=20 h=11**, may or may not appear in the greedy cover depending on whether it passes filter 2 — see [`minimum_surgical_set.json`](../outputs/head_surgery/minimum_surgical_set.json) for the authoritative list.

---

## 9. Stage E — Decoding-strategy ablation

### 9.1 Method

Grid search over decoding hyperparameters that typically affect Whisper hallucination behavior:

| Axis | Values |
|---|---|
| Beam width | {1, 5} |
| Repetition penalty | {1.0, 1.1, 1.3} |
| No-repeat-ngram size | {0, 3, 5} |
| Temperature fallback | {True, False} |

Full-factorial: 2 × 3 × 3 × 2 = **36 configs**, each run on the full 484-utt pool. Head-masking hook is **not** applied — this stage tests whether decoding alone can close the hallucination gap.

### 9.2 Result — Top-10 configs by lowest insertion rate (all beam=1)

| beam | rep_penalty | no_repeat_ngram | temp_fallback | total | rep | syn | con |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 1 | 1.1 | 0 | False | **1.23%** | 0.00% | 0.3% | 0.9% |
| 1 | 1.1 | 0 | True | 1.23% | 0.00% | 0.3% | 0.9% |
| 1 | 1.3 | 0/3/5 | True/False | 1.23% | 0.00% | 0.3% | 0.9% |

### 9.3 Result — summary by config family

| Family | Insertion rate |
|---|---:|
| Best (beam=1 + rep∈{1.1, 1.3}) | **1.23%** |
| Plain baseline (beam=1, rep=1.0, nr=0) | 1.27% |
| beam=5 + rep=1.3 OR n-gram ≥ 3 | 1.25–1.35% |
| beam=5 + rep=1.1, no n-gram | **8.21%** ‼ |
| beam=5 + rep=1.0, no n-gram | **14.78%** ‼‼ |

Decoding changes buy at most **0.04 pp**. The only large effect is a failure mode: naive beam search without n-gram blocking amplifies repetition loops ~10×.

**Wall-clock:** ~45 min.

---

## 10. Stage F — Silence-injection robustness

### 10.1 Method

Stress-test Whisper's hallucination rate under synthetic silence. Silence-perturbed clips are generated offline before inference ([_generate_silence_perturbations.py:40-67](../scripts/head_surgery/_generate_silence_perturbations.py#L40-L67)):

| Step | Operation |
|---|---|
| 1 | Load 16 kHz mono audio (same pipeline as §2.1) |
| 2 | Split a fraction (25% / 50% / 75% of duration) into **3 random-length blocks** (Dirichlet-style proportions) |
| 3 | Insert those zero-valued silence blocks at sorted random positions **inside** the clip (never at edges, so reference transcript remains valid) |
| 4 | Write back as 16 kHz WAV via `soundfile.write` |
| 5 | Rerun Whisper on the perturbed WAVs with the same preprocessing chain |

Three dB floors (−30, −35, −40) × three severities (25/50/75%) = **9 configs**. Seed: `20260418`.

### 10.2 Result

| severity | db_floor | insertion | rep | syn | con |
|:---|---:|---:|---:|---:|---:|
| 25% | −40 | 1.82% | 0.00% | 0.8% | 1.0% |
| 50% | −40 | 1.80% | 0.00% | 0.9% | 0.9% |
| 75% | −40 | 1.86% | 0.00% | 0.9% | 1.0% |
| 25% | −35 | 1.94% | 0.00% | 0.8% | 1.1% |
| 50% | −35 | 1.80% | 0.00% | 0.8% | 1.0% |
| 75% | −35 | 2.07% | 0.00% | 1.1% | 1.0% |
| 25% | −30 | 1.70% | 0.00% | 0.8% | 0.8% |
| 50% | −30 | 2.01% | 0.08% | 0.9% | 1.0% |
| 75% | −30 | 1.68% | 0.00% | 0.8% | 0.8% |

Baseline (no silence): 1.27%. Silence injection raises insertion rate by ~0.5–0.8 pp — small but consistent. No explosion into repetition loops.

**Wall-clock:** ~5 min.

---

## 11. Compute summary

| Stage | Gate | What | Wall-clock | Artifact |
|---|---|---|---|---|
| A | G1 | Baseline insertion on CV25 Indian N=484 | ~2 min | `baseline_metrics.json` |
| A.5 | G1.5 | Batch-size tuning (chose bs=32) | ~1 min | `tune_batch_size.json` |
| B | G2 | 50-utt pilot head-mask sweep | ~2 min | `pilot_sweep.csv` |
| C | G3+G4 | Full sweep — 309,760 rows (640 heads × 484 utts) | **~6 h 50 min** | `sweep.csv` (53 MB) |
| D | — | Scoring + 10,000× bootstrap per head + regression guard (top-50) | ~30 min | `head_scores.csv`, `top_k_heads.csv` |
| E | — | Decoding ablation — 36 configs | ~45 min | `decoding_scores.csv` |
| F | — | Energy VAD under silence injection — 9 configs | ~5 min | `vad_scores.csv` |
| G | — | Aggregate report + heatmap | <1 min | `head_surgery_report.md`, heatmap PNG |

**Total compute: ~8 h 15 min** (dominated by Stage C on 1× GPU). Calendar span: 2026-04-11 → 2026-04-18 (~7 days). Active implementation + execution: ~24 h.

---

## 12. Discussion

### Why the primary result is negative

The insertion rate metric is **inserted tokens ÷ total reference words**. At the CV25 baseline:

| Quantity | Value |
|---|---:|
| Total reference words (484 utts) | 4,885 |
| Hallucinated tokens | **62** |
| Utterances with ≥1 hallucination | 45 / 484 |
| Insertion rate | 62 / 4,885 = **1.27%** |
| Quantum (per-token Δ) | 1 / 4,885 = **0.020 pp** |

Per-head Δ is therefore quantized at ~0.02 pp per token. The best observed Δ (**−0.08 pp**, L=20 h=11) represents only ~4 tokens shifted across the whole 484-utt pool. With ~60 tokens of signal distributed across 640 independent head-tests — and no multiple-testing correction — the bootstrap cannot reliably pull effects of that size out of the noise floor. The test is **under-powered**, not "no effect to find."

Reading the p-values charitably:

- 1/640 significant at α=0.05 is **worse than chance** (≈32 expected false positives with no correction).
- The top-10 Δ values cluster at exactly +0.001 (≈4 tokens each), which is quantization bottoming out — not a signal ordering.

### What we'd need to see a real effect

| Option | Cost | Rationale |
|---|---|---|
| Evaluate on CV24 (Srishti's 9.62%) | bootstrap with ~50 events | ~8× more signal per head |
| Scale N to ~3,000 Indian utts | ~5× Stage C compute | power for ≤0.1 pp effects |
| Multi-head ablation | combinatorial blow-up | tests interaction effects the single-head sweep misses |

### Keystone-head finding is robust

The +100 pp catastrophe at L=0 h=5 is unmistakable regardless of statistical power — it is visible across all 484 utterances and on non-Indian audio. The ~10% failure cluster (7 heads) is also unambiguous.

### Caveats

- **Decoder self-attention only.** Encoder and cross-attention heads untested.
- **Single-head ablation.** No combinations or circuit-level probing.
- **Greedy + temp-fallback decoding.** Results may shift under beam search.
- **CV25-only.** Findings may not transfer to CV24 or to held-out Indian speech outside Common Voice.

---

## 13. Reproducibility

### Gate results

| Gate | Result | Notes |
|---|---|---|
| G1 — midterm CV24 baseline parity | **redefined** | CV24 unavailable; new baseline on CV25 is 1.27% |
| G1.5 — batch-size tune | ✅ PASS | bs=32 @ 14.12 utts/s |
| G2 — pilot head-mask signal | ⚠️ WARN | pilot baseline 0.97% < 2% quantization floor; hook activity verified via hypothesis-diff |
| G3 — batched ≈ serial | ✅ PASS | max \|Δ\| = 0.194% (1-utt RNG) |
| G4 — sweep completeness | ✅ PASS | 309,760 rows |
| G5 — non-Indian vs T7 | ⏭️ SKIP | T7 reference not present in env |

### Code

- Hook: [head_mask_hook.py](../scripts/head_surgery/head_mask_hook.py)
- Sweep: [run_diagnosis_sweep.py](../scripts/head_surgery/run_diagnosis_sweep.py)
- Scoring + bootstrap + guard: [score_heads.py](../scripts/head_surgery/score_heads.py)
- Decoding ablation: [decoding_ablation_grid.py](../scripts/head_surgery/decoding_ablation_grid.py)
- VAD: [energy_vad.py](../scripts/head_surgery/energy_vad.py)
- Report generator: [aggregate_report.py](../scripts/head_surgery/aggregate_report.py)
- PRD: [prd-head-surgery-diagnosis.md](../tasks/prd-head-surgery-diagnosis.md)
- Milestone log: [head-surgery-diagnosis-complete.md](../logs/head-surgery-diagnosis-complete.md)

---

## Appendix A — Full 50-head ranking

| layer | head | Δ insertion | p-value | reg. ok |
|---:|---:|---:|---:|:---:|
| 20 | 11 | 0.001 | 0.046 | ✅ |
| 10 | 8 | 0.001 | 0.374 | ✅ |
| 0 | 14 | 0.001 | 0.239 | ✅ |
| 7 | 6 | 0.001 | 0.128 | ✅ |
| 25 | 5 | 0.001 | 0.051 | ✅ |
| 22 | 19 | 0.001 | 0.051 | ✅ |
| 0 | 15 | 0.001 | 0.051 | ✅ |
| 3 | 3 | 0.000 | 0.140 | – |
| 15 | 6 | 0.000 | 0.138 | – |
| 12 | 8 | 0.000 | 0.138 | – |
| 15 | 18 | 0.000 | 0.364 | – |
| 17 | 8 | 0.000 | 0.139 | – |
| 23 | 16 | 0.000 | 0.138 | – |
| 24 | 18 | 0.000 | 0.136 | – |
| 23 | 11 | 0.000 | 0.136 | – |
| 19 | 17 | 0.000 | 0.134 | – |
| 22 | 1 | 0.000 | 0.138 | – |
| 22 | 2 | 0.000 | 0.136 | – |
| 21 | 15 | 0.000 | 0.136 | – |
| 25 | 18 | 0.000 | 0.138 | – |
| 24 | 6 | 0.000 | 0.141 | – |
| 31 | 9 | 0.000 | 0.138 | – |
| 20 | 18 | 0.000 | 0.138 | – |
| 18 | 17 | 0.000 | 0.228 | – |
| 16 | 2 | 0.000 | 0.136 | – |
| 20 | 6 | 0.000 | 0.136 | – |
| 20 | 5 | 0.000 | 0.292 | – |
| 16 | 1 | 0.000 | 0.364 | – |
| 30 | 16 | 0.000 | 0.130 | – |
| 29 | 6 | 0.000 | 0.136 | – |
| 11 | 2 | 0.000 | 0.138 | – |
| 16 | 13 | 0.000 | 0.135 | – |
| 13 | 0 | 0.000 | 0.133 | – |
| 11 | 11 | 0.000 | 0.133 | – |
| 13 | 17 | 0.000 | 0.132 | – |
| 22 | 5 | 0.000 | 0.138 | – |
| 25 | 16 | 0.000 | 0.130 | – |
| 25 | 0 | 0.000 | 0.138 | – |
| 4 | 11 | 0.000 | 0.374 | – |
| 21 | 2 | 0.000 | 0.366 | – |
| 0 | 12 | 0.000 | 0.441 | – |
| 4 | 8 | 0.000 | 0.374 | – |
| 30 | 0 | 0.000 | 0.366 | – |
| 21 | 10 | 0.000 | 0.402 | – |
| 19 | 12 | 0.000 | 0.375 | – |
| 23 | 1 | 0.000 | 0.366 | – |
| 22 | 9 | 0.000 | 0.366 | – |
| 31 | 8 | 0.000 | 0.366 | – |
| 21 | 18 | 0.000 | 0.375 | – |
| 26 | 1 | 0.000 | 0.375 | – |

*Full 640-head table: [outputs/head_surgery/head_scores.csv](../outputs/head_surgery/head_scores.csv).*

---

## Appendix B — Per-utterance baseline hallucinations

The 45 Indian-accent utterances with `baseline_count > 0` in [`baseline_predictions.csv`](../outputs/head_surgery/baseline_predictions.csv), partitioned by whether any valid single-head mask covers them (**15 helpable**) or not (**30 unhelpable**). Inserted tokens are extracted by the aligner in [`scripts/analysis/whisper_hallucination_analysis.py`](../scripts/analysis/whisper_hallucination_analysis.py) and tagged:

- `con` — content hallucination (novel content words)
- `syn` — syntactic completion (function words / fillers)
- `rep` — repetition
- *SUB-only* — divergences that the aligner scores as substitutions (typically hyphenation, casing, or proper-noun mangling), so no INS tokens are emitted even though `baseline_count > 0` by the head-surgery token counter.

### B.1 Helpable utterances (15)

| # | ID | baseline_count | Reference → Hypothesis | Inserted tokens |
|---:|---|---:|---|---|
| 1 | common_voice_en_17661177 | 3 | "Perhaps, you should just grow sea monkeys." → "Perhaps you should just **go and see** your aunt, please." | go[con]; and[syn]; see[con] |
| 2 | common_voice_en_18754521 | 4 | "It was built on land formerly held by the Macfadzeans." → "It was built on land **so we** were ahead by **a lot** of chance." | so[syn]; we[syn]; a[syn]; lot[con] |
| 3 | common_voice_en_18765645 | 1 | "Hasan Buzurg seemed intent on restoring unity to the Ilkhanate." → "Hassan Muzarak's Seemed Intent on Restoring Unity to the Khalid" | *SUB-only* |
| 4 | common_voice_en_18893500 | 2 | "Aurelian accepted Bahram I's gifts and the terms of peace offered." → "**early** and accepted Bahram is gift and the terms of peace **of** it" | early[con]; of[syn] |
| 5 | common_voice_en_20688664 | 4 | "Neither can be historically proven." → "Neither can make historically **bolder services for the** nation." | bolder[con]; services[con]; for[syn]; the[syn] |
| 6 | common_voice_en_25845294 | 1 | "It is located at a geographical headland and surrounds the town of Sisters Beach." → "It is located at the geographical **head** land and surrounds the town of Sisters Beach." | head[con] |
| 7 | common_voice_en_31752858 | 1 | "His first trainers was Vitaliy Khmelnytskyi." → "His first **credits** were with Ben Zuchinansky." | credits[con] |
| 8 | common_voice_en_34954466 | 4 | "The corner stone was laid by Sir Mortimer Clarke, Lieutenant Governor of Ontario." → "The **car was shown on** the late Wednesday in Montemar, Canada, left in an economy of Ontario." | car[con]; was[syn]; shown[con]; on[syn] |
| 9 | common_voice_en_37318569 | 1 | "It is located to the northwest." → "It is located to the **North** West." | North[con] |
| 10 | common_voice_en_37828418 | 1 | "'I am quite aware of that,' she replied." → "I am quite aware of that she **look** like" | look[con] |
| 11 | common_voice_en_38174426 | 1 | "The higher the degree of substitution, the more stable a carbocation generally is." → "the higher the degree of substitution the more stable a **car** vacation generally is" | car[con] |
| 12 | common_voice_en_38361992 | 1 | "When a person is sufficiently fatigued, microsleeps may be experienced." → "When a person is sufficiently fatigued, micro-sleeps may be experienced." | *SUB-only (hyphenation)* |
| 13 | common_voice_en_38663260 | 1 | "The band's longest engagement was at Hollenden Hotel's Vogue Room in Cleveland, Ohio." → "The band's longest engagement was at **the** Orlandin Hotel's Vogue Room in Cleveland, Ohio." | the[syn] |
| 14 | common_voice_en_40021346 | 1 | "The station was near the southern edge of Bedwellty Park." → "The station was near the southern edge of **Bad** Velti Park." | Bad[con] |
| 15 | common_voice_en_40886852 | 1 | "Norfolk Island is the only non-mainland Australian territory to have had self-governance." → "Norfolk, I-land is the only non-mainstream Australian territory to have had self-governance." | *SUB-only* |

### B.2 Unhelpable utterances (30)

`baseline_count` is not stored per-utterance for the unhelpable set (only helpable IDs appear in [`fixing_set_per_utterance.csv`](../outputs/head_surgery/fixing_set_per_utterance.csv)).

| # | ID | Reference → Hypothesis | Inserted tokens |
|---:|---|---|---|
| 16 | common_voice_en_17417536 | "Without haste, yet without rest unhasting, yet unresting." → "Without haste yet without rest, **on** hasting you turn resting." | rest,[con]; on[syn] |
| 17 | common_voice_en_17758246 | "The irregular pattern comes from the pseudorandomness of the function." → "the irregular pattern comes from the **pseudo** randomness of the function" | pseudo[con] |
| 18 | common_voice_en_18545942 | "Jujitsu is a form of martial arts." → "**Jiu** Jitsu is a form of martial arts." | Jiu[con] |
| 19 | common_voice_en_18976651 | "Nikiforos Hatzidakis was killed." → "**Nikki Forrest at** the Zee Ducky School" | Nikki[con]; Forrest[con]; at[syn] |
| 20 | common_voice_en_18976653 | "Banasura became invincible." → "**Bara** Surah became invincible" | Bara[con] |
| 21 | common_voice_en_19213280 | "More recently, members of the Eastercon convention have also been eligible to vote." → "More recently, members of the **Easter** Con Convention have also been eligible to vote." | Easter[con] |
| 22 | common_voice_en_19985202 | "The Football Association entered a Great Britain national amateur team to represent Great Britain." → "The Football Association of the Inter-Decade Britain National Team is representing the United Nations." | *SUB-only* |
| 23 | common_voice_en_20685733 | "Its county seat and largest city is Morgan." → "It's QuantiSync and I just sitting in mall" | *SUB-only* |
| 24 | common_voice_en_20688728 | "Watford is on the main Grand Union Canal route northwards from London." → "**But** Ford is on the main Grand Union Canal route northwards from Madan." | But[syn] |
| 25 | common_voice_en_21780812 | "The large Fortymile caribou herd roams near the highway." → "The large **40** mile caribou herd roams near the highway." | 40[con] |
| 26 | common_voice_en_26675808 | "It isn't theatrical!" → "It isn't **the** afterthought." | the[syn] |
| 27 | common_voice_en_26940741 | "The song consists of three verses in total." → "The **sound** and discs of J-Wheels are included." | sound[con] |
| 28 | common_voice_en_30512895 | "Blessed are those who have died in the Lord and have not seen them." → "**The** best are those who have died in the Lord and have not seen Him." | The[syn] |
| 29 | common_voice_en_33226454 | "Her other works include \"Christmas(Short film)\"." → "Her other works include **Christmas** short film" | Christmas[con] |
| 30 | common_voice_en_36641656 | "The economy is primarily based on coconut husking and farming." → "The economy is primarily based on **the** coconut husking and farming." | the[syn] |
| 31 | common_voice_en_36724596 | "They typically form proximally during Strombolian eruptions, and are common at strongly peralkaline volcanoes." → "They typically come approximately during **strong** beryllium eruptions and are common at strongly pergolined volcanoes." | strong[con] |
| 32 | common_voice_en_37032602 | "The settlement's principal tourist attraction is the famous Borisoglebsky Monastery, now a museum." → "The settlement's principal tourist attraction is the famous **Borys** Glybecki Monastery, now a museum." | Borys[con] |
| 33 | common_voice_en_37236919 | "A rip line simultaneously tears open the top of the balloon." → "A **rip-lens** Hamilton is retired so open the top of the bedroom" | rip-lens[con] |
| 34 | common_voice_en_37264767 | "Barsi's first role was in \"Fatal Vision\", playing the three-year-old Kimberley MacDonald." → "Barsi's first role was in Fatal Vision, playing the three-year-old **Kim** Ming McDonnell." | Kim[con] |
| 35 | common_voice_en_37431886 | "The launching station incorporates a tracking camera with two lenses." → "**These** are launching station incorporates of tracking camera with two lenses." | These[con] |
| 36 | common_voice_en_37523691 | "They toured with Paul Weller and Razorlight." → "They toured with ball weller and **razor** light." | razor[con] |
| 37 | common_voice_en_37527873 | "Furthermore, the equations of motion impose that the Romans mass is constant." → "furthermore the equations of motion impose that the roman's mass is constant" | *SUB-only (casing/punctuation)* |
| 38 | common_voice_en_38004990 | "Thus the Byzantines were forced to fight alone." → "Thus the **bison** tens were focused to fight lone." | bison[con] |
| 39 | common_voice_en_38297928 | "These four days corresponded to the thirteenth, fourteenth, fifteenth, and sixteenth of November." → "These four days corresponded to the 13th, 14th, 15th and **the** 16th of November." | the[syn] |
| 40 | common_voice_en_38625438 | "Ashur's brothers were Elam, Arphaxad, Lud, and Aram." → "Asur's brothers were **Ilum,** Arfak, Saad, Lord and Aram." | Ilum,[con] |
| 41 | common_voice_en_39568375 | "The birdlife to be found in this municipality is characteristic for the region." → "The **bird** life to be found in this municipality is characteristic for the region." | bird[con] |
| 42 | common_voice_en_39653346 | "Therefore, according to Kircher, Snakestones worked." → "Therefore, according to **Kertcher,** snake stones worked." | Kertcher,[con] |
| 43 | common_voice_en_39780450 | "Many frequent Buzztime players are enrolled in the \"Players Plus\" program." → "Many frequent **buzz** time players are enrolled in the Players Plus program." | buzz[con] |
| 44 | common_voice_en_40169034 | "The legendary founder of Littleport was King Canute." → "The legendary founder of **Little** Pot was King Canold." | Little[con] |
| 45 | common_voice_en_40863387 | "Pickhaver has three sisters Jane, Anne and Mary and a brother Mark." → "**Pig** Hever has three sisters Jane, Annie and Mary and a brother Mark." | Pig[con] |

### B.3 Observations

- **Insertion character is dominated by content hallucinations (`con`).** 38 of 45 utterances have at least one `con` insertion; only 7 are pure `syn` (function-word completion) or `SUB-only`.
- **Six utterances (#3, #12, #15, #22, #23, #37) show `baseline_count > 0` under the head-surgery counter but no INS under the aligner.** This happens when baseline differences are hyphenation, casing, or proper-noun substitutions rather than word insertions — the two counters use different criteria. These remain in the fixing-set denominator of 45 for consistency with §8.
- **The unhelpable group is enriched for proper-noun mis-recognition** (Ilkhanate, Hatzidakis, Banasura, Strombolian, Borisoglebsky, Buzztime, Littleport, Pickhaver, …). These errors are plausibly below the ceiling of any attention-masking intervention because the correct token has low probability mass at the top of the decoder distribution in the first place.
