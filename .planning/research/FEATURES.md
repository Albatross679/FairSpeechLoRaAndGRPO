# Feature Landscape — v2.0 Attention Head Surgery

**Domain:** Whisper-large-v3 decoder attention-head-level hallucination mitigation for accent fairness
**Researched:** 2026-04-11
**Confidence:** HIGH for Calm-Whisper protocol and decoding-ablation literature; MEDIUM for accent-targeted (vs non-speech-targeted) adaptation of the method; MEDIUM for selective-fine-tuning training-signal choices
**Scope note:** This file covers ONLY the v2.0 intervention. It supersedes the prior v1.0 GRPO FEATURES.md content that lived at this path.

## Context (one paragraph)

Midterm (§3.2) identifies Whisper-large-v3 as a worst-case Indian-accent hallucinator: 9.62% insertion rate on the 511-utterance Indian-accent CV24 subset; 50.7% of errors are insertions; insertion breakdown 43% repetition / 48% syntactic / 9% content. Non-monotonic with scale (small 3.22% -> medium 1.53% -> large-v3 9.62%); Gen-3 LLM-decoder models all <3.1%. Midterm §4.2 proposes a four-pronged v2.0 follow-up: per-head masking diagnosis, decoding ablation, energy-based VAD under silence injection, and selective head fine-tuning. The reference paper is Calm-Whisper (Wang et al., Interspeech 2025), which found that 3 of 20 decoder self-attention heads cause >75% of non-speech hallucinations and that fine-tuning only those heads with empty-label noise data yields an 80% reduction with <0.1% WER regression on LibriSpeech.

## Calm-Whisper protocol — exact facts (HIGH confidence, arxiv 2505.12969)

These numbers anchor all "is this feature faithful to the reference" checks. They are quoted from the paper, not paraphrased.

| Item | Value | Source |
|------|-------|--------|
| Total decoder self-attention heads (per layer) in Whisper-large-v3 | 20 | Calm-Whisper §Methods |
| Decoder layers in Whisper-large-v3 | 32 | Whisper-large-v3 spec |
| Heads identified as "crazy" after diagnosis | #1, #6, #11 | Calm-Whisper |
| Diagnosis procedure | Mask one head at a time (independent, NOT accumulated), measure hallucination rate | Calm-Whisper |
| Hallucination metric for diagnosis | Utterance-level binary: `hallucination_rate = N(len(transcription)>0) / N_total` on non-speech audio | Calm-Whisper |
| Diagnosis dataset | UrbanSound8k (non-speech) | Calm-Whisper |
| Training signal for fine-tuning | Cross-entropy toward empty string on pure-noise audio | Calm-Whisper |
| Training data | AudioSet (11,753 non-speech clips) + DEMAND + MUSAN, 105 h total | Calm-Whisper |
| Learning rate | 1e-6, ~15% warmup | Calm-Whisper |
| Batch size | 128 | Calm-Whisper |
| Optimal epochs | 5 | Calm-Whisper |
| Whisper params frozen | All except heads #1/#6/#11 in every decoder layer | Calm-Whisper |
| Main result | >80% reduction in non-speech hallucination; <0.1% WER degradation on LibriSpeech test-clean/test-other | Calm-Whisper |
| Decoding strategy ablation in Calm-Whisper | NONE — paper does not test beam/repetition/n-gram | Calm-Whisper |

**Critical adaptation caveat:** Calm-Whisper targets _non-speech_ hallucination (empty audio should yield empty output). v2.0 targets _accent_ hallucination (Indian-accent speech should yield correct, non-looped transcription). The diagnosis metric, training signal, and training data all need adaptation. The protocol (per-head independent masking + selective fine-tuning of identified heads) transfers; the metric, dataset, and loss do NOT transfer one-for-one.

## Table Stakes

Features without which the milestone cannot credibly report results or cite Calm-Whisper.

| # | Feature | Why Required | Complexity | Dependencies |
|---|---------|--------------|------------|--------------|
| T1 | **Per-head attention masking hook on Whisper decoder self-attention** | Foundational primitive. Without a way to zero/mask a single head at a given layer, none of the diagnosis or intervention work is possible. Should be a PyTorch forward hook on `model.model.decoder.layers[L].self_attn` that multiplies head `h` contributions by 0 before the output projection. Must be layer-scoped and head-scoped independently. | MEDIUM | — |
| T2 | **Single-head masking diagnosis sweep (20 heads × 32 layers = 640 conditions)** | Faithful replication of Calm-Whisper's protocol adapted to accent. Run inference on the 511-utterance Indian-accent CV24 subset 641 times (baseline + 640 masked). Whisper-large-v3 inference on 511 utterances is ~10 minutes on a single GPU; full sweep = ~4 days unbatched, hours if parallelized at the layer level. | HIGH (compute), MEDIUM (code) | T1 |
| T3 | **Accent-adapted "driving-ness" metric** | Calm-Whisper's binary hallucination rate (non-empty transcription on non-speech) does not apply. v2.0 must instead rank heads by their effect on: (a) insertion rate on the 511 Indian-accent utterances (primary), (b) repetition-only insertion rate (to match the "looping" literature), and (c) WER delta on a non-Indian CV24 accent bucket (regression guard). A head qualifies as "hallucination-driving" iff masking it **reduces** insertion rate by ≥ some threshold **without** degrading non-target WER beyond a budget. Threshold TBD by the REQUIREMENTS author. | MEDIUM | T2, existing `classify_insertions.py` |
| T4 | **Use existing insertion classifier to split repetition / syntactic / content** | Midterm already shows 43% / 48% / 9% split on large-v3. Without the split, cannot tell whether head surgery fixes the "loop" heads or the "semantic" heads. This code exists in `scripts/compute_insertion_metrics.py` (or equivalent — reuse, don't rebuild). | LOW | existing code |
| T5 | **Decoding-strategy ablation baseline grid** | Reviewer question: "Couldn't you just turn on `repetition_penalty=1.3`?" — must be answered with numbers, not hand-waving. Reference paper (arxiv 2501.11378) already shows beam-1 beats beam-5 on hallucination (19.5% → 37.4%). Ablation grid: beam ∈ {1, 5}, repetition_penalty ∈ {1.0, 1.1, 1.3}, no_repeat_ngram_size ∈ {0, 3, 5}, temperature fallback on/off. 2×3×3×2 = 36 configs × 511 utterances = manageable. | MEDIUM | existing `run_inference.py` |
| T6 | **Evaluation standard report set** | For the class project writeup, need: per-accent insertion rate (6 CV24 groups) × {baseline, masked, fine-tuned}, overall WER on CV24 + Fair-Speech + LibriSpeech test-clean (regression guard), MMR (max/min ratio) before/after, insertion breakdown repetition/syntactic/content before/after, per-head "driving-ness" ranking table. This is the "table of numbers" the downstream REQUIREMENTS author needs to scope. | MEDIUM | T2, T4, T5, existing metrics pipeline |
| T7 | **Reuse existing 216-run baseline matrix** | Midterm already computed WER, MMR, insertion rate, and insertion breakdown for Whisper-small/medium/large-v3 + all Gen-3 models × 12 perturbation conditions × 2 datasets. v2.0 must NOT re-run these; it must read them as the "before" column in every comparison table. Cost: zero if the CSVs are still on disk; days if they were lost. | LOW | existing CSVs |
| T8 | **Energy-based VAD preprocessing on silence-injected audio (25/50/75%)** | Directly asked for in midterm §4.2. Must implement an energy-threshold VAD (simple: RMS frames above a dB floor) as a pre-Whisper pass that strips injected silence. Evaluate delta in insertion rate on the existing silence-injection perturbation manifests. "Energy-based" specifically — no Silero dependency required (though Silero is the community default, see differentiator D3). | LOW | existing silence perturbation manifests |
| T9 | **Reproducibility: frozen seeds, frozen model revision hash, frozen CV24 subset IDs** | 9.62% is a small-number finding on n=511. Any comparison must use the same 511 utterances, same Whisper-large-v3 checkpoint hash, same tokenizer, same `generate()` config apart from the one variable being ablated. Without this, noise will swamp the signal. | LOW | — |

## Differentiators

Optional but valuable features that strengthen the class project without bloating scope.

| # | Feature | Value | Complexity | Dependencies |
|---|---------|-------|------------|--------------|
| D1 | **Selective fine-tuning of identified hallucination-driving heads** | The "surgery" half of the Calm-Whisper story. After diagnosis (T2, T3) ranks heads, unfreeze the top-k (k ∈ {1, 3, 5}) and fine-tune with a loss adapted to accent hallucination (see D1a/D1b/D1c for options). Calm-Whisper proves the recipe works for non-speech; v2.0 would be the first public adaptation to accent. **This is a differentiator, not a table stake, because** the diagnosis alone (T2+T3) is a complete, citable contribution — "we localize the accent-hallucination heads in Whisper-large-v3" — and fine-tuning can be deferred if compute or time runs short. | HIGH | T1, T2, T3, compute budget |
| D1a | **Training signal option A: MLE on clean Indian-accent audio** | Standard cross-entropy fine-tuning on a held-out slice of Indian-accent CV24 utterances (NOT the 511-eval subset). Simplest. Directly analogous to Calm-Whisper's "target heads + clean labels", just with speech instead of empty strings. Risk: the model already handles most Indian-accent audio correctly — MLE on correctly-handled samples provides no gradient for the hallucination case. | MEDIUM | D1, held-out Indian-accent split |
| D1b | **Training signal option B: contrastive / un-hallucination loss on repetition-hallucinated samples** | Build a small dataset of Whisper-large-v3's own hallucinated outputs on the diagnosis subset (inputs where the baseline inserts repetition loops), pair with the correct transcripts, and train the unfrozen heads to assign higher probability to the correct transcript than to the hallucinated one. Closer in spirit to Calm-Whisper's "train the bad heads to output the right thing on the bad inputs". Data is automatic (no new labels). | HIGH | D1, T2, T3 |
| D1c | **Training signal option C: RL with insertion-rate reward** | GRPO-style or similar, with reward = `-insertion_rate` or `-repetition_count`. Reuses the v1.0 GRPO infrastructure that is currently paused. Highest complexity; lowest marginal benefit relative to D1a/D1b for a class project. Listed for completeness; not recommended for v2.0 scope. | HIGH | D1, v1.0 GRPO code |
| D2 | **Accumulated / greedy head masking** | Calm-Whisper uses independent single-head masking. A stronger diagnosis protocol: mask the worst head, re-measure, mask the next worst on top, etc. Captures interactions (e.g., if heads #1 and #6 redundantly cause loops, independent masking understates the gain from masking both). Expensive: up to 20*19/2 second-order plus iterative cost. Scope only if T2 finishes early and interaction is suspected. | HIGH | T2 |
| D3 | **Silero VAD comparison baseline (T8 companion)** | Compare energy-based VAD (T8) against SileroVAD on the same silence-injection conditions. Literature (arxiv 2501.11378) shows SileroVAD reduces detected hallucinations to 0.2% vs WebRTC's 12.5-15.4%. Lets v2.0 report "our simple energy VAD matches / trails / beats the SOTA VAD by X points" — strong sentence for the writeup. | LOW | T8 |
| D4 | **Cross-accent head overlap analysis** | Run T2's diagnosis independently on each CV24 accent group (not just Indian). Question answered: are the hallucination-driving heads _accent-specific_ (different heads for different accents) or _hallucination-generic_ (same heads regardless of accent)? This is a genuinely novel question — Calm-Whisper only tested non-speech. Strong differentiator, but cost = N_accents × 640 inference runs. | HIGH | T2 |
| D5 | **Layer-selective vs all-layers fine-tuning ablation** | Calm-Whisper fine-tunes heads #1/#6/#11 in _every_ decoder layer. Ablation: does fine-tuning only the top-K _layer-head_ pairs (from T2's ranking) work as well as fine-tuning those head indices across all 32 layers? Informative for understanding whether hallucination is localized in specific layers or is a "head-index property". | MEDIUM | D1 |
| D6 | **Decoding-strategy + head-masking interaction** | After T5 and T2, cross: does `repetition_penalty=1.1 AND mask head #H` improve over the sum of each intervention alone? Cheap (a few extra configs) if both grids are already computed. Answers the reviewer question "is head surgery additive with cheap decoder tricks?" | LOW | T2, T5 |
| D7 | **Publication-quality per-head heatmap** | Visualization: 32×20 grid where cell (L, h) color-encodes the insertion-rate delta when masking head h at layer L. Calm-Whisper prints a table; a heatmap would be visually stronger for the class project report. | LOW | T2 |

## Anti-Features

Things that are tempting, look rigorous, and would wreck the scope.

| # | Anti-Feature | Why Tempting | Why To Avoid | Do Instead |
|---|--------------|--------------|--------------|------------|
| A1 | **Activation patching** (swap activations between clean and hallucinated runs) | It is the canonical mechanistic-interpretability tool (Beyond Transcription, arxiv 2508.15882) and sounds more sophisticated than masking. | Requires paired clean/corrupt inputs with aligned tokenization, careful hook management across multiple forward passes, and far more code than masking. Calm-Whisper got SOTA-level findings with plain masking. For a class project, masking is sufficient and defensible. | Stick to T1/T2 single-head masking. Mention activation patching in the writeup as future work. |
| A2 | **Fine-tuning the _entire_ decoder or the _entire_ model** | Biggest hammer; would definitely reduce hallucination. | (a) Defeats the point — the contribution is targeted surgery, not retraining. (b) Will regress WER on non-target accents. (c) Eats VRAM unnecessarily. (d) Calm-Whisper explicitly freezes everything except the crazy heads. | D1 (selective head fine-tuning only). |
| A3 | **Adding more datasets** (GigaSpeech, Artie bias corpus, AfricanAccentsEnglish, etc.) | Bigger N feels more rigorous. | v2.0 already has 3 datasets (CV24, Fair-Speech, LibriSpeech) and 216 baseline runs in the CSV matrix. Adding datasets means new prep scripts, new demographic label schemas, new manifest validation. Scope killer. | Use only the existing datasets. Report on the existing subgroups. |
| A4 | **Running the head sweep on Whisper-small AND -medium AND -large-v3** | Comparison across scale would be cute given the midterm's non-monotonic finding. | 3× the compute. The midterm's finding (large-v3 hallucinates, small/medium do not) _is_ the reason v2.0 only targets large-v3. Re-running on small/medium answers a different, less interesting question. | Large-v3 only. Mention scale behavior in the intro by citing the midterm. |
| A5 | **Building a new insertion classifier** | The existing midterm classifier is string-heuristic based; a learned classifier would be more accurate. | Out of scope. The 43/48/9 split is already the "source of truth" for this project. Rebuilding it mid-milestone would invalidate comparisons to the midterm baseline. | Reuse existing `classify_insertions.py`. If issues found, document as a limitation. |
| A6 | **Fairness metric innovation** (new metrics beyond WER / insertion-rate / MMR) | Novel metric = paper-worthy. | v2.0 is a class project writeup, not a metrics paper. MMR, per-group WER, insertion rate, and insertion-type breakdown are sufficient and already wired into the existing pipeline. | Reuse existing fairness metrics from midterm. |
| A7 | **LoRA on the unfrozen heads instead of direct parameter updates** | LoRA is the default in 2026; feels "modern". | Calm-Whisper uses direct parameter updates on 3 heads × 32 layers = a tiny parameter count (~0.3% of model). LoRA adds rank hyperparameters and more code for zero benefit at this scale. v1.0's LoRA infrastructure is validated but is the wrong tool here. | Direct param updates on the identified heads (D1). Freeze everything else via `requires_grad=False`. |
| A8 | **Custom training loop when HF Trainer would do** | "More control" over the training step. | Calm-Whisper's recipe (CE loss, LR 1e-6, warmup, batch 128, 5 epochs) is vanilla HF Trainer territory. Writing a custom loop = bug surface. | Use `transformers.Trainer` with a custom `compute_loss` only if D1b/D1c is chosen. Otherwise the default is fine. |
| A9 | **GRPO reward engineering revival inside v2.0** | v1.0 already has GRPO plumbing; tempting to reuse. | v1.0 was paused for a reason (scope). Pulling GRPO back in defeats the pivot. Mentioned as D1c only for completeness; strongly recommend NOT selecting it. | Keep v1.0 archived. D1a or D1b. |
| A10 | **Chunked long-form inference experiments** | Real-world Whisper use is long-form; could show "our fix transfers". | CV24 is short-form utterance data; the 511-utterance subset is already short. Long-form changes the decoding dynamic (temperature fallback, condition-on-previous-text) and is a different experiment. | Stay on the utterance-level CV24 subset. Mention long-form as future work. |

## Feature Dependencies

```
T7 (reuse 216-run baseline CSVs)
    └─ provides "before" column for all comparisons

T1 (per-head masking hook)
    ├──> T2 (640-cell diagnosis sweep)
    │       ├──> T3 (driving-ness metric)
    │       │       └──> T6 (evaluation report)
    │       │               └──> downstream REQUIREMENTS
    │       ├──> D2 (accumulated masking)
    │       ├──> D4 (cross-accent head overlap)
    │       ├──> D7 (heatmap viz)
    │       └──> D1 (selective fine-tuning)
    │               ├──> D1a (MLE training signal)
    │               ├──> D1b (contrastive training signal)
    │               ├──> D1c (RL training signal)   [not recommended]
    │               ├──> D5 (layer-selective vs all-layers ablation)
    │               └──> T6 (post-fine-tuning numbers in evaluation report)
    └──> (standalone) head-masking inference = ablation baseline for T5

T4 (insertion classifier reuse)
    └──> feeds T3 (repetition-only metric) and T6 (breakdown)

T5 (decoding-strategy ablation)
    ├──> T6 (decoding column in evaluation report)
    └──> D6 (decoding × masking interaction)  [requires T2 also]

T8 (energy-based VAD)
    ├──> T6 (silence-condition column in evaluation report)
    └──> D3 (Silero comparison)

T9 (frozen seeds/hash/subset IDs)
    └──> precondition for EVERYTHING measured; must be fixed before T2 starts

Conflicts:
    D1c (RL signal) ─conflicts─> scope (pulls v1.0 GRPO back in — see A9)
    A2 (full fine-tune) ─conflicts─> D1 (selective fine-tune)
```

### Ordering implications for REQUIREMENTS author

1. **T1 and T9 first** — masking hook + frozen reproducibility contract. Without these, any measurement made later is invalidated.
2. **T5 and T8 are independent of the main diagnosis path** and can run in parallel waves. They use the existing `run_inference.py` with different config flags; no new infra.
3. **T2 is the long pole.** Budget it explicitly. 640 conditions × Whisper-large-v3 inference on 511 utterances is the compute anchor of the milestone.
4. **T3 depends on T2 and T4**, both of which must be done.
5. **T6 (the evaluation report) is a sink** that aggregates T2, T3, T4, T5, T7, T8, and D1 if chosen.
6. **D1 cannot start before T2+T3 identify the target heads.** Hard gate.
7. **T8 can be scoped down to "energy-based VAD only, no Silero" if D3 is dropped.** This keeps the silence-injection arm of the milestone small.

## MVP Definition

### Minimum Credible Milestone (must-ship)

These must all ship for the v2.0 writeup to be defensible:

- [ ] **T1** per-head masking hook
- [ ] **T2** 640-cell diagnosis sweep on the 511 Indian-accent CV24 utterances
- [ ] **T3** accent-adapted driving-ness metric that produces a ranked head list
- [ ] **T4** reuse of existing insertion classifier for the 43/48/9 breakdown
- [ ] **T5** decoding-strategy ablation grid (36 configs)
- [ ] **T6** evaluation report with before/after numbers on the standard set
- [ ] **T7** reuse of 216-run baseline CSVs as "before" column
- [ ] **T8** energy-based VAD on the existing silence-injection perturbation manifests
- [ ] **T9** frozen seeds, checkpoint hash, and utterance ID list

At this scope, the milestone has: (a) a citable diagnosis contribution, (b) a principled decoding-strategy baseline that reviewers will ask about, (c) the VAD arm from midterm §4.2, and (d) the "before column" is free. Calm-Whisper is cited as methodology; no claim of fine-tuning results is made.

### Stretch (add if diagnosis finishes on time)

- [ ] **D1 + D1a** (selective fine-tuning with MLE signal) — turns the diagnosis into an intervention and matches the full Calm-Whisper loop
- [ ] **D7** per-head heatmap for the writeup
- [ ] **D3** Silero VAD comparison — cheap if T8 already exists
- [ ] **D6** decoding × masking interaction — cheap if both T2 and T5 exist

### Explicit Non-Goals for v2.0 (deferred to v2.1 or future work)

- [ ] **D1b / D1c** — alternative training signals (contrastive, RL)
- [ ] **D2** — accumulated / greedy head masking
- [ ] **D4** — cross-accent head overlap analysis
- [ ] **D5** — layer-selective vs all-layers fine-tuning ablation
- [ ] Activation patching (A1)
- [ ] Any new datasets (A3)
- [ ] Any other Whisper model size (A4)

## Feature Prioritization Matrix

| Feature | Delivers to the writeup | Compute cost | Engineering cost | Priority |
|---------|-------------------------|--------------|------------------|----------|
| T1  per-head masking hook | Enables everything | LOW | MEDIUM | P1 |
| T2  640-cell diagnosis sweep | Core contribution | HIGH | LOW (loop on T1) | P1 |
| T3  accent-adapted metric | Turns T2 into findings | LOW | MEDIUM | P1 |
| T4  reuse insertion classifier | Free breakdown numbers | LOW | LOW | P1 |
| T5  decoding-strategy ablation | Answers reviewer Q#1 | MEDIUM | LOW | P1 |
| T6  evaluation report | The artifact | LOW | MEDIUM | P1 |
| T7  reuse 216 baseline runs | Free "before" column | ZERO | LOW | P1 |
| T8  energy-based VAD | Answers midterm §4.2 | LOW | LOW | P1 |
| T9  frozen reproducibility | Guardrail | ZERO | LOW | P1 |
| D1  selective head fine-tune | Turns diagnosis into intervention | HIGH | HIGH | P2 |
| D1a MLE signal | Simplest path if D1 chosen | MEDIUM | LOW | P2 |
| D7  per-head heatmap | Writeup polish | ZERO | LOW | P2 |
| D3  Silero VAD comparison | "our simple VAD is competitive" line | LOW | LOW | P2 |
| D6  decoding × masking interaction | Bonus reviewer answer | LOW | LOW | P2 |
| D2  accumulated masking | Stronger diagnosis | HIGH | MEDIUM | P3 |
| D4  cross-accent head overlap | Novel question | HIGH | LOW | P3 |
| D5  layer vs head-index ablation | Methodological depth | MEDIUM | LOW | P3 |
| D1b contrastive signal | Better loss than D1a | MEDIUM | HIGH | P3 |
| D1c RL signal | Pulls v1.0 back in | HIGH | HIGH | DROP |

## Literature Feature Matrix

How the relevant papers handle the v2.0 feature set. "—" = not addressed.

| Feature | Calm-Whisper (Interspeech 2025) | Investigation of Whisper Hallucinations (arxiv 2501.11378) | Listen Like a Teacher (arxiv 2511.14219) | Beyond Transcription (AAAI 2026, arxiv 2508.15882) | v2.0 plan |
|---------|---------|---------|---------|---------|-----------|
| Per-head masking diagnosis | Yes — 20 heads, independent, non-speech binary metric | — | — | Uses activation patching + ablation across 32 layers | T1 + T2, adapted to accent |
| Heads-per-layer scope | All 32 layers | — | — | All 32 layers | All 32 layers |
| Diagnosis metric | Utterance-level binary on UrbanSound8k | Detected-hallucination rate + WER | — | Per-layer component effect | Insertion-rate delta + repetition-only delta on CV24 Indian-accent |
| Decoding ablation | None | Beam 1/3/5, silence threshold | — | — | T5 (beam, rep penalty, n-gram, temp fallback) |
| VAD preprocessing | None | WebRTC, SileroVAD | — | — | T8 (energy) + D3 (Silero) |
| Selective head fine-tuning | Yes — heads #1/#6/#11, CE on empty strings | — | Adaptive layer attention + KD (different approach) | — | D1 (stretch), adapted to speech labels |
| Training data | 105 h non-speech (AudioSet + DEMAND + MUSAN) | — | — | — | Held-out Indian-accent CV24 split |
| Reported gain | >80% non-speech hallucination reduction, <0.1% LibriSpeech WER regression | SileroVAD: 0.2% hallucination vs WebRTC 12.5% | — | — | Target TBD by REQUIREMENTS |

## Open Questions for the REQUIREMENTS Author

1. **Masking threshold for T3:** what insertion-rate reduction (absolute? relative? bootstrap-significant?) qualifies a head as "driving"? Calm-Whisper used raw rate drop on a binary metric; accent is continuous. **Unknown** — needs a decision in REQUIREMENTS.
2. **Regression budget for T3:** how much WER degradation on a non-Indian accent group is acceptable when masking a head? Calm-Whisper accepted <0.1% on LibriSpeech. v2.0 equivalent TBD.
3. **Fine-tuning (D1) in or out of MVP?** The template above places D1 as stretch. If the class project deadline is tight, ship T1-T9 only and cite Calm-Whisper as the "what-if". If there is room, D1+D1a is the natural extension.
4. **Parallelization of T2:** is the 640-cell sweep run serially (safe, ~days) or in a layer-parallel data-parallel fashion (faster, more bug surface)? Has implications for the infra phase.
5. **Which CV24 non-Indian group is the regression guard?** England and US have the largest n; midterm already shows England's WER quadruples under 75% silence so it may be a poor guard. Needs picking.

## Sources

- [Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down (arxiv)](https://arxiv.org/abs/2505.12969) — HIGH confidence primary source for diagnosis protocol, head identity, fine-tuning recipe
- [Calm-Whisper ISCA archive (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.html)
- [Calm-Whisper HTML full-text](https://arxiv.org/html/2505.12969v1)
- [Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio (arxiv 2501.11378)](https://arxiv.org/html/2501.11378v1) — HIGH confidence for beam-size numbers (beam 1: 19.5% / beam 5: up to 37.4% hallucination) and VAD numbers (SileroVAD 0.2% vs WebRTC 12.5-15.4%)
- [Beyond Transcription: Mechanistic Interpretability in ASR (arxiv 2508.15882, AAAI 2026)](https://arxiv.org/html/2508.15882) — MEDIUM confidence context for activation patching as an alternative to masking; cited in A1
- [Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation (arxiv 2511.14219)](https://arxiv.org/abs/2511.14219) — LOW confidence, extension of Calm-Whisper line; not central to v2.0
- [Whisper-CD: Contrastive Decoding (arxiv 2603.06193)](https://arxiv.org/html/2603.06193) — MEDIUM confidence for alternative decoding strategy reported to reduce WER by up to 24.3pp on CORAAL; cited as A10-adjacent future work
- [Silero VAD (github)](https://github.com/snakers4/silero-vad) — MEDIUM confidence; the community default for the D3 comparison
- [A possible solution to Whisper hallucination (openai/whisper discussion #679)](https://github.com/openai/whisper/discussions/679) — LOW confidence community discussion cited for rep-penalty folklore only
- `llm-asr-fairness-midterm.pdf` §3.2 (9.62% finding, 43/48/9 breakdown), §4.2 (proposed v2.0 methodology) — HIGH confidence (authoritative for this project)
- `.planning/PROJECT.md` — HIGH confidence (authoritative for scope)

---
*Features research for: Whisper-large-v3 attention-head-surgery accent hallucination mitigation, v2.0 milestone*
*Researched: 2026-04-11*
