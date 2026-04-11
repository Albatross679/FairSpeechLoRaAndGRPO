# LLM-ASR Fairness — Hallucination Mitigation via Attention Head Surgery

## What This Is

An attention-head-level intervention for Whisper-large-v3 ASR targeting the 9.62% Indian-accent insertion rate identified in midterm benchmarks. Following the Calm-Whisper methodology (Interspeech 2025) — which found that 3 of 20 decoder self-attention heads cause 75% of hallucinations — this milestone identifies the hallucination-driving heads on Qwen/Whisper decoders, ablates decoding strategies (beam size, repetition penalty, length penalty, no-repeat n-gram), tests energy-based VAD preprocessing under silence-injection conditions, and selectively fine-tunes the identified heads on accent-diverse audio. Target: reduce Indian-accent insertion rate without degrading WER on other demographic groups.

This replaces the GRPO fine-tuning work (milestone v1.0, paused) as the primary intervention for the CSE 5525 final project.

## Core Value

Reduce Whisper-large-v3's accent-specific hallucination rate (baseline 9.62% on Indian-accent utterances, 50.7% of total errors) via targeted attention-head surgery, without sacrificing overall WER or fairness on other demographic groups.

## Current Milestone: v2.0 Hallucination Mitigation via Attention Head Surgery

**Goal:** Identify and mitigate the Whisper-large-v3 decoder self-attention heads that drive accent-specific hallucination, targeting the 9.62% Indian-accent insertion rate.

**Target features:**
- Per-head masking diagnosis across all 20 decoder self-attention heads on 511 Indian-accent Common Voice utterances
- Decoding strategy ablation (beam size, repetition penalty, length penalty, no-repeat n-gram)
- Energy-based VAD preprocessing tested under silence-injection perturbations (25/50/75%)
- Selective fine-tuning of identified hallucination-driving heads on accent-diverse audio
- Evaluation: Indian-accent insertion rate + overall WER across all demographic groups + MMR

## Requirements

### Validated (reusable infrastructure from midterm + v1.0)

- ✓ ASR inference pipeline for 9 models across clean and perturbed conditions
- ✓ Fairness metrics computation (max-min ratio, gap %, std, bootstrap CIs)
- ✓ Common Voice 24 and Fair-Speech dataset preparation
- ✓ Perturbation generation (noise, reverb, silence, masking)
- ✓ Visualization pipeline for publication-ready figures
- ✓ Insertion classification (repetition / syntactic / content) from midterm
- ✓ 216 baseline inference runs (9 models × 12 conditions × 2 datasets)
- ✓ LoRA adapter integration on decoders (validated in v1.0 Phase 1.1)

### Active (v2.0)

*Requirements will be defined in REQUIREMENTS.md after scoping.*

### Out of Scope (v2.0)

- GRPO fairness-aware fine-tuning — paused as milestone v1.0, may resume after v2.0
- Intersectional analysis (accent × gender, ethnicity × age) — deferred
- Demographic conditioning / group-specific LoRA adapters — deferred
- Multi-speaker diarization fairness — blocked by dataset labels
- Audio-visual ASR fairness — blocked by dataset availability
- Full paper writing — this milestone focuses on the intervention and its evaluation

## Context

- **Course:** CSE 5525 Speech and Language Processing
- **Team:** Srishti Ginjala, Qifan Wen
- **Midterm finding (accent hallucination):** Whisper-large-v3 has a 9.62% insertion rate on Indian-accent speech (511 Common Voice utterances), with 50.7% of errors being insertions. Breakdown: 43% repetition, 48% syntactic, 9% content. Non-monotonic across Whisper scale: small 3.22% → medium 1.53% → large-v3 9.62%. All Gen-3 models (Qwen3, Canary, Granite) remain <3.1% — this is a Whisper-specific failure mode.
- **Reference methodology:** Calm-Whisper (Interspeech 2025) — 3 of 20 decoder self-attention heads cause 75% of hallucinations.
- **Existing codebase:** 28 Python scripts covering dataset prep → inference → metrics → visualization. Modular pipeline with CSV manifest data contracts. Per-head attention analysis is NEW — no existing infrastructure.
- **Hardware:** Single NVIDIA GPU (previously ~16 GB A4000; current GPU has 49 GB free per `nvidia-smi`). Whisper-large-v3 is 1.5B params — inference-only analysis fits easily; selective fine-tuning feasible with LoRA or full head unfreezing.
- **Evaluation datasets:**
  - Common Voice 24: 6 accent groups including Indian (n=511)
  - Fair-Speech: 7 ethnicity groups, controlled prompts (no vocabulary confound)
  - LibriSpeech test-clean: reference baseline
- **Primary target model:** Whisper-large-v3 (worst hallucinator). Whisper-small and -medium used as comparison points.

## Constraints

- **Hardware:** Single GPU — must fit Whisper-large-v3 (1.5B params) in memory alongside hooks capturing per-head attention
- **Tech stack:** Python / PyTorch / HuggingFace transformers. `WhisperForConditionalGeneration` exposes decoder self-attention layers; hook-based per-head masking required
- **Timeline:** CSE 5525 final project deliverable
- **Data:** Must use existing 511-utterance Indian-accent CV subset for diagnosis; cannot collect new labeled data
- **Methodology grounding:** Must replicate/adapt Calm-Whisper per-head masking protocol faithfully enough to cite

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Target Whisper-large-v3 | Worst Indian-accent insertion rate (9.62%); Calm-Whisper methodology is Whisper-specific | — Pending |
| Diagnosis before intervention | Cannot fine-tune what hasn't been localized; per-head masking identifies the 3-ish critical heads first | — Pending |
| Reuse existing inference pipeline | Avoid rebuilding eval infra; add per-head hooks as a wrapper around current `run_inference.py` | — Pending |
| Keep v1.0 artifacts archived, not deleted | User may resume GRPO work after v2.0 | ✓ Archived to `.planning/archive/v1.0-phases/` |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-11 — Milestone v1.0 (GRPO) paused and archived; v2.0 (attention head surgery) started*
