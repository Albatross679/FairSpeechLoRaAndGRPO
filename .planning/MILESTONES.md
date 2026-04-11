---
fileClass: Log
name: Milestones
description: Running log of GSD milestones for this project
created: 2026-04-11
updated: 2026-04-11
tags: [gsd, milestones]
aliases: []
status: complete
subtype: setup
---

# Milestones

Running history of milestone cycles for this project.

## v1.0 — GRPO Fairness-Aware ASR Fine-Tuning (PAUSED — partially complete)

**Status:** Paused 2026-04-11. May be resumed after v2.0.

**Goal:** Produce a GRPO training pipeline tracing the accuracy-fairness Pareto frontier for Qwen3-ASR-1.7B.

**Phases delivered:**
- ✓ Phase 01 — Validation (LoRA prototype, VRAM confirmed < 14 GB)
- ✓ Phase 02 — Standard LoRA baseline + evaluation bridge
- ✓ Phase 02.1 — PLoRA + ASHA + RsLoRA advanced HP sweep
- ◐ Phase 03 — Full SFT LoRA training (launched 2026-04-10, process terminated, outcome unverified)

**Phases NOT started:**
- Phase 04 — RL prototyping before scaling up RL
- Phase 04.1 — Experiments with different RL algorithms
- Phase 05 — Launch scaled-up RL
- Phase 06 — Rejection sampling fairness SFT
- Phase 07 — FairLoRA + Group-DRO + ICASSP comparisons

**Reason for pause:** Milestone was demanding more compute/time than planned. User opted to pivot to Section 4.2 of the midterm report (attention head surgery) as v2.0 and potentially return to v1.0 later.

**Artifacts preserved in:** `.planning/archive/v1.0-phases/`

## v2.0 — Hallucination Mitigation via Attention Head Surgery (ACTIVE)

**Started:** 2026-04-11

**Goal:** Identify and mitigate the Whisper-large-v3 decoder self-attention heads that drive accent-specific hallucination, targeting the 9.62% Indian-accent insertion rate.

**Reference:** midterm report §4.2; Calm-Whisper (Interspeech 2025).
