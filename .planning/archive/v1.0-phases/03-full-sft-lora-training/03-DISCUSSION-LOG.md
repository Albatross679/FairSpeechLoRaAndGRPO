# Phase 3: Full SFT LoRA Training - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 03-full-sft-lora-training
**Areas discussed:** Training config, Batch & step calc, VRAM impact, Epoch strategy
**Context:** Update based on Phase 2.1 results (locked_config changed significantly)

---

## Training Config

| Option | Description | Selected |
|--------|-------------|----------|
| Keep constant (Recommended) | The sweep optimized with constant — changing it introduces an untested variable | ✓ |
| Cosine decay | Common for long training runs. But wasn't what the sweep optimized for | |
| You decide | Let Claude choose based on loss curve behavior | |

**User's choice:** Keep constant LR scheduler, matching the sweep-validated config
**Notes:** lr_scheduler=constant, warmup_ratio=0.0 — both locked from Phase 2.1

---

## Batch & Step Calc

### Batch size question

| Option | Description | Selected |
|--------|-------------|----------|
| Honor sweep (batch=2) | Keep grad_accum=1 as the sweep optimized. ~40h for 2 epochs. | ✓ |
| Increase to grad_accum=2 | Effective batch=4, halves step count. Slightly different from sweep config. | |
| You decide | Let Claude choose based on practical considerations | |

**User's choice:** Honor sweep config (grad_accum_steps=1, effective batch=2)
**Notes:** Accepts ~20K steps/epoch wall-clock time

### Checkpoint/eval intervals

| Option | Description | Selected |
|--------|-------------|----------|
| Moderate (Recommended) | Save every 4000 steps, eval every 2000 steps | ✓ |
| Frequent | Save every 2000, eval every 1000 | |
| You decide | Let Claude pick | |

**User's choice:** Moderate — checkpoint every 4000 steps, eval every 2000 steps
**Notes:** User initially needed clarification on what these intervals mean

---

## VRAM Impact

| Option | Description | Selected |
|--------|-------------|----------|
| Looks fine, move on | ~10.5 GB peak is under 15 GB T4 budget | |
| Run VRAM check first | Add explicit VRAM profiling step before full training | ✓ |

**User's choice:** Run VRAM check first
**Notes:** Even though estimate is safe (~10-10.5 GB), user wants explicit verification before committing to a ~20h training run

---

## Epoch Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Start with 1, extend if needed | Run 1 epoch, check loss, continue if still decreasing | |
| Commit to 2 epochs | Full 2 epochs, more regularization reduces overfitting risk | |
| You decide | Let Claude monitor and decide | |

**User's choice:** Other — "Until converge, but we should have early stopping"
**Notes:** User wants convergence-driven training rather than fixed epochs. Added early stopping callback.

### Early stopping patience follow-up

| Option | Description | Selected |
|--------|-------------|----------|
| Patience = 3 (Recommended) | Stop after 3 evals (6000 steps) with no improvement | ✓ |
| Patience = 5 | Stop after 5 evals (10000 steps) with no improvement | |
| You decide | Let Claude set patience | |

**User's choice:** Patience = 3
**Notes:** With eval every 2000 steps, this means 6000 steps of no improvement triggers stop

---

## Claude's Discretion

- Max epoch cap (2 vs 3) — safety limit for convergence training
- W&B run organization and naming
- Whether to load best checkpoint or last checkpoint for final evaluation

## Deferred Ideas

None
