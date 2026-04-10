---
fileClass: Log
name: phase3-full-sft-lora-launch
description: Plan 03-03 full-scale SFT LoRA training launched detached (PID 253833) on A6000 with user-override go_with_warning verdict — ~66h expected wall-clock
created: 2026-04-10
updated: 2026-04-10
tags: [phase3, training, lora, full-dataset, launched, user-override, long-run]
aliases: []
status: complete
subtype: training
---

# Phase 3 Plan 03-03 Launch Record

## Summary

Full-scale SFT LoRA training for Qwen3-ASR-1.7B **launched detached** on
the A6000 host with the user-override `go_with_warning` verdict from
Plan 03-02. Expected wall-clock ~66 hours for 2 epochs, less if
`EarlyStoppingCallback(patience=3)` triggers. Bottleneck is known
(CPU-bound audio decode, mean_gpu_util ~45%) and the fix-forward path
(Option 1 — WebDataset pre-decode) is documented in
`outputs/full-lora-vram-tune/io_remediation_report.md` for after this
run completes.

This is a DECIDED TRADE-OFF: ship the baseline now, tune later.

## Process details

| Field | Value |
|-------|-------|
| **PID** | `253833` (written to `outputs/full-lora/train.pid`) |
| **PPID** | `1` (init) — detached, survives Claude Code session teardown |
| **Session leader** | Yes (`setsid` forked a new session) |
| **Launched** | 2026-04-10T19:31:02Z |
| **Host** | NVIDIA RTX A6000 48 GB, 4-core cgroup, overlayfs |
| **Python** | `/usr/bin/python` (system), unbuffered stdout (`-u`) |
| **Launch script** | `outputs/full-lora/launch.sh` (idempotent; re-run with `resume` arg to resume from latest checkpoint) |
| **Log file** | `outputs/full-lora/train.log` |
| **W&B project** | `asr-fairness-lora` (authenticated as `qifan_wen`) |

## Command (from launch.sh)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/training/train_standard_lora.py \
    --mode train \
    --full_dataset \
    --fs_manifest outputs/manifests/fs_train.csv \
    --cv_manifest outputs/manifests/cv_train.csv \
    --output_dir outputs/full-lora \
    --locked_config_path outputs/standard-lora/locked_config.json \
    --vram_config outputs/full-lora-vram-tune/vram_config.json \
    --lr 9.8e-4 \
    --num_epochs 2 \
    --save_total_limit 3 \
    --max_audio_secs 30 \
    --wandb_project asr-fairness-lora
```

Launched as:
```bash
setsid nohup bash outputs/full-lora/launch.sh > outputs/full-lora/train.log 2>&1 < /dev/null &
```

This gives the python process: new session (`setsid`), HUP-ignored (`nohup`), detached stdin (`< /dev/null`), stdout/stderr → `train.log`, backgrounded (`&`). On Claude Code session exit, the parent bash dies but the python process is reparented to init (PID 1) and continues running.

## Effective trainer configuration

Resolved from `locked_config.json` ∪ `vram_config.json.flags` ∪ explicit CLI:

| Knob | Value | Source |
|------|-------|--------|
| model | Qwen/Qwen3-ASR-1.7B | hardcoded MODEL_ID |
| LoRA rank | 4 | locked_config.json |
| LoRA alpha | 4 | locked_config.json |
| LoRA dropout | 0.05 | locked_config.json |
| target_modules | q/k/v/o + gate/up/down | locked_config.json (target_mlp=true) |
| weight_decay | 1.74e-4 | locked_config.json |
| lr_scheduler | constant | locked_config.json |
| warmup_ratio | 0.0 | locked_config.json |
| learning_rate | 9.8e-4 | explicit CLI (D-16 sqrt-scaling from 3.47e-4) |
| per_device_train_batch_size | 8 | vram_config.json.flags |
| gradient_accumulation_steps | 2 | vram_config.json.flags |
| effective batch | 16 | 8 × 2 |
| optim | adamw_torch_fused | vram_config.json.flags |
| gradient_checkpointing | **enabled** (default) | not disabled by vram_config |
| dataloader_num_workers | **2** | vram_config.json.flags (pre-flight win: +4.7 pp util) |
| dataloader_prefetch_factor | 2 | vram_config.json.flags |
| per_device_eval_batch_size | 8 | train_standard_lora.py default (was 1, now 8 per Plan 03-02) |
| save_total_limit | **3** | vram_config.json.flags (bumped from grid's 1) |
| num_epochs | 2 | explicit CLI (D-05 cap) |
| max_audio_secs | **30** | explicit CLI (safety against outlier OOM over 2 epochs) |
| EarlyStoppingCallback | patience=3 on eval_loss | train_standard_lora.py default |
| full_dataset | True | explicit CLI |
| wandb_project | asr-fairness-lora | explicit CLI |

## Expected timeline

| Milestone | Expected wall-clock |
|-----------|---------------------|
| `sf.info` scan for `--max_audio_secs 30` filter | ~25-30 min |
| Speaker-disjoint train/eval split on ~1.16M rows | ~30 s |
| Model load (Qwen3-ASR-1.7B + LoRA wrap) | ~30 s |
| **Training steady state** | 1.85 s/step × ~65,238 steps/epoch × 2 epochs = **~67 h** |
| Eval checkpoints | every ~5000 steps, ~13 per epoch |
| **Total (2 epochs, no early stop)** | **~68 h** from launch |
| **Total (early stop at 1-1.5 epochs)** | ~34-50 h from launch |

Estimated completion window: **2026-04-13 to 2026-04-15**.

## Checkpoint strategy

- `save_total_limit=3` → at most 3 `checkpoint-XXXX` directories at any time
- Each checkpoint ~1.5 GB → ~4.5 GB max checkpoint storage
- Disk budget: 96 GB free at launch → plenty of headroom
- Checkpoints written every ~5000 steps (~2.7 h at 1.85 s/step)
- **Resume pattern:** `bash outputs/full-lora/launch.sh resume` — the launcher auto-detects the latest `checkpoint-XXXX` and passes it via `--resume_from_checkpoint`. Because the launch config is fully sourced from `vram_config.json`, a resume is byte-identical to the initial launch (no trainer flag drift — T-03-16 mitigation).

## User override context

**Decision:** accept ~66h wall-clock and ship the baseline instead of pursuing Option 1 (WebDataset pre-decode, ~4-6h preprocessing + medium code changes) or Option 5 (migrate to 8+ core NVMe host, ~1h ops).

**Override audit trail:**
- `outputs/full-lora-vram-tune/vram_config.json` — `verdict: go_with_warning`, `verdict_reason` contains explicit `USER OVERRIDE 2026-04-10` stamp
- `outputs/full-lora-vram-tune/io_remediation_report.md` — 6 ranked options + sharpened decode-bound diagnosis
- `logs/phase3-vram-preflight-cellC-nw2.md` — pre-flight that measured 45.67% util and ruled out Options 2/3
- Commit `819f12a` — `chore(03-02): user-override vram_config.json verdict to go_with_warning`

**Why Options 1/5 were deferred:** user chose to unblock Plan 03-03 and downstream phases 4, 4.1, 5, 6 first. Re-running the 6-cell grid on a sharded dataset (Option 1) or a new host (Option 5) both gate Plan 03-03 for additional hours/days, and the baseline produced by this run is usable for downstream comparisons regardless of its training wall-clock.

## Monitoring

**Remote (W&B):** https://wandb.ai/qifan_wen/asr-fairness-lora — loss curves, eval metrics, GPU stats logged every 10 steps.

**Local process:**
- `cat outputs/full-lora/train.pid` — current PID
- `ps -p $(cat outputs/full-lora/train.pid) -o pid,ppid,etime,stat,%cpu,%mem,cmd` — process health
- `tail -f outputs/full-lora/train.log | tr '\r' '\n'` — live trainer stdout
- `nvidia-smi -l 10` — GPU utilization
- `ls -la outputs/full-lora/checkpoint-*` — checkpoint cadence
- `df -h outputs/` — disk headroom (watch for <10 GB free)

**Automated:** run the `babysit-training` skill periodically (hourly or on session resume) to aggregate process + loss + GPU + disk + W&B into one health report.

**Failure modes to watch for:**
1. **Outlier-length OOM** — mitigated by `--max_audio_secs 30`, but if an in-memory batch spike still OOMs, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` should handle it. Recovery: `bash outputs/full-lora/launch.sh resume`.
2. **Process killed / OOM on host** — re-parents to init, survives most shells ending. If the whole VM reboots, resume from latest checkpoint.
3. **Disk full** — `save_total_limit=3` caps checkpoints. Training output CSVs + W&B artifacts are small. 96 GB → ~4.5 GB used by checkpoints leaves plenty.
4. **W&B auth expires** — training continues without W&B if the API key expires; check log for "wandb:" errors.
5. **Gradient NaN / loss explosion** — D-16's sqrt-scaled LR (9.8e-4) is the main risk. Plan 01's dry-run at this LR converged cleanly over 50 steps; pre-flight (100 steps) also converged. Full-scale behavior is unvalidated. If loss goes NaN mid-run, kill + restart with a lower LR (e.g., 7.0e-4) — this becomes a new plan.

## Commits

- `819f12a` — `chore(03-02): user-override vram_config.json verdict to go_with_warning`
- (this launch record commit — see `docs(03-03): launch Plan 03-03 full-scale training`)

## Next steps after training completes

1. Run the plan's automated verify: `test -f outputs/full-lora/adapter/adapter_model.safetensors && python -c "..."` (see 03-03-PLAN.md `<verify>` block)
2. Create `03-03-SUMMARY.md` documenting final metrics, wall-clock, any deviations
3. Advance state via `gsd-tools state advance-plan` + `update-progress`
4. Proceed to Plan 03-04 (evaluation on Fair-Speech eval + Common Voice dev)
5. **Recommended follow-up plan:** Option 1 (WebDataset pre-decode) — still the right long-term fix for future phases. Not blocking for 03-04.
