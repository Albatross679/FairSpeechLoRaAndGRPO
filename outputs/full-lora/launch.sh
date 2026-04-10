#!/usr/bin/env bash
#
# Plan 03-03 full-scale SFT LoRA training launcher.
#
# Usage:
#   bash outputs/full-lora/launch.sh            # fresh launch
#   bash outputs/full-lora/launch.sh resume     # resume from latest checkpoint-XXXX
#
# This script is idempotent w.r.t. the vram_config.json contract — never
# hand-edit trainer flags here. If you need to change a trainer flag,
# update outputs/full-lora-vram-tune/vram_config.json instead.

set -euo pipefail
cd "$(dirname "$0")/../.."   # project root

# -- Pre-flight checks --------------------------------------------------------
nvidia_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1)
if [ "$nvidia_free_mib" -lt 40000 ]; then
  echo "ERROR: GPU free memory ${nvidia_free_mib} MiB < 40 GiB — another job is running?" >&2
  exit 1
fi

test -f outputs/manifests/fs_train.csv || { echo "ERROR: missing fs_train.csv" >&2; exit 1; }
test -f outputs/manifests/cv_train.csv || { echo "ERROR: missing cv_train.csv" >&2; exit 1; }
test -f outputs/standard-lora/locked_config.json || { echo "ERROR: missing locked_config.json" >&2; exit 1; }
test -f outputs/full-lora-vram-tune/vram_config.json || { echo "ERROR: missing vram_config.json (Plan 03-02 did not run)" >&2; exit 1; }

python -c "
import json, sys
c = json.load(open('outputs/full-lora-vram-tune/vram_config.json'))
if c['verdict'] not in ('go', 'go_with_warning'):
    print(f'ERROR: vram_config.json verdict={c[\"verdict\"]} — launch blocked', file=sys.stderr)
    sys.exit(1)
" || exit 1

# -- Resume handling ----------------------------------------------------------
RESUME_FLAG=""
if [ "${1:-}" = "resume" ]; then
  LATEST_CHECKPOINT=$(ls -1d outputs/full-lora/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
  if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: 'resume' requested but no outputs/full-lora/checkpoint-* found" >&2
    exit 1
  fi
  RESUME_FLAG="--resume_from_checkpoint $LATEST_CHECKPOINT"
  echo "Resuming from $LATEST_CHECKPOINT"
fi

# -- Launch -------------------------------------------------------------------
# expandable_segments:True reduces fragmentation OOMs over a long run.
# --vram_config is the Plan 03-02 frozen launch config (with num_workers=2 + save_total_limit=3 baked in).
# --lr stays explicit because it's a D-16 HP decision, not a VRAM tuning choice.
# --max_audio_secs 30 is insurance against outlier-length CV samples OOM'ing mid-run.
# --wandb_project asr-fairness-lora: remote monitoring (W&B).

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
exec python -u scripts/training/train_standard_lora.py \
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
    --wandb_project asr-fairness-lora \
    $RESUME_FLAG
