#!/usr/bin/env bash
# Supervised FairSpeech VM pilot-gate preparation.
#
# Intended invocation on the GPU VM:
#   tmux new -d -s fairspeech-prep \
#     'cd /workspace/FairSpeechLoRaAndGRPO && bash scripts/setup/run_fairspeech_vm_prep.sh 2>&1 | tee datasets/fairspeech_compression/logs/tmux_fairspeech_prep.log'
#
# Runtime-heavy artifacts intentionally live under /opt:
#   /opt/hf-cache
#   /opt/fairspeech-variants
#   /opt/fairspeech-results

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SOURCE_METADATA="${SOURCE_METADATA:-$PROJECT_ROOT/datasets/meta_speech_fairness/downloaded_file.tsv}"
SOURCE_AUDIO_DIR="${SOURCE_AUDIO_DIR:-$PROJECT_ROOT/datasets/meta_speech_fairness/asr_fairness_audio}"

DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/datasets/fairspeech_compression}"
LOG_DIR="${LOG_DIR:-$DATASET_DIR/logs}"
FULL_MANIFEST_DIR="$DATASET_DIR/manifests"
SUMMARY_DIR="$DATASET_DIR/summaries"
PLAN_DIR="$DATASET_DIR/batch_plans"
CALIBRATION_DIR="$DATASET_DIR/calibration"
MODEL_CACHE_DIR="$DATASET_DIR/model_cache"
PROFILE_DIR="$DATASET_DIR/profiles"

OPT_VARIANT_ROOT="${OPT_VARIANT_ROOT:-/opt/fairspeech-variants}"
PILOT_DIR="${PILOT_DIR:-$OPT_VARIANT_ROOT/pilot}"
PROFILE_WORK_DIR="${PROFILE_WORK_DIR:-$OPT_VARIANT_ROOT/profile_work}"
RESULT_ROOT="${RESULT_ROOT:-/opt/fairspeech-results}"
HF_HOME="${HF_HOME:-/opt/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
WANDB_DIR="${WANDB_DIR:-/opt/wandb}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/opt/pip-cache}"

EXPECTED_WAV_COUNT="${EXPECTED_WAV_COUNT:-26471}"
ROOT_MIN_FREE_GIB="${ROOT_MIN_FREE_GIB:-12}"
GPU_PREFLIGHT_THRESHOLD="${GPU_PREFLIGHT_THRESHOLD:-0.80}"
GPU_PROFILE_THRESHOLD="${GPU_PROFILE_THRESHOLD:-0.75}"
CANDIDATE_SECONDS="${CANDIDATE_SECONDS:-40,80,120,160}"
CANDIDATE_MAX_SAMPLES="${CANDIDATE_MAX_SAMPLES:-4,8,16}"
PILOT_MODEL="${PILOT_MODEL:-wav2vec2-large}"
START_STEP="${START_STEP:-preflight}"

MODEL_ORDER=(
  wav2vec2-large
  whisper-small
  whisper-medium
  whisper-large-v3
  qwen3-asr-0.6b
  qwen3-asr-1.7b
  canary-qwen-2.5b
  granite-speech-3.3-2b
  granite-speech-3.3-8b
)

VARIANTS=(
  baseline
  bottleneck_12k
  bottleneck_8k
  mp3_64k
  mp3_32k
  mp3_16k
)

mkdir -p \
  "$LOG_DIR" "$SUMMARY_DIR" "$PLAN_DIR" "$CALIBRATION_DIR" \
  "$MODEL_CACHE_DIR" "$PROFILE_DIR" "$PILOT_DIR" "$PROFILE_WORK_DIR" \
  "$RESULT_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$WANDB_DIR" "$PIP_CACHE_DIR"

export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE WANDB_DIR PIP_CACHE_DIR

STATUS_JSON="$DATASET_DIR/vm_prep_status.json"
MODEL_STATUS_JSONL="$MODEL_CACHE_DIR/model_cache_status.jsonl"
SMOKE_PASS_MODELS="$MODEL_CACHE_DIR/smoke_passing_models.txt"
PILOT_METADATA="$PILOT_DIR/pilot_metadata.tsv"
SMOKE_MANIFEST="$MODEL_CACHE_DIR/model_smoke_manifest.csv"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

write_status() {
  local status="$1"
  local step="$2"
  local message="${3:-}"
  STATUS="$status" STEP="$step" MESSAGE="$message" "$PYTHON_BIN" - "$STATUS_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": os.environ["STATUS"],
    "step": os.environ["STEP"],
    "message": os.environ.get("MESSAGE", ""),
    "timestamp_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

fail() {
  local step="$1"
  local message="$2"
  write_status "blocked" "$step" "$message"
  log "BLOCKED at $step: $message"
  exit 1
}

run_step() {
  local step="$1"
  shift
  log "START: $step"
  write_status "running" "$step" ""
  "$@"
  log "DONE: $step"
}

root_free_gib() {
  df -BG / | awk 'NR==2 {gsub("G","",$4); print $4}'
}

cache_size_bytes() {
  if [ -d "$HF_HOME" ]; then
    du -sb "$HF_HOME" 2>/dev/null | awk '{print $1}'
  else
    printf '0\n'
  fi
}

check_root_free_or_stop_downloads() {
  local free_gib
  free_gib="$(root_free_gib)"
  if [ "$free_gib" -lt "$ROOT_MIN_FREE_GIB" ]; then
    log "Root free space ${free_gib}GiB < ${ROOT_MIN_FREE_GIB}GiB; stopping further downloads."
    return 1
  fi
  return 0
}

cleanup_failed_model_cache() {
  local model="$1"
  MODEL="$model" HF_HUB_CACHE="$HF_HUB_CACHE" "$PYTHON_BIN" - <<'PY'
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scripts.inference.run_inference import MODEL_REGISTRY  # noqa: E402

model = os.environ["MODEL"]
hf_id = MODEL_REGISTRY[model]["hf_id"]
cache_root = Path(os.environ["HF_HUB_CACHE"])
repo_dir = cache_root / f"models--{hf_id.replace('/', '--')}"
lock_dir = cache_root / ".locks" / f"models--{hf_id.replace('/', '--')}"

for path in (repo_dir, lock_dir):
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed failed partial cache: {path}")
PY
}

check_gpu_memory() {
  local threshold="$1"
  [ -x "$(command -v nvidia-smi)" ] || fail "gpu-preflight" "nvidia-smi not found"
  THRESHOLD="$threshold" "$PYTHON_BIN" - <<'PY'
import os
import subprocess
import sys

threshold = float(os.environ["THRESHOLD"])
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()[0]
used, total = [int(x.strip()) for x in out.split(",")]
frac = used / total if total else 1.0
print(f"GPU memory: {used}/{total} MiB ({frac:.1%}); threshold={threshold:.0%}")
if frac > threshold:
    raise SystemExit(1)
PY
}

check_runtime_versions() {
  "$PYTHON_BIN" - <<'PY'
import re
import sys

import torch
import transformers


def version_tuple(raw: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", raw)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


torch_v = version_tuple(torch.__version__)
transformers_v = version_tuple(transformers.__version__)
print(f"Runtime versions: torch={torch.__version__}, transformers={transformers.__version__}")
if transformers_v[0] >= 5 and torch_v < (2, 6):
    raise SystemExit(
        "transformers>=5 with torch<2.6 blocks .bin weight loading; "
        "install project deps with transformers<5 or upgrade torch first."
    )
PY
}

append_model_status() {
  MODEL="$1" \
  DOWNLOAD_STATUS="$2" \
  SMOKE_STATUS="$3" \
  CACHE_MANIFEST="$4" \
  DOWNLOAD_LOG="$5" \
  SMOKE_LOG="$6" \
  ERROR_TEXT="${7:-}" \
  CACHE_SIZE_BYTES="$(cache_size_bytes)" \
  ROOT_FREE_GIB="$(root_free_gib)" \
  "$PYTHON_BIN" - "$MODEL_STATUS_JSONL" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest_path = Path(os.environ["CACHE_MANIFEST"])
cache_record = {}
if manifest_path.is_file():
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = data.get("models") or []
        if models:
            cache_record = models[0]
    except Exception as exc:  # noqa: BLE001
        cache_record = {"manifest_parse_error": str(exc)}

record = {
    "timestamp_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "model": os.environ["MODEL"],
    "download_status": os.environ["DOWNLOAD_STATUS"],
    "smoke_status": os.environ["SMOKE_STATUS"],
    "cache_manifest": str(manifest_path),
    "download_log": os.environ["DOWNLOAD_LOG"],
    "smoke_log": os.environ["SMOKE_LOG"],
    "hf_cache_size_bytes": int(os.environ["CACHE_SIZE_BYTES"]),
    "root_free_gib": int(os.environ["ROOT_FREE_GIB"]),
    "error": os.environ.get("ERROR_TEXT") or cache_record.get("error"),
    "cache_record": cache_record,
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
PY
}

preflight() {
  [ -x "$PYTHON_BIN" ] || fail "preflight" "Python not found at $PYTHON_BIN"
  check_runtime_versions || fail "preflight" "incompatible torch/transformers runtime"

  local branch
  branch="$(git branch --show-current)"
  [ "$branch" = "master" ] || fail "preflight" "expected branch master, found $branch"
  git diff --quiet || fail "preflight" "tracked working tree has unstaged changes"
  git diff --cached --quiet || fail "preflight" "tracked working tree has staged changes"

  local wav_count
  wav_count="$(find "$SOURCE_AUDIO_DIR" -maxdepth 1 -type f -name '*.wav' | wc -l | tr -d ' ')"
  [ "$wav_count" = "$EXPECTED_WAV_COUNT" ] || \
    fail "preflight" "expected $EXPECTED_WAV_COUNT FairSpeech WAVs, found $wav_count"

  local venv_target
  venv_target="$(readlink -f "$PROJECT_ROOT/.venv" 2>/dev/null || true)"
  [ "$venv_target" = "/opt/fairspeech-venv" ] || \
    fail "preflight" "expected .venv -> /opt/fairspeech-venv, found ${venv_target:-missing}"

  check_gpu_memory "$GPU_PREFLIGHT_THRESHOLD" || \
    fail "preflight" "GPU memory above ${GPU_PREFLIGHT_THRESHOLD}"
  check_root_free_or_stop_downloads || \
    fail "preflight" "root free space below ${ROOT_MIN_FREE_GIB}GiB before starting"

  log "Preflight ok: branch=$branch, FairSpeech WAVs=$wav_count, venv=$venv_target"
}

generate_full_manifests() {
  "$PYTHON_BIN" scripts/data/prepare_fairspeech_compression.py \
    --metadata "$SOURCE_METADATA" \
    --audio-dir "$SOURCE_AUDIO_DIR" \
    --output-dir "$DATASET_DIR" \
    --variants all \
    --expected-full-count "$EXPECTED_WAV_COUNT"
}

validate_full_baseline_manifest() {
  "$PYTHON_BIN" scripts/validation/validate_fairspeech_compression.py \
    "$FULL_MANIFEST_DIR/fairspeech_baseline_manifest.csv" \
    --expected-count "$EXPECTED_WAV_COUNT" \
    --summary-json "$SUMMARY_DIR/full_baseline_validation.json"
}

build_calibration_subset() {
  "$PYTHON_BIN" scripts/inference/build_duration_batch_plan.py \
    --manifest "$FULL_MANIFEST_DIR/fairspeech_baseline_manifest.csv" \
    --output-jsonl "$PLAN_DIR/fairspeech_baseline_total160s_plan.jsonl" \
    --summary-json "$PLAN_DIR/fairspeech_baseline_total160s_summary.json" \
    --calibration-csv "$CALIBRATION_DIR/fairspeech_duration_ethnicity_calibration.csv" \
    --profile-schema-json "$PROFILE_DIR/profile_schema.json" \
    --budget-mode total \
    --max-audio-seconds 160 \
    --max-samples 64
}

write_pilot_metadata() {
  "$PYTHON_BIN" - "$SOURCE_METADATA" "$CALIBRATION_DIR/fairspeech_duration_ethnicity_calibration.csv" "$PILOT_METADATA" <<'PY'
import csv
import sys
from pathlib import Path

source_tsv = Path(sys.argv[1])
calibration_csv = Path(sys.argv[2])
out_tsv = Path(sys.argv[3])

with calibration_csv.open("r", encoding="utf-8", newline="") as f:
    calibration = list(csv.DictReader(f))
ids = [row.get("hash_name") or row.get("utterance_id") for row in calibration]
ids = [x for x in ids if x]
order = {uid: idx for idx, uid in enumerate(ids)}

with source_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = [row for row in reader if row.get("hash_name") in order]
    fieldnames = reader.fieldnames or []

rows.sort(key=lambda row: order[row["hash_name"]])
if len(rows) != len(ids):
    missing = sorted(set(ids) - {row["hash_name"] for row in rows})
    raise SystemExit(f"Pilot metadata missing {len(missing)} calibration ids: {missing[:5]}")

out_tsv.parent.mkdir(parents=True, exist_ok=True)
with out_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote pilot metadata: {out_tsv} ({len(rows)} rows)")
PY
}

pilot_count() {
  "$PYTHON_BIN" - "$PILOT_METADATA" <<'PY'
import csv
import sys
from pathlib import Path
with Path(sys.argv[1]).open("r", encoding="utf-8", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f, delimiter="\t")))
PY
}

generate_pilot_audio() {
  local count
  count="$(pilot_count)"
  "$PYTHON_BIN" scripts/data/prepare_fairspeech_compression.py \
    --metadata "$PILOT_METADATA" \
    --audio-dir "$SOURCE_AUDIO_DIR" \
    --output-dir "$PILOT_DIR" \
    --variants all \
    --generate-audio \
    --overwrite \
    --expected-full-count "$count"
}

validate_pilot_audio() {
  local count
  count="$(pilot_count)"
  "$PYTHON_BIN" scripts/validation/validate_fairspeech_compression.py \
    "$PILOT_DIR"/manifests/fairspeech_baseline_manifest.csv \
    "$PILOT_DIR"/manifests/fairspeech_bottleneck_12k_manifest.csv \
    "$PILOT_DIR"/manifests/fairspeech_bottleneck_8k_manifest.csv \
    "$PILOT_DIR"/manifests/fairspeech_mp3_64k_manifest.csv \
    "$PILOT_DIR"/manifests/fairspeech_mp3_32k_manifest.csv \
    "$PILOT_DIR"/manifests/fairspeech_mp3_16k_manifest.csv \
    --expected-count "$count" \
    --check-silence \
    --summary-json "$SUMMARY_DIR/pilot_audio_validation.json"
}

write_smoke_manifest() {
  "$PYTHON_BIN" - "$PILOT_DIR/manifests/fairspeech_baseline_manifest.csv" "$SMOKE_MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
with source.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    first = next(reader)
    fieldnames = reader.fieldnames or list(first)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(first)
print(f"Wrote smoke manifest: {out}")
PY
}

smoke_audio_path() {
  "$PYTHON_BIN" - "$SMOKE_MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open("r", encoding="utf-8", newline="") as f:
    print(next(csv.DictReader(f))["audio_path"])
PY
}

download_and_smoke_models() {
  : > "$MODEL_STATUS_JSONL"
  : > "$SMOKE_PASS_MODELS"
  local stop_downloads=0

  for model in "${MODEL_ORDER[@]}"; do
    local cache_manifest="$MODEL_CACHE_DIR/${model}.json"
    local download_log="$LOG_DIR/download_${model}.log"
    local smoke_log="$LOG_DIR/smoke_${model}.log"
    local download_status="skipped"
    local smoke_status="skipped"
    local error_text=""

    if [ "$stop_downloads" -eq 1 ] || ! check_root_free_or_stop_downloads; then
      stop_downloads=1
      download_status="skipped_low_disk"
      smoke_status="skipped_low_disk"
      append_model_status "$model" "$download_status" "$smoke_status" "$cache_manifest" "$download_log" "$smoke_log" ""
      continue
    fi

    log "Downloading/cache-preparing model: $model"
    if "$PYTHON_BIN" scripts/inference/prepare_model_cache.py \
      --models "$model" \
      --output "$cache_manifest" \
      --cache-dir "$HF_HUB_CACHE" \
      --download \
      --device cuda \
      --smoke-audio "$(smoke_audio_path)" \
      > "$download_log" 2>&1; then
      download_status="downloaded"
    else
      download_status="failed"
      error_text="$(tail -n 20 "$download_log" | tr '\n' ' ' | cut -c1-2000)"
    fi

    if [ "$download_status" = "downloaded" ]; then
      log "Smoke inference: $model"
      if check_gpu_memory "$GPU_PREFLIGHT_THRESHOLD" \
        && "$PYTHON_BIN" scripts/inference/run_inference.py \
          --model "$model" \
          --manifest "$SMOKE_MANIFEST" \
          --output_dir "$RESULT_ROOT/model_smoke/$model" \
          --device cuda \
          --max_samples 1 \
          --batch_size 1 \
          --audio_variant model_cache_smoke \
          --dataset fs \
          > "$smoke_log" 2>&1; then
        smoke_status="passed"
        printf '%s\n' "$model" >> "$SMOKE_PASS_MODELS"
      else
        smoke_status="failed"
        error_text="${error_text} $(tail -n 30 "$smoke_log" | tr '\n' ' ' | cut -c1-3000)"
      fi
    fi

    append_model_status "$model" "$download_status" "$smoke_status" "$cache_manifest" "$download_log" "$smoke_log" "$error_text"

    if [ "$download_status" = "failed" ]; then
      cleanup_failed_model_cache "$model" || true
    fi
    if ! check_root_free_or_stop_downloads; then
      stop_downloads=1
    fi
  done
}

profile_smoke_passing_models() {
  if [ ! -s "$SMOKE_PASS_MODELS" ]; then
    fail "duration-profile" "no smoke-passing models to profile"
  fi

  while IFS= read -r model; do
    [ -n "$model" ] || continue
    local profile_jsonl="$PROFILE_DIR/${model}_duration_budget_profile.jsonl"
    local profile_log="$LOG_DIR/profile_${model}.log"
    log "Duration-budget profiling: $model"
    check_root_free_or_stop_downloads || fail "duration-profile" "root free space below ${ROOT_MIN_FREE_GIB}GiB before profiling $model"
    check_gpu_memory "$GPU_PREFLIGHT_THRESHOLD" || fail "duration-profile" "GPU busy before profiling $model"
    "$PYTHON_BIN" scripts/inference/profile_batch_policy.py \
      --model "$model" \
      --manifest "$PILOT_DIR/manifests/fairspeech_baseline_manifest.csv" \
      --output-jsonl "$profile_jsonl" \
      --work-dir "$PROFILE_WORK_DIR" \
      --candidate-seconds "$CANDIDATE_SECONDS" \
      --candidate-max-samples "$CANDIDATE_MAX_SAMPLES" \
      --budget-mode total \
      --device cuda \
      --gpu-safety-threshold "$GPU_PROFILE_THRESHOLD" \
      > "$profile_log" 2>&1 || fail "duration-profile" "profiling failed for $model; see $profile_log"
  done < "$SMOKE_PASS_MODELS"
}

selected_plan_for_model() {
  local model="$1"
  "$PYTHON_BIN" - "$PROFILE_DIR/${model}_duration_budget_profile.jsonl" <<'PY'
import json
import sys
from pathlib import Path

profile = Path(sys.argv[1])
selected = None
with profile.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("selected"):
            selected = row
if not selected:
    raise SystemExit(f"No selected policy in {profile}")
plan = selected.get("selected_batch_plan")
if not plan or not Path(plan).is_file():
    raise SystemExit(f"Selected policy has no persistent batch plan in {profile}: {plan}")
print(plan)
PY
}

run_pilot_inference() {
  if ! grep -qx "$PILOT_MODEL" "$SMOKE_PASS_MODELS"; then
    fail "pilot-inference" "$PILOT_MODEL did not pass smoke inference"
  fi

  local selected_plan
  check_root_free_or_stop_downloads || fail "pilot-inference" "root free space below ${ROOT_MIN_FREE_GIB}GiB before pilot inference"
  selected_plan="$(selected_plan_for_model "$PILOT_MODEL")"
  local output_dir="$RESULT_ROOT/pilot/$PILOT_MODEL"
  mkdir -p "$output_dir"

  for variant in "${VARIANTS[@]}"; do
    local manifest="$PILOT_DIR/manifests/fairspeech_${variant}_manifest.csv"
    local infer_log="$LOG_DIR/pilot_${PILOT_MODEL}_${variant}.log"
    log "Pilot inference: model=$PILOT_MODEL variant=$variant"
    check_gpu_memory "$GPU_PREFLIGHT_THRESHOLD" || \
      fail "pilot-inference" "GPU busy before $PILOT_MODEL/$variant"
    "$PYTHON_BIN" scripts/inference/run_inference.py \
      --model "$PILOT_MODEL" \
      --manifest "$manifest" \
      --output_dir "$output_dir" \
      --device cuda \
      --batch_plan "$selected_plan" \
      --audio_variant "$variant" \
      --dataset fs \
      --resume \
      > "$infer_log" 2>&1 || fail "pilot-inference" "pilot inference failed for $variant; see $infer_log"
  done

  local csv_count
  csv_count="$(find "$output_dir" -maxdepth 1 -type f -name 'predictions_*.csv' | wc -l | tr -d ' ')"
  [ "$csv_count" -ge 6 ] || fail "pilot-inference" "expected at least 6 prediction CSVs, found $csv_count"

  local metrics_dir="$RESULT_ROOT/pilot_metrics/$PILOT_MODEL"
  local plots_dir="$RESULT_ROOT/pilot_plots/$PILOT_MODEL"
  "$PYTHON_BIN" scripts/metrics/compute_fairspeech_compression_metrics.py \
    --predictions-dir "$output_dir" \
    --output-dir "$metrics_dir" \
    --min-group-size 1 \
    > "$LOG_DIR/pilot_${PILOT_MODEL}_metrics.log" 2>&1 || \
    fail "pilot-metrics" "metric computation failed; see $LOG_DIR/pilot_${PILOT_MODEL}_metrics.log"
  "$PYTHON_BIN" scripts/plots/generate_fairspeech_compression_plots.py \
    --metrics-dir "$metrics_dir" \
    --output-dir "$plots_dir" \
    > "$LOG_DIR/pilot_${PILOT_MODEL}_plots.log" 2>&1 || \
    fail "pilot-plots" "plot generation failed; see $LOG_DIR/pilot_${PILOT_MODEL}_plots.log"
}

finalize_status() {
  RESULT_ROOT="$RESULT_ROOT" \
  PILOT_DIR="$PILOT_DIR" \
  PROFILE_DIR="$PROFILE_DIR" \
  MODEL_STATUS_JSONL="$MODEL_STATUS_JSONL" \
  PILOT_MODEL="$PILOT_MODEL" \
  "$PYTHON_BIN" - "$STATUS_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
result_root = Path(os.environ["RESULT_ROOT"])
pilot_model = os.environ["PILOT_MODEL"]
prediction_dir = result_root / "pilot" / pilot_model
metrics_dir = result_root / "pilot_metrics" / pilot_model
plots_dir = result_root / "pilot_plots" / pilot_model

model_status_path = Path(os.environ["MODEL_STATUS_JSONL"])
model_statuses = []
if model_status_path.is_file():
    model_statuses = [
        json.loads(line) for line in model_status_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

payload = {
    "status": "complete",
    "step": "complete",
    "message": "Stopped at pilot gate before full 54-run evaluation.",
    "timestamp_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "pilot_dir": os.environ["PILOT_DIR"],
    "profile_dir": os.environ["PROFILE_DIR"],
    "model_status_jsonl": str(model_status_path),
    "model_status_count": len(model_statuses),
    "prediction_csv_count": len(list(prediction_dir.glob("predictions_*.csv"))),
    "prediction_dir": str(prediction_dir),
    "metrics_dir": str(metrics_dir),
    "plots_dir": str(plots_dir),
}
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

main() {
  local started=0
  maybe_run_step() {
    local step="$1"
    shift
    if [ "$started" -eq 0 ] && [ "$step" != "$START_STEP" ]; then
      log "SKIP before START_STEP=$START_STEP: $step"
      return 0
    fi
    started=1
    run_step "$step" "$@"
  }

  maybe_run_step "preflight" preflight
  maybe_run_step "full-manifests" generate_full_manifests
  maybe_run_step "full-baseline-validation" validate_full_baseline_manifest
  maybe_run_step "duration-calibration" build_calibration_subset
  maybe_run_step "pilot-metadata" write_pilot_metadata
  maybe_run_step "pilot-audio-generation" generate_pilot_audio
  maybe_run_step "pilot-audio-validation" validate_pilot_audio
  maybe_run_step "model-smoke-manifest" write_smoke_manifest
  maybe_run_step "model-download-and-smoke" download_and_smoke_models
  maybe_run_step "duration-budget-profiling" profile_smoke_passing_models
  maybe_run_step "wav2vec2-large-pilot" run_pilot_inference
  [ "$started" -eq 1 ] || fail "startup" "START_STEP=$START_STEP did not match any known step"
  finalize_status
  log "COMPLETE: supervised FairSpeech VM prep stopped at pilot gate."
}

main "$@"
