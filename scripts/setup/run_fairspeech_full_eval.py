#!/usr/bin/env python3
"""Supervise the full FairSpeech compression inference matrix.

This is the post-pilot launcher for the 9 model x 6 variant evaluation. It is
safe to import in tests and safe to run in dry stages: no model downloads or GPU
inference happen unless the ``run-matrix`` command is invoked.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COUNT = 26_471
DEFAULT_MAX_AUDIO_SECONDS = 160.0
DEFAULT_MAX_SAMPLES = 16
DEFAULT_ROOT_MIN_FREE_GIB = 12.0
DEFAULT_RUNTIME_MIN_FREE_GIB = 12.0
DEFAULT_GPU_THRESHOLD = 0.80

MODEL_ORDER = [
    "wav2vec2-large",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v3",
    "qwen3-asr-0.6b",
    "qwen3-asr-1.7b",
    "canary-qwen-2.5b",
    "granite-speech-3.3-2b",
    "granite-speech-3.3-8b",
]

VARIANTS = [
    "baseline",
    "bottleneck_12k",
    "bottleneck_8k",
    "mp3_64k",
    "mp3_32k",
    "mp3_16k",
]


@dataclass(frozen=True)
class RuntimePaths:
    dataset_dir: Path
    manifest_dir: Path
    plan_dir: Path
    summary_dir: Path
    run_dir: Path
    log_dir: Path
    model_status_jsonl: Path
    derived_audio_root: Path
    profile_work_root: Path
    result_root: Path
    hf_home: Path
    hf_hub_cache: Path
    transformers_cache: Path
    wandb_dir: Path
    pip_cache_dir: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RuntimePaths":
        dataset_dir = args.dataset_dir
        hf_home = args.hf_home
        hf_hub_cache = args.hf_hub_cache or hf_home / "hub"
        return cls(
            dataset_dir=dataset_dir,
            manifest_dir=dataset_dir / "manifests",
            plan_dir=dataset_dir / "batch_plans",
            summary_dir=dataset_dir / "summaries",
            run_dir=dataset_dir / "full_eval",
            log_dir=dataset_dir / "logs" / "full_eval",
            model_status_jsonl=args.model_status_jsonl
            or dataset_dir / "model_cache" / "model_cache_status.jsonl",
            derived_audio_root=args.derived_audio_root,
            profile_work_root=args.profile_work_root,
            result_root=args.result_root,
            hf_home=hf_home,
            hf_hub_cache=hf_hub_cache,
            transformers_cache=args.transformers_cache or hf_hub_cache,
            wandb_dir=args.wandb_dir,
            pip_cache_dir=args.pip_cache_dir,
        )

    def runtime_dirs(self) -> list[Path]:
        return [
            self.dataset_dir,
            self.manifest_dir,
            self.plan_dir,
            self.summary_dir,
            self.run_dir,
            self.log_dir,
            self.derived_audio_root,
            self.profile_work_root,
            self.result_root,
            self.hf_home,
            self.hf_hub_cache,
            self.transformers_cache,
            self.wandb_dir,
            self.pip_cache_dir,
        ]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_model_label(model: str) -> str:
    return model.replace("-", "_")


def safe_variant_label(variant: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in variant).strip("_")


def manifest_path(paths: RuntimePaths, variant: str) -> Path:
    return paths.manifest_dir / f"fairspeech_{variant}_manifest.csv"


def plan_path(paths: RuntimePaths, variant: str, max_audio_seconds: float, max_samples: int) -> Path:
    seconds = f"{max_audio_seconds:g}".replace(".", "p")
    return paths.plan_dir / f"fairspeech_{variant}_total{seconds}s_max{max_samples}_plan.jsonl"


def plan_summary_path(
    paths: RuntimePaths,
    variant: str,
    max_audio_seconds: float,
    max_samples: int,
) -> Path:
    seconds = f"{max_audio_seconds:g}".replace(".", "p")
    return paths.plan_dir / f"fairspeech_{variant}_total{seconds}s_max{max_samples}_summary.json"


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def disk_entry(label: str, path: Path) -> dict[str, Any]:
    existing = nearest_existing_parent(path)
    usage = shutil.disk_usage(existing)
    return {
        "label": label,
        "path": str(path),
        "existing_path": str(existing),
        "total_gib": round(gib(usage.total), 3),
        "used_gib": round(gib(usage.used), 3),
        "free_gib": round(gib(usage.free), 3),
    }


def disk_audit(
    paths: RuntimePaths,
    root_min_free_gib: float = DEFAULT_ROOT_MIN_FREE_GIB,
    runtime_min_free_gib: float = DEFAULT_RUNTIME_MIN_FREE_GIB,
) -> dict[str, Any]:
    entries = [
        disk_entry("root", Path("/")),
        disk_entry("opt", Path("/opt")),
        disk_entry("hf_home", paths.hf_home),
        disk_entry("hf_hub_cache", paths.hf_hub_cache),
        disk_entry("result_root", paths.result_root),
        disk_entry("derived_audio_root", paths.derived_audio_root),
        disk_entry("profile_work_root", paths.profile_work_root),
        disk_entry("wandb_dir", paths.wandb_dir),
        disk_entry("pip_cache_dir", paths.pip_cache_dir),
    ]
    errors: list[str] = []
    warnings: list[str] = []
    for entry in entries:
        free_gib = float(entry["free_gib"])
        label = str(entry["label"])
        path = Path(str(entry["path"]))
        if label == "root" and free_gib < root_min_free_gib:
            errors.append(f"/ free space {free_gib:.1f} GiB is below {root_min_free_gib:g} GiB")
        elif label != "root" and free_gib < runtime_min_free_gib:
            errors.append(
                f"{label} free space {free_gib:.1f} GiB is below {runtime_min_free_gib:g} GiB"
            )
        if label in {"hf_home", "hf_hub_cache", "result_root", "derived_audio_root"}:
            try:
                path.relative_to(PROJECT_ROOT)
                warnings.append(f"{label} is under the workspace; prefer /opt or attached storage")
            except ValueError:
                pass
    return {
        "status": "pass" if not errors else "fail",
        "timestamp_utc": timestamp(),
        "root_min_free_gib": root_min_free_gib,
        "runtime_min_free_gib": runtime_min_free_gib,
        "entries": entries,
        "warnings": warnings,
        "errors": errors,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_batches(
    manifest: Path,
    output_jsonl: Path,
    batches: list[dict[str, Any]],
    max_audio_seconds: float,
    max_samples: int,
) -> dict[str, Any]:
    sample_counts = [int(batch["n_samples"]) for batch in batches]
    bucket_counts = Counter(str(batch["duration_bucket"]) for batch in batches)
    sorted_counts = sorted(sample_counts)
    median_samples = 0
    if sorted_counts:
        mid = len(sorted_counts) // 2
        if len(sorted_counts) % 2:
            median_samples = sorted_counts[mid]
        else:
            median_samples = (sorted_counts[mid - 1] + sorted_counts[mid]) / 2
    return {
        "manifest": str(manifest),
        "output_jsonl": str(output_jsonl),
        "num_rows": sum(int(batch["n_samples"]) for batch in batches),
        "num_batches": len(batches),
        "budget_mode": "total",
        "selected_total_duration_seconds": max_audio_seconds,
        "max_samples_cap": max_samples,
        "median_batch_samples": median_samples,
        "max_observed_samples": max(sample_counts, default=0),
        "max_observed_sum_duration_seconds": max(
            (float(batch["sum_duration_seconds"]) for batch in batches),
            default=0.0,
        ),
        "max_observed_padded_seconds": max(
            (float(batch["padded_audio_seconds"]) for batch in batches),
            default=0.0,
        ),
        "duration_buckets": dict(sorted(bucket_counts.items())),
    }


def validate_batch_plan(
    manifest: Path,
    plan: Path,
    expected_count: int,
    max_audio_seconds: float,
    max_samples: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if not manifest.is_file():
        return {"status": "fail", "manifest": str(manifest), "plan": str(plan), "errors": ["missing manifest"]}
    if not plan.is_file():
        return {"status": "fail", "manifest": str(manifest), "plan": str(plan), "errors": ["missing plan"]}

    manifest_rows = read_csv_rows(manifest)
    manifest_ids = [str(row.get("utterance_id", "")) for row in manifest_rows]
    if len(manifest_rows) != expected_count:
        errors.append(f"expected {expected_count:,} manifest rows, found {len(manifest_rows):,}")
    if len(set(manifest_ids)) != len(manifest_ids):
        errors.append("manifest has duplicate utterance_id values")

    planned_ids: list[str] = []
    records = read_jsonl(plan)
    for idx, record in enumerate(records, start=1):
        n_samples = int(record.get("n_samples", 0))
        sum_duration = float(record.get("sum_duration_seconds", 0.0))
        if n_samples > max_samples:
            errors.append(f"batch {idx} has {n_samples} samples > {max_samples}")
        if sum_duration > max_audio_seconds + 1e-6:
            errors.append(f"batch {idx} has {sum_duration:.3f}s > {max_audio_seconds:g}s")
        ids = [str(uid) for uid in record.get("utterance_ids") or []]
        if len(ids) != n_samples:
            errors.append(f"batch {idx} utterance_ids length does not match n_samples")
        planned_ids.extend(ids)

    planned_counter = Counter(planned_ids)
    manifest_set = set(manifest_ids)
    planned_set = set(planned_ids)
    missing = sorted(manifest_set - planned_set)
    extra = sorted(planned_set - manifest_set)
    duplicated = sorted(uid for uid, count in planned_counter.items() if uid and count > 1)
    if missing:
        errors.append(f"plan is missing {len(missing):,} manifest utterance_ids; first={missing[:5]}")
    if extra:
        errors.append(f"plan has {len(extra):,} unknown utterance_ids; first={extra[:5]}")
    if duplicated:
        errors.append(f"plan has {len(duplicated):,} duplicated utterance_ids; first={duplicated[:5]}")

    return {
        "status": "pass" if not errors else "fail",
        "manifest": str(manifest),
        "plan": str(plan),
        "manifest_rows": len(manifest_rows),
        "plan_rows": len(planned_ids),
        "num_batches": len(records),
        "max_audio_seconds": max_audio_seconds,
        "max_samples": max_samples,
        "errors": errors,
    }


def build_full_batch_plans(
    paths: RuntimePaths,
    expected_count: int,
    max_audio_seconds: float,
    max_samples: int,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.inference.build_duration_batch_plan import (  # noqa: PLC0415
        build_batches,
        read_manifest,
        write_jsonl as write_batch_jsonl,
    )

    summaries = []
    for variant in VARIANTS:
        manifest = manifest_path(paths, variant)
        output_jsonl = plan_path(paths, variant, max_audio_seconds, max_samples)
        output_summary = plan_summary_path(paths, variant, max_audio_seconds, max_samples)
        if output_jsonl.exists() and output_summary.exists() and not overwrite:
            validation = validate_batch_plan(
                manifest,
                output_jsonl,
                expected_count,
                max_audio_seconds,
                max_samples,
            )
            summaries.append({**validation, "variant": variant, "reused": True})
            continue
        rows = read_manifest(manifest)
        batches = build_batches(
            rows,
            max_audio_seconds=max_audio_seconds,
            max_samples=max_samples,
            budget_mode="total",
        )
        write_batch_jsonl(output_jsonl, batches)
        summary = summarize_batches(manifest, output_jsonl, batches, max_audio_seconds, max_samples)
        write_json(output_summary, summary)
        validation = validate_batch_plan(
            manifest,
            output_jsonl,
            expected_count,
            max_audio_seconds,
            max_samples,
        )
        summaries.append({**summary, **validation, "variant": variant, "reused": False})
    return summaries


def latest_model_statuses(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        model = row.get("model") or row.get("model_key")
        if model:
            latest[str(model)] = row
    return latest


def select_models(raw: str, model_status_jsonl: Path) -> list[str]:
    if raw == "all":
        return MODEL_ORDER
    if raw == "from-status":
        statuses = latest_model_statuses(model_status_jsonl)
        passed = [
            model
            for model in MODEL_ORDER
            if statuses.get(model, {}).get("download_status") == "downloaded"
            and statuses.get(model, {}).get("smoke_status") == "passed"
        ]
        if not passed:
            raise SystemExit(
                f"No smoke-passing models found in {model_status_jsonl}. "
                "Run the cache/smoke gate first or pass --models explicitly."
            )
        return passed
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [model for model in requested if model not in MODEL_ORDER]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}. Known: {MODEL_ORDER}")
    return requested


def runtime_env(paths: RuntimePaths, cuda_visible_devices: str | None = None) -> dict[str, str]:
    env = {
        "HF_HOME": str(paths.hf_home),
        "HF_HUB_CACHE": str(paths.hf_hub_cache),
        "TRANSFORMERS_CACHE": str(paths.transformers_cache),
        "WANDB_DIR": str(paths.wandb_dir),
        "PIP_CACHE_DIR": str(paths.pip_cache_dir),
    }
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def prediction_path(result_root: Path, model: str, variant: str) -> Path:
    return result_root / "full" / model / f"predictions_{safe_model_label(model)}_{safe_variant_label(variant)}.csv"


def build_run_matrix(
    paths: RuntimePaths,
    models: list[str],
    expected_count: int,
    max_audio_seconds: float,
    max_samples: int,
    device: str,
    cuda_visible_devices: str | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    python = python_bin or sys.executable
    validations = [
        validate_batch_plan(
            manifest_path(paths, variant),
            plan_path(paths, variant, max_audio_seconds, max_samples),
            expected_count,
            max_audio_seconds,
            max_samples,
        )
        for variant in VARIANTS
    ]
    failed = [item for item in validations if item["status"] != "pass"]
    if failed:
        raise SystemExit(f"Batch-plan validation failed: {json.dumps(failed[:2], indent=2)}")

    env = runtime_env(paths, cuda_visible_devices)
    runs = []
    for model in models:
        output_dir = paths.result_root / "full" / model
        for variant in VARIANTS:
            manifest = manifest_path(paths, variant)
            batch_plan = plan_path(paths, variant, max_audio_seconds, max_samples)
            log_path = paths.log_dir / f"infer_{model}_{variant}.log"
            command = [
                python,
                str(PROJECT_ROOT / "scripts" / "inference" / "run_inference.py"),
                "--model",
                model,
                "--manifest",
                str(manifest),
                "--output_dir",
                str(output_dir),
                "--device",
                device,
                "--batch_plan",
                str(batch_plan),
                "--audio_variant",
                variant,
                "--dataset",
                "fs",
                "--resume",
            ]
            runs.append(
                {
                    "model": model,
                    "audio_variant": variant,
                    "manifest": str(manifest),
                    "batch_plan": str(batch_plan),
                    "output_dir": str(output_dir),
                    "prediction_csv": str(prediction_path(paths.result_root, model, variant)),
                    "log_path": str(log_path),
                    "command": command,
                }
            )

    return {
        "schema_version": 1,
        "created_utc": timestamp(),
        "project_root": str(PROJECT_ROOT),
        "expected_count": expected_count,
        "max_audio_seconds": max_audio_seconds,
        "max_samples": max_samples,
        "models": models,
        "variants": VARIANTS,
        "runtime_paths": {k: str(v) for k, v in asdict(paths).items()},
        "env": env,
        "validations": validations,
        "runs": runs,
    }


def validate_prediction_csv(prediction_csv: Path, manifest: Path, expected_count: int) -> dict[str, Any]:
    errors: list[str] = []
    if not prediction_csv.is_file():
        return {
            "status": "fail",
            "prediction_csv": str(prediction_csv),
            "manifest": str(manifest),
            "errors": ["missing prediction CSV"],
        }
    prediction_rows = read_csv_rows(prediction_csv)
    manifest_rows = read_csv_rows(manifest)
    pred_ids = [str(row.get("utterance_id", "")) for row in prediction_rows]
    manifest_ids = [str(row.get("utterance_id", "")) for row in manifest_rows]
    if len(prediction_rows) != expected_count:
        errors.append(f"expected {expected_count:,} prediction rows, found {len(prediction_rows):,}")
    if set(pred_ids) != set(manifest_ids):
        errors.append("prediction utterance_id coverage does not match manifest")
    if len(pred_ids) != len(set(pred_ids)):
        errors.append("prediction CSV has duplicate utterance_id values")
    if not any((row.get("hypothesis") or row.get("hypothesis_raw") or "").strip() for row in prediction_rows):
        errors.append("transcript columns are globally empty")
    required = {"model", "audio_variant", "wer"}
    columns = set(prediction_rows[0].keys()) if prediction_rows else set()
    missing = sorted(required - columns)
    if missing:
        errors.append(f"missing required prediction columns: {missing}")
    elif any((row.get("wer") is None or str(row.get("wer")).strip() == "") for row in prediction_rows):
        errors.append("one or more prediction rows has missing wer")
    return {
        "status": "pass" if not errors else "fail",
        "prediction_csv": str(prediction_csv),
        "manifest": str(manifest),
        "rows": len(prediction_rows),
        "errors": errors,
    }


def query_gpu_memory(cuda_visible_devices: str | None, threshold: float) -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"status": "fail", "errors": ["nvidia-smi not found"], "gpus": []}
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.check_output(cmd, text=True).strip().splitlines()
    requested = None
    if cuda_visible_devices:
        requested = {item.strip() for item in cuda_visible_devices.split(",") if item.strip()}
    gpus = []
    errors = []
    for line in raw:
        idx_raw, used_raw, total_raw = [part.strip() for part in line.split(",")]
        if requested is not None and idx_raw not in requested:
            continue
        used = int(used_raw)
        total = int(total_raw)
        fraction = used / total if total else 1.0
        gpus.append(
            {
                "index": idx_raw,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "memory_fraction": fraction,
            }
        )
        if fraction > threshold:
            errors.append(f"GPU {idx_raw} memory {fraction:.1%} exceeds {threshold:.0%}")
    if not gpus:
        errors.append("no GPUs matched CUDA_VISIBLE_DEVICES")
    return {"status": "pass" if not errors else "fail", "errors": errors, "gpus": gpus}


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    for path in paths.runtime_dirs():
        path.mkdir(parents=True, exist_ok=True)


def run_preflight(args: argparse.Namespace, paths: RuntimePaths) -> dict[str, Any]:
    ensure_runtime_dirs(paths)
    audit = disk_audit(paths, args.root_min_free_gib, args.runtime_min_free_gib)
    gpu = None
    if args.device.startswith("cuda") and not args.skip_gpu_check:
        gpu = query_gpu_memory(args.cuda_visible_devices, args.gpu_threshold)
        if gpu["status"] != "pass":
            audit["errors"].extend(gpu["errors"])
            audit["status"] = "fail"
    payload = {
        "status": audit["status"],
        "timestamp_utc": timestamp(),
        "disk": audit,
        "gpu": gpu,
        "runtime_env": runtime_env(paths, args.cuda_visible_devices),
    }
    write_json(paths.run_dir / "full_eval_preflight.json", payload)
    if payload["status"] != "pass":
        raise SystemExit(json.dumps(payload["disk"]["errors"], indent=2))
    return payload


def write_matrix(args: argparse.Namespace, paths: RuntimePaths) -> dict[str, Any]:
    models = select_models(args.models, paths.model_status_jsonl)
    matrix = build_run_matrix(
        paths,
        models,
        args.expected_count,
        args.max_audio_seconds,
        args.max_samples,
        args.device,
        args.cuda_visible_devices,
        args.python_bin,
    )
    write_json(args.matrix_json or paths.run_dir / "full_eval_run_matrix.json", matrix)
    return matrix


def append_status(paths: RuntimePaths, record: dict[str, Any]) -> None:
    record = {"timestamp_utc": timestamp(), **record}
    write_jsonl(paths.run_dir / "full_eval_status.jsonl", [record], append=True)


def validate_matrix_predictions(
    matrix: dict[str, Any],
    expected_count: int,
    output_json: Path | None = None,
) -> dict[str, Any]:
    validations = []
    for run in matrix["runs"]:
        validations.append(
            validate_prediction_csv(
                Path(run["prediction_csv"]),
                Path(run["manifest"]),
                expected_count,
            )
        )
    errors = [item for item in validations if item["status"] != "pass"]
    payload = {
        "status": "pass" if not errors else "fail",
        "timestamp_utc": timestamp(),
        "validated_csvs": len(validations),
        "passed_csvs": len(validations) - len(errors),
        "failed_csvs": len(errors),
        "validations": validations,
    }
    if output_json:
        write_json(output_json, payload)
    return payload


def run_matrix(args: argparse.Namespace, paths: RuntimePaths) -> None:
    if args.require_tmux and not os.environ.get("TMUX"):
        raise SystemExit("run-matrix must be launched inside tmux; pass --allow-outside-tmux to override")

    matrix_path = args.matrix_json or paths.run_dir / "full_eval_run_matrix.json"
    if matrix_path.is_file():
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    else:
        matrix = write_matrix(args, paths)

    base_env = os.environ.copy()
    base_env.update(matrix.get("env", {}))
    for run in matrix["runs"]:
        prediction_csv = Path(run["prediction_csv"])
        validation = validate_prediction_csv(prediction_csv, Path(run["manifest"]), args.expected_count)
        if validation["status"] == "pass":
            append_status(paths, {**run, "status": "skipped_complete", "validation": validation})
            continue

        audit = disk_audit(paths, args.root_min_free_gib, args.runtime_min_free_gib)
        if audit["status"] != "pass":
            append_status(paths, {**run, "status": "blocked_disk", "disk": audit})
            raise SystemExit(f"Disk preflight failed before {run['model']}/{run['audio_variant']}")
        if args.device.startswith("cuda") and not args.skip_gpu_check:
            gpu = query_gpu_memory(args.cuda_visible_devices, args.gpu_threshold)
            if gpu["status"] != "pass":
                append_status(paths, {**run, "status": "blocked_gpu", "gpu": gpu})
                raise SystemExit(f"GPU preflight failed before {run['model']}/{run['audio_variant']}")

        log_path = Path(run["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{timestamp()}] START {run['model']} {run['audio_variant']}\n")
            proc = subprocess.run(run["command"], stdout=log_file, stderr=subprocess.STDOUT, env=base_env)
            log_file.write(f"[{timestamp()}] EXIT returncode={proc.returncode}\n")
        elapsed = time.time() - started
        validation = validate_prediction_csv(prediction_csv, Path(run["manifest"]), args.expected_count)
        status = "complete" if proc.returncode == 0 and validation["status"] == "pass" else "failed"
        append_status(
            paths,
            {
                **run,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "validation": validation,
            },
        )
        if status != "complete" and not args.continue_on_error:
            raise SystemExit(f"Run failed for {run['model']}/{run['audio_variant']}; see {log_path}")


def run_subprocess(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{timestamp()}] {' '.join(command)}\n")
        proc = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}); see {log_path}")


def generate_full_audio(args: argparse.Namespace, paths: RuntimePaths) -> None:
    command = [
        args.python_bin,
        str(PROJECT_ROOT / "scripts" / "data" / "prepare_fairspeech_compression.py"),
        "--metadata",
        str(args.source_metadata),
        "--audio-dir",
        str(args.source_audio_dir),
        "--output-dir",
        str(paths.dataset_dir),
        "--audio-output-dir",
        str(paths.derived_audio_root),
        "--variants",
        "all",
        "--generate-audio",
        "--expected-full-count",
        str(args.expected_count),
        "--jobs",
        str(args.audio_jobs),
    ]
    if args.overwrite_audio:
        command.append("--overwrite")
    run_subprocess(command, paths.log_dir / "full_audio_generation.log")


def validate_full_audio(args: argparse.Namespace, paths: RuntimePaths) -> None:
    command = [
        args.python_bin,
        str(PROJECT_ROOT / "scripts" / "validation" / "validate_fairspeech_compression.py"),
        *[str(manifest_path(paths, variant)) for variant in VARIANTS],
        "--expected-count",
        str(args.expected_count),
        "--summary-json",
        str(paths.summary_dir / "full_audio_validation.json"),
    ]
    if not args.skip_silence_check:
        command.append("--check-silence")
        command.extend(["--silence-sample-frames", str(args.silence_sample_frames)])
    run_subprocess(command, paths.log_dir / "full_audio_validation.log")


def compute_metrics(args: argparse.Namespace, paths: RuntimePaths) -> None:
    command = [
        args.python_bin,
        str(PROJECT_ROOT / "scripts" / "metrics" / "compute_fairspeech_compression_metrics.py"),
        "--predictions-dir",
        str(paths.result_root / "full"),
        "--output-dir",
        str(paths.result_root / "full_metrics"),
        "--min-group-size",
        str(args.min_group_size),
        "--recursive",
    ]
    run_subprocess(command, paths.log_dir / "full_metrics.log")


def generate_plots(args: argparse.Namespace, paths: RuntimePaths) -> None:
    command = [
        args.python_bin,
        str(PROJECT_ROOT / "scripts" / "plots" / "generate_fairspeech_compression_plots.py"),
        "--metrics-dir",
        str(paths.result_root / "full_metrics"),
        "--output-dir",
        str(paths.result_root / "full_plots"),
    ]
    run_subprocess(command, paths.log_dir / "full_plots.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FairSpeech compression evaluation")
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "generate-audio",
            "validate-audio",
            "build-plans",
            "write-matrix",
            "run-matrix",
            "validate-predictions",
            "metrics",
            "plots",
            "all",
        ],
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "datasets" / "fairspeech_compression")
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "meta_speech_fairness" / "downloaded_file.tsv",
    )
    parser.add_argument(
        "--source-audio-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "meta_speech_fairness" / "asr_fairness_audio",
    )
    parser.add_argument("--derived-audio-root", type=Path, default=Path("/opt/fairspeech-variants/full"))
    parser.add_argument("--profile-work-root", type=Path, default=Path("/opt/fairspeech-variants/profile_work"))
    parser.add_argument("--result-root", type=Path, default=Path("/opt/fairspeech-results"))
    parser.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/opt/hf-cache")))
    parser.add_argument("--hf-hub-cache", type=Path, default=None)
    parser.add_argument("--transformers-cache", type=Path, default=None)
    parser.add_argument("--wandb-dir", type=Path, default=Path("/opt/wandb"))
    parser.add_argument("--pip-cache-dir", type=Path, default=Path("/opt/pip-cache"))
    parser.add_argument("--model-status-jsonl", type=Path, default=None)
    parser.add_argument("--matrix-json", type=Path, default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--models", default="from-status", help="'from-status', 'all', or comma model keys")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--max-audio-seconds", type=float, default=DEFAULT_MAX_AUDIO_SECONDS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--root-min-free-gib", type=float, default=DEFAULT_ROOT_MIN_FREE_GIB)
    parser.add_argument("--runtime-min-free-gib", type=float, default=DEFAULT_RUNTIME_MIN_FREE_GIB)
    parser.add_argument("--gpu-threshold", type=float, default=DEFAULT_GPU_THRESHOLD)
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--audio-jobs", type=int, default=8)
    parser.add_argument("--skip-silence-check", action="store_true")
    parser.add_argument("--silence-sample-frames", type=int, default=4096)
    parser.add_argument("--overwrite-plans", action="store_true")
    parser.add_argument("--allow-outside-tmux", dest="require_tmux", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--min-group-size", type=int, default=50)
    parser.set_defaults(require_tmux=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RuntimePaths.from_args(args)

    if args.command in {"preflight", "all"}:
        run_preflight(args, paths)
    if args.command in {"generate-audio", "all"}:
        ensure_runtime_dirs(paths)
        generate_full_audio(args, paths)
    if args.command in {"validate-audio", "all"}:
        validate_full_audio(args, paths)
    if args.command in {"build-plans", "all"}:
        ensure_runtime_dirs(paths)
        summaries = build_full_batch_plans(
            paths,
            args.expected_count,
            args.max_audio_seconds,
            args.max_samples,
            overwrite=args.overwrite_plans,
        )
        write_json(paths.run_dir / "full_batch_plan_validation.json", summaries)
        failures = [item for item in summaries if item["status"] != "pass"]
        if failures:
            raise SystemExit(f"Batch-plan build/validation failed: {failures[:2]}")
    if args.command in {"write-matrix", "all"}:
        write_matrix(args, paths)
    if args.command in {"run-matrix", "all"}:
        run_matrix(args, paths)
    if args.command in {"validate-predictions", "all"}:
        matrix_path = args.matrix_json or paths.run_dir / "full_eval_run_matrix.json"
        if not matrix_path.is_file():
            raise SystemExit(f"Missing run matrix: {matrix_path}")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        validation = validate_matrix_predictions(
            matrix,
            args.expected_count,
            paths.run_dir / "full_prediction_validation.json",
        )
        if validation["status"] != "pass":
            raise SystemExit("Prediction validation failed")
    if args.command in {"metrics", "all"}:
        compute_metrics(args, paths)
    if args.command in {"plots", "all"}:
        generate_plots(args, paths)


if __name__ == "__main__":
    main()
