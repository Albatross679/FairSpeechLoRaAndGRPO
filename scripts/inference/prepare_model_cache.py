#!/usr/bin/env python3
"""Prepare/download model cache manifest for the FairSpeech inference benchmark.

Default mode is safe: it writes a manifest for the nine benchmark models without
loading weights or downloading anything. Add ``--download`` explicitly on the VM
when ready to populate the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "fairspeech_compression" / "model_cache_manifest.json"

# Keep this order aligned with the teammate benchmark and docs Chapter 2.
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


def import_registry() -> dict[str, dict[str, str]]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.inference.run_inference import MODEL_REGISTRY  # noqa: PLC0415

    return MODEL_REGISTRY


def parse_models(raw: str) -> list[str]:
    if raw == "all":
        return MODEL_ORDER
    requested = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in requested if x not in MODEL_ORDER]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}. Known: {MODEL_ORDER}")
    return requested


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            pass
    return total


def package_version(pkg: str) -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version(pkg)
    except Exception:
        return None


def snapshot_download(hf_id: str, cache_dir: str | None, revision: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "huggingface_hub is required for --download. Install transformers/huggingface_hub first."
        ) from exc

    path = hf_snapshot_download(repo_id=hf_id, cache_dir=cache_dir, revision=revision)
    return Path(path)


def smoke_command(model_key: str, smoke_audio: Path, output_dir: Path, device: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "inference" / "run_inference.py"),
        "--model", model_key,
        "--manifest", str(output_dir / f"smoke_{model_key}.csv"),
        "--output_dir", str(output_dir / "smoke_outputs"),
        "--device", device,
        "--max_samples", "1",
        "--batch_size", "1",
        "--audio_variant", "model_cache_smoke",
    ]


def write_smoke_manifest(path: Path, smoke_audio: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance_id", "audio_path", "sentence", "ethnicity"])
        writer.writeheader()
        writer.writerow({
            "utterance_id": smoke_audio.stem,
            "audio_path": str(smoke_audio),
            "sentence": "",
            "ethnicity": "",
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASR model cache manifest")
    parser.add_argument("--models", default="all", help="Comma-separated model keys or 'all'")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download", action="store_true", help="Download model snapshots into cache")
    parser.add_argument("--device", default="cuda", help="Device for optional smoke command metadata")
    parser.add_argument("--smoke-audio", type=Path, default=None, help="Optional one-file audio path for command template only")
    args = parser.parse_args()

    registry = import_registry()
    selected = parse_models(args.models)
    records = []
    for model_key in selected:
        info = registry[model_key]
        hf_id = info["hf_id"]
        snapshot_path: Path | None = None
        status = "planned"
        error = None
        if args.download:
            try:
                snapshot_path = snapshot_download(hf_id, args.cache_dir, args.revision)
                status = "downloaded"
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = str(exc)

        record = {
            "model_key": model_key,
            "hf_id": hf_id,
            "generation": info.get("generation"),
            "architecture": info.get("architecture"),
            "params": info.get("params"),
            "type": info.get("type"),
            "cache_dir": args.cache_dir,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "snapshot_exists": bool(snapshot_path and snapshot_path.exists()),
            "snapshot_size_bytes": directory_size_bytes(snapshot_path) if snapshot_path else None,
            "revision": args.revision,
            "status": status,
            "error": error,
        }
        records.append(record)
        print(f"{model_key}: {status} ({hf_id})")

    manifest = {
        "models": records,
        "package_versions": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "torchaudio": package_version("torchaudio"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "qwen_asr": package_version("qwen-asr") or package_version("qwen_asr"),
            "nemo_toolkit": package_version("nemo_toolkit"),
        },
        "executables": {
            "python": sys.executable,
            "ffmpeg": shutil.which("ffmpeg"),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
    }

    if args.smoke_audio:
        smoke_dir = args.output.parent / "model_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for model_key in selected:
            write_smoke_manifest(smoke_dir / f"smoke_{model_key}.csv", args.smoke_audio)
        manifest["smoke_audio"] = str(args.smoke_audio)
        manifest["smoke_commands"] = {
            model_key: smoke_command(model_key, args.smoke_audio, smoke_dir, args.device)
            for model_key in selected
        }
        # We intentionally do not run smoke commands here. The VM pilot gate should run them.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote model cache manifest: {args.output}")

    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
