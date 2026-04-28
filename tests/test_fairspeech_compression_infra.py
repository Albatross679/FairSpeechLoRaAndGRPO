"""CPU-only tests for FairSpeech compression infrastructure."""

from __future__ import annotations

import csv
import json
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_wav(path: Path, sample_rate: int = 16000, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _write_metadata(path: Path, hash_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "hash_name",
                "transcription",
                "age",
                "gender",
                "first_language",
                "socioeconomic_bkgd",
                "ethnicity",
            ],
        )
        writer.writeheader()
        for idx, uid in enumerate(hash_names):
            writer.writerow({
                "hash_name": uid,
                "transcription": f"hello world {idx}",
                "age": "18 - 22",
                "gender": "female",
                "first_language": "English",
                "socioeconomic_bkgd": "Low",
                "ethnicity": "White",
            })


def test_build_source_and_variant_manifests(tmp_path):
    from scripts.data.prepare_fairspeech_compression import (
        build_source_manifest,
        variant_manifest_rows,
    )

    audio_dir = tmp_path / "audio"
    uids = ["a1", "b2"]
    for uid in uids:
        _write_wav(audio_dir / f"{uid}.wav")
    metadata = tmp_path / "downloaded_file.tsv"
    _write_metadata(metadata, uids)

    rows = build_source_manifest(metadata, audio_dir)
    assert len(rows) == 2
    assert rows[0]["source_sample_rate"] == 16000
    assert rows[0]["source_channels"] == 1
    assert rows[0]["ethnicity"] == "White"

    baseline_rows = variant_manifest_rows(rows, tmp_path / "out", "baseline", generated_audio=False)
    assert baseline_rows[0]["audio_path"] == rows[0]["source_audio_path"]
    assert baseline_rows[0]["variant"] == "baseline"

    mp3_rows = variant_manifest_rows(rows, tmp_path / "out", "mp3_32k", generated_audio=False)
    assert mp3_rows[0]["audio_path"].endswith("audio/mp3_32k/a1.wav")
    assert mp3_rows[0]["source_audio_path"] == rows[0]["source_audio_path"]


def test_variant_manifest_can_store_audio_outside_manifest_root(tmp_path):
    from scripts.data.prepare_fairspeech_compression import (
        build_source_manifest,
        variant_manifest_rows,
    )

    audio_dir = tmp_path / "audio"
    _write_wav(audio_dir / "a1.wav")
    metadata = tmp_path / "downloaded_file.tsv"
    _write_metadata(metadata, ["a1"])

    source_rows = build_source_manifest(metadata, audio_dir)
    manifest_root = tmp_path / "repo_dataset"
    audio_root = tmp_path / "opt_payload"
    rows = variant_manifest_rows(
        source_rows,
        manifest_root,
        "mp3_32k",
        generated_audio=True,
        audio_output_dir=audio_root,
    )

    assert rows[0]["audio_path"] == str(audio_root / "audio" / "mp3_32k" / "a1.wav")
    assert rows[0]["derived_audio_path"] == rows[0]["audio_path"]


def test_duration_batch_plan_uses_total_duration_by_default():
    from scripts.inference.build_duration_batch_plan import build_batches

    rows = []
    for idx, dur in enumerate([5, 5, 5, 20, 20, 40, 40]):
        rows.append({
            "_row_index": str(idx),
            "utterance_id": f"utt{idx}",
            "duration_seconds": str(dur),
            "ethnicity": "group",
        })

    batches = build_batches(rows, max_audio_seconds=80, max_samples=8, bucket_edges=[10, 30, 60])
    assert batches[0]["n_samples"] == 3  # 3 × 5s = 15 padded seconds
    assert any(batch["n_samples"] == 2 and batch["max_duration_seconds"] == 20 for batch in batches)
    assert any(batch["n_samples"] == 2 and batch["max_duration_seconds"] == 40 for batch in batches)
    assert max(batch["sum_duration_seconds"] for batch in batches) <= 80
    assert all(batch["budget_mode"] == "total" for batch in batches)
    assert all("padded_audio_seconds" in batch for batch in batches)


def test_duration_batch_plan_can_use_padded_seconds():
    from scripts.inference.build_duration_batch_plan import build_batches

    rows = []
    for idx, dur in enumerate([5, 5, 5, 20, 20, 40]):
        rows.append({
            "_row_index": str(idx),
            "utterance_id": f"utt{idx}",
            "duration_seconds": str(dur),
            "ethnicity": "group",
        })

    batches = build_batches(
        rows,
        max_audio_seconds=40,
        max_samples=8,
        bucket_edges=[10, 30, 60],
        budget_mode="padded",
    )
    assert max(batch["padded_audio_seconds"] for batch in batches) <= 40
    assert all(batch["budget_mode"] == "padded" for batch in batches)


def test_iter_manifest_batches_maps_by_utterance_id(tmp_path):
    import pandas as pd
    from scripts.inference.run_inference import iter_manifest_batches

    df = pd.DataFrame({
        "utterance_id": ["u1", "u3"],  # u2 simulates a resumed/already-finished row
        "audio_path": ["/tmp/u1.wav", "/tmp/u3.wav"],
        "sentence": ["one", "three"],
    })
    plan = tmp_path / "plan.jsonl"
    plan.write_text(
        json.dumps({"batch_id": "b0", "utterance_ids": ["u1", "u2"], "row_indices": [0, 1]}) + "\n"
        + json.dumps({"batch_id": "b1", "utterance_ids": ["u3"], "row_indices": [2]}) + "\n",
        encoding="utf-8",
    )

    class Args:
        batch_plan = str(plan)
        batch_size = 16

    batches = list(iter_manifest_batches(df, Args()))
    assert len(batches) == 2
    assert batches[0][1] == [0]
    assert batches[1][1] == [1]


def test_pilot_metrics_can_lower_min_group_size():
    from scripts.metrics.compute_fairspeech_compression_metrics import (
        compute_fairness_table,
        compute_group_table,
        compute_paired_delta,
    )

    rows = [
        {
            "model": "wav2vec2-large",
            "audio_variant": "baseline",
            "utterance_id": "a1",
            "ethnicity": "A",
            "reference": "hello world",
            "hypothesis": "hello world",
            "wer": "0.0",
        },
        {
            "model": "wav2vec2-large",
            "audio_variant": "baseline",
            "utterance_id": "b1",
            "ethnicity": "B",
            "reference": "hello world",
            "hypothesis": "hello",
            "wer": "0.5",
        },
        {
            "model": "wav2vec2-large",
            "audio_variant": "mp3_16k",
            "utterance_id": "a1",
            "ethnicity": "A",
            "reference": "hello world",
            "hypothesis": "hello",
            "wer": "0.5",
        },
        {
            "model": "wav2vec2-large",
            "audio_variant": "mp3_16k",
            "utterance_id": "b1",
            "ethnicity": "B",
            "reference": "hello world",
            "hypothesis": "hello",
            "wer": "0.5",
        },
    ]

    assert compute_group_table(rows, "ethnicity") == []
    pilot_groups = compute_group_table(rows, "ethnicity", min_group_size=1)
    assert len(pilot_groups) == 4
    assert compute_fairness_table(pilot_groups)
    assert len(compute_paired_delta(rows, "baseline", "ethnicity", min_group_size=1)) == 2


def _runtime_paths(tmp_path: Path):
    from scripts.setup.run_fairspeech_full_eval import RuntimePaths

    dataset_dir = tmp_path / "dataset"
    return RuntimePaths(
        dataset_dir=dataset_dir,
        manifest_dir=dataset_dir / "manifests",
        plan_dir=dataset_dir / "batch_plans",
        summary_dir=dataset_dir / "summaries",
        run_dir=dataset_dir / "full_eval",
        log_dir=dataset_dir / "logs" / "full_eval",
        model_status_jsonl=dataset_dir / "model_cache" / "model_cache_status.jsonl",
        derived_audio_root=tmp_path / "opt" / "fairspeech-variants" / "full",
        profile_work_root=tmp_path / "opt" / "fairspeech-variants" / "profile_work",
        result_root=tmp_path / "opt" / "fairspeech-results",
        hf_home=tmp_path / "opt" / "hf-cache",
        hf_hub_cache=tmp_path / "opt" / "hf-cache" / "hub",
        transformers_cache=tmp_path / "opt" / "hf-cache" / "hub",
        wandb_dir=tmp_path / "opt" / "wandb",
        pip_cache_dir=tmp_path / "opt" / "pip-cache",
    )


def _write_full_eval_manifest(path: Path, durations: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "audio_path", "sentence", "ethnicity", "duration_seconds"],
        )
        writer.writeheader()
        for idx, duration in enumerate(durations):
            writer.writerow({
                "utterance_id": f"utt{idx}",
                "audio_path": f"/tmp/utt{idx}.wav",
                "sentence": f"hello {idx}",
                "ethnicity": "group",
                "duration_seconds": duration,
            })


def test_full_eval_builds_six_guarded_plans_and_matrix(tmp_path):
    from scripts.setup.run_fairspeech_full_eval import (
        VARIANTS,
        build_full_batch_plans,
        build_run_matrix,
        manifest_path,
        plan_path,
    )

    paths = _runtime_paths(tmp_path)
    for variant in VARIANTS:
        _write_full_eval_manifest(manifest_path(paths, variant), [3.0, 3.0, 3.0])

    summaries = build_full_batch_plans(
        paths,
        expected_count=3,
        max_audio_seconds=6.0,
        max_samples=2,
        overwrite=True,
    )

    assert len(summaries) == 6
    assert all(summary["status"] == "pass" for summary in summaries)
    assert all(summary["max_samples"] == 2 for summary in summaries)
    assert all(plan_path(paths, variant, 6.0, 2).is_file() for variant in VARIANTS)

    matrix = build_run_matrix(
        paths,
        models=["wav2vec2-large", "whisper-small"],
        expected_count=3,
        max_audio_seconds=6.0,
        max_samples=2,
        device="cpu",
        cuda_visible_devices=None,
        python_bin="/python",
    )

    assert len(matrix["runs"]) == 12
    assert matrix["max_audio_seconds"] == 6.0
    assert matrix["max_samples"] == 2
    assert matrix["runs"][0]["command"][-1] == "--resume"


def test_full_eval_prediction_validation_checks_rows_coverage_and_empty_transcripts(tmp_path):
    from scripts.setup.run_fairspeech_full_eval import validate_prediction_csv

    manifest = tmp_path / "manifest.csv"
    _write_full_eval_manifest(manifest, [1.0, 1.0])
    prediction = tmp_path / "predictions_wav2vec2_large_baseline.csv"
    with prediction.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "hypothesis", "wer", "model", "audio_variant"],
        )
        writer.writeheader()
        writer.writerow({
            "utterance_id": "utt0",
            "hypothesis": "hello zero",
            "wer": "0.0",
            "model": "wav2vec2-large",
            "audio_variant": "baseline",
        })
        writer.writerow({
            "utterance_id": "utt1",
            "hypothesis": "hello one",
            "wer": "0.5",
            "model": "wav2vec2-large",
            "audio_variant": "baseline",
        })

    assert validate_prediction_csv(prediction, manifest, expected_count=2)["status"] == "pass"

    blank = tmp_path / "predictions_blank.csv"
    with blank.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "hypothesis", "wer", "model", "audio_variant"],
        )
        writer.writeheader()
        writer.writerow({
            "utterance_id": "utt0",
            "hypothesis": "",
            "wer": "1.0",
            "model": "wav2vec2-large",
            "audio_variant": "baseline",
        })
        writer.writerow({
            "utterance_id": "utt0",
            "hypothesis": "",
            "wer": "1.0",
            "model": "wav2vec2-large",
            "audio_variant": "baseline",
        })

    failed = validate_prediction_csv(blank, manifest, expected_count=2)
    assert failed["status"] == "fail"
    assert any("coverage" in error for error in failed["errors"])
    assert any("duplicate" in error for error in failed["errors"])
    assert any("globally empty" in error for error in failed["errors"])


def test_compression_metrics_can_load_recursive_prediction_dirs(tmp_path):
    from scripts.metrics.compute_fairspeech_compression_metrics import load_all_predictions

    nested = tmp_path / "full" / "wav2vec2-large"
    nested.mkdir(parents=True)
    with (nested / "predictions_wav2vec2_large_baseline.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "reference", "hypothesis", "model", "audio_variant", "wer"],
        )
        writer.writeheader()
        writer.writerow({
            "utterance_id": "utt0",
            "reference": "hello",
            "hypothesis": "hello",
            "model": "wav2vec2-large",
            "audio_variant": "baseline",
            "wer": "0.0",
        })

    assert load_all_predictions(tmp_path / "full") == []
    rows = load_all_predictions(tmp_path / "full", recursive=True)
    assert len(rows) == 1
    assert rows[0]["audio_variant"] == "baseline"
