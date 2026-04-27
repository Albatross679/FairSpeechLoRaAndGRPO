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
