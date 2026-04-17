"""Diagnosis-sweep drivers for Stages A, B, C of v2.0 head surgery.

Invocations:
    python -m scripts.head_surgery.run_diagnosis_sweep baseline
    python -m scripts.head_surgery.run_diagnosis_sweep pilot --pilot-layer 15 --n-utts 50
    python -m scripts.head_surgery.run_diagnosis_sweep full --batch-size <from-tune>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.head_mask_hook import (
    BatchedHeadMaskHook,
    SerialHeadMaskHook,
)
from scripts.head_surgery.insertion_classifier import insertion_rate_breakdown

OUT_DIR = Path("outputs/head_surgery")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Whisper loading & inference (reuses midterm config) ──────────────────

def load_whisper(device: str = "cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(rc.MODEL_ID, revision=rc.MODEL_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        rc.MODEL_ID,
        revision=rc.MODEL_REVISION,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device).eval()
    return model, processor


def load_manifest_for_ids(ids: List[str], manifest_csv: str, id_col: str = None):
    """Return a DataFrame of manifest rows matching `ids` in order."""
    df = pd.read_csv(manifest_csv)
    if id_col is None:
        id_col = next(
            c for c in df.columns
            if c in ("utt_id", "utterance_id", "client_id", "path")
        )
    df[id_col] = df[id_col].astype(str)
    subset = df[df[id_col].isin(set(ids))].copy()
    # Preserve the input `ids` order
    subset["__order__"] = subset[id_col].map({i: k for k, i in enumerate(ids)})
    subset = subset.sort_values("__order__").drop(columns=["__order__"]).reset_index(drop=True)
    if len(subset) != len(ids):
        missing = set(ids) - set(subset[id_col])
        raise RuntimeError(f"{len(missing)} IDs missing from manifest; first={list(missing)[:3]}")
    return subset, id_col


def _infer_whisper_batch(model, processor, audio_arrays, device: str):
    sr = 16000
    inputs = processor(audio_arrays, sampling_rate=sr, return_tensors="pt", padding=True)
    features = inputs.input_features.to(
        device, dtype=torch.float16 if "cuda" in device else torch.float32
    )
    with torch.no_grad():
        pred_ids = model.generate(features, **rc.GENERATE_CONFIG)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)


# ── Stage A: baseline rerun ──────────────────────────────────────────────

def run_baseline(manifest_csv: str, batch_size: int = 8, device: str = "cuda") -> dict:
    """Run Whisper-large-v3 baseline on the frozen Indian-accent IDs; return metrics dict."""
    import soundfile as sf

    ids = rc.load_indian_accent_ids()
    subset, id_col = load_manifest_for_ids(ids, manifest_csv)
    model, processor = load_whisper(device=device)

    audio_col = next(
        c for c in subset.columns if c in ("audio_path", "audio", "path")
    )
    ref_col = next(
        c for c in subset.columns if c in ("reference", "transcript", "sentence")
    )

    predictions: List[Tuple[str, str, str]] = []  # (id, ref, hyp)
    t0 = time.time()
    for i in range(0, len(subset), batch_size):
        batch = subset.iloc[i:i + batch_size]
        audios = [sf.read(str(p))[0] for p in batch[audio_col]]
        hyps = _infer_whisper_batch(model, processor, audios, device)
        for (_, row), hyp in zip(batch.iterrows(), hyps):
            predictions.append((str(row[id_col]), str(row[ref_col]), hyp.strip()))

    from scripts.inference.run_inference import normalize_text
    normed_pairs = [(normalize_text(r), normalize_text(h)) for _, r, h in predictions]
    breakdown = insertion_rate_breakdown(normed_pairs)

    out_csv = OUT_DIR / "baseline_predictions.csv"
    pd.DataFrame(predictions, columns=["id", "reference", "hypothesis"]).to_csv(out_csv, index=False)

    metrics = {
        "n": len(predictions),
        "insertion_rate_total": breakdown["total"],
        "insertion_rate_repetition": breakdown["repetition"],
        "insertion_rate_syntactic": breakdown["syntactic"],
        "insertion_rate_content": breakdown["content"],
        "seconds": round(time.time() - t0, 1),
        "model_revision": rc.MODEL_REVISION,
        "generate_config": rc.GENERATE_CONFIG,
    }
    (OUT_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def gate_G1(metrics: dict) -> None:
    observed = metrics["insertion_rate_total"]
    target = rc.GATE_G1_MIDTERM_INSERTION_RATE
    tol = rc.GATE_G1_TOLERANCE_PP
    diff = abs(observed - target)
    print(f"[G1] insertion rate = {observed*100:.2f}% (target {target*100:.2f}% ± {tol*100:.1f}pp, diff {diff*100:.2f}pp)")
    if diff > tol:
        print(f"[G1 NOTE] {observed*100:.2f}% differs from midterm {target*100:.2f}% by >{tol*100:.1f}pp. "
              f"On CV25 this is expected; G1 is redefined as 'establish CV25 baseline' (see repro_config docstring).")
    else:
        print("[G1 PASS] baseline reproduces midterm within tolerance.")


# ── CLI dispatch ─────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseline")
    p_base.add_argument("--manifest", required=True)
    p_base.add_argument("--batch-size", type=int, default=8)
    p_base.add_argument("--device", default="cuda")

    # pilot / full parsers added in later tasks
    args = p.parse_args()

    if args.cmd == "baseline":
        m = run_baseline(args.manifest, batch_size=args.batch_size, device=args.device)
        gate_G1(m)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    _cli()
