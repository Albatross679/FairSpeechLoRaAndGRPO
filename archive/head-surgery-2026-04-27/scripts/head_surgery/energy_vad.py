"""Stage F — energy-based VAD preprocessing (T8).

Frame-level RMS threshold. A frame is 'silence' if 20*log10(RMS) < db_floor
for at least min_silence_ms consecutive frames. Silence regions are removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


FRAME_MS = 20
HOP_MS = 10


def _frame(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int, int]:
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    n = 1 + max(0, (len(audio) - frame_len) // hop_len)
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len][:n]
    return frames, frame_len, hop_len


def filter_silence(audio: np.ndarray, sr: int,
                   db_floor: float = -35.0,
                   min_silence_ms: int = 200) -> np.ndarray:
    """Return audio with below-threshold runs of length >= min_silence_ms removed."""
    if len(audio) == 0:
        return audio
    audio = audio.astype(np.float32)
    frames, frame_len, hop_len = _frame(audio, sr)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    silent = db < db_floor
    min_frames = max(1, min_silence_ms // HOP_MS)
    to_drop = np.zeros(len(audio), dtype=bool)
    i = 0
    while i < len(silent):
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < len(silent) and silent[j]:
            j += 1
        if (j - i) >= min_frames:
            start_sample = i * hop_len
            end_sample = min(len(audio), j * hop_len + frame_len)
            to_drop[start_sample:end_sample] = True
        i = j
    return audio[~to_drop]


def run_vad_arm(silence_manifests_csv: str, batch_size: int, device: str = "cuda") -> dict:
    """For each severity × db_floor cell, filter audio and rerun Whisper baseline."""
    from scripts.head_surgery.insertion_classifier import insertion_rate_breakdown
    from scripts.head_surgery.run_diagnosis_sweep import (
        _infer_whisper_batch, load_audio_16k, load_whisper, OUT_DIR,
    )
    from scripts.inference.run_inference import normalize_text

    df = pd.read_csv(silence_manifests_csv)
    severity_col = next(c for c in df.columns if c in ("severity", "perturbation"))
    audio_col = next(c for c in df.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in df.columns if c in ("reference", "transcript", "sentence"))

    model, processor = load_whisper(device=device)
    results = []
    for db_floor in (-40.0, -35.0, -30.0):
        for sev_val, g in df.groupby(severity_col):
            audios_filt = []
            for p in g[audio_col]:
                a = load_audio_16k(p)
                audios_filt.append(filter_silence(a, 16000, db_floor=db_floor))
            refs = [normalize_text(r) for r in g[ref_col]]
            hyps = []
            for j in range(0, len(audios_filt), batch_size):
                hyps.extend(t.strip() for t in _infer_whisper_batch(
                    model, processor, audios_filt[j:j + batch_size], device
                ))
            pairs = [(r, normalize_text(h)) for r, h in zip(refs, hyps)]
            br = insertion_rate_breakdown(pairs)
            results.append({"severity": sev_val, "db_floor": db_floor, **br})
            print(f"[vad] sev={sev_val} db_floor={db_floor}: ins={br['total']*100:.2f}%")
    out = OUT_DIR / "vad_scores.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    return {"out": str(out), "cells": len(results)}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--silence-manifest", required=True,
                   help="CSV listing silence-injection perturbation audio (severity + audio_path cols)")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_vad_arm(args.silence_manifest, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    _cli()
