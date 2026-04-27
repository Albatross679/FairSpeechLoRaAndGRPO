"""One-shot follow-up: full 4-metric table for the §7.4 top-10 hallucination-driving heads.

For each of the 10 heads in §7.4, capture:
  - Indian insertion rate when masked (from existing sweep.csv)
  - Indian WER when masked (from existing sweep.csv)
  - Non-Indian insertion rate when masked (NEW — GPU rerun on non-Indian set)
  - Non-Indian WER when masked (NEW — also from the same rerun, since
    head_scores.csv only has it for 7 of the 10 heads — the bottom 3 fall
    outside Stage D's top-50 regression-guard scope)

Output: outputs/head_surgery/top10_drivers_full_metrics.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jiwer
import pandas as pd
import torch

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.head_mask_hook import BatchedHeadMaskHook
from scripts.head_surgery.insertion_classifier import insertion_rate_breakdown
from scripts.head_surgery.run_diagnosis_sweep import (
    _infer_whisper_batch, load_audio_16k, load_whisper,
)
from scripts.inference.run_inference import normalize_text

OUT = Path("outputs/head_surgery/top10_drivers_full_metrics.json")
TOP10 = [
    (20, 11), (0, 15), (22, 19), (7, 6), (10, 8),
    (25, 5), (0, 14), (13, 17), (12, 8), (11, 11),
]


def main():
    # Indian metrics from existing sweep.csv (no GPU)
    sweep = pd.read_csv("outputs/head_surgery/sweep.csv")
    indian_metrics = {}
    for L, h in TOP10:
        rows = sweep[(sweep["layer"] == L) & (sweep["head"] == h)]
        refs = [normalize_text(r) for r in rows["reference"]]
        hyps = [normalize_text(str(hh)) for hh in rows["hypothesis"]]
        pairs = list(zip(refs, hyps))
        br = insertion_rate_breakdown(pairs)
        wer = jiwer.wer(refs, hyps)
        indian_metrics[(L, h)] = {"insertion_rate": br["total"], "wer": wer}
    print("Indian metrics computed from sweep.csv (no GPU).")

    # Non-Indian metrics — rerun inference for each head
    manifest = "datasets/cv_test_manifest_non_indian_sample.csv"
    df = pd.read_csv(manifest)
    audio_col = next(c for c in df.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in df.columns if c in ("reference", "transcript", "sentence"))

    print(f"Loading {len(df)} non-Indian audios from {manifest} ...")
    audios = [load_audio_16k(p) for p in df[audio_col]]
    refs = df[ref_col].astype(str).tolist()
    refs_norm = [normalize_text(r) for r in refs]

    model, processor = load_whisper(device="cuda")
    BS = 32
    num_heads = rc.NUM_DECODER_SELF_ATTN_HEADS

    def _run_with_mask(layer: int, head: int) -> list[str]:
        if layer == -1:
            hyps = []
            for j in range(0, len(audios), BS):
                hyps.extend(_infer_whisper_batch(model, processor, audios[j:j + BS], "cuda"))
            return [hh.strip() for hh in hyps]
        hook = BatchedHeadMaskHook(model, layer_idx=layer).install()
        try:
            hyps = []
            for j in range(0, len(audios), BS):
                chunk = audios[j:j + BS]
                mask = torch.ones(len(chunk), num_heads)
                mask[:, head] = 0.0
                hook.set_batch_mask(mask)
                hyps.extend(_infer_whisper_batch(model, processor, chunk, "cuda"))
            return [hh.strip() for hh in hyps]
        finally:
            hook.remove()

    results = {}
    t0 = time.time()
    print(f"\n{'condition':<14} {'I-ins':>8} {'I-WER':>8} {'N-ins':>8} {'N-WER':>8}  ({len(df)} non-Indian utts)")

    # Baseline
    hyps_b = _run_with_mask(-1, -1)
    pairs_b = [(r, normalize_text(hh)) for r, hh in zip(refs_norm, hyps_b)]
    br_b = insertion_rate_breakdown(pairs_b)
    wer_b = jiwer.wer(refs_norm, [normalize_text(hh) for hh in hyps_b])
    results["baseline"] = {
        "indian_insertion_rate": 0.0127, "indian_wer": 0.1093,
        "non_indian_insertion_rate": br_b["total"], "non_indian_wer": wer_b,
    }
    print(f"{'baseline':<14} {1.27:>7.2f}% {10.93:>7.2f}% {br_b['total']*100:>7.2f}% {wer_b*100:>7.2f}%  [{(time.time()-t0)/60:.1f}min]")

    # The 10 driver heads
    for L, h in TOP10:
        hyps = _run_with_mask(L, h)
        pairs = [(r, normalize_text(hh)) for r, hh in zip(refs_norm, hyps)]
        br = insertion_rate_breakdown(pairs)
        wer = jiwer.wer(refs_norm, [normalize_text(hh) for hh in hyps])
        ind = indian_metrics[(L, h)]
        results[f"L{L}_h{h}"] = {
            "layer": L, "head": h,
            "indian_insertion_rate": ind["insertion_rate"],
            "indian_wer": ind["wer"],
            "non_indian_insertion_rate": br["total"],
            "non_indian_wer": wer,
        }
        elapsed = (time.time() - t0) / 60
        print(f"({L:>2},{h:>2})       {ind['insertion_rate']*100:>7.2f}% {ind['wer']*100:>7.2f}% {br['total']*100:>7.2f}% {wer*100:>7.2f}%  [{elapsed:.1f}min]")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT} (total {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
