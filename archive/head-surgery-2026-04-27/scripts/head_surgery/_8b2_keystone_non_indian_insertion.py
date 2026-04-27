"""One-shot follow-up: compute non-Indian insertion rate for the 8 §8.2 keystone heads.

Stage D's regression guard recorded only the aggregate non-Indian WER per head.
This script reruns inference on the same non-Indian manifest with each keystone
head masked, captures per-utterance hypotheses, and computes insertion rate
(via the project's existing classifier + EnglishTextNormalizer).

Output: stdout summary + outputs/head_surgery/keystone_non_indian.json.
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

OUT = Path("outputs/head_surgery/keystone_non_indian.json")
KEYSTONES = [(0, 5), (0, 13), (0, 18), (0, 1), (29, 18), (27, 15), (11, 9), (13, 19)]


def main():
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
        if layer == -1:  # baseline (no mask)
            hyps = []
            for j in range(0, len(audios), BS):
                hyps.extend(_infer_whisper_batch(model, processor, audios[j:j + BS], "cuda"))
            return [h.strip() for h in hyps]
        hook = BatchedHeadMaskHook(model, layer_idx=layer).install()
        try:
            hyps = []
            for j in range(0, len(audios), BS):
                chunk = audios[j:j + BS]
                mask = torch.ones(len(chunk), num_heads)
                mask[:, head] = 0.0
                hook.set_batch_mask(mask)
                hyps.extend(_infer_whisper_batch(model, processor, chunk, "cuda"))
            return [h.strip() for h in hyps]
        finally:
            hook.remove()

    results = {}
    t0 = time.time()
    print(f"\n{'condition':<14} {'ins_rate':>10} {'WER':>10}  ({len(df)} utts)")

    # Baseline
    hyps = _run_with_mask(-1, -1)
    pairs = [(r, normalize_text(h)) for r, h in zip(refs_norm, hyps)]
    br = insertion_rate_breakdown(pairs)
    wer = jiwer.wer(refs_norm, [normalize_text(h) for h in hyps])
    results["baseline"] = {"insertion_rate": br["total"], "wer": wer}
    print(f"{'baseline':<14} {br['total']*100:>9.2f}% {wer*100:>9.2f}%  [{(time.time()-t0)/60:.1f}min]")

    # Keystones
    for L, h in KEYSTONES:
        hyps = _run_with_mask(L, h)
        pairs = [(r, normalize_text(hh)) for r, hh in zip(refs_norm, hyps)]
        br = insertion_rate_breakdown(pairs)
        wer = jiwer.wer(refs_norm, [normalize_text(hh) for hh in hyps])
        key = f"L{L}_h{h}"
        results[key] = {
            "layer": L, "head": h,
            "insertion_rate": br["total"],
            "insertion_rep": br["repetition"],
            "insertion_syn": br["syntactic"],
            "insertion_con": br["content"],
            "wer": wer,
        }
        elapsed = (time.time() - t0) / 60
        print(f"({L:>2},{h:>2})       {br['total']*100:>9.2f}% {wer*100:>9.2f}%  [{elapsed:.1f}min]")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT} (total {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
