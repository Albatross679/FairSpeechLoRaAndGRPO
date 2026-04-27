"""Stage E — decoding-strategy ablation (T5).

2 × 3 × 3 × 2 = 36 configs: beam ∈ {1,5} × rep_penalty ∈ {1.0,1.1,1.3} ×
no_repeat_ngram ∈ {0,3,5} × temperature_fallback ∈ {off, on}.
Each runs the full Indian-accent utterance subset. Output:
  outputs/head_surgery/decoding_grid.csv    (one row per utterance × config)
  outputs/head_surgery/decoding_scores.csv  (per-config aggregates)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List

import pandas as pd
import soundfile as sf
import torch

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.insertion_classifier import insertion_rate_breakdown
from scripts.head_surgery.run_diagnosis_sweep import (
    OUT_DIR, load_audio_16k, load_manifest_for_ids, load_whisper,
)
from scripts.inference.run_inference import normalize_text

BEAMS = [1, 5]
REP_PENALTIES = [1.0, 1.1, 1.3]
NO_REPEAT_NGRAMS = [0, 3, 5]
TEMP_FALLBACKS = [False, True]
TEMP_FALLBACK_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _generate_with_config(model, processor, audios, beam, rep_pen, no_rep_ng, temp_fb, device):
    inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    features = inputs.input_features.to(
        device, dtype=torch.float16 if "cuda" in device else torch.float32
    )
    kwargs = dict(rc.GENERATE_CONFIG)
    kwargs["num_beams"] = beam
    kwargs["repetition_penalty"] = rep_pen
    kwargs["no_repeat_ngram_size"] = no_rep_ng
    if temp_fb:
        kwargs["temperature"] = TEMP_FALLBACK_VALUES
        kwargs["do_sample"] = True
    with torch.no_grad():
        ids = model.generate(features, **kwargs)
    return processor.batch_decode(ids, skip_special_tokens=True)


def run_decoding_grid(manifest_csv: str, batch_size: int, device: str = "cuda") -> dict:
    ids = rc.load_indian_accent_ids()
    subset, id_col = load_manifest_for_ids(ids, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in subset.columns if c in ("reference", "transcript", "sentence"))
    audios = [load_audio_16k(p) for p in subset[audio_col]]
    refs = subset[ref_col].astype(str).tolist()
    utt_ids = subset[id_col].astype(str).tolist()

    model, processor = load_whisper(device=device)
    rows, scores = [], []
    configs = list(itertools.product(BEAMS, REP_PENALTIES, NO_REPEAT_NGRAMS, TEMP_FALLBACKS))
    # beam=5 multiplies KV-cache memory by ~5x and OOM'd at bs=24 on a 48 GB A6000.
    # Halve the batch for beam=5 configs; keep the user-requested batch for beam=1.
    def _bs_for(beam):
        return max(1, batch_size // 4) if beam >= 5 else batch_size

    for k, (beam, rp, nr, tf) in enumerate(configs):
        bs_k = _bs_for(beam)
        hyps: List[str] = []
        for j in range(0, len(audios), bs_k):
            chunk = audios[j:j + bs_k]
            hyps.extend(t.strip() for t in _generate_with_config(
                model, processor, chunk, beam, rp, nr, tf, device
            ))
        pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps)]
        br = insertion_rate_breakdown(pairs)
        for uid, ref, hyp in zip(utt_ids, refs, hyps):
            rows.append({"beam": beam, "rep_penalty": rp, "no_repeat_ngram": nr,
                         "temp_fallback": tf, "id": uid, "reference": ref, "hypothesis": hyp})
        scores.append({"beam": beam, "rep_penalty": rp, "no_repeat_ngram": nr,
                       "temp_fallback": tf, **br})
        print(f"[decoding {k+1}/{len(configs)}] beam={beam} rep={rp} "
              f"nr={nr} tf={tf} bs={bs_k}: ins={br['total']*100:.2f}%", flush=True)
        # Checkpoint partial results after each config — previous run OOM'd at
        # config 19 and lost all 18 completed results.
        pd.DataFrame(scores).to_csv(OUT_DIR / "decoding_scores.csv", index=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT_DIR / "decoding_grid.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT_DIR / "decoding_scores.csv", index=False)
    return {"configs": len(configs)}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_decoding_grid(args.manifest, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    _cli()
