"""Stage A.5 — empirical batch-size tuner for hooked Whisper inference.

Picks the largest utterances_per_batch where peak VRAM < 90% of device memory
AND tokens/sec >= 95% of the best observed.
Output: outputs/head_surgery/tune_batch_size.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import soundfile as sf
import torch

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.head_mask_hook import BatchedHeadMaskHook
from scripts.head_surgery.run_diagnosis_sweep import (
    OUT_DIR,
    _infer_whisper_batch,
    load_audio_16k,
    load_manifest_for_ids,
    load_whisper,
)

CANDIDATE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def _time_one_setting(model, processor, audios, mask_layer, batch_size, n_warmup, n_measure, device):
    hook = BatchedHeadMaskHook(model, layer_idx=mask_layer).install()
    try:
        torch.cuda.reset_peak_memory_stats(device) if "cuda" in device else None
        num_heads, _ = (model.config.decoder_attention_heads, None)

        def _run_batch(chunk):
            mask = torch.ones(len(chunk), num_heads)
            # Arbitrarily zero head `i % num_heads` for sample i (covers many head indices).
            for i in range(len(chunk)):
                mask[i, i % num_heads] = 0.0
            hook.set_batch_mask(mask)
            _infer_whisper_batch(model, processor, chunk, device)

        # Warmup
        for _ in range(n_warmup):
            chunk = audios[:batch_size]
            if len(chunk) < batch_size:
                return None
            _run_batch(chunk)

        # Measure
        t0 = time.time()
        total = 0
        for _ in range(n_measure):
            chunk = audios[:batch_size]
            _run_batch(chunk)
            total += len(chunk)
        dt = time.time() - t0
        utts_per_sec = total / dt if dt > 0 else 0.0
        peak = (
            torch.cuda.max_memory_allocated(device) if "cuda" in device else 0
        )
        return {"batch_size": batch_size, "utts_per_sec": utts_per_sec, "peak_bytes": peak, "ok": True}
    finally:
        hook.remove()
        if "cuda" in device:
            torch.cuda.empty_cache()


def tune(manifest_csv: str, mask_layer: int = 15, device: str = "cuda",
         n_warmup: int = 2, n_measure: int = 5) -> dict:
    ids = rc.load_indian_accent_ids()
    subset, _ = load_manifest_for_ids(ids, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    audios = [load_audio_16k(p) for p in subset[audio_col].head(max(CANDIDATE_BATCH_SIZES))]

    model, processor = load_whisper(device=device)
    device_total = (
        torch.cuda.get_device_properties(device).total_memory if "cuda" in device else 10 ** 12
    )

    results: List[dict] = []
    for bs in CANDIDATE_BATCH_SIZES:
        try:
            r = _time_one_setting(model, processor, audios, mask_layer, bs, n_warmup, n_measure, device)
            if r is None:
                continue
            results.append(r)
            print(f"[tune] bs={bs}: {r['utts_per_sec']:.2f} utts/s, peak={r['peak_bytes']/1e9:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": bs, "utts_per_sec": 0.0, "peak_bytes": -1, "ok": False, "oom": True})
            print(f"[tune] bs={bs}: OOM")
            break

    ok = [r for r in results if r["ok"] and r["peak_bytes"] < 0.9 * device_total]
    if not ok:
        raise SystemExit("[G1.5 FAIL] no batch size fits in 90% of device memory")
    best_throughput = max(r["utts_per_sec"] for r in ok)
    qualifying = [r for r in ok if r["utts_per_sec"] >= 0.95 * best_throughput]
    chosen = max(qualifying, key=lambda r: r["batch_size"])

    payload = {
        "chosen_batch_size": chosen["batch_size"],
        "chosen_throughput_utts_per_sec": chosen["utts_per_sec"],
        "chosen_peak_bytes": chosen["peak_bytes"],
        "device_total_bytes": device_total,
        "mask_layer": mask_layer,
        "sweep": results,
    }
    (OUT_DIR / "tune_batch_size.json").write_text(json.dumps(payload, indent=2))
    print(f"[G1.5 PASS] chosen batch_size={chosen['batch_size']} "
          f"({chosen['utts_per_sec']:.2f} utts/s, {chosen['peak_bytes']/1e9:.2f} GB)")
    return payload


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--mask-layer", type=int, default=15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    tune(args.manifest, mask_layer=args.mask_layer, device=args.device)


if __name__ == "__main__":
    _cli()
