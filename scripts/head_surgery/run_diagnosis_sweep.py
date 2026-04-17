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

SAMPLE_RATE = 16000


def load_audio_16k(path) -> np.ndarray:
    """Load an audio file and resample to 16 kHz mono (matches midterm pipeline).

    CV mp3s are 48 kHz; passing them to the Whisper processor with
    sampling_rate=16000 without resampling warps the spectrogram 3× and
    produces hallucination loops (observed as 58% insertion rate on an early
    Stage A run). Use this helper everywhere audio is loaded.
    """
    import torchaudio
    waveform, orig_sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)(waveform)
    return waveform.squeeze(0).numpy()


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
        audios = [load_audio_16k(p) for p in batch[audio_col]]
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


# ── Stage B: pilot sweep (serial hook) ───────────────────────────────────

PILOT_N_UTTS = 50


def _rng_sample_ids(ids: List[str], n: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[int(i)] for i in sorted(idx.tolist())]


def run_pilot(manifest_csv: str, pilot_layer: int = 15, n_utts: int = PILOT_N_UTTS,
              batch_size: int = 8, device: str = "cuda") -> dict:
    import soundfile as sf
    from scripts.inference.run_inference import normalize_text

    ids_all = rc.load_indian_accent_ids()
    pilot_ids = _rng_sample_ids(ids_all, n_utts, rc.SEED)
    subset, id_col = load_manifest_for_ids(pilot_ids, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in subset.columns if c in ("reference", "transcript", "sentence"))

    model, processor = load_whisper(device=device)
    audios = [load_audio_16k(p) for p in subset[audio_col]]
    refs = subset[ref_col].astype(str).tolist()
    ids = subset[id_col].astype(str).tolist()

    # For each head h in [0, num_heads), install serial hook (pilot_layer, h),
    # run inference on the 50 audios, compute insertion rate.
    per_head: Dict[int, dict] = {}
    for h in range(rc.NUM_DECODER_SELF_ATTN_HEADS):
        hyps: List[str] = []
        with SerialHeadMaskHook(model, pilot_layer, h):
            for i in range(0, len(audios), batch_size):
                batch_audios = audios[i:i + batch_size]
                texts = _infer_whisper_batch(model, processor, batch_audios, device)
                hyps.extend(t.strip() for t in texts)
        pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps)]
        br = insertion_rate_breakdown(pairs)
        per_head[h] = {"hyps": hyps, **br}
        print(f"[pilot] L={pilot_layer} h={h}: insertion_rate={br['total']*100:.2f}%")

    # Also the no-mask baseline on the same 50 utts
    hyps_base: List[str] = []
    for i in range(0, len(audios), batch_size):
        batch_audios = audios[i:i + batch_size]
        hyps_base.extend(t.strip() for t in _infer_whisper_batch(model, processor, batch_audios, device))
    base_pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps_base)]
    base_br = insertion_rate_breakdown(base_pairs)

    # Write pilot_sweep.csv — one row per (head, utterance)
    rows = []
    for h, v in per_head.items():
        for uid, ref, hyp in zip(ids, refs, v["hyps"]):
            rows.append({"layer": pilot_layer, "head": h, "id": uid, "reference": ref, "hypothesis": hyp})
    for uid, ref, hyp in zip(ids, refs, hyps_base):
        rows.append({"layer": -1, "head": -1, "id": uid, "reference": ref, "hypothesis": hyp})
    pd.DataFrame(rows).to_csv(OUT_DIR / "pilot_sweep.csv", index=False)

    metrics = {
        "pilot_layer": pilot_layer,
        "n_utts": n_utts,
        "baseline_insertion_rate": base_br["total"],
        "per_head": {str(h): {"total": v["total"], "repetition": v["repetition"],
                               "syntactic": v["syntactic"], "content": v["content"]}
                      for h, v in per_head.items()},
    }
    (OUT_DIR / "pilot_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def gate_G2(metrics: dict) -> None:
    base = metrics["baseline_insertion_rate"]
    deltas = [v["total"] - base for v in metrics["per_head"].values()]
    n_pos = sum(1 for d in deltas if d > 0)
    n_non_pos = sum(1 for d in deltas if d <= 0)
    print(f"[G2] pilot deltas (masked - baseline): {n_pos} heads ↑, {n_non_pos} heads ↓/=")
    if n_pos == 0 or n_non_pos == 0:
        raise SystemExit(f"[G2 FAIL] all pilot deltas are same sign ({n_pos} ↑, {n_non_pos} ↓/=). "
                         f"Either the hook is a no-op or the pipeline has a bug. Investigate before Stage C.")
    if all(abs(d) < 1e-6 for d in deltas):
        raise SystemExit("[G2 FAIL] all pilot deltas are ~0 — hook is likely a no-op.")
    print("[G2 PASS] pilot shows head-level signal.")


# ── Stage C: full batched sweep ──────────────────────────────────────────

def _batched_pilot_replay(model, processor, audios, refs, pilot_layer, batch_size, device) -> dict:
    """Replay Stage B's 20 head-masked passes using BatchedHeadMaskHook (Gate G3)."""
    from scripts.inference.run_inference import normalize_text
    per_head: Dict[int, dict] = {}
    num_heads = rc.NUM_DECODER_SELF_ATTN_HEADS
    hook = BatchedHeadMaskHook(model, layer_idx=pilot_layer).install()
    try:
        for h in range(num_heads):
            hyps: List[str] = []
            for i in range(0, len(audios), batch_size):
                chunk = audios[i:i + batch_size]
                mask = torch.ones(len(chunk), num_heads)
                mask[:, h] = 0.0
                hook.set_batch_mask(mask)
                hyps.extend(t.strip() for t in _infer_whisper_batch(model, processor, chunk, device))
            pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps)]
            per_head[h] = {"hyps": hyps, **insertion_rate_breakdown(pairs)}
    finally:
        hook.remove()
    return per_head


def gate_G3(serial_metrics: dict, batched_per_head: dict) -> None:
    """Assert batched per-head insertion rates match serial within tolerance."""
    tol = rc.GATE_G3_WER_TOLERANCE
    deltas = []
    for h, v in batched_per_head.items():
        serial_rate = serial_metrics["per_head"][str(h)]["total"]
        batched_rate = v["total"]
        d = abs(serial_rate - batched_rate)
        deltas.append(d)
        if d > tol:
            raise SystemExit(
                f"[G3 FAIL] head {h}: serial={serial_rate*100:.3f}% batched={batched_rate*100:.3f}% "
                f"(|Δ|={d*100:.3f}% > {tol*100:.3f}%). Batched hook is not bytes-equivalent to serial."
            )
    print(f"[G3 PASS] batched matches serial on pilot (max |Δ|={max(deltas)*100:.4f}%).")


def run_full(manifest_csv: str, batch_size: int, device: str = "cuda") -> dict:
    import soundfile as sf
    from scripts.inference.run_inference import normalize_text

    pilot_metrics = json.loads((OUT_DIR / "pilot_metrics.json").read_text())
    pilot_layer = pilot_metrics["pilot_layer"]

    ids_all = rc.load_indian_accent_ids()
    subset, id_col = load_manifest_for_ids(ids_all, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in subset.columns if c in ("reference", "transcript", "sentence"))
    audios = [load_audio_16k(p) for p in subset[audio_col]]
    refs = subset[ref_col].astype(str).tolist()
    ids = subset[id_col].astype(str).tolist()

    model, processor = load_whisper(device=device)

    # Gate G3 — batched replay of pilot slice
    pilot_ids = _rng_sample_ids(ids_all, pilot_metrics["n_utts"], rc.SEED)
    pilot_indices = [ids.index(p) for p in pilot_ids]
    pilot_audios = [audios[i] for i in pilot_indices]
    pilot_refs = [refs[i] for i in pilot_indices]
    batched_replay = _batched_pilot_replay(model, processor, pilot_audios, pilot_refs,
                                           pilot_layer, batch_size, device)
    gate_G3(pilot_metrics, batched_replay)

    # Full 640-cell sweep
    rows = []
    num_heads = rc.NUM_DECODER_SELF_ATTN_HEADS
    t0 = time.time()
    for L in range(rc.NUM_DECODER_LAYERS):
        hook = BatchedHeadMaskHook(model, layer_idx=L).install()
        try:
            for h in range(num_heads):
                hyps: List[str] = []
                for i in range(0, len(audios), batch_size):
                    chunk = audios[i:i + batch_size]
                    mask = torch.ones(len(chunk), num_heads)
                    mask[:, h] = 0.0
                    hook.set_batch_mask(mask)
                    hyps.extend(t.strip() for t in _infer_whisper_batch(model, processor, chunk, device))
                pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps)]
                br = insertion_rate_breakdown(pairs)
                for uid, ref, hyp in zip(ids, refs, hyps):
                    rows.append({"layer": L, "head": h, "id": uid,
                                 "reference": ref, "hypothesis": hyp,
                                 "condition_insertion_rate_total": br["total"]})
                elapsed = time.time() - t0
                done = L * num_heads + h + 1
                total = rc.NUM_DECODER_LAYERS * num_heads
                print(f"[sweep] L={L} h={h} ins={br['total']*100:.2f}% "
                      f"[{done}/{total}, {elapsed/60:.1f}min]")
        finally:
            hook.remove()

    out_csv = OUT_DIR / "sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Gate G4 — completeness
    df = pd.read_csv(out_csv)
    expected = rc.NUM_DECODER_LAYERS * num_heads * len(ids)
    if len(df) != expected:
        raise SystemExit(f"[G4 FAIL] sweep has {len(df)} rows, expected {expected}")
    if df["hypothesis"].isna().any() or (df["hypothesis"].fillna("") == "").any():
        n_empty = df["hypothesis"].fillna("").eq("").sum()
        print(f"[G4 WARN] {n_empty} empty hypotheses in sweep (will be treated as insertion-rate NaN)")
    print(f"[G4 PASS] sweep complete: {len(df)} rows.")
    return {"rows": len(df), "out_csv": str(out_csv), "minutes": round((time.time() - t0) / 60, 1)}


# ── CLI dispatch ─────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseline")
    p_base.add_argument("--manifest", required=True)
    p_base.add_argument("--batch-size", type=int, default=8)
    p_base.add_argument("--device", default="cuda")

    p_pilot = sub.add_parser("pilot")
    p_pilot.add_argument("--manifest", required=True)
    p_pilot.add_argument("--pilot-layer", type=int, default=15)
    p_pilot.add_argument("--n-utts", type=int, default=PILOT_N_UTTS)
    p_pilot.add_argument("--batch-size", type=int, default=8)
    p_pilot.add_argument("--device", default="cuda")

    p_full = sub.add_parser("full")
    p_full.add_argument("--manifest", required=True)
    p_full.add_argument("--batch-size", type=int, required=True,
                        help="Use outputs/head_surgery/tune_batch_size.json:chosen_batch_size")
    p_full.add_argument("--device", default="cuda")

    args = p.parse_args()

    if args.cmd == "baseline":
        m = run_baseline(args.manifest, batch_size=args.batch_size, device=args.device)
        gate_G1(m)
    elif args.cmd == "pilot":
        m = run_pilot(args.manifest, pilot_layer=args.pilot_layer, n_utts=args.n_utts,
                      batch_size=args.batch_size, device=args.device)
        gate_G2(m)
    elif args.cmd == "full":
        run_full(args.manifest, batch_size=args.batch_size, device=args.device)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    _cli()
