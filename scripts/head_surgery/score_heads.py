"""Stage D — per-head scoring, paired bootstrap, regression guard.

Inputs:
  outputs/head_surgery/sweep.csv          (from Stage C)
  outputs/head_surgery/baseline_predictions.csv (from Stage A)

Outputs:
  outputs/head_surgery/head_scores.csv    (all (L, h) rows with all metrics)
  outputs/head_surgery/top_k_heads.csv    (ranked top-K after regression guard)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.insertion_classifier import (
    categorize_insertions,
    insertion_rate_breakdown,
)

OUT_DIR = Path("outputs/head_surgery")


def _ins_count(ref: str, hyp: str) -> Tuple[int, int, int, int, int]:
    """Return (total, repetition, syntactic, content, ref_words) for one utterance."""
    cats = categorize_insertions(ref or "", hyp or "")
    total = len(cats)
    rep = sum(1 for c in cats if c["category"] == "repetition")
    syn = sum(1 for c in cats if c["category"] == "syntactic_completion")
    con = sum(1 for c in cats if c["category"] == "content_hallucination")
    nwords = len((ref or "").split())
    return total, rep, syn, con, nwords


def paired_bootstrap_delta_p(
    base_counts: np.ndarray,
    masked_counts: np.ndarray,
    ref_words: np.ndarray,
    n_iter: int = rc.BOOTSTRAP_ITERATIONS,
    seed: int = rc.SEED,
) -> float:
    """p-value for H0: masked insertion rate >= baseline (one-sided).

    Delta = rate_baseline - rate_masked (positive = improvement). For each bootstrap
    resample of utterance indices, compute the Delta. p = fraction of bootstrap Delta <= 0.
    """
    rng = np.random.default_rng(seed)
    n = len(base_counts)
    deltas = np.empty(n_iter, dtype=np.float64)
    ref_words = np.asarray(ref_words, dtype=np.float64)
    base_counts = np.asarray(base_counts, dtype=np.float64)
    masked_counts = np.asarray(masked_counts, dtype=np.float64)
    for k in range(n_iter):
        idx = rng.integers(0, n, size=n)
        tw = ref_words[idx].sum()
        if tw == 0:
            deltas[k] = 0.0
            continue
        rb = base_counts[idx].sum() / tw
        rm = masked_counts[idx].sum() / tw
        deltas[k] = rb - rm
    return float((deltas <= 0).mean())


def compute_head_scores(sweep_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    from scripts.inference.run_inference import normalize_text

    sweep = pd.read_csv(sweep_csv)
    baseline = pd.read_csv(baseline_csv)
    baseline["ref_n"] = baseline["reference"].fillna("").apply(normalize_text)
    baseline["hyp_n"] = baseline["hypothesis"].fillna("").apply(normalize_text)
    base_tot = np.zeros(len(baseline)); base_rep = np.zeros(len(baseline))
    base_syn = np.zeros(len(baseline)); base_con = np.zeros(len(baseline))
    ref_words = np.zeros(len(baseline))
    for i, row in baseline.iterrows():
        t, rep, syn, con, nw = _ins_count(row["ref_n"], row["hyp_n"])
        base_tot[i], base_rep[i], base_syn[i], base_con[i], ref_words[i] = t, rep, syn, con, nw
    total_refw = ref_words.sum() or 1
    base_rate = base_tot.sum() / total_refw
    base_rep_rate = base_rep.sum() / total_refw
    base_syn_rate = base_syn.sum() / total_refw
    base_con_rate = base_con.sum() / total_refw
    idx_of = {str(r["id"]): i for i, r in baseline.iterrows()}

    rows = []
    for (L, h), g in sweep.groupby(["layer", "head"]):
        if int(L) == -1:  # pilot-baseline rows
            continue
        g_tot = np.zeros(len(baseline)); g_rep = np.zeros(len(baseline))
        g_syn = np.zeros(len(baseline)); g_con = np.zeros(len(baseline))
        for _, r in g.iterrows():
            ref_n = normalize_text(r.get("reference", ""))
            hyp_n = normalize_text(r.get("hypothesis", ""))
            t, rep, syn, con, nw = _ins_count(ref_n, hyp_n)
            i = idx_of.get(str(r["id"]))
            if i is None:
                continue
            g_tot[i], g_rep[i], g_syn[i], g_con[i] = t, rep, syn, con
        rate_tot = g_tot.sum() / total_refw
        rate_rep = g_rep.sum() / total_refw
        rate_syn = g_syn.sum() / total_refw
        rate_con = g_con.sum() / total_refw
        p_val = paired_bootstrap_delta_p(base_tot, g_tot, ref_words)
        rows.append({
            "layer": int(L), "head": int(h),
            "insertion_rate_masked": rate_tot,
            "delta_insertion_rate": base_rate - rate_tot,
            "delta_repetition":     base_rep_rate - rate_rep,
            "delta_syntactic":      base_syn_rate - rate_syn,
            "delta_content":        base_con_rate - rate_con,
            "p_value_delta":        p_val,
            "regression_checked":   False,
            "non_indian_wer_masked": None,
            "regression_ok":        None,
        })
    return pd.DataFrame(rows)


# ── Regression guard ─────────────────────────────────────────────────────

def compute_regression_guard(
    head_scores: pd.DataFrame,
    non_indian_manifest_csv: str,
    batch_size: int,
    device: str = "cuda",
    top_k: int = rc.REGRESSION_GUARD_TOP_K,
) -> pd.DataFrame:
    """Run Whisper-large-v3 with head (L,h) masked on non-Indian accents.

    Only the top-`top_k` rows by |delta_insertion_rate| are checked to bound compute.
    Computes composite (concatenated) WER across non-Indian accent groups.
    """
    import jiwer
    import soundfile as sf
    from scripts.head_surgery.head_mask_hook import BatchedHeadMaskHook
    from scripts.head_surgery.run_diagnosis_sweep import (
        _infer_whisper_batch, load_whisper,
    )
    from scripts.inference.run_inference import normalize_text
    import torch

    non_ind = pd.read_csv(non_indian_manifest_csv)
    audio_col = next(c for c in non_ind.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in non_ind.columns if c in ("reference", "transcript", "sentence"))
    audios = [sf.read(str(p))[0] for p in non_ind[audio_col]]
    refs = [normalize_text(r) for r in non_ind[ref_col]]

    model, processor = load_whisper(device=device)
    num_heads = rc.NUM_DECODER_SELF_ATTN_HEADS

    hyps_base = []
    for i in range(0, len(audios), batch_size):
        hyps_base.extend(_infer_whisper_batch(model, processor, audios[i:i + batch_size], device))
    baseline_wer = jiwer.wer(refs, [normalize_text(h) for h in hyps_base])
    print(f"[guard] non-Indian composite baseline WER = {baseline_wer*100:.2f}%")

    candidates = head_scores.reindex(
        head_scores["delta_insertion_rate"].abs().sort_values(ascending=False).index
    ).head(top_k)

    updated = head_scores.copy()
    for i, row in candidates.iterrows():
        L, h = int(row["layer"]), int(row["head"])
        hook = BatchedHeadMaskHook(model, layer_idx=L).install()
        try:
            hyps: List[str] = []
            for j in range(0, len(audios), batch_size):
                chunk = audios[j:j + batch_size]
                mask = torch.ones(len(chunk), num_heads)
                mask[:, h] = 0.0
                hook.set_batch_mask(mask)
                hyps.extend(_infer_whisper_batch(model, processor, chunk, device))
        finally:
            hook.remove()
        wer = jiwer.wer(refs, [normalize_text(x) for x in hyps])
        reg_ok = (wer - baseline_wer) <= rc.REGRESSION_BUDGET_PP
        updated.at[i, "non_indian_wer_masked"] = wer
        updated.at[i, "regression_ok"] = bool(reg_ok)
        updated.at[i, "regression_checked"] = True
        print(f"[guard] L={L} h={h}: non-Indian WER={wer*100:.2f}% (Delta={(wer-baseline_wer)*100:+.2f}pp) "
              f"ok={reg_ok}")

    expected_path = Path("outputs/head_surgery/t7_non_indian_baseline_wer.json")
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())["composite_wer"]
        diff = abs(baseline_wer - expected)
        if diff > rc.GATE_G5_BASELINE_WER_TOLERANCE_PP:
            raise SystemExit(
                f"[G5 FAIL] non-Indian baseline WER={baseline_wer*100:.2f}% "
                f"vs T7 expected {expected*100:.2f}% (|Delta|={diff*100:.2f}pp)."
            )
        print(f"[G5 PASS] non-Indian baseline matches T7 within {rc.GATE_G5_BASELINE_WER_TOLERANCE_PP*100:.1f}pp.")
    else:
        print("[G5 SKIP] outputs/head_surgery/t7_non_indian_baseline_wer.json missing; "
              "cannot cross-check against T7.")
    return updated


def write_top_k(head_scores: pd.DataFrame, k: int = rc.TOP_K_FOR_REPORT) -> pd.DataFrame:
    qualifying = head_scores[
        (head_scores["regression_ok"] == True) |
        (head_scores["regression_checked"] == False)
    ]
    top = qualifying.sort_values("delta_insertion_rate", ascending=False).head(k).reset_index(drop=True)
    top.to_csv(OUT_DIR / "top_k_heads.csv", index=False)
    return top


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--non-indian-manifest", required=True,
                   help="CSV of non-Indian CV accent utterances (composite regression guard)")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k-guard", type=int, default=rc.REGRESSION_GUARD_TOP_K)
    args = p.parse_args()

    scores = compute_head_scores(
        OUT_DIR / "sweep.csv", OUT_DIR / "baseline_predictions.csv",
    )
    scores = compute_regression_guard(
        scores, args.non_indian_manifest, batch_size=args.batch_size,
        device=args.device, top_k=args.top_k_guard,
    )
    scores.to_csv(OUT_DIR / "head_scores.csv", index=False)
    top = write_top_k(scores)
    print(f"[score] wrote {OUT_DIR/'head_scores.csv'} ({len(scores)} rows) and top-{len(top)} heads.")


if __name__ == "__main__":
    _cli()
