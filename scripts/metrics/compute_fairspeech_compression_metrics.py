#!/usr/bin/env python3
"""Compute FairSpeech compression/resampling fairness metrics.

Consumes prediction CSVs from run_inference.py where each row has utterance_id,
reference, hypothesis, model, ethnicity, and audio_variant/variant. Produces:
- WER by ethnicity and variant
- MMR and relative gap per model × variant
- insertion rate and insertion subtype rates
- paired ΔWER versus a baseline variant
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

MIN_GROUP_SIZE = 50
FUNCTION_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "is", "was", "are", "were",
    "and", "or", "but", "that", "this", "it", "he", "she", "they", "we", "i", "you", "my",
    "your", "his", "her", "its", "our", "their", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "may", "might", "can", "could", "not", "no",
    "so", "if", "then", "than", "as", "with", "by", "from", "up", "about", "into", "through",
    "during", "before", "after", "again", "there", "here", "when", "where", "why", "how", "all",
    "each", "every", "both", "more", "most", "other", "some", "such", "only", "same", "too",
    "very", "just",
}


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("prediction_file", str(path))
        if not row.get("audio_variant"):
            row["audio_variant"] = row.get("variant") or row.get("perturbation") or "baseline"
    return rows


def load_all_predictions(predictions_dir: Path, recursive: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = "**/predictions_*.csv" if recursive else "predictions_*.csv"
    for path in sorted(predictions_dir.glob(pattern)):
        rows.extend(read_predictions(path))
    return rows


def filter_complete_run_rows(rows: list[dict[str, str]], expected_rows: int | None) -> list[dict[str, str]]:
    """Keep only model x variant slots with the expected row count."""
    if not expected_rows:
        return rows
    complete: list[dict[str, str]] = []
    for _, group_data in group_rows(rows, ("model", "audio_variant")).items():
        if len(group_data) == expected_rows:
            complete.extend(group_data)
    return complete


def jiwer_module():
    try:
        import jiwer  # type: ignore

        return jiwer
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("jiwer is required for metric computation. Install project dependencies first.") from exc


def wer(refs: list[str], hyps: list[str]) -> float:
    jiwer = jiwer_module()
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not valid:
        return math.nan
    r, h = zip(*valid)
    return float(jiwer.wer(list(r), list(h)))


def classify_insertions(ref: str, hyp: str) -> Counter[str]:
    if not ref or not hyp:
        return Counter()
    jiwer = jiwer_module()
    try:
        out = jiwer.process_words(ref, hyp)
    except Exception:
        return Counter()

    counts: Counter[str] = Counter()
    hyp_words = hyp.split()
    for chunk in out.alignments[0]:
        if chunk.type != "insert":
            continue
        for pos in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
            if pos >= len(hyp_words):
                continue
            word = hyp_words[pos].lower()
            preceding = hyp_words[max(0, pos - 20):pos]
            repeated = word in preceding
            for n in (2, 3, 4):
                if repeated or pos < n:
                    continue
                ngram = tuple(hyp_words[pos - n + 1:pos + 1])
                for start in range(max(0, pos - 20), pos - n + 1):
                    if tuple(hyp_words[start:start + n]) == ngram:
                        repeated = True
                        break
            if repeated:
                counts["repetition"] += 1
            elif word in FUNCTION_WORDS:
                counts["syntactic"] += 1
            else:
                counts["content"] += 1
    return counts


def error_counts(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    jiwer = jiwer_module()
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not valid:
        return {"ref_words": 0, "insertions": 0, "insertion_rate": math.nan}
    r, h = zip(*valid)
    out = jiwer.process_words(list(r), list(h))
    ref_words = out.hits + out.substitutions + out.deletions
    subtype_counts: Counter[str] = Counter()
    for ref, hyp in valid:
        subtype_counts.update(classify_insertions(ref, hyp))
    return {
        "ref_words": ref_words,
        "insertions": out.insertions,
        "insertion_rate": out.insertions / ref_words if ref_words else math.nan,
        "repetition_insertions": subtype_counts["repetition"],
        "syntactic_insertions": subtype_counts["syntactic"],
        "content_insertions": subtype_counts["content"],
        "repetition_insertion_rate": subtype_counts["repetition"] / ref_words if ref_words else math.nan,
        "syntactic_insertion_rate": subtype_counts["syntactic"] / ref_words if ref_words else math.nan,
        "content_insertion_rate": subtype_counts["content"] / ref_words if ref_words else math.nan,
    }


def group_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    return grouped


def compute_group_table(
    rows: list[dict[str, str]],
    group_col: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped = group_rows(rows, ("model", "audio_variant", group_col))
    for (model, variant, group), group_data in sorted(grouped.items()):
        if not group or len(group_data) < min_group_size:
            continue
        refs = [r.get("reference", "") for r in group_data]
        hyps = [r.get("hypothesis", "") for r in group_data]
        counts = error_counts(refs, hyps)
        out.append({
            "model": model,
            "audio_variant": variant,
            "group_axis": group_col,
            "group": group,
            "n": len(group_data),
            "wer": wer(refs, hyps),
            **counts,
        })
    return out


def row_word_counts(row: dict[str, str]) -> tuple[int, int] | None:
    """Return reference-word count and edit count for one utterance."""
    try:
        ref_words = int(float(row.get("num_ref_words") or 0))
    except ValueError:
        ref_words = len((row.get("reference") or "").split())
    if ref_words <= 0:
        return None
    try:
        row_wer = float(row.get("wer") or 0.0)
    except ValueError:
        return None
    edits = int(round(row_wer * ref_words))
    return ref_words, edits


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ratio_ci(
    ref_words: list[int],
    edits: list[int],
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not ref_words or not edits:
        return math.nan, math.nan, math.nan
    point = sum(edits) / sum(ref_words)
    n = len(ref_words)
    if n_resamples <= 0:
        return point, math.nan, math.nan

    try:
        import numpy as np  # type: ignore

        refs = np.asarray(ref_words, dtype=np.int64)
        errs = np.asarray(edits, dtype=np.int64)
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            denom = int(refs[idx].sum())
            samples.append(float(errs[idx].sum() / denom) if denom else math.nan)
    except Exception:
        rng = random.Random(seed)
        samples = []
        for _ in range(n_resamples):
            denom = 0
            numer = 0
            for _ in range(n):
                idx = rng.randrange(n)
                denom += ref_words[idx]
                numer += edits[idx]
            samples.append(numer / denom if denom else math.nan)

    finite = [value for value in samples if not math.isnan(value)]
    return point, percentile(finite, 2.5), percentile(finite, 97.5)


def compute_bootstrap_group_ci(
    rows: list[dict[str, str]],
    group_col: str,
    n_resamples: int,
    seed: int,
    min_group_size: int = MIN_GROUP_SIZE,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped = group_rows(rows, ("model", "audio_variant", group_col))
    for group_idx, ((model, variant, group), group_data) in enumerate(sorted(grouped.items())):
        if not group or len(group_data) < min_group_size:
            continue
        ref_words: list[int] = []
        edits: list[int] = []
        for row in group_data:
            counts = row_word_counts(row)
            if counts is None:
                continue
            ref_count, edit_count = counts
            ref_words.append(ref_count)
            edits.append(edit_count)
        if len(ref_words) < min_group_size:
            continue
        point, ci_low, ci_high = bootstrap_ratio_ci(
            ref_words,
            edits,
            n_resamples=n_resamples,
            seed=seed + group_idx,
        )
        out.append({
            "model": model,
            "audio_variant": variant,
            "group_axis": group_col,
            "group": group,
            "n": len(ref_words),
            "ref_words": sum(ref_words),
            "wer": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": ci_high - ci_low if not math.isnan(ci_low) and not math.isnan(ci_high) else math.nan,
            "n_resamples": n_resamples,
            "seed": seed + group_idx,
        })
    return out


def ci_overlap(row_a: dict[str, object], row_b: dict[str, object]) -> bool:
    a_low = float(row_a["ci_low"])
    a_high = float(row_a["ci_high"])
    b_low = float(row_b["ci_low"])
    b_high = float(row_b["ci_high"])
    return max(a_low, b_low) <= min(a_high, b_high)


def compute_ci_overlap_table(bootstrap_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_run: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in bootstrap_rows:
        by_run[(str(row["model"]), str(row["audio_variant"]), str(row["group_axis"]))].append(row)

    out: list[dict[str, object]] = []
    for (model, variant, axis), rows in sorted(by_run.items()):
        by_group = {str(row["group"]): row for row in rows}
        comparisons: list[tuple[str, dict[str, object], dict[str, object]]] = []
        if "Black/AA" in by_group and "White" in by_group:
            comparisons.append(("blackaa_vs_white", by_group["Black/AA"], by_group["White"]))
        if len(rows) >= 2:
            best = min(rows, key=lambda row: float(row["wer"]))
            worst = max(rows, key=lambda row: float(row["wer"]))
            if best["group"] != worst["group"]:
                comparisons.append(("worst_vs_best", worst, best))

        seen: set[tuple[str, str, str]] = set()
        for comparison, row_a, row_b in comparisons:
            key = (comparison, str(row_a["group"]), str(row_b["group"]))
            if key in seen:
                continue
            seen.add(key)
            overlap = ci_overlap(row_a, row_b)
            out.append({
                "model": model,
                "audio_variant": variant,
                "group_axis": axis,
                "comparison": comparison,
                "group_a": row_a["group"],
                "group_a_wer": row_a["wer"],
                "group_a_ci_low": row_a["ci_low"],
                "group_a_ci_high": row_a["ci_high"],
                "group_b": row_b["group"],
                "group_b_wer": row_b["wer"],
                "group_b_ci_low": row_b["ci_low"],
                "group_b_ci_high": row_b["ci_high"],
                "ci_overlap": overlap,
                "ci_interpretation": "overlap_uncertain" if overlap else "separated_stable_gap",
            })
    return out


def compute_fairness_table(group_table: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in group_table:
        grouped[(str(row["model"]), str(row["audio_variant"]), str(row["group_axis"]))].append(row)

    out: list[dict[str, object]] = []
    for (model, variant, axis), rows in sorted(grouped.items()):
        valid = [r for r in rows if not math.isnan(float(r["wer"]))]
        if len(valid) < 2:
            continue
        best = min(valid, key=lambda r: float(r["wer"]))
        worst = max(valid, key=lambda r: float(r["wer"]))
        best_wer = float(best["wer"])
        worst_wer = float(worst["wer"])
        out.append({
            "model": model,
            "audio_variant": variant,
            "group_axis": axis,
            "best_group": best["group"],
            "best_wer": best_wer,
            "worst_group": worst["group"],
            "worst_wer": worst_wer,
            "mmr": worst_wer / best_wer if best_wer > 0 else math.inf,
            "relative_gap": (worst_wer - best_wer) / best_wer if best_wer > 0 else math.inf,
            "absolute_gap": worst_wer - best_wer,
        })
    return out


def compute_paired_delta(
    rows: list[dict[str, str]],
    baseline_variant: str,
    group_col: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> list[dict[str, object]]:
    by_key = group_rows(rows, ("model", "utterance_id"))
    deltas: list[dict[str, object]] = []
    for (model, _), items in by_key.items():
        baseline = None
        for item in items:
            if item.get("audio_variant") == baseline_variant:
                baseline = item
                break
        if baseline is None:
            continue
        base_wer = float(baseline.get("wer") or 0)
        for item in items:
            variant = item.get("audio_variant", "")
            if variant == baseline_variant:
                continue
            deltas.append({
                "model": model,
                "audio_variant": variant,
                "utterance_id": item.get("utterance_id", ""),
                group_col: item.get(group_col, ""),
                "delta_wer": float(item.get("wer") or 0) - base_wer,
            })

    grouped = group_rows([{k: str(v) for k, v in row.items()} for row in deltas], ("model", "audio_variant", group_col))
    out: list[dict[str, object]] = []
    for (model, variant, group), group_data in sorted(grouped.items()):
        values = [float(row["delta_wer"]) for row in group_data]
        if not group or len(values) < min_group_size:
            continue
        out.append({
            "model": model,
            "audio_variant": variant,
            "group_axis": group_col,
            "group": group,
            "n": len(values),
            "mean_delta_wer_vs_baseline": sum(values) / len(values),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FairSpeech compression fairness metrics")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-col", default="ethnicity")
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Read predictions_*.csv recursively for per-model output directories.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=MIN_GROUP_SIZE,
        help="Minimum rows per demographic group; use 1 for tiny pilot gates.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=0,
        help="If set, only metric slots with exactly this many rows are included.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=0,
        help="Number of utterance-level bootstrap resamples for group WER CIs.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Base random seed for bootstrap resampling.",
    )
    args = parser.parse_args()

    raw_rows = load_all_predictions(args.predictions_dir, recursive=args.recursive)
    rows = filter_complete_run_rows(raw_rows, args.expected_rows)
    if not rows:
        raise SystemExit(f"No predictions_*.csv files found in {args.predictions_dir}")

    group_table = compute_group_table(rows, args.group_col, min_group_size=args.min_group_size)
    fairness_table = compute_fairness_table(group_table)
    paired_delta = compute_paired_delta(
        rows,
        args.baseline_variant,
        args.group_col,
        min_group_size=args.min_group_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fairspeech_compression_group_metrics.csv", group_table)
    write_csv(args.output_dir / "fairspeech_compression_fairness_metrics.csv", fairness_table)
    write_csv(args.output_dir / "fairspeech_compression_paired_delta.csv", paired_delta)

    bootstrap_group_ci: list[dict[str, object]] = []
    bootstrap_ci_overlap: list[dict[str, object]] = []
    if args.bootstrap_resamples > 0:
        bootstrap_group_ci = compute_bootstrap_group_ci(
            rows,
            args.group_col,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            min_group_size=args.min_group_size,
        )
        bootstrap_ci_overlap = compute_ci_overlap_table(bootstrap_group_ci)
        write_csv(args.output_dir / "fairspeech_compression_bootstrap_group_ci.csv", bootstrap_group_ci)
        write_csv(args.output_dir / "fairspeech_compression_bootstrap_ci_overlap.csv", bootstrap_ci_overlap)

    summary = {
        "raw_prediction_rows": len(raw_rows),
        "prediction_rows": len(rows),
        "expected_rows": args.expected_rows,
        "group_metric_rows": len(group_table),
        "fairness_metric_rows": len(fairness_table),
        "paired_delta_rows": len(paired_delta),
        "bootstrap_group_ci_rows": len(bootstrap_group_ci),
        "bootstrap_ci_overlap_rows": len(bootstrap_ci_overlap),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "group_col": args.group_col,
        "baseline_variant": args.baseline_variant,
        "min_group_size": args.min_group_size,
    }
    (args.output_dir / "fairspeech_compression_metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
