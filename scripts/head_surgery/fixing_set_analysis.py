"""Post-hoc fixing-set analysis on the 640-head × 484-utterance sweep.

For each Indian-accent utterance with ≥1 hallucinated token at baseline, find
every (layer, head) whose masking strictly reduces its insertion count; then
solve min-set-cover over those heads under two additional filters —
(i) the head does not introduce new hallucinations on any other utterance, and
(ii) the head passes the Stage D non-Indian-WER regression guard (regression_ok).

Inputs (all under outputs/head_surgery/):
  sweep.csv                  — Stage C
  baseline_predictions.csv   — Stage A
  head_scores.csv            — Stage D

Outputs:
  fixing_set_per_utterance.csv
  coverage_matrix.npz
  minimum_surgical_set.json

See docs/superpowers/plans/2026-04-18-head-surgery-fixing-set-analysis.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

OUT_DIR = Path("outputs/head_surgery")


def count_insertions(reference: str, hypothesis: str) -> int:
    """Number of inserted tokens in hypothesis relative to reference.

    Reference and hypothesis are normalized via Whisper's EnglishTextNormalizer
    (matching the rest of the project). Returns 0 if either side is empty.
    """
    from scripts.head_surgery.insertion_classifier import categorize_insertions
    from scripts.inference.run_inference import normalize_text

    ref = normalize_text(reference or "")
    hyp = normalize_text(hypothesis or "")
    if not ref or not hyp:
        return 0
    return len(categorize_insertions(ref, hyp))


def build_count_table(sweep_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    """Long-form DataFrame: columns = [condition, layer, head, id, count].

    `condition` is "baseline" for the no-mask rows (layer=head=-1 sentinel not
    used here — baseline comes from baseline_predictions.csv), otherwise the
    string "L{L}_h{H}". Every utterance appears once per condition.
    """
    sweep = pd.read_csv(sweep_csv)
    baseline = pd.read_csv(baseline_csv)

    # Baseline rows
    baseline_counts = [
        {"condition": "baseline", "layer": -1, "head": -1,
         "id": str(r["id"]), "count": count_insertions(r["reference"], r["hypothesis"])}
        for _, r in baseline.iterrows()
    ]
    # Masked rows
    masked_counts = []
    for _, r in sweep.iterrows():
        L, h = int(r["layer"]), int(r["head"])
        if L == -1 or h == -1:
            # Pilot-baseline sentinel rows, if any — skip (baseline is in baseline_csv)
            continue
        masked_counts.append({
            "condition": f"L{L}_h{h}",
            "layer": L,
            "head": h,
            "id": str(r["id"]),
            "count": count_insertions(r["reference"], r["hypothesis"]),
        })
    return pd.DataFrame(baseline_counts + masked_counts)


def identify_affected(counts: pd.DataFrame) -> List[str]:
    """Utterance IDs with baseline insertion count > 0, in CSV order."""
    base = counts[counts["condition"] == "baseline"].copy()
    return base.loc[base["count"] > 0, "id"].astype(str).tolist()


def build_coverage_matrix(
    counts: pd.DataFrame,
    head_scores_csv: Path,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]]]:
    """Binary coverage matrix under three filters.

    A head (L, h) is VALID iff all three hold:
      (i)   it strictly reduces insertion count on ≥1 affected utterance,
      (ii)  it does NOT introduce new insertions on any utterance
            (for every utterance u: masked_count[u] ≤ baseline_count[u]),
      (iii) head_scores[regression_ok] is True OR regression_checked is False
            (unchecked heads are treated as pass-through, matching §8 of the
             edited report).

    Returns:
      matrix[n_affected × n_valid]  — 1 iff head helps that affected utterance
      utt_ids                       — row labels
      heads                         — column labels as (layer, head) tuples
    """
    affected = identify_affected(counts)
    if not affected:
        return np.zeros((0, 0), dtype=bool), [], []

    base_lookup = (
        counts[counts["condition"] == "baseline"]
        .set_index("id")["count"].astype(int).to_dict()
    )

    # Pivot to wide form: rows=id, cols=condition, values=count. Keep baseline.
    masked = counts[counts["condition"] != "baseline"].copy()
    wide = masked.pivot_table(
        index="id", columns=["layer", "head"], values="count", aggfunc="first"
    ).fillna(-1).astype(int)

    # Filter (iii) — regression guard
    scores = pd.read_csv(head_scores_csv)
    scores["accept"] = scores["regression_ok"].where(
        scores["regression_checked"] == True, other=True
    ).fillna(True).astype(bool)
    accepted = {(int(r["layer"]), int(r["head"])) for _, r in scores.iterrows() if r["accept"]}

    # Apply filters (i) and (ii) per column
    valid_heads: List[Tuple[int, int]] = []
    col_vectors: List[np.ndarray] = []
    affected_index = {u: i for i, u in enumerate(affected)}

    for (L, h) in wide.columns:
        key = (int(L), int(h))
        if key not in accepted:
            continue
        col = wide[(L, h)]  # int series indexed by utterance id

        # Filter (ii) — no new global harm. For every utt, masked ≤ baseline.
        harms = False
        for utt_id, masked_count in col.items():
            if masked_count < 0:
                continue  # missing — treat as unchanged (skip)
            if masked_count > base_lookup.get(str(utt_id), 0):
                harms = True
                break
        if harms:
            continue

        # Filter (i) — helps ≥ 1 affected utterance. Build the coverage column.
        vec = np.zeros(len(affected), dtype=bool)
        any_help = False
        for utt_id in affected:
            masked_count = int(col.get(utt_id, -1))
            if masked_count < 0:
                continue
            if masked_count < base_lookup[utt_id]:
                vec[affected_index[utt_id]] = True
                any_help = True
        if not any_help:
            continue

        valid_heads.append(key)
        col_vectors.append(vec)

    if not valid_heads:
        return np.zeros((len(affected), 0), dtype=bool), affected, []
    matrix = np.stack(col_vectors, axis=1)
    return matrix, affected, valid_heads


def greedy_cover(
    matrix: np.ndarray,
    heads: List[Tuple[int, int]],
) -> List[Tuple[Tuple[int, int], List[int]]]:
    """Greedy min-set-cover. Returns [((layer, head), newly_covered_row_indices)].

    Terminates when no remaining column adds any uncovered row (so unhelpable
    rows, i.e. rows with no 1s, are silently left uncovered).
    """
    if matrix.size == 0 or matrix.shape[1] == 0:
        return []
    n_rows = matrix.shape[0]
    uncovered = np.ones(n_rows, dtype=bool)
    remaining_cols = set(range(matrix.shape[1]))
    result: List[Tuple[Tuple[int, int], List[int]]] = []
    while uncovered.any() and remaining_cols:
        best_col, best_hits = -1, 0
        for c in remaining_cols:
            hits = int((matrix[:, c] & uncovered).sum())
            if hits > best_hits:
                best_col, best_hits = c, hits
        if best_hits == 0:
            break
        newly = list(np.where(matrix[:, best_col] & uncovered)[0])
        result.append((heads[best_col], newly))
        uncovered[newly] = False
        remaining_cols.discard(best_col)
    return result


def ilp_cover(
    matrix: np.ndarray,
    heads: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Exact min-set-cover via scipy.optimize.milp.

    Returns the sorted list of chosen (layer, head) tuples. Rows with no 1
    (unhelpable utterances) are dropped before solving — the solver requires
    every constraint to be satisfiable.
    """
    if matrix.size == 0 or matrix.shape[1] == 0:
        return []
    from scipy.optimize import LinearConstraint, milp, Bounds

    # Drop unhelpable rows (rows of all zeros) — otherwise the MILP is infeasible.
    row_has_cover = matrix.any(axis=1)
    A = matrix[row_has_cover]  # [n_feasible_rows × n_heads]
    n_rows, n_cols = A.shape
    if n_rows == 0:
        return []

    c = np.ones(n_cols)                    # minimize sum of x (number of heads chosen)
    constraint = LinearConstraint(A.astype(float), lb=1, ub=np.inf)  # each row covered ≥ 1
    bounds = Bounds(lb=0, ub=1)            # x ∈ {0,1}
    integrality = np.ones(n_cols)          # all variables integer
    res = milp(c, constraints=constraint, bounds=bounds, integrality=integrality)
    if not res.success:
        raise RuntimeError(f"MILP solve failed: {res.message}")
    picked_idx = [i for i, v in enumerate(res.x) if v > 0.5]
    picked = [heads[i] for i in picked_idx]
    return sorted(picked)


def _write_per_utterance_csv(
    counts: pd.DataFrame,
    affected: List[str],
    matrix: np.ndarray,
    heads: List[Tuple[int, int]],
    out_path: Path,
) -> None:
    base_lookup = (
        counts[counts["condition"] == "baseline"]
        .set_index("id")["count"].astype(int).to_dict()
    )
    # Long-form: one row per (utterance, helping-head)
    rows = []
    if matrix.size:
        for u_idx, u in enumerate(affected):
            ref = counts[(counts["condition"] == "baseline") & (counts["id"] == u)].iloc[0]
            for h_idx, (L, h) in enumerate(heads):
                if not matrix[u_idx, h_idx]:
                    continue
                masked = counts[
                    (counts["layer"] == L) & (counts["head"] == h) & (counts["id"] == u)
                ]["count"].iloc[0]
                rows.append({
                    "id": u, "baseline_count": base_lookup[u],
                    "layer": L, "head": h,
                    "masked_count": int(masked),
                    "reduction": int(base_lookup[u] - masked),
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_matrix_npz(matrix, utt_ids, heads, out_path: Path) -> None:
    np.savez(
        out_path,
        matrix=matrix,
        utt_ids=np.array(utt_ids, dtype=object),
        heads=np.array(heads, dtype=np.int32).reshape(-1, 2) if heads else np.zeros((0, 2), dtype=np.int32),
    )


def run(
    sweep_csv: Path,
    baseline_csv: Path,
    head_scores_csv: Path,
) -> dict:
    """End-to-end driver. Writes three artifacts under OUT_DIR and returns a summary."""
    t0 = time.time()
    counts = build_count_table(sweep_csv, baseline_csv)
    matrix, utt_ids, heads = build_coverage_matrix(counts, head_scores_csv)

    greedy = greedy_cover(matrix, heads)
    ilp = ilp_cover(matrix, heads) if matrix.size else []

    # Unhelpable: rows with no 1s in the matrix
    if matrix.size:
        row_has_cover = matrix.any(axis=1)
        unhelpable = [utt_ids[i] for i, flag in enumerate(row_has_cover) if not flag]
    else:
        unhelpable = list(utt_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_utt_csv = OUT_DIR / "fixing_set_per_utterance.csv"
    matrix_npz = OUT_DIR / "coverage_matrix.npz"
    set_json = OUT_DIR / "minimum_surgical_set.json"

    _write_per_utterance_csv(counts, utt_ids, matrix, heads, per_utt_csv)
    _write_matrix_npz(matrix, utt_ids, heads, matrix_npz)

    summary = {
        "n_affected": len(utt_ids),
        "n_valid_heads": len(heads),
        "unhelpable_utterances": unhelpable,
        "greedy": [
            {"layer": L, "head": h, "newly_covered_count": len(covered),
             "newly_covered_ids": [utt_ids[i] for i in covered]}
            for (L, h), covered in greedy
        ],
        "ilp": [{"layer": L, "head": h} for (L, h) in ilp],
        "runtime_seconds": round(time.time() - t0, 2),
    }
    set_json.write_text(json.dumps(summary, indent=2))
    return summary


def _cli():
    p = argparse.ArgumentParser(description="Fixing-set analysis on existing sweep artifacts.")
    p.add_argument("--sweep-csv", default=str(OUT_DIR / "sweep.csv"))
    p.add_argument("--baseline-csv", default=str(OUT_DIR / "baseline_predictions.csv"))
    p.add_argument("--head-scores-csv", default=str(OUT_DIR / "head_scores.csv"))
    args = p.parse_args()
    summary = run(Path(args.sweep_csv), Path(args.baseline_csv), Path(args.head_scores_csv))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
