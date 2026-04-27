"""Tests for scripts/head_surgery/fixing_set_analysis.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "head_surgery"


def test_prerequisite_csvs_present_and_shaped():
    """If prerequisites are missing, the analysis cannot run — surface it clearly."""
    sweep = OUT_DIR / "sweep.csv"
    baseline = OUT_DIR / "baseline_predictions.csv"
    scores = OUT_DIR / "head_scores.csv"
    assert sweep.exists(), f"missing {sweep} — Stage C must have run"
    assert baseline.exists(), f"missing {baseline} — Stage A must have run"
    assert scores.exists(), f"missing {scores} — Stage D must have run"
    sweep_df = pd.read_csv(sweep, nrows=2)
    for col in ["layer", "head", "id", "reference", "hypothesis"]:
        assert col in sweep_df.columns, f"sweep.csv missing column {col}"
    baseline_df = pd.read_csv(baseline, nrows=2)
    for col in ["id", "reference", "hypothesis"]:
        assert col in baseline_df.columns, f"baseline_predictions.csv missing column {col}"
    scores_df = pd.read_csv(scores, nrows=2)
    for col in ["layer", "head", "regression_ok"]:
        assert col in scores_df.columns, f"head_scores.csv missing column {col}"


from scripts.head_surgery.fixing_set_analysis import count_insertions


def test_count_insertions_identical_is_zero():
    assert count_insertions("the cat sat", "the cat sat") == 0


def test_count_insertions_extra_word():
    # "really" is inserted — classifier treats it as content or syntactic filler
    assert count_insertions("the cat sat", "the really cat sat") == 1


def test_count_insertions_repetition_loop():
    # "thank you" repeated 3× extra times → 6 extra tokens
    assert count_insertions("thank you", "thank you thank you thank you thank you") == 6


def test_count_insertions_handles_empty():
    # Empty refs and hyps should not crash and should return 0
    assert count_insertions("", "") == 0
    assert count_insertions("", "anything") == 0
    assert count_insertions("some reference", "") == 0


from scripts.head_surgery.fixing_set_analysis import build_count_table


def _fake_sweep_rows():
    # Three utterances, two heads, covering distinct behaviors
    rows = [
        # Baseline behavior (encoded via layer=-1, head=-1) — Stage B convention
        {"layer": -1, "head": -1, "id": "u1", "reference": "the cat", "hypothesis": "the cat cat"},     # baseline +1
        {"layer": -1, "head": -1, "id": "u2", "reference": "a dog",   "hypothesis": "a dog"},            # baseline 0
        {"layer": -1, "head": -1, "id": "u3", "reference": "red fox", "hypothesis": "red fox fox fox"},  # baseline +2
        # Head (0, 0) fixes u1 but introduces a new insertion on u2
        {"layer": 0, "head": 0, "id": "u1", "reference": "the cat", "hypothesis": "the cat"},            # masked 0
        {"layer": 0, "head": 0, "id": "u2", "reference": "a dog",   "hypothesis": "a dog dog"},          # masked +1
        {"layer": 0, "head": 0, "id": "u3", "reference": "red fox", "hypothesis": "red fox fox fox"},    # masked +2
        # Head (1, 0) fixes u3 without harming anything
        {"layer": 1, "head": 0, "id": "u1", "reference": "the cat", "hypothesis": "the cat cat"},        # masked +1
        {"layer": 1, "head": 0, "id": "u2", "reference": "a dog",   "hypothesis": "a dog"},              # masked 0
        {"layer": 1, "head": 0, "id": "u3", "reference": "red fox", "hypothesis": "red fox fox"},        # masked +1
    ]
    return pd.DataFrame(rows)


def test_build_count_table_shape_and_values(tmp_path):
    sweep_df = _fake_sweep_rows()
    baseline_df = sweep_df[sweep_df["layer"] == -1][["id", "reference", "hypothesis"]].reset_index(drop=True)
    # Spill to disk the way the real files are shaped
    s_csv = tmp_path / "sweep.csv"
    b_csv = tmp_path / "baseline_predictions.csv"
    sweep_df[sweep_df["layer"] != -1].to_csv(s_csv, index=False)
    baseline_df.to_csv(b_csv, index=False)

    counts = build_count_table(s_csv, b_csv)
    # Expect one row per (id × condition) where condition ∈ {"baseline"} ∪ {(L,h)}
    assert set(counts["id"]) == {"u1", "u2", "u3"}
    baseline_counts = counts[counts["condition"] == "baseline"].set_index("id")["count"].to_dict()
    assert baseline_counts == {"u1": 1, "u2": 0, "u3": 2}
    # Head (0,0) on u1 should be 0 (fixed), u2 should be 1 (new harm)
    m00 = counts[(counts["layer"] == 0) & (counts["head"] == 0)].set_index("id")["count"].to_dict()
    assert m00 == {"u1": 0, "u2": 1, "u3": 2}
    m10 = counts[(counts["layer"] == 1) & (counts["head"] == 0)].set_index("id")["count"].to_dict()
    assert m10 == {"u1": 1, "u2": 0, "u3": 1}


from scripts.head_surgery.fixing_set_analysis import (
    identify_affected,
    build_coverage_matrix,
)


def test_identify_affected_picks_nonzero_baseline():
    counts = pd.DataFrame([
        {"condition": "baseline", "layer": -1, "head": -1, "id": "u1", "count": 2},
        {"condition": "baseline", "layer": -1, "head": -1, "id": "u2", "count": 0},
        {"condition": "baseline", "layer": -1, "head": -1, "id": "u3", "count": 5},
    ])
    assert identify_affected(counts) == ["u1", "u3"]


def test_build_coverage_matrix_applies_all_three_filters(tmp_path):
    # Same fake data as Task 3, plus a head_scores row per masked (L, h).
    sweep_df = _fake_sweep_rows()
    baseline_df = sweep_df[sweep_df["layer"] == -1][["id", "reference", "hypothesis"]].reset_index(drop=True)
    s_csv = tmp_path / "sweep.csv"
    b_csv = tmp_path / "baseline_predictions.csv"
    sweep_df[sweep_df["layer"] != -1].to_csv(s_csv, index=False)
    baseline_df.to_csv(b_csv, index=False)
    counts = build_count_table(s_csv, b_csv)

    # head_scores — both heads "regression_ok=True"
    scores = pd.DataFrame([
        {"layer": 0, "head": 0, "regression_ok": True,  "regression_checked": True},
        {"layer": 1, "head": 0, "regression_ok": True,  "regression_checked": True},
    ])
    scores_csv = tmp_path / "head_scores.csv"
    scores.to_csv(scores_csv, index=False)

    matrix, utt_ids, heads = build_coverage_matrix(counts, scores_csv)
    # Affected: u1 (baseline=1), u3 (baseline=2) → two rows
    assert utt_ids == ["u1", "u3"]
    # Valid heads under the three filters:
    #   (0, 0): fixes u1 (1→0), but HARMS u2 (0→1) → filter (ii) eliminates it.
    #   (1, 0): fixes u3 (2→1), no harm on u2 (stays 0), u1 stays 1 (no harm) → valid.
    assert heads == [(1, 0)]
    # Coverage: (1, 0) helps u3 (1→0 reduction) but NOT u1 (no reduction on u1).
    assert matrix.shape == (2, 1)
    assert matrix[utt_ids.index("u1"), 0] == 0
    assert matrix[utt_ids.index("u3"), 0] == 1


def test_build_coverage_matrix_filters_regression_failures(tmp_path):
    sweep_df = _fake_sweep_rows()
    baseline_df = sweep_df[sweep_df["layer"] == -1][["id", "reference", "hypothesis"]].reset_index(drop=True)
    s_csv = tmp_path / "sweep.csv"
    b_csv = tmp_path / "baseline_predictions.csv"
    sweep_df[sweep_df["layer"] != -1].to_csv(s_csv, index=False)
    baseline_df.to_csv(b_csv, index=False)
    counts = build_count_table(s_csv, b_csv)

    # Mark (1, 0) as regression_ok=False — it should be filtered out.
    scores = pd.DataFrame([
        {"layer": 0, "head": 0, "regression_ok": True,  "regression_checked": True},
        {"layer": 1, "head": 0, "regression_ok": False, "regression_checked": True},
    ])
    scores_csv = tmp_path / "head_scores.csv"
    scores.to_csv(scores_csv, index=False)
    matrix, utt_ids, heads = build_coverage_matrix(counts, scores_csv)
    # Neither head is valid → matrix has zero columns.
    assert matrix.shape == (2, 0)
    assert heads == []


def test_build_coverage_matrix_accepts_unchecked_heads(tmp_path):
    sweep_df = _fake_sweep_rows()
    baseline_df = sweep_df[sweep_df["layer"] == -1][["id", "reference", "hypothesis"]].reset_index(drop=True)
    s_csv = tmp_path / "sweep.csv"
    b_csv = tmp_path / "baseline_predictions.csv"
    sweep_df[sweep_df["layer"] != -1].to_csv(s_csv, index=False)
    baseline_df.to_csv(b_csv, index=False)
    counts = build_count_table(s_csv, b_csv)

    # regression_checked=False means the Stage D guard did not evaluate this head
    # (the top-50 cap). Treat it as "pass" rather than exclude — matches report §8 logic.
    scores = pd.DataFrame([
        {"layer": 0, "head": 0, "regression_ok": True,  "regression_checked": True},
        {"layer": 1, "head": 0, "regression_ok": None,  "regression_checked": False},
    ])
    scores_csv = tmp_path / "head_scores.csv"
    scores.to_csv(scores_csv, index=False)
    matrix, utt_ids, heads = build_coverage_matrix(counts, scores_csv)
    # (1, 0) unchecked but valid under the other two filters → kept.
    assert heads == [(1, 0)]


from scripts.head_surgery.fixing_set_analysis import greedy_cover


def test_greedy_cover_empty():
    assert greedy_cover(np.zeros((0, 0), dtype=bool), []) == []


def test_greedy_cover_no_columns():
    assert greedy_cover(np.zeros((5, 0), dtype=bool), []) == []


def test_greedy_cover_picks_largest_then_rest():
    # 5 utterances × 4 heads with a known optimum of 2.
    # h0 covers {0, 1, 2}, h1 covers {2, 3}, h2 covers {4}, h3 covers {1}.
    # Greedy: pick h0 (3), then must pick h1 (for 3) + h2 (for 4). Size 3.
    # (Optimum is h0+h1+h2 = 3 here — they are the same; this is a sanity baseline.)
    matrix = np.array([
        [1, 0, 0, 0],  # utt 0: covered by h0
        [1, 0, 0, 1],  # utt 1: covered by h0, h3
        [1, 1, 0, 0],  # utt 2: covered by h0, h1
        [0, 1, 0, 0],  # utt 3: covered by h1
        [0, 0, 1, 0],  # utt 4: covered by h2
    ], dtype=bool)
    heads = [(0, 0), (0, 1), (0, 2), (0, 3)]
    cover = greedy_cover(matrix, heads)
    # Expect h0 first (covers 3), then h1 (covers u3), then h2 (covers u4).
    assert len(cover) == 3
    assert cover[0][0] == (0, 0)
    assert {c[0] for c in cover} == {(0, 0), (0, 1), (0, 2)}
    # Each entry should list its newly covered utterance indices
    assert cover[0][1] == [0, 1, 2]


def test_greedy_cover_leaves_uncovered_unhelpable_rows():
    # Row 0 has no ones — no head can cover it. Greedy must terminate when no
    # column adds coverage, returning the partial cover only.
    matrix = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
    ], dtype=bool)
    cover = greedy_cover(matrix, [(0, 0), (1, 0)])
    assert len(cover) == 2
    covered = set().union(*(set(c[1]) for c in cover))
    assert covered == {1, 2}  # row 0 is unhelpable; not in cover


from scripts.head_surgery.fixing_set_analysis import ilp_cover


def test_ilp_cover_matches_greedy_on_easy_case():
    matrix = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=bool)
    heads = [(0, 0), (0, 1), (0, 2)]
    picked = ilp_cover(matrix, heads)
    # Optimum here is 2: {(0,0), (0,2)} covers all rows.
    assert len(picked) == 2
    assert set(picked) == {(0, 0), (0, 2)}


def test_ilp_cover_finds_strictly_smaller_set_than_greedy():
    # Classic greedy-worst-case: greedy picks h0 first (covers 4), then
    # needs h1, h2, h3, h4 to cover the remaining 1s. Optimum is {h5, h6}.
    #
    # Rows 0..3 each covered by h0 AND one of {h5, h6}
    # Row 4 covered by h5 only; row 5 by h6 only.
    matrix = np.array([
        # h0 h1 h2 h3 h4 h5 h6
        [ 1, 0, 0, 0, 0, 1, 0],  # row 0
        [ 1, 0, 0, 0, 0, 1, 0],  # row 1
        [ 1, 0, 0, 0, 0, 0, 1],  # row 2
        [ 1, 0, 0, 0, 0, 0, 1],  # row 3
        [ 0, 0, 0, 0, 0, 1, 0],  # row 4
        [ 0, 0, 0, 0, 0, 0, 1],  # row 5
    ], dtype=bool)
    heads = [(0, i) for i in range(7)]
    picked = ilp_cover(matrix, heads)
    # Optimum = 2: {(0,5), (0,6)} covers every row.
    assert len(picked) == 2
    assert set(picked) == {(0, 5), (0, 6)}


def test_ilp_cover_handles_empty_inputs():
    assert ilp_cover(np.zeros((0, 0), dtype=bool), []) == []
    assert ilp_cover(np.zeros((3, 0), dtype=bool), []) == []


from scripts.head_surgery.fixing_set_analysis import run


def test_run_produces_all_three_artifacts(tmp_path, monkeypatch):
    # Point OUT_DIR at tmp_path for this test only.
    import scripts.head_surgery.fixing_set_analysis as mod
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)

    sweep_df = _fake_sweep_rows()
    baseline_df = sweep_df[sweep_df["layer"] == -1][["id", "reference", "hypothesis"]].reset_index(drop=True)
    (tmp_path).mkdir(parents=True, exist_ok=True)
    s_csv = tmp_path / "sweep.csv"
    b_csv = tmp_path / "baseline_predictions.csv"
    sweep_df[sweep_df["layer"] != -1].to_csv(s_csv, index=False)
    baseline_df.to_csv(b_csv, index=False)
    scores = pd.DataFrame([
        {"layer": 0, "head": 0, "regression_ok": True,  "regression_checked": True},
        {"layer": 1, "head": 0, "regression_ok": True,  "regression_checked": True},
    ])
    sc_csv = tmp_path / "head_scores.csv"
    scores.to_csv(sc_csv, index=False)

    summary = run(s_csv, b_csv, sc_csv)
    # Artifacts
    assert (tmp_path / "fixing_set_per_utterance.csv").exists()
    assert (tmp_path / "coverage_matrix.npz").exists()
    assert (tmp_path / "minimum_surgical_set.json").exists()
    # Summary shape
    assert "n_affected" in summary
    assert "greedy" in summary and "ilp" in summary
    assert summary["n_affected"] == 2  # u1, u3
