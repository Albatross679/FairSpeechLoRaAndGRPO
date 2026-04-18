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
