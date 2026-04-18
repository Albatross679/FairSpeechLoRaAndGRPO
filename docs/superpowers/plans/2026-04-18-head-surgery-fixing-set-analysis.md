# Head-Surgery Fixing-Set Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a post-hoc analysis on the existing 309,760-row `sweep.csv` that answers "for each of the ~45 utterances with hallucinations at baseline, which (layer, head) masks reduce at least one hallucinated token on that utterance — and what is the minimum set of heads whose union covers all affected utterances without introducing new hallucinations or breaking the non-Indian regression guard?"

**Architecture:** One new module `scripts/head_surgery/fixing_set_analysis.py` (~350 LoC) that ingests three existing CSVs, computes per-(utterance, head) insertion counts via the existing classifier, builds a binary coverage matrix under three filters (helps, no-new-harm globally, passes non-Indian guard), runs both a greedy and an ILP min-set-cover, and emits three artifacts plus an appended §8b in `docs/head_surgery_report_edited.md`. No GPU inference — pure pandas + numpy + scipy.optimize.milp. One new test module with synthetic fixtures for the set-cover logic.

**Tech Stack:** Python 3.10+, pandas 2.1+, numpy 1.26+, scipy 1.11+ (for `scipy.optimize.milp`), the project's existing `scripts.head_surgery.insertion_classifier` wrapper, the project's existing `scripts.inference.run_inference.normalize_text` for text normalization, pytest 8.

**Project conventions (read before starting):** See [`CLAUDE.md`](../../../CLAUDE.md). Every behavioral change warrants a `logs/<topic>.md` log entry with `fileClass: Log` frontmatter. Artifacts go under `outputs/head_surgery/` (gitignored). Python tests live under `tests/`. The `scripts/head_surgery/` package is already wired; just drop a new module into it.

**Input preconditions (all verified present on disk):**
- `outputs/head_surgery/sweep.csv` — 309,761 rows (header + 640 conditions × 484 utterances); columns `layer, head, id, reference, hypothesis, condition_insertion_rate_total`.
- `outputs/head_surgery/baseline_predictions.csv` — 484 rows; columns `id, reference, hypothesis`.
- `outputs/head_surgery/head_scores.csv` — 640 rows; columns `layer, head, …, regression_checked, regression_ok`.
- `scripts/head_surgery/insertion_classifier.py` — exposes `categorize_insertions(ref, hyp) -> list[{"word","category"}]`. Count of insertions on an utterance = `len(categorize_insertions(ref, hyp))`.
- `scripts/inference/run_inference.py` — exposes `normalize_text(s) -> str` (Whisper's `EnglishTextNormalizer`). Must be applied to both reference and hypothesis before classifier, matching the rest of the project.

**Output artifacts (all under `outputs/head_surgery/`):**
- `fixing_set_per_utterance.csv` — one row per (affected utterance, helping head); columns `id, reference, baseline_count, layer, head, masked_count, reduction, regression_ok`.
- `coverage_matrix.npz` — `numpy.savez` of `{"matrix": bool[n_affected × n_valid_heads], "utt_ids": list[str], "heads": list[(int,int)]}`.
- `minimum_surgical_set.json` — both greedy and ILP solutions, counts, unhelpable utterances, runtime in seconds.
- `docs/head_surgery_report_edited.md` appended with §8b.

---

## Task 1: Scaffolding + prerequisite verification

**Files:**
- Create: `scripts/head_surgery/fixing_set_analysis.py` (skeleton)
- Create: `tests/test_fixing_set_analysis.py` (skeleton)

- [ ] **Step 1: Write a failing prerequisite-presence test**

Create `tests/test_fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py::test_prerequisite_csvs_present_and_shaped -v`
Expected: PASS (the CSVs already exist per the in-scope execution state). If it fails, the engineer stops here and backfills the missing CSV before continuing.

- [ ] **Step 2: Create the module skeleton**

Create `scripts/head_surgery/fixing_set_analysis.py`:

```python
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
```

- [ ] **Step 3: Commit scaffolding**

```bash
git add scripts/head_surgery/fixing_set_analysis.py tests/test_fixing_set_analysis.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): scaffold fixing-set analysis module

Skeleton + prerequisite-presence test. The analysis runs on the existing
sweep.csv / baseline_predictions.csv / head_scores.csv artifacts — no GPU.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Per-utterance insertion count function

The core primitive: for one (reference, hypothesis) pair, return the number of inserted tokens per the project's existing classifier.

**Files:**
- Modify: `scripts/head_surgery/fixing_set_analysis.py`
- Modify: `tests/test_fixing_set_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k count_insertions`
Expected: FAIL with `ImportError` on `count_insertions`.

- [ ] **Step 2: Implement `count_insertions`**

Append to `scripts/head_surgery/fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k count_insertions`
Expected: all four PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/head_surgery/fixing_set_analysis.py tests/test_fixing_set_analysis.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): per-utterance insertion counter

Thin wrapper around the project's existing EnglishTextNormalizer +
insertion classifier. Returns the integer count; 0 for empty inputs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Build the per-(utterance, head) count table

Aggregate sweep.csv + baseline_predictions.csv into one long-form DataFrame with an insertion count per (utterance, head) plus a baseline row per utterance.

**Files:**
- Modify: `scripts/head_surgery/fixing_set_analysis.py`
- Modify: `tests/test_fixing_set_analysis.py`

- [ ] **Step 1: Write failing tests using an in-memory fixture**

Append to `tests/test_fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k build_count_table`
Expected: FAIL with `ImportError` on `build_count_table`.

- [ ] **Step 2: Implement `build_count_table`**

Append to `scripts/head_surgery/fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k build_count_table`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/head_surgery/fixing_set_analysis.py tests/test_fixing_set_analysis.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): build long-form (utterance, head) insertion-count table

Ingests sweep.csv + baseline_predictions.csv, emits one count per
(condition, utterance). Condition = "baseline" or "L{L}_h{h}". Skips any
layer=-1 / head=-1 sentinel rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Identify affected utterances and build the coverage matrix

This is the analysis's core data structure. The matrix is `[|affected| × n_valid_heads]` with a cell = 1 iff the head is valid *and* helps that utterance.

**Files:**
- Modify: `scripts/head_surgery/fixing_set_analysis.py`
- Modify: `tests/test_fixing_set_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k coverage_matrix`
Expected: FAIL with `ImportError` on `identify_affected` / `build_coverage_matrix`.

- [ ] **Step 2: Implement `identify_affected` and `build_coverage_matrix`**

Append to `scripts/head_surgery/fixing_set_analysis.py`:

```python
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
      (ii)  it does NOT introduce any new insertions globally
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k coverage_matrix`
Expected: all three PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/head_surgery/fixing_set_analysis.py tests/test_fixing_set_analysis.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): identify affected utterances + build coverage matrix

Coverage matrix applies three filters: head helps ≥1 affected utterance,
head introduces no new global harm, head passes the Stage D non-Indian
regression guard (unchecked heads treated as pass-through).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Greedy min-set-cover

Standard greedy — pick the column covering the most uncovered rows, repeat. Easy to interpret and gives a log-factor approximation. Cheap (microseconds).

**Files:**
- Modify: `scripts/head_surgery/fixing_set_analysis.py`
- Modify: `tests/test_fixing_set_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k greedy_cover`
Expected: FAIL with `ImportError` on `greedy_cover`.

- [ ] **Step 2: Implement `greedy_cover`**

Append to `scripts/head_surgery/fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k greedy_cover`
Expected: all three PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/head_surgery/fixing_set_analysis.py tests/test_fixing_set_analysis.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): greedy min-set-cover for fixing-set analysis

Returns an ordered list of (head, newly_covered_utterances). Terminates
when no remaining head adds coverage; unhelpable rows are left out.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ILP min-set-cover via `scipy.optimize.milp`

Exact solve, used to verify how close greedy is to optimal. Small problems (≤ ~50 rows, ≤ ~600 cols) solve in milliseconds.

**Files:**
- Modify: `scripts/head_surgery/fixing_set_analysis.py`
- Modify: `tests/test_fixing_set_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k ilp_cover`
Expected: FAIL with `ImportError` on `ilp_cover`.

- [ ] **Step 2: Implement `ilp_cover`**

Append to `scripts/head_surgery/fixing_set_analysis.py`:

```python
def ilp_cover(
    matrix: np.ndarray,
    heads: List[Tuple[int, int]],
) -> List[Tuple[Tuple[int, int], ...]]:
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k ilp_cover`
Expected: all three PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/head_surgery/fixing_set_analysis.py tests/test_fixing_set_analysis.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): exact ILP min-set-cover via scipy.optimize.milp

Baseline to compare greedy against. Drops unhelpable rows (all-zero) before
solving to keep the MILP feasible.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Driver that glues it all together and writes artifacts

**Files:**
- Modify: `scripts/head_surgery/fixing_set_analysis.py`
- Modify: `tests/test_fixing_set_analysis.py`

- [ ] **Step 1: Write a smoke test for the driver using tmp_path**

Append to `tests/test_fixing_set_analysis.py`:

```python
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
```

Run: `pytest tests/test_fixing_set_analysis.py -v -k run_produces`
Expected: FAIL with `ImportError` on `run`.

- [ ] **Step 2: Implement `run` and the CLI**

Append to `scripts/head_surgery/fixing_set_analysis.py`:

```python
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
```

- [ ] **Step 3: Run the smoke test**

Run: `pytest tests/test_fixing_set_analysis.py -v`
Expected: all tests PASS. This is the end of the unit suite.

- [ ] **Step 4: Commit**

```bash
git add scripts/head_surgery/fixing_set_analysis.py tests/test_fixing_set_analysis.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): fixing-set analysis driver + CLI

Wires build_count_table, build_coverage_matrix, greedy_cover, and
ilp_cover into one driver. Writes three artifacts (per-utterance CSV,
coverage matrix NPZ, minimum surgical set JSON) under outputs/head_surgery/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Run on real data

**Files:**
- (reads) `outputs/head_surgery/{sweep.csv, baseline_predictions.csv, head_scores.csv}`
- (writes) `outputs/head_surgery/{fixing_set_per_utterance.csv, coverage_matrix.npz, minimum_surgical_set.json}`

- [ ] **Step 1: Execute the driver**

Run: `python -m scripts.head_surgery.fixing_set_analysis`
Expected: JSON summary printed to stdout with fields `n_affected`, `n_valid_heads`, `unhelpable_utterances`, `greedy`, `ilp`, `runtime_seconds`. Runtime target ≤ 120 s. If it exceeds 5 min, kill and investigate — likely the per-row insertion counting is the hot loop; cache via `counts = build_count_table(...)` already does this but if memory is constrained switch to dtype-int8 for the matrix.

- [ ] **Step 2: Sanity-check the numbers against §8 of the edited report**

Expected qualitative checks:
- `n_affected` should be in the range [30, 60]. The edited report says 45/484 utterances had ≥1 hallucination.
- `n_valid_heads` is ≤ 640 and empirically likely ≤ 50 (most heads either help nothing or globally harm).
- `ilp` length should be ≤ `len(greedy)` (ILP is optimal, greedy is an approximation).
- L=0 h=5 (the catastrophic keystone) should NOT appear in either cover — it introduces massive global harm, so filter (ii) eliminates it.
- L=20 h=11 (the sole bootstrap-significant head from §7.4) *may* appear in greedy, depending on whether it also passes filter (ii).

- [ ] **Step 3: Write a log entry**

Create `logs/head-surgery-fixing-set-analysis.md`:

```markdown
---
fileClass: Log
name: head-surgery-fixing-set-analysis
description: Post-hoc fixing-set analysis on sweep.csv — which (L, h) pairs cover which of the ~45 affected utterances, and what is the minimum surgical set?
status: complete
subtype: evaluation
created: 2026-04-18
updated: 2026-04-18
tags: [head-surgery, fixing-set, post-hoc, no-gpu]
aliases: []
---

# Fixing-set analysis

## Inputs
- `outputs/head_surgery/sweep.csv` (309,760 rows)
- `outputs/head_surgery/baseline_predictions.csv` (484 rows)
- `outputs/head_surgery/head_scores.csv` (640 rows)

## Method
Three-filter coverage matrix (helps, no global harm, regression_ok) +
greedy and ILP min-set-cover. Implementation:
`scripts/head_surgery/fixing_set_analysis.py`.

## Results
- Affected utterances: **<n_affected>**
- Valid heads (after all three filters): **<n_valid_heads>**
- Greedy cover size: **<len(greedy)>**
- ILP optimum size: **<len(ilp)>**
- Unhelpable utterances: **<len(unhelpable_utterances)>** — `[<list>]`
- Runtime: **<runtime_seconds> s**

## Artifacts
- `outputs/head_surgery/fixing_set_per_utterance.csv`
- `outputs/head_surgery/coverage_matrix.npz`
- `outputs/head_surgery/minimum_surgical_set.json`

## Interpretation note
The min-cover is a *hypothesis* about multi-head masking — it assumes the
effects of masking multiple heads simultaneously are at least as good as
the union of single-head effects. That assumption is unverified; the
sweep is single-head. Validating requires a separate GPU run that masks
the candidate set together.
```

Fill in the `<…>` fields from the Step 1 output.

- [ ] **Step 4: Commit**

```bash
git add outputs/head_surgery/fixing_set_per_utterance.csv \
        outputs/head_surgery/coverage_matrix.npz \
        outputs/head_surgery/minimum_surgical_set.json \
        logs/head-surgery-fixing-set-analysis.md
git commit -m "$(cat <<'EOF'
chore(head_surgery): run fixing-set analysis on sweep.csv

No GPU. Artifacts + log entry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

> **Gitignore note:** `outputs/` is gitignored per the project convention; the `git add` above will fail for the CSV/NPZ/JSON. That is expected — commit only the log entry:
>
> ```bash
> git add logs/head-surgery-fixing-set-analysis.md
> git commit -m "docs(head_surgery): log entry for fixing-set analysis run"
> ```

---

## Task 9: Append §8b to the edited report

**Files:**
- Modify: `docs/head_surgery_report_edited.md` (append new §8b after §8)

- [ ] **Step 1: Locate the insertion point**

Run: `grep -n "^## 9. Stage E" docs/head_surgery_report_edited.md`
Expected: one line like `294:## 9. Stage E — Decoding-strategy ablation`. §8b goes immediately before this.

- [ ] **Step 2: Insert the §8b block**

Use the `Edit` tool with `old_string` anchored on the §9 header and `new_string` prepending the §8b content:

```markdown
## 8b. Stage D (cont.) — Fixing-set analysis

### 8b.1 Method

Reframes the question from "which single head reduces hallucination the most on average?" (§7) to "for each of the {n_affected} Indian-accent utterances with ≥1 hallucinated token, which (L, h) masks eliminate at least one token of that utterance — and what is the minimum head set whose union covers all of them?"

A head (L, h) is considered **valid** for the fixing set only if all three hold:
1. It strictly reduces the insertion count on ≥1 affected utterance.
2. It introduces no new insertions on any utterance in the pool (global no-harm).
3. It passes the non-Indian regression guard from §8 (`regression_ok=True` or `regression_checked=False`).

A binary coverage matrix `[n_affected × n_valid_heads]` is then solved two ways: **greedy** (picks the column covering the most uncovered rows, repeats) and **ILP optimal** via `scipy.optimize.milp` (for comparison). Source: [`scripts/head_surgery/fixing_set_analysis.py`](../scripts/head_surgery/fixing_set_analysis.py).

### 8b.2 Result — coverage statistics

| Metric | Value |
|---|---:|
| Affected utterances (baseline insertion count > 0) | **{n_affected}** |
| Valid heads after three filters | **{n_valid_heads}** (of 640) |
| Greedy cover size | **{len(greedy)}** |
| ILP optimum cover size | **{len(ilp)}** |
| Unhelpable utterances (no single-head mask can fix) | **{len(unhelpable)}** |
| Analysis runtime | {runtime_seconds} s (no GPU) |

### 8b.3 Result — greedy ordering

| Order | (Layer, Head) | Newly covered utterances | Cumulative coverage |
|---:|---|---:|---:|
| *(one row per entry in `minimum_surgical_set.json::greedy`, cumulative count)* | | | |

### 8b.4 Unhelpable utterances

{len(unhelpable)} utterances have at least one hallucinated token at baseline that **no valid single-head mask** can eliminate under the three-filter criterion. Their IDs are listed in [`minimum_surgical_set.json`](../outputs/head_surgery/minimum_surgical_set.json). These represent the floor of what single-head masking can achieve on this dataset.

### 8b.5 Interpretation caveat

The min-cover is a *hypothesis about multi-head masking*, not a measured result. It assumes that masking multiple heads simultaneously produces at least the union of their single-head effects. The full sweep is single-head; validating the min-cover requires a separate GPU run that installs the entire candidate set together. Interaction effects may reduce coverage (e.g., if two heads are redundant circuits, masking both is no better than masking one).

### 8b.6 Cross-reference to §8

The catastrophic keystone head **L=0 h=5** (§8.2, +100.16 pp) is by construction excluded from the fixing set (filter 2). The sole bootstrap-significant head from §7.4, **L=20 h=11**, may or may not appear in the greedy cover depending on whether it passes filter 2 — see [`minimum_surgical_set.json`](../outputs/head_surgery/minimum_surgical_set.json) for the authoritative list.

---

```

Replace each `{…}` placeholder with the exact value from `outputs/head_surgery/minimum_surgical_set.json` (loaded from Task 8 step 1). Build §8b.3's table by iterating `summary["greedy"]` and accumulating `newly_covered_count` for the Cumulative column.

- [ ] **Step 3: Verify the report renders**

Run: `head -320 docs/head_surgery_report_edited.md | tail -80`
Expected: shows §8b.1 through §8b.6 in order, with all `{…}` placeholders replaced. §9 (Stage E) immediately follows §8b.6.

- [ ] **Step 4: Commit**

```bash
git add docs/head_surgery_report_edited.md
git commit -m "$(cat <<'EOF'
docs(head_surgery): §8b fixing-set analysis in edited report

Adds a new section between §8 (keystone heads) and §9 (decoding
ablation). Reframes the per-head Δ from §7 as a coverage problem over
the ~45 affected utterances. Includes the single-head-masking caveat.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update the milestone-complete log

The milestone log at `logs/head-surgery-diagnosis-complete.md` already notes stages A–G. Add a pointer to the fixing-set analysis so future readers find it.

**Files:**
- Modify: `logs/head-surgery-diagnosis-complete.md`

- [ ] **Step 1: Append a "Post-hoc analyses" section**

Open `logs/head-surgery-diagnosis-complete.md`. Append after the last section:

```markdown

## Post-hoc analyses (no additional GPU)

- **Fixing-set analysis** (2026-04-18) — [log entry](head-surgery-fixing-set-analysis.md), [report §8b](../docs/head_surgery_report_edited.md), artifacts under `outputs/head_surgery/`. Reframes the per-head Δ as a coverage problem; greedy vs ILP min-set-cover over the affected utterances.
```

- [ ] **Step 2: Commit**

```bash
git add logs/head-surgery-diagnosis-complete.md
git commit -m "$(cat <<'EOF'
docs(head_surgery): link fixing-set analysis from milestone log

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review summary (plan author)

**Spec coverage:**

| Spec requirement | Implemented in Task |
|---|---|
| Per-(utterance, head) insertion count | Tasks 2, 3 |
| Identify affected utterances (baseline > 0) | Task 4 |
| Coverage matrix with filter "helps ≥ 1" | Task 4 |
| Coverage matrix with filter "no new harm globally" | Task 4 |
| Coverage matrix with filter "regression_ok / unchecked-pass" | Task 4 |
| Greedy min-set-cover | Task 5 |
| ILP min-set-cover | Task 6 |
| Unhelpable-utterances surfacing | Task 7 step 2 (driver) |
| Three artifacts (CSV, NPZ, JSON) | Task 7 step 2 |
| Run on real data and sanity-check | Task 8 |
| §8b in `docs/head_surgery_report_edited.md` | Task 9 |
| No GPU; pure pandas + numpy + scipy | All tasks — no `torch` import anywhere |
| Integrate with `scripts/head_surgery/` package | Task 1 scaffolding + all subsequent tasks extend one module |
| Log entry per CLAUDE.md convention | Task 8 step 3 |
| Unit tests for set-cover logic on synthetic fixture | Tasks 2–7 each pair failing test → implementation |

**Placeholder scan:** the `{…}` markers in Task 9 Step 2 are explicit instructions to interpolate runtime values from `minimum_surgical_set.json`, not plan placeholders. No `TBD`/`TODO`/`implement later` in the plan body.

**Type consistency:** function signatures used across tasks are internally consistent — `count_insertions(ref, hyp) -> int`, `build_count_table(sweep_csv, baseline_csv) -> DataFrame`, `identify_affected(counts) -> List[str]`, `build_coverage_matrix(counts, head_scores_csv) -> (matrix, utt_ids, heads)`, `greedy_cover(matrix, heads) -> List[((L,h), List[int])]`, `ilp_cover(matrix, heads) -> List[(L,h)]`, `run(sweep_csv, baseline_csv, head_scores_csv) -> dict`. All test fixtures use the same `_fake_sweep_rows()` helper (defined in Task 3, reused in Tasks 4 and 7) — consistent column set.

**Scope:** ten tasks, ~350 LoC, one new module, one new test module, one report edit, two log updates. Fits one engineer-session.
