"""Frozen reproducibility contract for v2.0 head-surgery diagnosis (T9).

All numbers derived from / compared against the midterm must reference these
constants. Do not edit without re-running Gate G1 (see
tasks/prd-head-surgery-diagnosis.md §6).

NOTE on CV corpus version: the project originally targeted Common Voice 24
(midterm baseline 9.62% insertion rate). CV24 is unavailable in this
environment, so the pipeline runs on CV25. The 510-utterance Indian-accent
subset is CV25's strict single-label match for
"india and south asia (india, pakistan, sri lanka)". Gate G1 is redefined to
establish the CV25 baseline rather than reproduce midterm.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_ID = "openai/whisper-large-v3"
# Pinned HuggingFace revision for reproducibility.
MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"

# ── Determinism ───────────────────────────────────────────────────────────
SEED = 20260417

# ── Generation config ──────────────────────────────────────────────────────
GENERATE_CONFIG = {
    "max_new_tokens": 440,
    "language": "en",
    "task": "transcribe",
    "num_beams": 1,
    "do_sample": False,
    "temperature": 0.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
}

# ── Evaluation subset ──────────────────────────────────────────────────────
_IDS_JSON = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "head_surgery" / "indian_accent_ids.json"
)

EXPECTED_N_INDIAN_ACCENT_IDS = 484  # CV25 strict single-label match, intersected with on-disk clips
# NOTE: the initial CV25 strict filter yielded 510 IDs; 26 are absent because the
# B2 tarball was truncated during extraction. See indian_accent_ids.json "note" field.


@lru_cache(maxsize=1)
def _load_indian_accent_ids_cached() -> Tuple[str, ...]:
    payload = json.loads(_IDS_JSON.read_text())
    ids = sorted(payload["ids"])
    if len(ids) != EXPECTED_N_INDIAN_ACCENT_IDS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_INDIAN_ACCENT_IDS} Indian-accent IDs "
            f"(CV25 strict single-label snapshot); got {len(ids)}. "
            f"Refresh the fixture from datasets/cv-corpus-25.0-2026-03-09/en/test.tsv."
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError(
            f"Indian-accent IDs are not unique: {len(ids)} total, {len(set(ids))} distinct."
        )
    return tuple(ids)


def load_indian_accent_ids() -> List[str]:
    """CV25 Indian-accent utterance IDs (sorted, T9 snapshot).

    Returns a fresh list each call; the cache stores an immutable tuple, so
    callers cannot corrupt the frozen snapshot by mutating the returned list.
    """
    return list(_load_indian_accent_ids_cached())


# ── Whisper-large-v3 architecture constants ────────────────────────────────
NUM_DECODER_LAYERS = 32
NUM_DECODER_SELF_ATTN_HEADS = 20
HEAD_DIM = 64  # 1280 hidden / 20 heads


# ── Gate tolerances ────────────────────────────────────────────────────────
# NOTE: G1 is redefined for CV25 — see module docstring.
GATE_G1_MIDTERM_INSERTION_RATE = 0.0962
GATE_G1_TOLERANCE_PP = 0.005
GATE_G3_WER_TOLERANCE = 1e-4
GATE_G5_BASELINE_WER_TOLERANCE_PP = 0.002

# ── Regression guard ───────────────────────────────────────────────────────
REGRESSION_BUDGET_PP = 0.005
REGRESSION_GUARD_TOP_K = 50

# ── Scoring ────────────────────────────────────────────────────────────────
BOOTSTRAP_ITERATIONS = 10_000
TOP_K_FOR_REPORT = 10
