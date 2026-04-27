"""Thin wrapper around scripts/analysis/whisper_hallucination_analysis.py (T4).

Exposes:
  categorize_insertions(ref, hyp) -> list[{"word", "category"}]
    where category ∈ {"repetition", "syntactic_completion", "content_hallucination"}
  insertion_rate_breakdown(ref_hyp_pairs) -> dict(total, repetition, syntactic, content)
    where each value is (#insertions / total_ref_words).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

from scripts.analysis.whisper_hallucination_analysis import (
    extract_and_categorize_insertions,
)


def categorize_insertions(ref: str, hyp: str) -> List[dict]:
    return extract_and_categorize_insertions(ref, hyp)


def insertion_rate_breakdown(pairs: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    total_ref_words = 0
    counts = Counter()
    for ref, hyp in pairs:
        ref = (ref or "").strip()
        hyp = (hyp or "").strip()
        if not ref:
            continue
        total_ref_words += len(ref.split())
        for ins in categorize_insertions(ref, hyp):
            counts[ins["category"]] += 1
            counts["total"] += 1
    if total_ref_words == 0:
        return {"total": 0.0, "repetition": 0.0, "syntactic": 0.0, "content": 0.0,
                "total_ref_words": 0}
    return {
        "total": counts["total"] / total_ref_words,
        "repetition": counts["repetition"] / total_ref_words,
        "syntactic": counts["syntactic_completion"] / total_ref_words,
        "content": counts["content_hallucination"] / total_ref_words,
        "total_ref_words": total_ref_words,
    }
