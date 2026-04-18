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
