"""D7 stretch — per-head heatmap of Δ insertion rate on Indian-accent CV."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("docs/head_surgery_heatmap.png")
SCORES_CSV = Path("outputs/head_surgery/head_scores.csv")


def main():
    if not SCORES_CSV.exists():
        raise SystemExit(f"{SCORES_CSV} missing — run Stage D (score_heads) first.")
    scores = pd.read_csv(SCORES_CSV)
    L = int(scores["layer"].max()) + 1
    H = int(scores["head"].max()) + 1
    grid = np.full((L, H), np.nan)
    for _, r in scores.iterrows():
        grid[int(r["layer"]), int(r["head"])] = float(r["delta_insertion_rate"])
    fig, ax = plt.subplots(figsize=(8, 10))
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Decoder layer")
    ax.set_title("Δ insertion rate (baseline − masked) on Indian-accent CV\n"
                 "Red = hallucination-driving; blue = hallucination-suppressing")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Δ insertion rate")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
