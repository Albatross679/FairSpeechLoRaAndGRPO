#!/usr/bin/env python3
"""
Parse bootstrap CIs from SLURM log file and save to structured JSON.

Parses both Fair-Speech and Common Voice analysis sections from:
  logs/fs_fs_analysis_cpu_4065109.out

Output: results/bootstrap_cis.json
"""

import re
import json
from pathlib import Path

LOG_PATH = Path("/users/PAS2030/srishti/asr_fairness/logs/fs_fs_analysis_cpu_4065109.out")
OUT_PATH = Path("/users/PAS2030/srishti/asr_fairness/results/bootstrap_cis.json")

# Regex for WER lines:
#   GroupName              : WER=X.XX% [Y.YY%, Z.ZZ%] (n=N,NNN)
WER_RE = re.compile(
    r"^\s+"
    r"(?P<group>.+?)"
    r"\s+:\s+WER=(?P<wer>\d+\.\d+)%"
    r"\s+\[(?P<ci_low>\d+\.\d+)%,\s*(?P<ci_high>\d+\.\d+)%\]"
    r"\s+\(n=(?P<n>[\d,]+)\)"
)

# Regex for skipping lines: "GroupName: N samples (< 50, skipping)"
SKIP_RE = re.compile(r"^\s+.+:\s+\d+\s+samples\s+\(<\s*\d+,\s*skipping\)")

# Regex for model header: "Analyzing: model-name"
MODEL_RE = re.compile(r"^Analyzing:\s+(.+)$")

# Regex for demographic axis: "Analyzing demographic_axis..."
AXIS_RE = re.compile(r"^\s+Analyzing\s+(\w+)\.\.\.$")


def parse_log(log_path: Path) -> dict:
    """Parse the entire log file and return structured data."""
    lines = log_path.read_text().splitlines()

    result = {"fairspeech": {}, "commonvoice": {}}

    # Determine section boundaries
    # Fair-Speech: from start until "Step 2:" or "Step 5:"
    # Common Voice: from "Step 5:" until end
    section = None  # will be set to 'fairspeech' or 'commonvoice'
    current_model = None
    current_axis = None

    for i, line in enumerate(lines):
        # Detect section transitions
        if "Step 1: Computing fairness metrics on Fair-Speech" in line:
            section = "fairspeech"
            continue
        if "Step 5: Recomputing CV metrics" in line:
            section = "commonvoice"
            current_model = None
            current_axis = None
            continue
        # Stop parsing bootstrap CIs at non-model-analysis sections
        if any(x in line for x in [
            "Step 2:", "Step 3:", "Step 4:",
            "H1-c: Black/AA vs White Gap Analysis",
            "Generating outputs...",
        ]):
            # These sections don't contain bootstrap CIs in the model format
            current_model = None
            current_axis = None
            continue

        if section is None:
            continue

        # Check for model header
        m = MODEL_RE.search(line)
        if m:
            current_model = m.group(1).strip()
            current_axis = None
            if current_model not in result[section]:
                result[section][current_model] = {}
            continue

        if current_model is None:
            continue

        # Check for axis header
        m = AXIS_RE.match(line)
        if m:
            current_axis = m.group(1).strip()
            if current_axis not in result[section][current_model]:
                result[section][current_model][current_axis] = {}
            continue

        if current_axis is None:
            continue

        # Skip "skipping" lines
        if SKIP_RE.match(line):
            continue

        # Skip pairwise test lines and their results
        if "Pairwise tests:" in line:
            current_axis = None  # stop parsing this axis (pairwise lines follow)
            continue

        # Check for WER line
        m = WER_RE.match(line)
        if m:
            group_name = m.group("group").strip()
            n_str = m.group("n").replace(",", "")
            entry = {
                "wer": round(float(m.group("wer")) / 100, 6),
                "ci_low": round(float(m.group("ci_low")) / 100, 6),
                "ci_high": round(float(m.group("ci_high")) / 100, 6),
                "n": int(n_str),
            }
            result[section][current_model][current_axis][group_name] = entry

    return result


def check_ci_overlap(data: dict) -> None:
    """Check Black/AA vs White CI overlap for all 9 models on Fair-Speech ethnicity."""
    print("\n" + "=" * 70)
    print("CI OVERLAP CHECK: Black/AA vs White (Fair-Speech Ethnicity)")
    print("=" * 70)
    print(f"{'Model':<30} {'Black/AA CI':<24} {'White CI':<24} {'Overlap?':<10} {'Gap'}")
    print("-" * 110)

    fs = data.get("fairspeech", {})
    for model in sorted(fs.keys()):
        eth = fs[model].get("ethnicity", {})
        black = eth.get("Black/AA")
        white = eth.get("White")
        if black is None or white is None:
            print(f"{model:<30} MISSING DATA")
            continue

        b_lo, b_hi = black["ci_low"], black["ci_high"]
        w_lo, w_hi = white["ci_low"], white["ci_high"]

        # CIs overlap if max(lo) < min(hi)
        overlaps = max(b_lo, w_lo) < min(b_hi, w_hi)
        if overlaps:
            overlap_str = "YES"
            gap_str = "overlapping"
        else:
            overlap_str = "NO"
            gap_pct = (min(b_lo, w_lo) - max(b_hi, w_hi))  # negative = no overlap gap
            # gap between closest edges
            gap_val = max(b_lo, w_lo) - min(b_hi, w_hi)
            gap_str = f"gap = {gap_val*100:.2f} pp"

        print(
            f"{model:<30} "
            f"[{b_lo*100:5.2f}%, {b_hi*100:5.2f}%]   "
            f"[{w_lo*100:5.2f}%, {w_hi*100:5.2f}%]   "
            f"{overlap_str:<10} "
            f"{gap_str}"
        )


def print_summary(data: dict) -> None:
    """Print summary of extracted data."""
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    total_models = 0
    total_axes = 0
    total_groups = 0

    for dataset in ["fairspeech", "commonvoice"]:
        ds_data = data.get(dataset, {})
        n_models = len(ds_data)
        total_models += n_models
        print(f"\n  {dataset}:")
        print(f"    Models: {n_models}")
        for model in sorted(ds_data.keys()):
            axes = ds_data[model]
            n_axes = len(axes)
            total_axes += n_axes
            groups_per_axis = []
            for axis_name, groups in sorted(axes.items()):
                n_groups = len(groups)
                total_groups += n_groups
                groups_per_axis.append(f"{axis_name}({n_groups})")
            print(f"      {model}: {', '.join(groups_per_axis)}")

    print(f"\n  TOTALS: {total_models} models, {total_axes} model-axis combos, {total_groups} group entries")


def main():
    print(f"Parsing log: {LOG_PATH}")
    data = parse_log(LOG_PATH)

    # Save JSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to: {OUT_PATH}")

    # Print summary
    print_summary(data)

    # CI overlap check
    check_ci_overlap(data)


if __name__ == "__main__":
    main()
