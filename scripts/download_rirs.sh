#!/bin/bash
# Download OpenSLR-26 simulated RIRs and hand-pick 1 representative per RT60 band.
# Source: https://www.openslr.org/26/
# Target RT60 bands: ~0.3s (small room), ~0.6s (medium room), ~1.0s (large hall)

set -euo pipefail

PROJECT_DIR="/users/PAS2030/srishti/asr_fairness"
RIR_DIR="${PROJECT_DIR}/data/rirs"
DOWNLOAD_URL="https://www.openslr.org/resources/26/sim_rir_16k.zip"

mkdir -p "${RIR_DIR}"
cd "${RIR_DIR}"

if [ -f "rir_manifest.json" ]; then
    echo "RIR manifest already exists. Skipping download."
    echo "To re-download, delete ${RIR_DIR}/rir_manifest.json"
    exit 0
fi

echo "Downloading OpenSLR-26 simulated RIRs..."
wget -c "${DOWNLOAD_URL}" -O sim_rir_16k.zip

echo "Extracting..."
unzip -o sim_rir_16k.zip -d raw/

echo ""
echo "Analyzing RIRs and selecting representatives per RT60 band..."

python3 << 'PYEOF'
import os
import json
import numpy as np
import soundfile as sf
from pathlib import Path

raw_dir = Path("raw")
selected_dir = Path("selected")
selected_dir.mkdir(exist_ok=True)

def estimate_rt60(rir, sr):
    """Estimate RT60 from a room impulse response using Schroeder's method."""
    # Square and normalize
    rir_sq = rir ** 2
    # Schroeder integration (backward cumulative sum)
    sch = np.cumsum(rir_sq[::-1])[::-1]
    sch_db = 10 * np.log10(sch / sch[0] + 1e-12)
    
    # Find -5dB and -25dB points for T20 estimation (extrapolate to -60dB)
    t5 = np.argmax(sch_db < -5) / sr
    t25 = np.argmax(sch_db < -25) / sr
    
    if t25 > t5 and t5 > 0:
        rt60 = 3 * (t25 - t5)  # Extrapolate T20 -> T60
    else:
        # Fallback: find -60dB point directly
        t60 = np.argmax(sch_db < -60) / sr
        rt60 = t60 if t60 > 0 else 0.0
    
    return rt60

# Scan all RIR files
rir_catalog = []
for wav_path in sorted(raw_dir.rglob("*.wav")):
    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data[:, 0]
        if sr != 16000:
            continue  # Skip non-16kHz (shouldn't happen for this dataset)
        rt60 = estimate_rt60(data, sr)
        rir_catalog.append({
            "path": str(wav_path.resolve()),
            "rt60_estimated": round(rt60, 3),
            "length_samples": len(data),
            "peak_position": int(np.argmax(np.abs(data))),
        })
    except Exception as e:
        print(f"  Skipping {wav_path}: {e}")

print(f"Scanned {len(rir_catalog)} RIR files")

# Sort by RT60
rir_catalog.sort(key=lambda x: x["rt60_estimated"])

# Define target RT60 bands
targets = {
    "0.3s": (0.2, 0.4),   # Small room
    "0.6s": (0.5, 0.7),   # Medium room
    "1.0s": (0.85, 1.2),  # Large hall
}

selected = {}
for label, (lo, hi) in targets.items():
    candidates = [r for r in rir_catalog if lo <= r["rt60_estimated"] <= hi]
    if not candidates:
        # Relax bounds
        candidates = sorted(rir_catalog, key=lambda r: abs(r["rt60_estimated"] - (lo + hi) / 2))
        candidates = candidates[:5]
    
    # Pick the one closest to the band center
    target_center = (lo + hi) / 2
    best = min(candidates, key=lambda r: abs(r["rt60_estimated"] - target_center))
    
    # Copy to selected/
    src = best["path"]
    dst = str(selected_dir / f"rir_rt60_{label.replace('.', '_')}.wav")
    data, sr = sf.read(src)
    if data.ndim > 1:
        data = data[:, 0]
    sf.write(dst, data, 16000)
    
    selected[label] = {
        "path": os.path.abspath(dst),
        "rt60_estimated": best["rt60_estimated"],
        "source_path": src,
        "length_samples": best["length_samples"],
    }
    print(f"  RT60 {label}: selected {os.path.basename(src)} (measured RT60={best['rt60_estimated']:.3f}s)")

# Write manifest
manifest = {
    "description": "Hand-picked RIRs for ASR perturbation experiments, 1 per RT60 band",
    "rt60_bands": selected,
    "total_scanned": len(rir_catalog),
    "rt60_distribution": {
        "min": rir_catalog[0]["rt60_estimated"],
        "max": rir_catalog[-1]["rt60_estimated"],
        "median": rir_catalog[len(rir_catalog)//2]["rt60_estimated"],
    }
}

with open("rir_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written to rir_manifest.json")
print(f"Selected RIRs in: {selected_dir.resolve()}/")
PYEOF

# Clean up raw files (keep only selected)
echo ""
read -p "Delete raw RIR files and zip? Keep only selected/ [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf raw/ sim_rir_16k.zip
    echo "  Cleaned up raw files"
fi

echo ""
echo "Done! RIRs ready at: ${RIR_DIR}/selected/"
