#!/bin/bash
# Download MUSAN noise subset for perturbation experiments.
# Source: https://www.openslr.org/17/
# Only extracts the noise/ subset (~930 files, ~2GB after extraction).

set -euo pipefail

PROJECT_DIR="/users/PAS2030/srishti/asr_fairness"
MUSAN_DIR="${PROJECT_DIR}/data/musan"
DOWNLOAD_URL="https://www.openslr.org/resources/17/musan.tar.gz"

mkdir -p "${MUSAN_DIR}"
cd "${MUSAN_DIR}"

if [ -d "noise" ] && [ "$(find noise -name '*.wav' | wc -l)" -gt 800 ]; then
    echo "MUSAN noise subset already exists with $(find noise -name '*.wav' | wc -l) files. Skipping download."
    exit 0
fi

echo "Downloading MUSAN from ${DOWNLOAD_URL}..."
echo "  (This is ~11GB compressed — may take a while)"

# Download
wget -c "${DOWNLOAD_URL}" -O musan.tar.gz

# Extract only the noise subset
echo "Extracting noise subset..."
tar xzf musan.tar.gz --wildcards 'musan/noise/*' --strip-components=1

echo ""
NOISE_COUNT=$(find noise -name '*.wav' | wc -l)
echo "Extracted ${NOISE_COUNT} noise files to ${MUSAN_DIR}/noise/"

# Verify sample rate and convert if needed
echo "Checking sample rates..."
NEEDS_RESAMPLE=0
for f in $(find noise -name '*.wav' | head -5); do
    SR=$(soxi -r "$f" 2>/dev/null || python3 -c "import soundfile; print(soundfile.info('$f').samplerate)")
    if [ "$SR" != "16000" ]; then
        NEEDS_RESAMPLE=1
        echo "  Found non-16kHz file ($SR Hz): $f"
        break
    fi
done

if [ "$NEEDS_RESAMPLE" -eq 1 ]; then
    echo "Resampling all noise files to 16kHz mono..."
    python3 << 'PYEOF'
import os
import soundfile as sf
import numpy as np

noise_dir = "noise"
count = 0
for root, dirs, files in os.walk(noise_dir):
    for fname in files:
        if not fname.endswith('.wav'):
            continue
        fpath = os.path.join(root, fname)
        data, sr = sf.read(fpath)
        needs_write = False
        # Convert to mono
        if data.ndim > 1:
            data = data.mean(axis=1)
            needs_write = True
        # Resample if needed
        if sr != 16000:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            except ImportError:
                import torchaudio
                import torch
                t = torch.from_numpy(data).unsqueeze(0).float()
                t = torchaudio.transforms.Resample(sr, 16000)(t)
                data = t.squeeze(0).numpy()
            needs_write = True
        if needs_write:
            sf.write(fpath, data, 16000)
            count += 1
print(f"  Resampled {count} files to 16kHz mono")
PYEOF
else
    echo "  All files already 16kHz — no resampling needed."
fi

# Categorize noise files by type for stratification
echo "Categorizing noise files by type..."
python3 << 'PYEOF'
import os
import json

noise_dir = "noise"
categories = {}
for root, dirs, files in os.walk(noise_dir):
    for fname in files:
        if not fname.endswith('.wav'):
            continue
        fpath = os.path.join(root, fname)
        # MUSAN noise is organized as noise/free-sound/ and noise/sound-bible/
        # Subcategories are encoded in directory structure
        parts = os.path.relpath(fpath, noise_dir).split(os.sep)
        source = parts[0] if len(parts) > 1 else "unknown"
        if source not in categories:
            categories[source] = []
        categories[source].append(os.path.abspath(fpath))

manifest = {
    "total_files": sum(len(v) for v in categories.values()),
    "sources": {k: len(v) for k, v in categories.items()},
    "files_by_source": categories,
}
manifest_path = os.path.join(os.path.dirname(noise_dir), "noise_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  Created noise manifest: {manifest_path}")
for src, files in categories.items():
    print(f"    {src}: {len(files)} files")
PYEOF

# Clean up tarball
echo ""
read -p "Delete musan.tar.gz (~11GB) to save space? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f musan.tar.gz
    echo "  Deleted musan.tar.gz"
fi

echo ""
echo "Done! MUSAN noise subset ready at: ${MUSAN_DIR}/noise/"
