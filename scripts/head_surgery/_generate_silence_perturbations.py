"""One-shot: generate silence-injected audio for Stage F VAD arm.

Reads datasets/cv_test_manifest_indian.csv, for each clip creates three
perturbed versions — silence_25pct, silence_50pct, silence_75pct —
stored as 16 kHz WAV under datasets/silence_perturbations/. Writes
datasets/cv_silence_manifest.csv in the schema run_vad_arm expects.

Silence injection strategy: take the 16 kHz waveform and insert blocks
of zeros whose total length equals `fraction * len(audio)`, split across
`num_insertions` random positions. The utterance's reference transcript
is unchanged — the goal is to see whether the decoder hallucinates
during the silent stretches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, "/workspace/project")
from scripts.head_surgery.run_diagnosis_sweep import load_audio_16k

OUT_DIR = Path("/workspace/project/datasets/silence_perturbations")
MANIFEST_IN = Path("/workspace/project/datasets/cv_test_manifest_indian.csv")
MANIFEST_OUT = Path("/workspace/project/datasets/cv_silence_manifest.csv")

SEVERITIES = {
    "silence_25pct": 0.25,
    "silence_50pct": 0.50,
    "silence_75pct": 0.75,
}
NUM_INSERTIONS = 3  # split the silence into 3 blocks per severity
SEED = 20260418


def inject_silence(audio: np.ndarray, fraction: float, rng: np.random.Generator,
                   n_blocks: int = NUM_INSERTIONS) -> np.ndarray:
    """Insert `fraction * len(audio)` of silence as n_blocks blocks at random positions.

    The original audio is preserved in order; silence blocks are inserted between
    random non-block positions so the reference transcript stays valid.
    """
    if fraction <= 0 or len(audio) == 0:
        return audio
    total_silence = int(len(audio) * fraction)
    # Split the total silence length into n_blocks positive-integer pieces.
    # Use a Dirichlet-like split: draw n_blocks uniform numbers, normalize.
    props = rng.uniform(0.3, 1.0, size=n_blocks)
    props /= props.sum()
    block_lens = (props * total_silence).astype(int)
    block_lens[-1] = total_silence - block_lens[:-1].sum()  # fix rounding
    # Pick insertion points (sorted) inside the original audio (not at the edges).
    N = len(audio)
    insert_pts = sorted(rng.integers(1, N, size=n_blocks).tolist())
    # Build output: interleave original segments with silence blocks.
    segments = []
    prev = 0
    for pt, bl in zip(insert_pts, block_lens):
        segments.append(audio[prev:pt])
        segments.append(np.zeros(bl, dtype=audio.dtype))
        prev = pt
    segments.append(audio[prev:])
    return np.concatenate(segments)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST_IN, dtype=str, keep_default_na=False)
    print(f"source manifest: {len(df)} clips")

    rng = np.random.default_rng(SEED)
    rows = []
    for i, row in df.iterrows():
        audio = load_audio_16k(row["audio_path"])
        base_id = row["utterance_id"]
        for sev_name, frac in SEVERITIES.items():
            perturbed = inject_silence(audio, frac, rng)
            out_path = OUT_DIR / f"{base_id}__{sev_name}.wav"
            sf.write(str(out_path), perturbed, 16000)
            rows.append({
                "severity": sev_name,
                "audio_path": str(out_path),
                "reference": row["sentence"],
                "utterance_id": base_id,
                "accent": row.get("accent", "indian"),
            })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}] generated", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(MANIFEST_OUT, index=False)
    print(f"wrote {MANIFEST_OUT} ({len(out_df)} rows across {len(SEVERITIES)} severities)")


if __name__ == "__main__":
    main()
