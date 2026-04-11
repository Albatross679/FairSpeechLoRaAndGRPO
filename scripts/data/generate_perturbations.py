#!/usr/bin/env python3
"""
Generate perturbed audio for ASR fairness perturbation experiments.

Perturbation types:
  1. Additive noise (MUSAN) at SNR 20/10/0 dB
  2. Reverberation (RIR convolution) at RT60 0.3/0.6/1.0s
  3. Silence injection at 25%/50%/75% of utterance duration
  4. Partial audio masking at 10%/20%/30%

Usage:
    # Generate all perturbations for Fair-Speech:
    python scripts/data/generate_perturbations.py --dataset fs

    # Single perturbation+level for Common Voice:
    python scripts/data/generate_perturbations.py --dataset cv --perturbation snr_20db

    # Dry run (no audio written):
    python scripts/data/generate_perturbations.py --dataset fs --dry-run --max-samples 10
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_DIR = Path("/users/PAS2030/srishti/asr_fairness")
DATA_DIR = PROJECT_DIR / "data"
PERTURBED_DIR = PROJECT_DIR / "perturbed_audio"
METADATA_DIR = PERTURBED_DIR / "metadata"

MANIFESTS = {
    "cv": DATA_DIR / "cv_test_manifest.csv",
    "fs": DATA_DIR / "fs_manifest.csv",
}

DATASET_NAMES = {
    "cv": "common_voice",
    "fs": "fair_speech",
}

SAMPLE_RATE = 16000

# ── All 12 perturbation conditions ────────────────────────────────────────
PERTURBATION_CONFIGS = {
    # Additive noise
    "snr_20db": {"type": "noise", "snr_db": 20},
    "snr_10db": {"type": "noise", "snr_db": 10},
    "snr_0db":  {"type": "noise", "snr_db": 0},
    # Reverberation
    "reverb_0.3s": {"type": "reverb", "rt60": "0.3s"},
    "reverb_0.6s": {"type": "reverb", "rt60": "0.6s"},
    "reverb_1.0s": {"type": "reverb", "rt60": "1.0s"},
    # Silence injection
    "silence_25pct": {"type": "silence", "fraction": 0.25},
    "silence_50pct": {"type": "silence", "fraction": 0.50},
    "silence_75pct": {"type": "silence", "fraction": 0.75},
    # Partial masking
    "mask_10pct": {"type": "masking", "fraction": 0.10, "num_chunks": 2},
    "mask_20pct": {"type": "masking", "fraction": 0.20, "num_chunks": 3},
    "mask_30pct": {"type": "masking", "fraction": 0.30, "num_chunks": 4},
}


# ═════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file, resample to target sr, return 1D numpy array."""
    data, orig_sr = sf.read(path)

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if orig_sr != sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
        except ImportError:
            import torchaudio
            import torch
            t = torch.from_numpy(data).unsqueeze(0).float()
            t = torchaudio.transforms.Resample(orig_sr, sr)(t)
            data = t.squeeze(0).numpy()

    return data.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# NOISE MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

class NoisePool:
    """Manages MUSAN noise files with stratification by source type."""

    def __init__(self, musan_dir: Path):
        manifest_path = musan_dir / "noise_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Noise manifest not found at {manifest_path}.\n"
                f"Run: bash scripts/download_musan.sh"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.files_by_source = manifest["files_by_source"]
        self.all_files = []
        for source, files in self.files_by_source.items():
            self.all_files.extend(files)

        self.sources = list(self.files_by_source.keys())
        print(f"  NoisePool: {len(self.all_files)} files across {len(self.sources)} sources: "
              f"{', '.join(f'{s}({len(fs)})' for s, fs in self.files_by_source.items())}")

        # Pre-load all noise files into memory for speed (~2GB)
        self._cache = {}

    def get_noise(self, utt_id: str, required_length: int) -> np.ndarray:
        """Get a deterministic noise segment for a given utterance.

        Uses hash(utt_id) for deterministic selection within a stratified pool:
        - First, select the noise source (alternating by hash)
        - Then, select a file within that source
        """
        h = int(hashlib.md5(utt_id.encode()).hexdigest(), 16)

        # Stratify: pick source, then file within source
        source_idx = h % len(self.sources)
        source = self.sources[source_idx]
        source_files = self.files_by_source[source]
        file_idx = (h // len(self.sources)) % len(source_files)
        noise_path = source_files[file_idx]

        # Load (with cache)
        if noise_path not in self._cache:
            data, sr = sf.read(noise_path)
            if data.ndim > 1:
                data = data[:, 0]
            self._cache[noise_path] = data.astype(np.float32)
        noise = self._cache[noise_path]

        # Loop noise if shorter than speech
        if len(noise) < required_length:
            repeats = required_length // len(noise) + 1
            noise = np.tile(noise, repeats)

        # Deterministic offset
        max_offset = len(noise) - required_length
        offset = h % (max_offset + 1) if max_offset > 0 else 0
        noise = noise[offset:offset + required_length]

        return noise


# ═════════════════════════════════════════════════════════════════════════════
# RIR MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

class RIRPool:
    """Manages hand-picked RIRs."""

    def __init__(self, rir_dir: Path):
        manifest_path = rir_dir / "rir_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"RIR manifest not found at {manifest_path}.\n"
                f"Run: bash scripts/download_rirs.sh"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.rirs = {}
        for rt60_label, info in manifest["rt60_bands"].items():
            data, sr = sf.read(info["path"])
            if data.ndim > 1:
                data = data[:, 0]
            # Normalize RIR peak to 1
            data = data / np.max(np.abs(data))
            self.rirs[rt60_label] = data.astype(np.float32)
            print(f"  RIR {rt60_label}: {len(data)} samples, measured RT60={info['rt60_estimated']:.3f}s")

    def get_rir(self, rt60_label: str) -> np.ndarray:
        return self.rirs[rt60_label]


# ═════════════════════════════════════════════════════════════════════════════
# PERTURBATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def add_noise_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Add noise to speech at a specified SNR level."""
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10 or speech_power < 1e-10:
        return speech

    target_snr_linear = 10 ** (target_snr_db / 10)
    noise_scale = np.sqrt(speech_power / (target_snr_linear * noise_power))

    noisy = speech + noise_scale * noise

    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val

    return noisy


def apply_reverb(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve speech with a room impulse response, energy-normalized."""
    reverbed = fftconvolve(speech, rir, mode='full')[:len(speech)]

    # Match original energy
    original_energy = np.sqrt(np.mean(speech ** 2))
    reverbed_energy = np.sqrt(np.mean(reverbed ** 2))
    if reverbed_energy > 1e-10:
        reverbed = reverbed * (original_energy / reverbed_energy)

    # Prevent clipping
    max_val = np.max(np.abs(reverbed))
    if max_val > 1.0:
        reverbed = reverbed / max_val

    return reverbed


def inject_silence(speech: np.ndarray, sr: int, silence_fraction: float,
                   utt_id: str) -> tuple:
    """Insert proportional silence gap at a deterministic position.

    Returns: (modified_speech, insertion_point_sec, silence_duration_sec)
    """
    h = int(hashlib.md5(utt_id.encode()).hexdigest(), 16)
    rng = np.random.RandomState(h % (2**31) + 42)

    utterance_duration_sec = len(speech) / sr
    silence_duration_sec = utterance_duration_sec * silence_fraction
    silence_samples = int(silence_duration_sec * sr)
    silence = np.zeros(silence_samples, dtype=np.float32)

    # Position: random between 0.3-0.7 of utterance
    position_fraction = rng.uniform(0.3, 0.7)
    insertion_point = int(len(speech) * position_fraction)

    # Apply short fade (5ms) to avoid click artifacts
    fade_len = int(0.005 * sr)
    speech = speech.copy()

    if fade_len > 0 and insertion_point > fade_len:
        fade_out = np.linspace(1, 0, fade_len).astype(np.float32)
        fade_in = np.linspace(0, 1, fade_len).astype(np.float32)
        speech[insertion_point - fade_len:insertion_point] *= fade_out
        if insertion_point + fade_len < len(speech):
            speech[insertion_point:insertion_point + fade_len] *= fade_in

    # Insert silence
    result = np.concatenate([
        speech[:insertion_point],
        silence,
        speech[insertion_point:]
    ])

    return result, insertion_point / sr, silence_duration_sec


def mask_audio_chunks(speech: np.ndarray, sr: int, mask_fraction: float,
                      num_chunks: int, utt_id: str) -> tuple:
    """Zero out random contiguous chunks of audio.

    Returns: (masked_speech, mask_regions_sec)
    """
    h = int(hashlib.md5(utt_id.encode()).hexdigest(), 16)
    rng = np.random.RandomState(h % (2**31) + 42)

    total_mask_samples = int(len(speech) * mask_fraction)
    chunk_size = total_mask_samples // num_chunks

    masked = speech.copy()
    mask_regions = []
    fade_len = int(0.005 * sr)

    # Generate non-overlapping chunk positions
    used = np.zeros(len(speech), dtype=bool)

    for _ in range(num_chunks):
        # Find valid start positions
        valid_starts = []
        margin = fade_len + 1
        for s in range(margin, len(speech) - chunk_size - margin, max(1, chunk_size // 4)):
            if not np.any(used[max(0, s - fade_len):min(len(speech), s + chunk_size + fade_len)]):
                valid_starts.append(s)

        if not valid_starts:
            break

        start = rng.choice(valid_starts)
        end = start + chunk_size

        # Apply fades
        if start >= fade_len:
            fade_out = np.linspace(1, 0, fade_len).astype(np.float32)
            masked[start - fade_len:start] *= fade_out
        masked[start:end] = 0.0
        if end + fade_len <= len(speech):
            fade_in = np.linspace(0, 1, fade_len).astype(np.float32)
            masked[end:end + fade_len] *= fade_in

        used[start:end] = True
        mask_regions.append((round(start / sr, 4), round(end / sr, 4)))

    return masked, mask_regions


# ═════════════════════════════════════════════════════════════════════════════
# MANIFEST GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_perturbed_manifest(original_manifest_path: Path, perturbation: str,
                                 perturbed_audio_dir: Path, dataset: str) -> Path:
    """Create a perturbed manifest CSV with updated audio paths."""
    import pandas as pd

    df = pd.read_csv(original_manifest_path)
    # Update audio paths to point to perturbed versions
    perturbed_paths = []
    for _, row in df.iterrows():
        utt_id = row["utterance_id"]
        perturbed_path = perturbed_audio_dir / f"{utt_id}.wav"
        perturbed_paths.append(str(perturbed_path))
    df["audio_path"] = perturbed_paths

    manifest_path = perturbed_audio_dir / f"{DATASET_NAMES[dataset]}_{perturbation}_manifest.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def process_perturbation(manifest_df, perturbation_name: str, config: dict,
                         output_dir: Path, noise_pool=None, rir_pool=None,
                         dry_run: bool = False, max_samples: int = -1):
    """Apply one perturbation at one level to all utterances."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(manifest_df) if max_samples < 0 else min(max_samples, len(manifest_df))
    ptype = config["type"]

    silence_metadata = []
    mask_metadata = []

    skipped = 0
    for i in tqdm(range(n), desc=perturbation_name, unit="utt"):
        row = manifest_df.iloc[i]
        utt_id = str(row["utterance_id"])
        audio_path = row["audio_path"]
        output_path = output_dir / f"{utt_id}.wav"

        # Skip if already exists
        if output_path.exists():
            continue

        # Load audio
        try:
            speech = load_audio(audio_path)
        except Exception as e:
            print(f"  WARNING: Failed to load {audio_path}: {e}")
            skipped += 1
            continue

        if len(speech) < SAMPLE_RATE * 0.1:  # Skip <0.1s utterances
            skipped += 1
            continue

        # Apply perturbation
        if ptype == "noise":
            noise = noise_pool.get_noise(utt_id, len(speech))
            perturbed = add_noise_at_snr(speech, noise, config["snr_db"])

        elif ptype == "reverb":
            rir = rir_pool.get_rir(config["rt60"])
            perturbed = apply_reverb(speech, rir)

        elif ptype == "silence":
            perturbed, ins_point, sil_dur = inject_silence(
                speech, SAMPLE_RATE, config["fraction"], utt_id
            )
            silence_metadata.append({
                "utt_id": utt_id,
                "silence_fraction": config["fraction"],
                "actual_silence_sec": round(sil_dur, 4),
                "insertion_point_sec": round(ins_point, 4),
                "original_duration_sec": round(len(speech) / SAMPLE_RATE, 4),
                "perturbed_duration_sec": round(len(perturbed) / SAMPLE_RATE, 4),
            })

        elif ptype == "masking":
            perturbed, regions = mask_audio_chunks(
                speech, SAMPLE_RATE, config["fraction"], config["num_chunks"], utt_id
            )
            mask_metadata.append({
                "utt_id": utt_id,
                "mask_fraction": config["fraction"],
                "num_chunks": config["num_chunks"],
                "regions": regions,
                "original_duration_sec": round(len(speech) / SAMPLE_RATE, 4),
            })

        else:
            raise ValueError(f"Unknown perturbation type: {ptype}")

        # Write perturbed audio
        if not dry_run:
            sf.write(str(output_path), perturbed, SAMPLE_RATE)

    # Write metadata
    if silence_metadata and not dry_run:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = METADATA_DIR / f"silence_insertion_points_{perturbation_name}.jsonl"
        with open(meta_path, "w") as f:
            for entry in silence_metadata:
                f.write(json.dumps(entry) + "\n")
        print(f"  Silence metadata: {meta_path} ({len(silence_metadata)} entries)")

    if mask_metadata and not dry_run:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = METADATA_DIR / f"mask_regions_{perturbation_name}.jsonl"
        with open(meta_path, "w") as f:
            for entry in mask_metadata:
                f.write(json.dumps(entry) + "\n")
        print(f"  Mask metadata: {meta_path} ({len(mask_metadata)} entries)")

    if skipped > 0:
        print(f"  Skipped {skipped} utterances (missing/short audio)")

    return n - skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbed audio for ASR fairness experiments"
    )
    parser.add_argument("--dataset", required=True, choices=["cv", "fs"],
                        help="Dataset: cv (Common Voice) or fs (Fair-Speech)")
    parser.add_argument("--perturbation", type=str, default=None,
                        choices=list(PERTURBATION_CONFIGS.keys()),
                        help="Single perturbation to generate (default: all)")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Max utterances to process (-1 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write audio files, just validate")
    args = parser.parse_args()

    import pandas as pd

    dataset_name = DATASET_NAMES[args.dataset]
    manifest_path = MANIFESTS[args.dataset]

    print(f"\n{'='*60}")
    print(f"Perturbation Audio Generation")
    print(f"{'='*60}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output:   {PERTURBED_DIR / dataset_name}")
    print(f"{'='*60}\n")

    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Manifest: {len(df):,} utterances")

    if args.max_samples > 0:
        df = df.head(args.max_samples)
        print(f"  Limited to {args.max_samples} samples")

    # Determine which perturbations to run
    if args.perturbation:
        perturbations = {args.perturbation: PERTURBATION_CONFIGS[args.perturbation]}
    else:
        perturbations = PERTURBATION_CONFIGS

    # Lazy-load noise and RIR pools
    noise_pool = None
    rir_pool = None

    needs_noise = any(c["type"] == "noise" for c in perturbations.values())
    needs_rir = any(c["type"] == "reverb" for c in perturbations.values())

    if needs_noise:
        print("\nLoading MUSAN noise pool...")
        musan_dir = DATA_DIR / "musan"
        noise_pool = NoisePool(musan_dir)

    if needs_rir:
        print("\nLoading RIR pool...")
        rir_dir = DATA_DIR / "rirs"
        rir_pool = RIRPool(rir_dir)

    # Process each perturbation
    start_time = time.time()
    for pert_name, config in perturbations.items():
        output_dir = PERTURBED_DIR / dataset_name / pert_name
        print(f"\n{'─'*40}")
        print(f"Perturbation: {pert_name} ({config})")
        print(f"Output: {output_dir}")
        print(f"{'─'*40}")

        n_processed = process_perturbation(
            df, pert_name, config, output_dir,
            noise_pool=noise_pool, rir_pool=rir_pool,
            dry_run=args.dry_run, max_samples=args.max_samples,
        )

        # Generate perturbed manifest
        if not args.dry_run:
            manifest_out = generate_perturbed_manifest(
                manifest_path, pert_name, output_dir, args.dataset
            )
            print(f"  Manifest: {manifest_out}")

        print(f"  Processed: {n_processed:,} utterances")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! Total time: {elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
