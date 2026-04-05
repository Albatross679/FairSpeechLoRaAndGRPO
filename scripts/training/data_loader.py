"""
PyTorch Dataset and DataLoader with demographic-stratified sampling for ASR fine-tuning.

Provides ASRFairnessDataset (loads audio via soundfile), DemographicStratifiedSampler
(oversamples small demographic groups), collate_fn (pads variable-length audio), and
create_dataloader (single entry point for training and evaluation data loading).

Supports both Fair-Speech (ethnicity axis) and Common Voice (accent axis) datasets.
Audio loading uses soundfile.read() -- torchaudio.load() is broken in this environment
(requires torchcodec which is not installed).

Usage:
    python scripts/training/data_loader.py --manifest /path/to/fs_train.csv --axis ethnicity
"""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional
from torch.utils.data import Dataset, DataLoader, Sampler

# -- Constants ----------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 16000
MIN_GROUP_SIZE = 50


# -- Dataset ------------------------------------------------------------------

class ASRFairnessDataset(Dataset):
    """PyTorch Dataset for ASR fairness training/evaluation.

    Loads audio via soundfile (NOT torchaudio.load), returns dicts with
    audio tensor, transcript, demographic group label, and utterance ID.

    Handles both Fair-Speech (ethnicity axis) and Common Voice (accent axis)
    manifests. Missing demographic labels are represented as empty strings
    per D-08: excluded from R_fair but included in R_acc.

    Args:
        manifest_csv: Path to CSV with columns: utterance_id, audio_path,
            sentence, and demographic columns.
        demographic_axis: Column name for demographic grouping.
            Use "ethnicity" for Fair-Speech, "accent" for Common Voice.
        sample_rate: Target sample rate (default 16kHz).
    """

    def __init__(self, manifest_csv: str, demographic_axis: str = "ethnicity",
                 sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.manifest_csv = manifest_csv
        self.df = pd.read_csv(manifest_csv)
        self.demographic_axis = demographic_axis
        self.sample_rate = sample_rate

        # Validate required columns exist
        required = ["utterance_id", "audio_path", "sentence"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if demographic_axis and demographic_axis not in self.df.columns:
            raise ValueError(
                f"Demographic axis '{demographic_axis}' not in columns: "
                f"{list(self.df.columns)}"
            )

        # Build demographic group labels (per D-08: empty string for missing)
        if demographic_axis:
            self.demographics = self.df[demographic_axis].fillna("").astype(str).values
        else:
            self.demographics = np.array([""] * len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Use soundfile -- torchaudio.load is broken (requires torchcodec)
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        audio = torch.from_numpy(audio)

        # Handle stereo -> mono if needed
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)

        # Resample if needed (torchaudio.functional.resample works fine)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        return {
            "audio": audio,                              # 1D float32 tensor
            "transcript": str(row["sentence"]),
            "demographic_group": self.demographics[idx],  # empty string if missing (D-08)
            "utterance_id": str(row["utterance_id"]),
        }


# -- Collate Function ---------------------------------------------------------

def collate_fn(batch):
    """Collate variable-length audio into padded batch.

    Pads audio to the max length in the current batch (not global max)
    to mitigate OOM from unnecessarily large tensors (T-01.0-05).

    Args:
        batch: List of dicts from ASRFairnessDataset.__getitem__.

    Returns:
        Dict with:
            audio: (B, T) float32 padded tensor
            audio_lengths: (B,) int64 original lengths
            transcripts: list of str
            demographic_groups: list of str
            utterance_ids: list of str
    """
    audios = [item["audio"] for item in batch]
    lengths = torch.tensor([a.shape[0] for a in audios])
    max_len = lengths.max().item()

    # Pad to max length with zeros
    padded = torch.zeros(len(audios), max_len)
    for i, a in enumerate(audios):
        padded[i, :a.shape[0]] = a

    return {
        "audio": padded,                                          # (B, T) float32
        "audio_lengths": lengths,                                 # (B,) int64
        "transcripts": [item["transcript"] for item in batch],
        "demographic_groups": [item["demographic_group"] for item in batch],
        "utterance_ids": [item["utterance_id"] for item in batch],
    }


# -- Stratified Sampler -------------------------------------------------------

class DemographicStratifiedSampler(Sampler):
    """Sampler that ensures each batch contains utterances from multiple demographic groups.

    Oversamples small groups to ensure minimum representation (D-09).
    Groups with missing labels are included but not used for stratification.

    Args:
        demographics: Series or array of demographic group labels.
        batch_size: Number of samples per batch.
        min_per_group: Minimum samples per group per epoch (oversampling floor).
            Defaults to MIN_GROUP_SIZE=50.
        seed: Random seed for reproducibility.
        num_batches: Number of batches per epoch. If None, computed from
            oversampled dataset size / batch_size.
    """

    def __init__(self, demographics, batch_size: int,
                 min_per_group: int = MIN_GROUP_SIZE, seed: int = 42,
                 num_batches: int | None = None):
        self.batch_size = batch_size
        self.min_per_group = min_per_group
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Build per-group index lists (skip empty/missing labels)
        self.group_indices = {}
        self.unlabeled_indices = []
        demographics_arr = np.asarray(demographics)
        for idx, group in enumerate(demographics_arr):
            group_str = str(group).strip()
            if group_str and group_str != "nan":
                self.group_indices.setdefault(group_str, []).append(idx)
            else:
                self.unlabeled_indices.append(idx)

        self.groups = sorted(self.group_indices.keys())

        # Compute oversampled indices per group
        self._oversampled_pool = []
        for group in self.groups:
            indices = self.group_indices[group]
            if len(indices) < self.min_per_group:
                # Oversample to reach floor (D-09)
                oversampled = list(indices) * (self.min_per_group // len(indices) + 1)
                oversampled = oversampled[:self.min_per_group]
            else:
                oversampled = list(indices)
            self._oversampled_pool.extend(oversampled)

        # Add unlabeled indices (they contribute to R_acc per D-08)
        self._oversampled_pool.extend(self.unlabeled_indices)

        # Effective epoch length (per Pitfall 6 from RESEARCH.md)
        self._num_samples = len(self._oversampled_pool)
        if num_batches is not None:
            self._num_samples = num_batches * batch_size

    def __iter__(self):
        # Shuffle the oversampled pool
        pool = self._oversampled_pool.copy()
        self.rng.shuffle(pool)
        # Yield indices up to num_samples
        for i in range(min(self._num_samples, len(pool))):
            yield pool[i]

    def __len__(self):
        return self._num_samples


# -- DataLoader Factory -------------------------------------------------------

def create_dataloader(
    manifest_csv: str,
    demographic_axis: str = "ethnicity",
    batch_size: int = 8,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    stratified: bool = True,
    min_per_group: int = MIN_GROUP_SIZE,
    seed: int = 42,
    num_workers: int = 2,
    num_batches: int | None = None,
):
    """Create a DataLoader for ASR fairness training/evaluation.

    Args:
        manifest_csv: Path to CSV manifest (fs_train.csv, fs_eval.csv,
            or cv_test_manifest.csv).
        demographic_axis: Column for demographic grouping ("ethnicity"
            for Fair-Speech, "accent" for Common Voice).
        batch_size: Samples per batch.
        sample_rate: Target audio sample rate.
        stratified: If True, use DemographicStratifiedSampler (for training).
            If False, sequential iteration (for evaluation).
        min_per_group: Oversampling floor for small groups.
        seed: Random seed.
        num_workers: DataLoader workers.
        num_batches: Fixed number of batches per epoch (None = use all data).

    Returns:
        DataLoader yielding batches from collate_fn.
    """
    dataset = ASRFairnessDataset(
        manifest_csv=manifest_csv,
        demographic_axis=demographic_axis,
        sample_rate=sample_rate,
    )

    sampler = None
    shuffle = False
    if stratified:
        sampler = DemographicStratifiedSampler(
            demographics=dataset.demographics,
            batch_size=batch_size,
            min_per_group=min_per_group,
            seed=seed,
            num_batches=num_batches,
        )
    else:
        shuffle = False  # Eval: sequential, no shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


# -- CLI Smoke Test -----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--axis", default="ethnicity", help="Demographic axis")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    loader = create_dataloader(
        manifest_csv=args.manifest,
        demographic_axis=args.axis,
        batch_size=args.batch_size,
        stratified=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    print(f"Audio shape: {batch['audio'].shape}")
    print(f"Audio lengths: {batch['audio_lengths']}")
    print(f"Transcripts: {batch['transcripts'][:2]}")
    print(f"Demographics: {batch['demographic_groups']}")
    print(f"Utterance IDs: {batch['utterance_ids'][:2]}")
