"""PyTorch Dataset and DataLoader with demographic-stratified sampling for ASR fine-tuning.

Provides ASRFairnessDataset for loading audio from manifest CSVs, a
DemographicStratifiedSampler that oversamples small groups to ensure balanced
batching, a collate_fn that pads variable-length audio, and a create_dataloader
factory function for one-call setup.

Audio loading uses soundfile (NOT torchaudio.load, which is broken without torchcodec).
"""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional
from torch.utils.data import Dataset, DataLoader, Sampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16000
MIN_GROUP_SIZE = 50


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ASRFairnessDataset(Dataset):
    """Dataset that loads audio from a manifest CSV with demographic labels.

    Each sample is a dict with keys: audio (1-D float32 tensor), transcript,
    demographic_group, and utterance_id.

    Works for both Fair-Speech (ethnicity axis) and Common Voice (accent axis).

    Args:
        manifest_csv: Path to CSV with columns: utterance_id, audio_path,
            sentence, and demographic columns.
        demographic_axis: Column name for demographic grouping.
            Use "ethnicity" for Fair-Speech, "accent" for Common Voice.
        sample_rate: Target sample rate (default 16 kHz).
    """

    def __init__(self, manifest_csv: str, demographic_axis: str = "ethnicity",
                 sample_rate: int = DEFAULT_SAMPLE_RATE):
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
            self.demographics = (
                self.df[demographic_axis].fillna("").astype(str).values
            )
        else:
            self.demographics = np.array([""] * len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Use soundfile -- torchaudio.load is broken (requires torchcodec)
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        audio = torch.from_numpy(audio)

        # Resample if needed (torchaudio.functional.resample works fine)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        return {
            "audio": audio,                              # 1-D float32 tensor
            "transcript": str(row["sentence"]),
            "demographic_group": self.demographics[idx],  # empty string if missing (D-08)
            "utterance_id": str(row["utterance_id"]),
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length audio into a padded batch.

    Returns:
        Dict with keys:
            audio: (B, T) float32 padded tensor
            audio_lengths: (B,) int64 original lengths
            transcripts: list[str]
            demographic_groups: list[str]
            utterance_ids: list[str]
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
