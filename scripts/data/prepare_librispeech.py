"""
Prepare LibriSpeech test-clean manifest for ASR fairness evaluation.

Loads from HuggingFace cache, saves audio as 16kHz WAV files,
and creates ls_manifest.csv with columns matching the inference pipeline.
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

PROJECT_DIR = "/users/PAS2030/srishti/asr_fairness"
AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "librispeech", "audio")
MANIFEST_PATH = os.path.join(PROJECT_DIR, "data", "ls_manifest.csv")

os.makedirs(AUDIO_DIR, exist_ok=True)

print("Loading LibriSpeech test-clean from HuggingFace cache...")
# Load without automatic audio decoding to avoid torchcodec dependency
ds = load_dataset("librispeech_asr", "clean", split="test")
# Cast audio to decode via soundfile instead of torchcodec
ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
print(f"  {len(ds)} utterances loaded")

rows = []
for i, sample in enumerate(ds):
    uid = f"ls_{sample['id']}"
    audio_path = os.path.join(AUDIO_DIR, f"{uid}.wav")

    # Save audio as 16kHz WAV if not already present
    if not os.path.exists(audio_path):
        # Read from the raw audio file path provided by HF
        src_path = sample["audio"]["path"]
        if src_path and os.path.exists(src_path):
            data, sr = sf.read(src_path)
            # Resample to 16kHz if needed
            if sr != 16000:
                import torchaudio
                import torch
                tensor = torch.tensor(data).float()
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=16000)
                data = tensor.squeeze(0).numpy()
                sr = 16000
            sf.write(audio_path, data, sr)
        else:
            # Fallback: read raw bytes
            audio_bytes = sample["audio"]["bytes"]
            if audio_bytes:
                import io
                data, sr = sf.read(io.BytesIO(audio_bytes))
                sf.write(audio_path, data, sr)
            else:
                print(f"  WARNING: No audio data for {uid}")
                continue

    rows.append({
        "utterance_id": uid,
        "audio_path": audio_path,
        "sentence": sample["text"],
        "sentence_raw": sample["text"],
    })

    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{len(ds)}] processed")

df = pd.DataFrame(rows)
df.to_csv(MANIFEST_PATH, index=False)
print(f"\nSaved: {MANIFEST_PATH} ({len(df)} utterances)")

# Quick sanity check
missing = df[~df["audio_path"].apply(os.path.exists)]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} audio files missing!")
else:
    print("All audio files verified.")
