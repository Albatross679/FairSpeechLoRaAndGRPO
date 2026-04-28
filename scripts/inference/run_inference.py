#!/usr/bin/env python3
"""
Unified ASR inference for fairness evaluation — Phase 1.

Supports:
  Gen 1: wav2vec2-large  (facebook/wav2vec2-large-960h-lv60-self)
  Gen 2: whisper-large-v3 (openai/whisper-large-v3)
  Gen 3: qwen3-asr-1.7b  (Qwen/Qwen3-ASR-1.7B) via qwen-asr package

Usage:
    python scripts/inference/run_inference.py --model wav2vec2-large --device cuda
    python scripts/inference/run_inference.py --model whisper-large-v3 --device cuda
    python scripts/inference/run_inference.py --model qwen3-asr-1.7b --device cuda
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "wav2vec2-large": {
        "hf_id": "facebook/wav2vec2-large-960h-lv60-self",
        "generation": 1,
        "architecture": "CTC encoder-only",
        "params": "317M",
        "type": "wav2vec2",
    },
    "whisper-small": {
        "hf_id": "openai/whisper-small",
        "generation": 2,
        "architecture": "Encoder-Decoder",
        "params": "244M",
        "type": "whisper",
    },
    "whisper-medium": {
        "hf_id": "openai/whisper-medium",
        "generation": 2,
        "architecture": "Encoder-Decoder",
        "params": "764M",
        "type": "whisper",
    },
    "whisper-large-v3": {
        "hf_id": "openai/whisper-large-v3",
        "generation": 2,
        "architecture": "Encoder-Decoder",
        "params": "1.5B",
        "type": "whisper",
    },
    "qwen3-asr-0.6b": {
        "hf_id": "Qwen/Qwen3-ASR-0.6B",
        "generation": 3,
        "architecture": "Audio enc + Qwen3 LLM",
        "params": "0.6B",
        "type": "qwen3-asr",
    },
    "granite-speech-3.3-2b": {
        "hf_id": "ibm-granite/granite-speech-3.3-2b",
        "generation": 3,
        "architecture": "Conformer + Q-former + Granite",
        "params": "2B",
        "type": "granite",
    },
    "granite-speech-3.3-8b": {
        "hf_id": "ibm-granite/granite-speech-3.3-8b",
        "generation": 3,
        "architecture": "Conformer + Q-former + Granite",
        "params": "8B",
        "type": "granite",
    },
    "qwen3-asr-1.7b": {
        "hf_id": "Qwen/Qwen3-ASR-1.7B",
        "generation": 3,
        "architecture": "Audio enc + Qwen3 LLM",
        "params": "1.7B",
        "type": "qwen3-asr",
    },
    "canary-qwen-2.5b": {
        "hf_id": "nvidia/canary-qwen-2.5b",
        "generation": 3,
        "architecture": "FastConformer + Qwen3 (LoRA)",
        "params": "2.5B",
        "type": "canary",
    }
}

SAMPLE_RATE = 16000
DATA_DIR = "/users/PAS2030/srishti/asr_fairness/data"
RESULTS_DIR = "/users/PAS2030/srishti/asr_fairness/results/commonvoice"
PERTURBED_DIR = "/users/PAS2030/srishti/asr_fairness/perturbed_audio"

PERTURBATION_LABELS = [
    "clean",
    "snr_20db", "snr_10db", "snr_0db",
    "reverb_0.3s", "reverb_0.6s", "reverb_1.0s",
    "silence_25pct", "silence_50pct", "silence_75pct",
    "mask_10pct", "mask_20pct", "mask_30pct",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR inference for fairness audit")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--manifest", type=str, default=os.path.join(DATA_DIR, "cv_test_manifest.csv"))
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--perturbation", type=str, default="clean",
                        choices=PERTURBATION_LABELS,
                        help="Perturbation condition (default: clean = original audio)")
    parser.add_argument("--perturbations", type=str, default=None,
                        help="Comma-separated perturbation conditions to run sequentially "
                             "(model loaded once). Overrides --perturbation.")
    parser.add_argument("--dataset", type=str, default="cv", choices=["cv", "fs"],
                        help="Dataset name for resolving perturbed audio paths")
    parser.add_argument("--audio_variant", type=str, default=None,
                        help="Free-form audio variant label for compression/resampling manifests "
                             "(does not override audio paths).")
    parser.add_argument("--batch_plan", type=str, default=None,
                        help="Optional JSONL duration-bucketed batch plan produced by "
                             "scripts/inference/build_duration_batch_plan.py")
    parser.add_argument("--allow_batch_failures", action="store_true",
                        help="Write empty transcripts for failed inference batches instead of failing fast.")
    return parser.parse_args()


def safe_label(label: str) -> str:
    """Convert a free-form condition label to a filename-safe suffix."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def iter_manifest_batches(manifest_df, args):
    """Yield (batch_df, row_positions, batch_id) using fixed or planned batches.

    Batch-plan JSONL records should include ``utterance_ids``. When resuming,
    the manifest dataframe has already been filtered, so utterance IDs are used
    to map each planned batch onto the remaining rows. ``row_indices`` are only
    a fallback for older plans.
    """
    if not getattr(args, "batch_plan", None):
        n = len(manifest_df)
        for i in range(0, n, args.batch_size):
            positions = list(range(i, min(i + args.batch_size, n)))
            yield manifest_df.iloc[positions], positions, f"fixed_{i // args.batch_size:06d}"
        return

    id_to_pos = {}
    if "utterance_id" in manifest_df.columns:
        id_to_pos = {
            str(uid): pos for pos, uid in enumerate(manifest_df["utterance_id"].astype(str).tolist())
        }

    with open(args.batch_plan, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            batch_id = record.get("batch_id", f"planned_{line_no:06d}")
            positions = []
            utterance_ids = record.get("utterance_ids") or []
            if utterance_ids and id_to_pos:
                positions = [id_to_pos[str(uid)] for uid in utterance_ids if str(uid) in id_to_pos]
            elif record.get("row_indices"):
                positions = [int(idx) for idx in record["row_indices"] if int(idx) < len(manifest_df)]
            if not positions:
                continue
            yield manifest_df.iloc[positions], positions, batch_id


# ── Text normalization ──────────────────────────────────────────────────────
# Use Whisper's EnglishTextNormalizer for consistent WER across all models.
# This handles:
#   - Case normalization ("YES" → "yes")
#   - Punctuation removal ("hello." → "hello")
#   - Number word ↔ digit normalization ("six" → "6", "Six." → "6")
#   - Contraction expansion ("she'll" → "she will")
# Applied identically to BOTH reference and hypothesis for fair comparison.
try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError:
        class EnglishTextNormalizer:  # type: ignore[no-redef]
            """Lightweight fallback so registry/tools import without Whisper installed."""

            def __call__(self, text: str) -> str:
                import re
                text = (text or "").lower()
                text = re.sub(r"[^a-z0-9' ]+", " ", text)
                return re.sub(r"\s+", " ", text).strip()
_normalizer = EnglishTextNormalizer()

def normalize_text(text: str) -> str:
    """Normalize text for WER using Whisper's English normalizer."""
    if not text or not isinstance(text, str):
        return ""
    return _normalizer(text)


class IncrementalCSVWriter:
    """Writes full result rows to the output CSV incrementally during inference.

    This ensures that even if a job is killed/timed-out, all completed
    utterances are already on disk in the final output file.
    """

    def __init__(self, output_path, manifest_df, model_name, model_info, perturbation, append=False):
        self.output_path = output_path
        self.manifest_df = manifest_df
        self.model_name = model_name
        self.model_info = model_info
        self.perturbation = perturbation
        self.count = 0
        self._header_written = append  # skip header if appending to existing file
        import jiwer
        self._jiwer = jiwer

    def _build_row(self, pred):
        """Convert a raw prediction dict to a full result row."""
        idx = pred["idx"]
        row = self.manifest_df.iloc[idx]
        ref = normalize_text(row["sentence"])
        hyp = normalize_text(pred["hypothesis_raw"])

        if ref and hyp:
            try:
                wer_val = self._jiwer.wer(ref, hyp)
            except Exception:
                wer_val = 1.0
        else:
            wer_val = 1.0 if ref else 0.0

        result_row = {
            "utterance_id": row["utterance_id"],
            "reference": ref,
            "hypothesis": hyp,
            "hypothesis_raw": pred["hypothesis_raw"],
            "wer": round(wer_val, 4),
            "num_hyp_words": len(hyp.split()) if hyp else 0,
            "num_ref_words": len(ref.split()) if ref else 0,
            "perturbation": self.perturbation,
            "audio_variant": row.get("variant", self.perturbation),
            "gender": row.get("gender", ""),
            "accent": row.get("accent", ""),
            "age": row.get("age", ""),
            "model": self.model_name,
            "generation": self.model_info["generation"],
            "architecture": self.model_info["architecture"],
        }
        for extra_col in ["ethnicity", "first_language", "l1_group", "socioeconomic"]:
            if extra_col in row.index if hasattr(row, 'index') else extra_col in row:
                result_row[extra_col] = row.get(extra_col, "")
        return result_row

    def flush(self, predictions):
        """Write a batch of predictions to CSV. Called after each inference batch."""
        if not predictions:
            return
        rows = [self._build_row(p) for p in predictions]
        df = pd.DataFrame(rows)
        df.to_csv(self.output_path, mode="a", header=not self._header_written, index=False)
        self._header_written = True
        self.count += len(rows)


# ── Audio loading ───────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = SAMPLE_RATE):
    """Load audio file, resample to target sr, return numpy array."""
    import torchaudio
    import torch
    try:
        waveform, orig_sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)
        return waveform.squeeze(0).numpy()
    except Exception as e:
        print(f"  WARNING: Failed to load {path}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# MODEL RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

# ── Gen 1: wav2vec2 ─────────────────────────────────────────────────────────
def load_wav2vec2(args):
    """Load wav2vec2 model and processor."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    hf_id = MODEL_REGISTRY[args.model]["hf_id"]
    print(f"Loading {hf_id}...")
    processor = Wav2Vec2Processor.from_pretrained(hf_id)
    model = Wav2Vec2ForCTC.from_pretrained(hf_id).to(args.device).eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params on {args.device}")
    return {"model": model, "processor": processor}


def infer_wav2vec2(manifest_df, args, loaded, writer=None):
    """CTC-based inference with wav2vec2."""
    import torch
    model = loaded["model"]
    processor = loaded["processor"]

    predictions = []
    n = len(manifest_df)
    processed = 0

    for batch_num, (batch, batch_positions, batch_id) in enumerate(iter_manifest_batches(manifest_df, args)):
        audio_list, valid_idx = [], []
        batch_preds = []

        for row_pos, (_, row) in zip(batch_positions, batch.iterrows()):
            audio = load_audio(row["audio_path"])
            if audio is not None:
                audio_list.append(audio)
                valid_idx.append(row_pos)
            else:
                batch_preds.append({"idx": row_pos, "hypothesis_raw": ""})

        if not audio_list:
            if writer:
                writer.flush(batch_preds)
            predictions.extend(batch_preds)
            continue

        try:
            inputs = processor(audio_list, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(args.device)
            with torch.no_grad():
                logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            texts = processor.batch_decode(pred_ids)
            for idx, text in zip(valid_idx, texts):
                batch_preds.append({"idx": idx, "hypothesis_raw": text.strip()})
        except Exception as e:
            if not args.allow_batch_failures:
                raise RuntimeError(f"Wav2Vec2 batch {batch_id} failed") from e
            print(f"  WARNING: Batch {batch_id} failed: {e}")
            for idx in valid_idx:
                batch_preds.append({"idx": idx, "hypothesis_raw": ""})

        if writer:
            writer.flush(batch_preds)
        predictions.extend(batch_preds)

        processed += len(batch_positions)
        if batch_num % 20 == 0:
            print(f"  [{processed:,}/{n:,}] ({100*processed/n:.0f}%)")

    return predictions


def run_wav2vec2(manifest_df, args, writer=None):
    """CTC-based inference with wav2vec2 (legacy single-call interface)."""
    loaded = load_wav2vec2(args)
    return infer_wav2vec2(manifest_df, args, loaded, writer=writer)


# ── Gen 2: Whisper ──────────────────────────────────────────────────────────
def load_whisper(args):
    """Load Whisper model and processor."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    hf_id = MODEL_REGISTRY[args.model]["hf_id"]
    print(f"Loading {hf_id}...")
    processor = WhisperProcessor.from_pretrained(hf_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.float16 if "cuda" in args.device else torch.float32
    ).to(args.device).eval()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params on {args.device}")
    return {"model": model, "processor": processor}


def infer_whisper(manifest_df, args, loaded, writer=None):
    """Encoder-decoder inference with Whisper."""
    import torch
    model = loaded["model"]
    processor = loaded["processor"]

    predictions = []
    n = len(manifest_df)
    processed = 0

    for batch_num, (batch, batch_positions, batch_id) in enumerate(iter_manifest_batches(manifest_df, args)):
        audio_list, valid_idx = [], []
        batch_preds = []

        for row_pos, (_, row) in zip(batch_positions, batch.iterrows()):
            audio = load_audio(row["audio_path"])
            if audio is not None:
                audio_list.append(audio)
                valid_idx.append(row_pos)
            else:
                batch_preds.append({"idx": row_pos, "hypothesis_raw": ""})

        if not audio_list:
            if writer:
                writer.flush(batch_preds)
            predictions.extend(batch_preds)
            continue

        try:
            inputs = processor(
                audio_list,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            features = inputs.input_features.to(args.device, dtype=torch.float16 if "cuda" in args.device else torch.float32)
            generate_kwargs = {
                "max_new_tokens": 440,
                "language": "en",
                "task": "transcribe",
            }
            if "attention_mask" in inputs:
                generate_kwargs["attention_mask"] = inputs.attention_mask.to(args.device)
            with torch.no_grad():
                pred_ids = model.generate(features, **generate_kwargs)
            texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
            for idx, text in zip(valid_idx, texts):
                batch_preds.append({"idx": idx, "hypothesis_raw": text.strip()})
        except Exception as e:
            if not args.allow_batch_failures:
                raise RuntimeError(f"Whisper batch {batch_id} failed") from e
            print(f"  WARNING: Batch {batch_id} failed: {e}")
            for idx in valid_idx:
                batch_preds.append({"idx": idx, "hypothesis_raw": ""})

        if writer:
            writer.flush(batch_preds)
        predictions.extend(batch_preds)

        processed += len(batch_positions)
        if batch_num % 20 == 0:
            print(f"  [{processed:,}/{n:,}] ({100*processed/n:.0f}%)")

    return predictions


def run_whisper(manifest_df, args, writer=None):
    """Whisper inference (legacy single-call interface)."""
    loaded = load_whisper(args)
    return infer_whisper(manifest_df, args, loaded, writer=writer)


# ── Gen 3: Qwen3-ASR ───────────────────────────────────────────────────────
def load_qwen3_asr(args):
    """Load Qwen3-ASR model."""
    import torch
    from qwen_asr import Qwen3ASRModel

    hf_id = MODEL_REGISTRY[args.model]["hf_id"]
    print(f"Loading {hf_id} via qwen-asr...")
    model = Qwen3ASRModel.from_pretrained(
        hf_id,
        dtype=torch.bfloat16,
        device_map=f"{args.device}:0" if args.device == "cuda" else args.device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=256,
    )
    print(f"  Loaded on {args.device}")
    return {"model": model}


def infer_qwen3_asr(manifest_df, args, loaded, writer=None):
    """LLM-based inference with Qwen3-ASR via qwen-asr package."""
    model = loaded["model"]

    predictions = []
    n = len(manifest_df)
    processed = 0

    # Qwen3-ASR accepts file paths directly — batch them
    for batch_num, (batch, batch_positions, batch_id) in enumerate(iter_manifest_batches(manifest_df, args)):
        audio_paths = batch["audio_path"].tolist()
        indices = batch_positions
        batch_preds = []

        try:
            results = model.transcribe(
                audio=audio_paths,
                language="English",
            )
            for idx, result in zip(indices, results):
                batch_preds.append({"idx": idx, "hypothesis_raw": result.text.strip()})
        except Exception as e:
            if not args.allow_batch_failures:
                raise RuntimeError(f"Qwen3-ASR batch {batch_id} failed") from e
            print(f"  WARNING: Batch {batch_id} failed: {e}")
            for idx in indices:
                batch_preds.append({"idx": idx, "hypothesis_raw": ""})

        if writer:
            writer.flush(batch_preds)
        predictions.extend(batch_preds)

        processed += len(batch_positions)
        if batch_num % 20 == 0:
            print(f"  [{processed:,}/{n:,}] ({100*processed/n:.0f}%)")

    return predictions


def run_qwen3_asr(manifest_df, args, writer=None):
    """Qwen3-ASR inference (legacy single-call interface)."""
    loaded = load_qwen3_asr(args)
    return infer_qwen3_asr(manifest_df, args, loaded, writer=writer)

def _extract_granite_transcription(text):
    """Extract the actual transcription from Granite's potentially chatty output.

    Granite may wrap the transcription in conversational text like:
      'Sure, here\'s the transcription: "Hey Facebook, answer the call."'
    This extracts just the quoted transcription, or returns the full text if
    no quoting pattern is detected.
    """
    import re
    # Try to find text in double quotes (the transcription)
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        # Return the longest quoted string (likely the transcription)
        return max(quoted, key=len)
    # If no quotes, check for common preamble patterns and strip them
    preamble_patterns = [
        r"^(?:Sure,?\s*)?(?:Here(?:'s| is)\s+)?(?:the\s+)?(?:transcription|transcribed)[^:]*:\s*",
        r"^The user(?:'s)?\s+(?:message|speech|audio)[^:]*:\s*",
        r"^Of course[^:]*:\s*",
    ]
    for pat in preamble_patterns:
        cleaned = re.sub(pat, "", text, flags=re.IGNORECASE)
        if cleaned != text:
            return cleaned.strip().strip('"')
    return text


# ── Gen 3: Granite-Speech ───────────────────────────────────────────────────
def load_granite(args):
    """Load Granite-Speech model and processor."""
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    hf_id = MODEL_REGISTRY[args.model]["hf_id"]
    print(f"Loading {hf_id}...")
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_id, device_map=args.device, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    print(f"  Loaded on {args.device}")

    system_prompt = "You are a speech transcription system. Output ONLY the exact transcription of the audio. Do not add any commentary, explanation, or formatting."
    user_prompt = "<|audio|>Transcribe this audio exactly."
    chat = [
        dict(role="system", content=system_prompt),
        dict(role="user", content=user_prompt),
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return {"model": model, "processor": processor, "tokenizer": tokenizer, "prompt": prompt}


def infer_granite(manifest_df, args, loaded, writer=None):
    """LLM-based inference with Granite-Speech."""
    import torch
    import torchaudio
    model = loaded["model"]
    processor = loaded["processor"]
    tokenizer = loaded["tokenizer"]
    prompt = loaded["prompt"]

    predictions = []
    n = len(manifest_df)
    processed = 0

    for batch_num, (batch, batch_positions, batch_id) in enumerate(iter_manifest_batches(manifest_df, args)):
        audio_list, valid_idx = [], []
        batch_preds = []

        for row_pos, (_, row) in zip(batch_positions, batch.iterrows()):
            try:
                wav, sr = torchaudio.load(row["audio_path"])
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
                audio_list.append(wav[0].numpy())
                valid_idx.append(row_pos)
            except Exception as e:
                if not args.allow_batch_failures:
                    raise RuntimeError(f"Failed to load audio for Granite batch {batch_id}: {row['audio_path']}") from e
                print(f"  WARNING: Failed to load {row['audio_path']}: {e}")
                batch_preds.append({"idx": row_pos, "hypothesis_raw": ""})

        if not audio_list:
            if writer:
                writer.flush(batch_preds)
            predictions.extend(batch_preds)
            continue

        try:
            prompts = [prompt] * len(audio_list)
            model_inputs = processor(
                text=prompts,
                audio=audio_list,
                return_tensors="pt",
                padding=True
            ).to(args.device)

            with torch.no_grad():
                model_outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1
                )

            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[:, num_input_tokens:]

            texts = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
            for idx, text in zip(valid_idx, texts):
                batch_preds.append({"idx": idx, "hypothesis_raw": _extract_granite_transcription(text.strip())})
        except Exception as e:
            if not args.allow_batch_failures:
                raise RuntimeError(f"Granite batch {batch_id} failed") from e
            print(f"  WARNING: Batch {batch_id} failed: {e}")
            for idx in valid_idx:
                batch_preds.append({"idx": idx, "hypothesis_raw": ""})

        if writer:
            writer.flush(batch_preds)
        predictions.extend(batch_preds)

        processed += len(batch_positions)
        if batch_num % 20 == 0:
            print(f"  [{processed:,}/{n:,}] ({100*processed/n:.0f}%)")

    return predictions


def run_granite(manifest_df, args, writer=None):
    """Granite inference (legacy single-call interface)."""
    loaded = load_granite(args)
    return infer_granite(manifest_df, args, loaded, writer=writer)

# ── Gen 3: Canary-Qwen ──────────────────────────────────────────────────────
def run_canary(manifest_df, args, writer=None):
    """SALM-based inference with Canary-Qwen-2.5B (FastConformer + Qwen3 LoRA)."""
    import torch
    try:
        from nemo.collections.speechlm2.models import SALM
    except ImportError:
        raise ImportError(
            "Canary-Qwen requires NeMo >= 2.5.0 with speechlm2 support.\n"
            "Install: pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'"
        )

    hf_id = MODEL_REGISTRY[args.model]["hf_id"]
    print(f"Loading Canary-Qwen model via SALM: {hf_id}")
    model = SALM.from_pretrained(hf_id)
    model.eval()
    if args.device.startswith("cuda"):
        model = model.to(args.device) if hasattr(model, "to") else model.cuda()
    print(f"  Loaded on {args.device}")

    # The SALM API uses model.generate(prompts=...) with audio locator tags.
    # Each prompt is a list of message dicts (chat format).
    user_prompt_template = f"Transcribe the following: {model.audio_locator_tag}"

    predictions = []
    n = len(manifest_df)
    processed = 0

    for batch_num, (batch_df, batch_positions, batch_id) in enumerate(iter_manifest_batches(manifest_df, args)):
        audio_paths = batch_df["audio_path"].tolist()
        batch_preds = []

        # Build prompts: each is a single-turn chat with audio
        prompts = []
        for audio_path in audio_paths:
            prompts.append([{
                "role": "user",
                "content": user_prompt_template,
                "audio": [audio_path],
            }])

        try:
            with torch.no_grad():
                answer_ids = model.generate(
                    prompts=prompts,
                    max_new_tokens=256,
            )
            # Decode token IDs to text
            for row_pos, ids in zip(batch_positions, answer_ids):
                text = model.tokenizer.ids_to_text(ids.cpu().tolist())
                batch_preds.append({
                    "idx": row_pos,
                    "utterance_id": manifest_df.iloc[row_pos]["utterance_id"],
                    "hypothesis_raw": text.strip(),
                })
        except Exception as e:
            if not args.allow_batch_failures:
                raise RuntimeError(f"Canary-Qwen batch {batch_id} failed") from e
            print(f"  WARNING: Batch {batch_id} failed: {e}")
            for row_pos in batch_positions:
                batch_preds.append({
                    "idx": row_pos,
                    "utterance_id": manifest_df.iloc[row_pos]["utterance_id"],
                    "hypothesis_raw": "",
                })

        if writer:
            writer.flush(batch_preds)
        predictions.extend(batch_preds)

        processed += len(batch_positions)
        if batch_num % 20 == 0:
            print(f"  [{processed:,}/{n:,}] ({100 * processed / n:.0f}%)")

    return predictions


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    model_info = MODEL_REGISTRY[args.model]
    model_key = args.model.replace("-", "_")

    perturbation = args.perturbation
    is_perturbed = perturbation != "clean"
    condition_label = args.audio_variant or perturbation

    print(f"\n{'='*60}")
    print(f"ASR Fairness Audit — Inference")
    print(f"{'='*60}")
    print(f"  Model:  {args.model} (Gen {model_info['generation']}, {model_info['architecture']}, {model_info['params']})")
    print(f"  Manifest: {args.manifest}")
    print(f"  Perturbation: {perturbation}")
    print(f"  Audio variant: {condition_label}")
    print(f"  Device:   {args.device}")
    print(f"  Batch:    {args.batch_plan or args.batch_size}")
    print(f"{'='*60}\n")

    # ── Load manifest ───────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    print(f"Manifest: {len(df):,} utterances")

    # Override audio paths for perturbed conditions
    dataset_name = {"cv": "common_voice", "fs": "fair_speech"}[args.dataset]
    if is_perturbed:
        perturbed_base = os.path.join(PERTURBED_DIR, dataset_name, perturbation)
        df["audio_path_original"] = df["audio_path"]
        df["audio_path"] = df["utterance_id"].apply(
            lambda uid: os.path.join(perturbed_base, f"{uid}.wav")
        )
        print(f"  Overriding audio paths → {perturbed_base}/")

    # Pre-filter missing audio
    before = len(df)
    df = df[df["audio_path"].apply(os.path.isfile)].reset_index(drop=True)
    if len(df) < before:
        print(f"  Filtered {before - len(df):,} missing audio files")
    print(f"  Valid: {len(df):,}")

    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)
        print(f"  Limited to {args.max_samples} samples")

    # ── Output path & resume logic ──────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    # Suffix output filename with perturbation for non-clean conditions
    if args.audio_variant:
        output_path = os.path.join(args.output_dir, f"predictions_{model_key}_{safe_label(args.audio_variant)}.csv")
    elif is_perturbed:
        pert_suffix = perturbation.replace(".", "_")  # reverb_0.3s -> reverb_0_3s
        output_path = os.path.join(args.output_dir, f"predictions_{model_key}_{pert_suffix}.csv")
    else:
        output_path = os.path.join(args.output_dir, f"predictions_{model_key}.csv")

    n_existing = 0
    if args.resume and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        done_ids = set(existing["utterance_id"])
        n_existing = len(done_ids)
        df = df[~df["utterance_id"].isin(done_ids)].reset_index(drop=True)
        print(f"  Resuming: {n_existing:,} done, {len(df):,} remaining")
        if len(df) == 0:
            print("  All done!")
            return

    # ── Run inference with incremental CSV writing ─────────────────
    writer = IncrementalCSVWriter(
        output_path, df, args.model, model_info, condition_label,
        append=(n_existing > 0),
    )
    start = time.time()

    if model_info["type"] == "wav2vec2":
        run_wav2vec2(df, args, writer=writer)
    elif model_info["type"] == "whisper":
        run_whisper(df, args, writer=writer)
    elif model_info["type"] == "qwen3-asr":
        run_qwen3_asr(df, args, writer=writer)
    elif model_info["type"] == "granite":
        run_granite(df, args, writer=writer)
    elif model_info["type"] == "canary":
        run_canary(df, args, writer=writer)
    else:
        raise ValueError(f"Unknown model type: {model_info['type']}")

    elapsed = time.time() - start
    total_rows = n_existing + writer.count
    print(f"\nInference: {elapsed/60:.1f} min, {writer.count:,} new utterances")
    print(f"Saved: {output_path} ({total_rows:,} total rows)")

    # ── Read back for summary ────────────────────────────────────
    import jiwer
    results_df = pd.read_csv(output_path)

    # ── Quick summary ───────────────────────────────────────────────
    # Ensure string type and filter NaN/empty values for WER computation
    results_df["reference"] = results_df["reference"].fillna("").astype(str)
    results_df["hypothesis"] = results_df["hypothesis"].fillna("").astype(str)
    valid = results_df[(results_df["reference"] != "") & (results_df["hypothesis"] != "")]
    if len(valid) > 0:
        overall_wer = jiwer.wer(valid["reference"].tolist(), valid["hypothesis"].tolist())
        print(f"\n{'─'*40}")
        print(f"Overall WER: {overall_wer*100:.2f}% (n={len(valid):,})")

        for axis in ["gender", "accent", "age"]:
            axis_data = valid[valid[axis].notna() & (valid[axis] != "")]
            if len(axis_data) == 0:
                continue
            groups = axis_data.groupby(axis)
            print(f"\n  {axis.upper()}:")
            for name, group in sorted(groups, key=lambda x: -len(x[1])):
                if len(group) >= 50:
                    g_wer = jiwer.wer(group["reference"].tolist(), group["hypothesis"].tolist())
                    print(f"    {name:20s}: {g_wer*100:.2f}% (n={len(group):,})")

    # Save metadata
    meta = {
        "model": args.model,
        "model_info": model_info,
        "perturbation": perturbation,
        "total_utterances": len(results_df),
        "overall_wer": float(overall_wer) if len(valid) > 0 else None,
        "elapsed_seconds": elapsed,
        "device": args.device,
        "batch_size": args.batch_size,
        "batch_plan": args.batch_plan,
        "audio_variant": condition_label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.audio_variant:
        meta_path = os.path.join(args.output_dir, f"meta_{model_key}_{safe_label(args.audio_variant)}.json")
    elif is_perturbed:
        pert_suffix = perturbation.replace(".", "_")
        meta_path = os.path.join(args.output_dir, f"meta_{model_key}_{pert_suffix}.json")
    else:
        meta_path = os.path.join(args.output_dir, f"meta_{model_key}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
