"""
Validate LoRA prototype training results.

Checks:
1. VRAM: Peak memory < 14GB during training
2. Transcription quality: WER < 80%, no chatty patterns, no empty hypotheses
3. Adapter round-trip: save -> load -> identical output tokens
4. Loss trend: training loss decreases over steps

Usage:
    python scripts/training/validate_lora_prototype.py \
        --adapter_dir outputs/lora-prototype/adapter \
        --log_dir outputs/lora-prototype \
        --eval_manifest outputs/lora-prototype/prototype_subset.csv \
        --num_eval_samples 10
"""

import argparse
import json
import os
import re
import sys
import tempfile

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from whisper.normalizers import EnglishTextNormalizer

# -- Constants ----------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
VRAM_LIMIT_GB = 14.0
WER_THRESHOLD = 0.80
CHATTY_PATTERNS = [
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "sure",
    "of course",
    "here is",
    "here's",
    "the user",
    "transcription",
    "transcribed",
    "written format",
]
DEFAULT_NUM_EVAL_SAMPLES = 10
DEFAULT_SAMPLE_RATE = 16000


# -- Forward Patch (same as lora_prototype.py, Pattern 2 from RESEARCH.md) ----

def patch_outer_forward(model):
    """Patch Qwen3ASRForConditionalGeneration to delegate forward to thinker.

    The outer model only has generate(), not forward(). HuggingFace Trainer
    requires forward() for training loss computation. This patch delegates
    to thinker.forward() which has the full forward + loss computation.

    Based on official SFT script pattern (D-15, RESEARCH.md Pattern 2).
    """
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no .thinker.forward. "
            "Ensure model is Qwen3ASRForConditionalGeneration."
        )

    def forward(self, input_ids=None, attention_mask=None,
                input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# -- Model Loading Helpers ----------------------------------------------------

def load_base_model_and_processor():
    """Load Qwen3-ASR-1.7B via qwen-asr wrapper and extract HF model.

    Returns:
        Tuple of (model, processor).
    """
    from qwen_asr import Qwen3ASRModel

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model = asr_wrapper.model          # Qwen3ASRForConditionalGeneration
    processor = asr_wrapper.processor   # Qwen3ASRProcessor
    return model, processor


def load_adapter_model(adapter_dir):
    """Load base model with LoRA adapter from disk.

    Uses Pattern 5 from RESEARCH.md: load base model, patch forward,
    then load adapter via PeftModel.from_pretrained.

    Args:
        adapter_dir: Path to saved adapter (contains adapter_config.json
            and adapter_model.safetensors).

    Returns:
        Tuple of (model_with_adapter, processor).
    """
    from peft import PeftModel

    base_model, processor = load_base_model_and_processor()
    patch_outer_forward(base_model)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def load_eval_samples(eval_manifest, num_samples, sample_rate=DEFAULT_SAMPLE_RATE):
    """Load audio samples and references from eval manifest.

    Args:
        eval_manifest: Path to CSV with columns: audio_path, sentence.
        num_samples: Number of samples to load.
        sample_rate: Target sample rate.

    Returns:
        List of dicts with keys: audio (tensor), reference (str), audio_path (str).
    """
    import torchaudio.functional

    df = pd.read_csv(eval_manifest)
    df = df.head(num_samples)

    samples = []
    for _, row in df.iterrows():
        audio_path = row["audio_path"]
        if not os.path.isfile(audio_path):
            print(f"  WARNING: Audio file not found: {audio_path}")
            continue

        audio, sr = sf.read(audio_path, dtype="float32")
        audio = torch.from_numpy(audio)

        # Handle stereo -> mono
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)

        # Resample if needed
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)

        samples.append({
            "audio": audio,
            "reference": str(row["sentence"]),
            "audio_path": audio_path,
        })

    return samples


def transcribe_sample(model, processor, audio, sample_rate=DEFAULT_SAMPLE_RATE):
    """Transcribe a single audio sample using model.generate().

    Uses deterministic generation (do_sample=False) for reproducibility.

    Args:
        model: LoRA adapter model in eval mode.
        processor: Qwen3ASRProcessor.
        audio: 1D float32 tensor.
        sample_rate: Audio sample rate.

    Returns:
        Transcription string.
    """
    # Build chat-format input for Qwen3-ASR
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio.numpy()},
            ],
        },
    ]

    # Process inputs through the processor
    inputs = processor(
        conversations=conversation,
        audios=[audio.numpy()],
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # Generate with deterministic settings
    with torch.no_grad():
        # Use the base model's generate method (PeftModel delegates)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Decode only the new tokens (skip input tokens)
    input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
    new_tokens = output_ids[0, input_len:]
    transcription = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    return transcription


# -- Validation Checks --------------------------------------------------------

def check_vram(adapter_dir, issues):
    """Check VRAM usage by loading model and running a forward+backward pass.

    Loads the adapter model, creates a dummy batch, runs one forward pass
    and one backward pass, then records peak VRAM via
    torch.cuda.max_memory_allocated().

    Args:
        adapter_dir: Path to saved adapter.
        issues: List to append issues to.
    """
    if not torch.cuda.is_available():
        print("  SKIP: No CUDA device available, cannot measure VRAM")
        issues.append("Cannot measure VRAM: no CUDA device available")
        return

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    print("  Loading model with adapter for VRAM measurement...")
    from peft import PeftModel

    base_model, processor = load_base_model_and_processor()
    patch_outer_forward(base_model)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.cuda()
    model.train()  # Training mode for backward pass

    print("  Running forward+backward pass on dummy batch...")
    try:
        # Create a minimal dummy input
        # Use processor to generate a realistic-sized input
        dummy_audio = torch.randn(16000)  # 1 second of audio
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": dummy_audio.numpy()},
                ],
            },
        ]

        inputs = processor(
            conversations=conversation,
            audios=[dummy_audio.numpy()],
            sampling_rate=DEFAULT_SAMPLE_RATE,
            return_tensors="pt",
        )

        # Create labels from input_ids (shifted internally by model)
        inputs["labels"] = inputs["input_ids"].clone()

        # Move to GPU
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Backward pass
        if loss is not None:
            loss.backward()
        else:
            print("  WARNING: No loss returned from forward pass")

    except Exception as e:
        print(f"  WARNING: Forward/backward pass failed: {e}")
        issues.append(f"VRAM measurement failed: {e}")
        return
    finally:
        # Record peak VRAM before cleanup
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(f"  VRAM peak: {peak_gb:.2f}GB (limit: {VRAM_LIMIT_GB}GB) -- "
          f"{'PASS' if peak_gb < VRAM_LIMIT_GB else 'FAIL'}")

    if peak_gb >= VRAM_LIMIT_GB:
        issues.append(
            f"VRAM peak {peak_gb:.2f}GB exceeds {VRAM_LIMIT_GB}GB limit"
        )

    # Cleanup to free GPU memory for subsequent checks
    del model, base_model, processor, inputs
    if 'outputs' in dir():
        del outputs
    torch.cuda.empty_cache()


def check_transcription_quality(model, processor, eval_samples, issues):
    """Check transcription quality on eval samples.

    Verifies:
    - No empty hypotheses
    - No chatty patterns in output
    - Overall WER < 80%

    Args:
        model: LoRA adapter model in eval mode.
        processor: Qwen3ASRProcessor.
        eval_samples: List of dicts with audio and reference.
        issues: List to append issues to.
    """
    normalizer = EnglishTextNormalizer()

    empty_count = 0
    chatty_count = 0
    all_references = []
    all_hypotheses = []

    for i, sample in enumerate(eval_samples):
        audio = sample["audio"]
        reference = sample["reference"]

        print(f"  [{i+1}/{len(eval_samples)}] Transcribing...")
        try:
            hypothesis = transcribe_sample(model, processor, audio)
        except Exception as e:
            print(f"    ERROR: Transcription failed: {e}")
            issues.append(f"Transcription failed for sample {i+1}: {e}")
            hypothesis = ""

        # Check for empty hypothesis
        if not hypothesis.strip():
            empty_count += 1
            print(f"    EMPTY hypothesis")
        else:
            print(f"    Ref: {reference[:80]}...")
            print(f"    Hyp: {hypothesis[:80]}...")

        # Check for chatty patterns (case-insensitive substring match)
        hyp_lower = hypothesis.lower()
        for pattern in CHATTY_PATTERNS:
            if pattern in hyp_lower:
                chatty_count += 1
                print(f"    CHATTY: matched pattern '{pattern}'")
                break

        # Normalize for WER computation
        ref_norm = normalizer(reference)
        hyp_norm = normalizer(hypothesis) if hypothesis.strip() else ""

        if ref_norm.strip():
            all_references.append(ref_norm)
            all_hypotheses.append(hyp_norm)

    # Report empty hypotheses
    if empty_count > 0:
        issues.append(
            f"{empty_count}/{len(eval_samples)} hypotheses are empty"
        )
        print(f"\n  Empty hypotheses: {empty_count}/{len(eval_samples)}")

    # Report chatty patterns
    if chatty_count > 0:
        issues.append(
            f"{chatty_count}/{len(eval_samples)} hypotheses contain "
            f"chatty/wrapper text"
        )
        print(f"  Chatty hypotheses: {chatty_count}/{len(eval_samples)}")

    # Compute overall WER
    if all_references:
        overall_wer = jiwer.wer(all_references, all_hypotheses)
        print(f"\n  Overall WER: {overall_wer*100:.2f}% "
              f"(threshold: {WER_THRESHOLD*100:.0f}%) -- "
              f"{'PASS' if overall_wer < WER_THRESHOLD else 'FAIL'}")

        if overall_wer >= WER_THRESHOLD:
            issues.append(
                f"Overall WER {overall_wer*100:.2f}% exceeds "
                f"{WER_THRESHOLD*100:.0f}% threshold"
            )
    else:
        issues.append("No valid references for WER computation")
        print("  WARNING: No valid references for WER computation")


def check_adapter_roundtrip(adapter_dir, eval_samples, issues):
    """Check adapter save/load round-trip produces identical output.

    Per D-11: exact tensor comparison between original and reloaded adapter.

    Steps:
    1. Load base model + adapter from adapter_dir -> model_A
    2. Run model_A.generate() on first eval sample -> tokens_A
    3. Save model_A adapter to temp directory
    4. Load base model + adapter from temp directory -> model_B
    5. Run model_B.generate() on same sample -> tokens_B
    6. Compare with torch.equal(tokens_A, tokens_B)

    Also verifies adapter file sizes are reasonable (> 1KB, < 100MB).

    Args:
        adapter_dir: Path to saved adapter.
        eval_samples: List of eval samples (uses first sample).
        issues: List to append issues to.
    """
    from peft import PeftModel

    if not eval_samples:
        issues.append("No eval samples available for round-trip test")
        return

    # Check adapter file sizes
    print("  Checking adapter file sizes...")
    adapter_files = []
    for f in os.listdir(adapter_dir):
        fpath = os.path.join(adapter_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            adapter_files.append((f, size))
            print(f"    {f}: {size:,} bytes")

    safetensors_files = [
        (f, s) for f, s in adapter_files
        if f.endswith(".safetensors") or f.endswith(".bin")
    ]
    if not safetensors_files:
        issues.append("No adapter weight files found (.safetensors or .bin)")
        return

    for fname, fsize in safetensors_files:
        if fsize < 1024:
            issues.append(f"Adapter file {fname} is too small ({fsize} bytes)")
        elif fsize > 100 * 1024 * 1024:
            issues.append(
                f"Adapter file {fname} is too large "
                f"({fsize / 1024 / 1024:.1f}MB > 100MB)"
            )

    # Set deterministic seeds
    torch.manual_seed(42)

    # Step 1: Load original adapter
    print("  Loading model_A (original adapter)...")
    model_a, processor = load_adapter_model(adapter_dir)

    # Step 2: Generate with model_A on fixed input
    sample = eval_samples[0]
    audio = sample["audio"]

    print("  Generating with model_A...")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio.numpy()},
            ],
        },
    ]

    inputs = processor(
        conversations=conversation,
        audios=[audio.numpy()],
        sampling_rate=DEFAULT_SAMPLE_RATE,
        return_tensors="pt",
    )

    device = next(model_a.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        torch.manual_seed(42)
        tokens_a = model_a.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Step 3: Save adapter to temp directory
    tmpdir = tempfile.mkdtemp(prefix="lora_roundtrip_")
    print(f"  Saving adapter to temp dir: {tmpdir}")
    model_a.save_pretrained(tmpdir)
    processor.save_pretrained(tmpdir)

    # Cleanup model_A to free memory
    del model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 4: Load from temp directory
    print("  Loading model_B (reloaded adapter)...")
    model_b, processor_b = load_adapter_model(tmpdir)

    # Step 5: Generate with model_B on same input
    print("  Generating with model_B...")
    # Rebuild inputs with new processor (should be identical)
    inputs_b = processor_b(
        conversations=conversation,
        audios=[audio.numpy()],
        sampling_rate=DEFAULT_SAMPLE_RATE,
        return_tensors="pt",
    )

    device_b = next(model_b.parameters()).device
    inputs_b = {k: v.to(device_b) if isinstance(v, torch.Tensor) else v
                for k, v in inputs_b.items()}

    with torch.no_grad():
        torch.manual_seed(42)
        tokens_b = model_b.generate(
            **inputs_b,
            max_new_tokens=256,
            do_sample=False,
        )

    # Step 6: Compare tokens
    # Move both to CPU for comparison
    tokens_a_cpu = tokens_a.cpu()
    tokens_b_cpu = tokens_b.cpu()

    equal = torch.equal(tokens_a_cpu, tokens_b_cpu)
    print(f"  Adapter round-trip: {'PASS' if equal else 'FAIL'}")

    if not equal:
        issues.append("Adapter round-trip failed: output tokens differ")
        # Print some diagnostic info
        if tokens_a_cpu.shape != tokens_b_cpu.shape:
            print(f"    Shape mismatch: A={tokens_a_cpu.shape}, "
                  f"B={tokens_b_cpu.shape}")
        else:
            diff_mask = tokens_a_cpu != tokens_b_cpu
            n_diff = diff_mask.sum().item()
            total = tokens_a_cpu.numel()
            print(f"    {n_diff}/{total} tokens differ")

    # Cleanup
    del model_b, processor_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clean up temp directory
    import shutil
    try:
        shutil.rmtree(tmpdir)
        print(f"  Cleaned up temp dir: {tmpdir}")
    except Exception as e:
        print(f"  WARNING: Failed to clean temp dir: {e}")


def check_loss_trend(log_dir, issues):
    """Check that training loss shows a decreasing trend.

    Reads trainer_state.json from log_dir and compares first-quarter
    mean loss to last-quarter mean loss.

    Args:
        log_dir: Path to training output directory containing
            trainer_state.json (or a checkpoint subdirectory with it).
        issues: List to append issues to.
    """
    # Look for trainer_state.json in log_dir or checkpoint subdirectories
    state_path = os.path.join(log_dir, "trainer_state.json")
    if not os.path.isfile(state_path):
        # Check checkpoint subdirectories
        for entry in sorted(os.listdir(log_dir)):
            candidate = os.path.join(log_dir, entry, "trainer_state.json")
            if os.path.isfile(candidate):
                state_path = candidate
                break

    if not os.path.isfile(state_path):
        issues.append(
            f"trainer_state.json not found in {log_dir} "
            f"or its checkpoint subdirectories"
        )
        print(f"  FAIL: trainer_state.json not found")
        return

    print(f"  Reading: {state_path}")
    with open(state_path, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    losses = [
        entry["loss"] for entry in log_history
        if "loss" in entry
    ]

    if len(losses) < 2:
        issues.append(
            f"Not enough loss entries to determine trend "
            f"(found {len(losses)}, need at least 2)"
        )
        print(f"  FAIL: Only {len(losses)} loss entries found")
        return

    # Compute first and last quarter means
    n = len(losses)
    quarter = max(1, n // 4)
    first_quarter = losses[:quarter]
    last_quarter = losses[-quarter:]

    first_quarter_mean = np.mean(first_quarter)
    last_quarter_mean = np.mean(last_quarter)

    # Print loss curve summary
    print(f"  Total loss entries: {n}")
    print(f"  First step loss: {losses[0]:.4f}")
    print(f"  Last step loss:  {losses[-1]:.4f}")
    print(f"  Min loss:        {min(losses):.4f}")
    print(f"  First quarter mean: {first_quarter_mean:.4f}")
    print(f"  Last quarter mean:  {last_quarter_mean:.4f}")

    if last_quarter_mean >= first_quarter_mean:
        trend = "NOT DECREASING"
        issues.append(
            f"Loss not decreasing: first quarter mean={first_quarter_mean:.4f}, "
            f"last quarter mean={last_quarter_mean:.4f}"
        )
    else:
        trend = "DECREASING"

    print(f"  Trend: {trend} -- "
          f"{'PASS' if trend == 'DECREASING' else 'FAIL'}")


# -- CLI ----------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate LoRA prototype training results"
    )
    parser.add_argument(
        "--adapter_dir", required=True,
        help="Path to saved adapter directory "
             "(contains adapter_config.json and weights)"
    )
    parser.add_argument(
        "--log_dir", required=True,
        help="Path to training output directory "
             "(contains trainer_state.json or checkpoint subdirs)"
    )
    parser.add_argument(
        "--eval_manifest", required=True,
        help="Path to CSV manifest for evaluation samples "
             "(columns: audio_path, sentence)"
    )
    parser.add_argument(
        "--num_eval_samples", type=int, default=DEFAULT_NUM_EVAL_SAMPLES,
        help=f"Number of samples to transcribe "
             f"(default: {DEFAULT_NUM_EVAL_SAMPLES})"
    )

    return parser.parse_args()


# -- Main ---------------------------------------------------------------------

def main():
    """Run all 4 validation checks and report results."""
    args = parse_args()
    issues = []

    print(f"\n{'='*60}")
    print("LoRA Prototype Validation")
    print(f"{'='*60}")
    print(f"  Adapter dir:    {args.adapter_dir}")
    print(f"  Log dir:        {args.log_dir}")
    print(f"  Eval manifest:  {args.eval_manifest}")
    print(f"  Eval samples:   {args.num_eval_samples}")

    # Validate inputs exist
    if not os.path.isdir(args.adapter_dir):
        print(f"\nERROR: Adapter directory not found: {args.adapter_dir}")
        sys.exit(1)
    if not os.path.isdir(args.log_dir):
        print(f"\nERROR: Log directory not found: {args.log_dir}")
        sys.exit(1)
    if not os.path.isfile(args.eval_manifest):
        print(f"\nERROR: Eval manifest not found: {args.eval_manifest}")
        sys.exit(1)

    # -- Check 1: VRAM Usage ---------------------------------------------------
    print(f"\n{'='*60}")
    print("Check 1: VRAM Usage")
    print(f"{'='*60}")
    check_vram(args.adapter_dir, issues)

    # -- Check 2: Training Loss Trend ------------------------------------------
    print(f"\n{'='*60}")
    print("Check 2: Training Loss Trend")
    print(f"{'='*60}")
    check_loss_trend(args.log_dir, issues)

    # -- Load model and eval samples for Checks 3 and 4 -----------------------
    print(f"\n{'='*60}")
    print("Loading model and eval samples for transcription checks")
    print(f"{'='*60}")

    eval_samples = load_eval_samples(
        args.eval_manifest,
        args.num_eval_samples,
    )
    print(f"  Loaded {len(eval_samples)} eval samples")

    if not eval_samples:
        issues.append("No eval samples loaded -- cannot run transcription checks")
        print("  ERROR: No eval samples loaded")
    else:
        # Load adapter model once for checks 3 and 4
        print("  Loading adapter model...")
        model, processor = load_adapter_model(args.adapter_dir)

        # -- Check 3: Transcription Quality ------------------------------------
        print(f"\n{'='*60}")
        print("Check 3: Transcription Quality")
        print(f"{'='*60}")
        check_transcription_quality(model, processor, eval_samples, issues)

        # Cleanup model to free memory before round-trip test
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Check 4: Adapter Round-Trip ---------------------------------------
        print(f"\n{'='*60}")
        print("Check 4: Adapter Save/Load Round-Trip")
        print(f"{'='*60}")
        check_adapter_roundtrip(args.adapter_dir, eval_samples, issues)

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Validation Summary: {len(issues)} issue(s)")
    print(f"{'='*60}")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    if not issues:
        print("  All checks passed!")
    print(f"{'='*60}")

    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
