"""
Evaluation bridge: LoRA adapter → inference → fairness metrics.

Loads a trained LoRA adapter on Qwen3-ASR-1.7B, runs inference on
Fair-Speech (ethnicity axis) and Common Voice (accent axis), computes
per-group WER and fairness metrics (max-min ratio, relative gap%,
WER std, bootstrap 95% CIs).

Output format matches existing benchmarking pipeline (run_inference.py
prediction CSV columns, compute_fairness_metrics.py JSON structure).

Usage:
    python scripts/training/evaluate_adapter.py \
        --adapter_path outputs/standard-lora/adapter \
        --model_name standard-lora \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_dev.csv \
        --output_dir outputs/standard-lora/eval
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional

# -- Constants ----------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MIN_GROUP_SIZE = 50
SEED = 42
DEFAULT_OUTPUT_DIR = "outputs/standard-lora/eval"

# -- Text normalization (same as run_inference.py) ----------------------------

try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    from whisper_normalizer.english import EnglishTextNormalizer
_normalizer = EnglishTextNormalizer()

def normalize_text(text):
    if not text or not isinstance(text, str):
        return ""
    return _normalizer(text)


# -- Model helpers ------------------------------------------------------------

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    def forward(self, input_ids=None, attention_mask=None,
                input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs)
    cls.forward = forward
    cls._forward_patched = True


def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[VRAM {label}] Allocated: {allocated:.2f}GB, Peak: {peak:.2f}GB")


def load_model_with_adapter(adapter_path, device="cuda"):
    """Load Qwen3-ASR base model + LoRA adapter for inference."""
    from qwen_asr import Qwen3ASRModel
    from peft import PeftModel

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=None)
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)

    # Patch embeddings for PeftModel
    cls = model.__class__
    if not hasattr(cls, "_embeddings_patched"):
        cls.get_input_embeddings = lambda self: self.thinker.model.embed_tokens
        cls.set_input_embeddings = lambda self, v: setattr(
            self.thinker.model, "embed_tokens", v)
        cls._embeddings_patched = True

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model = model.to(device)

    print_gpu_memory("After adapter load")
    return model, processor


# -- Audio loading ------------------------------------------------------------

def load_audio(path, sample_rate=16000):
    """Load audio via soundfile (same as data_loader.py)."""
    audio, sr = sf.read(path, dtype="float32")
    audio = torch.from_numpy(audio)
    if audio.dim() > 1:
        audio = audio.mean(dim=-1)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.numpy()


# -- Inference ----------------------------------------------------------------

def run_inference_on_manifest(model, processor, manifest_df, device="cuda"):
    """Run single-sample inference on a manifest DataFrame."""
    predictions = []
    n = len(manifest_df)
    model_dtype = next(model.parameters()).dtype
    running_wer_sum = 0.0
    running_count = 0

    for i, (_, row) in enumerate(manifest_df.iterrows()):
        audio_np = load_audio(row["audio_path"])
        if audio_np is None:
            predictions.append({
                "idx": i, "hypothesis_raw": "", "hypothesis": "",
                "reference": normalize_text(str(row.get("sentence", "")))
            })
            continue

        # Build chat-format input with assistant prefix
        # The LoRA adapter was trained with prefix masking — it only learned
        # to predict transcript tokens after "<asr_text>". We must include
        # "language English<asr_text>" in the prompt so the model generates
        # only the transcript portion.
        conversation = [{"role": "user",
                         "content": [{"type": "audio", "audio": audio_np}]}]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        text += "language English<asr_text>"

        inputs = processor(text=text, audio=[audio_np],
                           return_tensors="pt", padding=False)
        inputs_gpu = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.is_floating_point():
                    v = v.to(model_dtype)
            inputs_gpu[k] = v

        with torch.no_grad():
            gen_output = model.generate(
                **inputs_gpu, max_new_tokens=256, do_sample=False)

        if hasattr(gen_output, "sequences"):
            output_ids = gen_output.sequences
        elif isinstance(gen_output, torch.Tensor):
            output_ids = gen_output
        else:
            output_ids = gen_output[0]

        input_len = inputs_gpu.get("input_ids", torch.tensor([])).shape[-1]
        new_tokens = output_ids[0, input_len:]
        hypothesis_raw = processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True)

        ref = normalize_text(str(row.get("sentence", "")))
        hyp = normalize_text(hypothesis_raw)

        if ref and hyp:
            try:
                wer_val = jiwer.wer(ref, hyp)
            except Exception:
                wer_val = 1.0
        else:
            wer_val = 1.0 if ref else 0.0

        running_wer_sum += wer_val
        running_count += 1

        predictions.append({
            "idx": i,
            "hypothesis_raw": hypothesis_raw,
            "hypothesis": hyp,
            "reference": ref,
            "wer": round(wer_val, 4),
        })

        if (i + 1) % 100 == 0 or (i + 1) == n:
            avg_wer = running_wer_sum / running_count if running_count else 0
            print(f"  [{i+1}/{n}] WER so far: {avg_wer:.4f}")

    return predictions


# -- Fairness metrics (matches compute_fairness_metrics.py) -------------------

def compute_group_wer(df):
    """Corpus-level WER for a group (not per-utterance average)."""
    refs = df["reference"].fillna("").astype(str).tolist()
    hyps = df["hypothesis"].fillna("").astype(str).tolist()
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if not valid:
        return {"wer": None, "n_utterances": 0}
    refs_clean, hyps_clean = zip(*valid)
    wer_val = jiwer.wer(list(refs_clean), list(hyps_clean))
    output = jiwer.process_words(list(refs_clean), list(hyps_clean))
    return {
        "wer": wer_val,
        "n_utterances": len(refs_clean),
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "hits": output.hits,
    }


def bootstrap_wer(df, n_bootstrap=1000):
    """Bootstrap 95% CI for WER."""
    refs = df["reference"].fillna("").astype(str).tolist()
    hyps = df["hypothesis"].fillna("").astype(str).tolist()
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if len(valid) < 10:
        return None, None, None
    refs_clean, hyps_clean = zip(*valid)
    refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)
    n = len(refs_clean)
    rng = np.random.RandomState(SEED)
    wers = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        try:
            wers.append(jiwer.wer(
                [refs_clean[i] for i in indices],
                [hyps_clean[i] for i in indices]))
        except Exception:
            continue
    if not wers:
        return None, None, None
    return np.mean(wers), np.percentile(wers, 2.5), np.percentile(wers, 97.5)


def compute_fairness_metrics(group_wers):
    """Fairness metrics from per-group WERs."""
    valid = {k: v for k, v in group_wers.items()
             if v["wer"] is not None and v["n_utterances"] >= MIN_GROUP_SIZE}
    if len(valid) < 2:
        return {"error": "Too few valid groups"}

    wer_values = {k: v["wer"] for k, v in valid.items()}
    wer_list = list(wer_values.values())
    best = min(wer_values, key=wer_values.get)
    worst = max(wer_values, key=wer_values.get)

    return {
        "best_group": best, "best_wer": wer_values[best],
        "worst_group": worst, "worst_wer": wer_values[worst],
        "max_min_ratio": max(wer_list) / min(wer_list) if min(wer_list) > 0 else float("inf"),
        "relative_gap_pct": (wer_values[worst] - wer_values[best]) / wer_values[best] * 100 if wer_values[best] > 0 else float("inf"),
        "wer_std": float(np.std(wer_list)),
        "wer_range": max(wer_list) - min(wer_list),
    }


# -- Evaluation pipeline for one dataset -------------------------------------

def evaluate_on_dataset(model, processor, manifest_path, dataset_name,
                        demographic_axis, model_name, output_dir, device,
                        n_bootstrap=1000, skip_bootstrap=False):
    """Full evaluation pipeline for one dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} ({demographic_axis} axis)")
    print(f"{'='*60}")

    df = pd.read_csv(manifest_path)
    print(f"  Manifest: {manifest_path} ({len(df)} rows)")

    # Run inference
    t0 = time.time()
    predictions = run_inference_on_manifest(model, processor, df, device)
    elapsed = time.time() - t0
    print(f"  Inference: {elapsed/60:.1f} min ({elapsed/len(df):.2f}s/sample)")

    # Build prediction DataFrame
    pred_rows = []
    for pred in predictions:
        row = df.iloc[pred["idx"]]
        pred_rows.append({
            "utterance_id": row.get("utterance_id", ""),
            "reference": pred["reference"],
            "hypothesis": pred["hypothesis"],
            "hypothesis_raw": pred["hypothesis_raw"],
            "wer": pred["wer"],
            "num_hyp_words": len(pred["hypothesis"].split()) if pred["hypothesis"] else 0,
            "num_ref_words": len(pred["reference"].split()) if pred["reference"] else 0,
            "perturbation": "clean",
            "gender": row.get("gender", ""),
            "accent": row.get("accent", ""),
            "age": row.get("age", ""),
            "model": model_name,
            "generation": 3,
            "architecture": "Audio enc + Qwen3 LLM + LoRA",
            "ethnicity": row.get("ethnicity", ""),
            "first_language": row.get("first_language", ""),
        })

    pred_df = pd.DataFrame(pred_rows)

    # Save prediction CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"predictions_{model_name}_{dataset_name}.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"  Predictions saved: {csv_path}")

    # Overall WER
    overall = compute_group_wer(pred_df)
    print(f"  Overall WER: {overall['wer']:.4f} ({overall['n_utterances']} utterances)")

    # Per-group WER
    if demographic_axis and demographic_axis in pred_df.columns:
        grouped = pred_df[pred_df[demographic_axis].notna() &
                          (pred_df[demographic_axis] != "")].groupby(demographic_axis)
    else:
        grouped = []

    group_wers = {}
    group_cis = {}
    for group_name, group_df in grouped:
        gwer = compute_group_wer(group_df)
        group_wers[group_name] = gwer
        if not skip_bootstrap and gwer["n_utterances"] >= MIN_GROUP_SIZE:
            mean, ci_lo, ci_hi = bootstrap_wer(group_df, n_bootstrap)
            group_cis[group_name] = {"mean": mean, "ci_lower": ci_lo, "ci_upper": ci_hi}
        else:
            group_cis[group_name] = {"mean": None, "ci_lower": None, "ci_upper": None}

    # Fairness metrics
    fairness = compute_fairness_metrics(group_wers)

    # Print per-group results
    print(f"\n  {'Group':<20} {'WER':>8} {'N':>6} {'95% CI':>20}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*20}")
    for gname, gwer in sorted(group_wers.items(), key=lambda x: x[1].get("wer", 999) or 999):
        ci = group_cis.get(gname, {})
        n = gwer["n_utterances"]
        w = gwer["wer"]
        if w is None:
            continue
        marker = " *" if n < MIN_GROUP_SIZE else ""
        ci_str = (f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                  if ci.get("ci_lower") is not None else "N/A")
        print(f"  {gname:<20} {w:>8.4f} {n:>6}{marker} {ci_str:>20}")

    if "error" not in fairness:
        print(f"\n  Fairness: max-min={fairness['max_min_ratio']:.2f}, "
              f"gap={fairness['relative_gap_pct']:.1f}%, "
              f"std={fairness['wer_std']:.4f}")

    # Build result dict
    per_group = {}
    for gname, gwer in group_wers.items():
        if gwer["wer"] is None:
            continue
        ci = group_cis.get(gname, {})
        per_group[gname] = {
            "wer": gwer["wer"], "n": gwer["n_utterances"],
            "ci_lower": ci.get("ci_lower"), "ci_upper": ci.get("ci_upper"),
        }

    return {
        "demographic_axis": demographic_axis,
        "overall_wer": overall["wer"],
        "n_utterances": overall["n_utterances"],
        "per_group": per_group,
        "fairness_metrics": fairness,
    }


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--model_name", default="standard-lora")
    parser.add_argument("--fs_manifest", default="outputs/manifests/fs_train.csv")
    parser.add_argument("--cv_manifest", default="outputs/manifests/cv_dev.csv")
    parser.add_argument("--ls_manifest", default=None)
    parser.add_argument("--eval_manifest_override", default=None,
                        help="Explicit eval manifest, bypassing fallback chain")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_bootstrap", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit samples per dataset (-1 = all)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EVALUATION BRIDGE: Adapter -> Inference -> Fairness")
    print(f"{'='*60}")
    print(f"  Adapter: {args.adapter_path}")
    print(f"  Model name: {args.model_name}")
    print_gpu_memory("Before model load")

    # Load model + adapter once
    model, processor = load_model_with_adapter(args.adapter_path, args.device)

    results = {"model_name": args.model_name, "adapter_path": args.adapter_path,
               "evaluation_date": time.strftime("%Y-%m-%d"), "datasets": {}}

    # -- Fair-Speech evaluation --
    fs_manifest = args.eval_manifest_override or args.fs_manifest
    # Check for eval subset from training (speaker-disjoint from training data)
    eval_subset = os.path.join(os.path.dirname(args.adapter_path), "eval_subset.csv")
    if os.path.exists(eval_subset) and not args.eval_manifest_override:
        fs_manifest = eval_subset
        print(f"\n  Using training eval subset: {eval_subset}")

    if os.path.exists(fs_manifest):
        fs_df = pd.read_csv(fs_manifest)
        if args.max_samples > 0:
            fs_df = fs_df.head(args.max_samples)
            fs_tmp = os.path.join(args.output_dir, "_fs_subset.csv")
            os.makedirs(args.output_dir, exist_ok=True)
            fs_df.to_csv(fs_tmp, index=False)
            fs_manifest = fs_tmp
        results["datasets"]["fairspeech"] = evaluate_on_dataset(
            model, processor, fs_manifest, "fairspeech", "ethnicity",
            args.model_name, args.output_dir, args.device,
            args.n_bootstrap, args.skip_bootstrap)

    # -- Common Voice evaluation --
    if os.path.exists(args.cv_manifest):
        cv_manifest = args.cv_manifest
        cv_df = pd.read_csv(cv_manifest)
        if args.max_samples > 0:
            cv_df = cv_df.head(args.max_samples)
            cv_tmp = os.path.join(args.output_dir, "_cv_subset.csv")
            os.makedirs(args.output_dir, exist_ok=True)
            cv_df.to_csv(cv_tmp, index=False)
            cv_manifest = cv_tmp
        results["datasets"]["commonvoice"] = evaluate_on_dataset(
            model, processor, cv_manifest, "commonvoice", "accent",
            args.model_name, args.output_dir, args.device,
            args.n_bootstrap, args.skip_bootstrap)

    # -- LibriSpeech evaluation (optional) --
    if args.ls_manifest and os.path.exists(args.ls_manifest):
        results["datasets"]["librispeech"] = evaluate_on_dataset(
            model, processor, args.ls_manifest, "librispeech", None,
            args.model_name, args.output_dir, args.device,
            args.n_bootstrap, args.skip_bootstrap)

    # Save analysis JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"analysis_{args.model_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Analysis saved: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for ds_name, ds_results in results["datasets"].items():
        fm = ds_results.get("fairness_metrics", {})
        print(f"\n  {ds_name}: WER={ds_results['overall_wer']:.4f} "
              f"({ds_results['n_utterances']} utt)")
        if "error" not in fm:
            print(f"    max-min ratio: {fm['max_min_ratio']:.2f}")
            print(f"    gap%: {fm['relative_gap_pct']:.1f}%")
            print(f"    best: {fm['best_group']} ({fm['best_wer']:.4f})")
            print(f"    worst: {fm['worst_group']} ({fm['worst_wer']:.4f})")


if __name__ == "__main__":
    main()
