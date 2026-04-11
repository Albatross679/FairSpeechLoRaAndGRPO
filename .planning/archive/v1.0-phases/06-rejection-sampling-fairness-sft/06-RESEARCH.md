# Phase 6: Rejection Sampling -- Fairness-Aware SFT - Research

**Researched:** 2026-04-08
**Domain:** Rejection sampling distillation from RL-trained ASR model to SFT
**Confidence:** MEDIUM-HIGH

## Summary

Phase 6 distills the fairness-aware GRPO checkpoint from Phase 5 into a standard SFT model via rejection sampling. The core idea: generate many transcription candidates from the GRPO model for each utterance, filter to keep only those that are both accurate (low WER) and fair (low demographic gap), then fine-tune a fresh LoRA adapter on this curated dataset. The result is a model with fairness properties baked in through its training data, without needing RL at inference time.

This is a well-established pattern in the LLM post-training pipeline (LLaMA 2, DeepSeek-R1, Constitutional AI) but has NOT been applied to ASR fairness before. The novelty here is the fairness-aware filtering criterion, which goes beyond simple reward-model scoring to consider demographic parity across groups. The existing project infrastructure (data loaders, evaluation bridge, model loading, LoRA config) handles 90% of the implementation -- the new work is (1) a multi-completion generation script, (2) a fairness-aware filtering pipeline, and (3) a second-round SFT training run.

Key risk: ASR transcription has limited output diversity compared to open-ended text generation (the "unique-answer problem" from Phase 5). Temperature-based sampling at 0.7-1.0 may produce only minor variations per utterance. The filtering step must define thresholds carefully to retain enough data for meaningful SFT while actually enforcing quality and fairness.

**Primary recommendation:** Build a three-stage pipeline: (1) generate N=10 candidates per utterance with temperature=0.8, (2) filter using joint WER + demographic-gap thresholds, retaining the best candidate per utterance, (3) SFT a fresh LoRA adapter on the filtered dataset using the same infrastructure as Phase 3.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 4.57.6 | Model loading, `model.generate()` with `num_return_sequences` | Already installed, used throughout project [VERIFIED: pip show] |
| peft | 0.18.1 | LoRA adapter creation + save/load for SFT round | Already installed, used in Phases 2-5 [VERIFIED: pip show] |
| torch | 2.8.0+cu128 | Inference + training | Already installed [VERIFIED: pip show] |
| jiwer | 4.0.0 | Per-candidate WER computation for filtering | Already installed, used in reward + evaluation [VERIFIED: pip show] |
| qwen-asr | 0.0.6 | Qwen3-ASR model loading | Already installed [VERIFIED: pip show] |
| soundfile | 0.13.1 | Audio loading | Already installed [VERIFIED: pip show] |
| pandas | (installed) | Manifest manipulation, filtering, demographic grouping | Already installed, used in data pipeline |
| numpy | (installed) | Array ops for reward scoring and thresholding | Already installed |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| wandb | 0.25.1 | Track generation + SFT metrics | Already installed [VERIFIED: pip show] |
| whisper-normalizer | (installed) | Text normalization for WER consistency | Already used in evaluate_adapter.py |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `model.generate(num_return_sequences=N)` | vLLM batch inference | vLLM not installed, adds dependency; HF generate is sufficient for ~45K utterances x 10 candidates [VERIFIED: vLLM not installed] |
| Custom filtering script | TRL BestOfNSampler | TRL not installed; BestOfNSampler is deprecated in TRL 0.25+ [CITED: huggingface.co/docs/trl/best_of_n]; custom is simpler for our fairness criteria |
| Fresh LoRA from base model | Continue training GRPO adapter with SFT | Fresh adapter avoids RL artifacts; merge-then-retrain is cleaner separation [ASSUMED] |

**Installation:**
```bash
# No new packages needed. Everything is already installed.
```

## Architecture Patterns

### Recommended Project Structure

```
scripts/training/
    generate_candidates.py       # NEW: Generate N candidates per utterance from GRPO checkpoint
    filter_rejection_samples.py  # NEW: Filter by WER + fairness criteria
    train_rejection_sft.py       # NEW: SFT on filtered dataset (reuses train_standard_lora.py patterns)

outputs/rejection-sampling/
    candidates/                  # Raw candidate CSVs (utterance_id, candidate_idx, hypothesis, wer)
    filtered/                    # Filtered dataset CSV (one best candidate per utterance)
    adapter/                     # Final rejection-sampled SFT adapter
    eval/                        # Evaluation results (reuse evaluate_adapter.py)
```

### Pattern 1: Three-Stage Pipeline (Generate -> Filter -> Train)

**What:** Cleanly separate generation, filtering, and training into independent scripts with CSV intermediates.

**When to use:** Always. This separation enables (a) regenerating candidates without retraining, (b) experimenting with different filtering thresholds without regenerating, and (c) reusing the existing SFT infrastructure for the training step.

**Example:**
```python
# Stage 1: generate_candidates.py
# For each utterance in the training manifest, generate N candidate transcriptions
# from the GRPO checkpoint using temperature sampling

model, processor = load_model_with_adapter(grpo_adapter_path)

for batch in dataloader:
    audio = batch["audio"]
    # Build chat-format input (same as evaluate_adapter.py)
    inputs = prepare_inputs(processor, audio)
    
    # Generate N candidates with temperature sampling
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=N,  # N=10 candidates per utterance
    )
    
    # Decode and compute WER for each candidate
    for candidate_idx in range(N):
        hypothesis = decode(outputs[candidate_idx])
        wer = jiwer.wer(reference, hypothesis)
        save_candidate(utterance_id, candidate_idx, hypothesis, wer, demographic_group)
```

```python
# Stage 2: filter_rejection_samples.py
# Load all candidates, apply joint WER + fairness filter

candidates_df = pd.read_csv("outputs/rejection-sampling/candidates/all_candidates.csv")

# Per-utterance: select best candidate (lowest WER)
best_per_utterance = candidates_df.loc[
    candidates_df.groupby("utterance_id")["wer"].idxmin()
]

# Apply WER threshold: reject utterances where best candidate WER > threshold
accepted = best_per_utterance[best_per_utterance["wer"] <= WER_THRESHOLD]

# Apply fairness filter: compute per-group WER on accepted set,
# reject utterances from overperforming groups if gap exceeds threshold
filtered = apply_fairness_filter(accepted, max_gap=FAIRNESS_GAP_THRESHOLD)

filtered.to_csv("outputs/rejection-sampling/filtered/filtered_dataset.csv", index=False)
```

```python
# Stage 3: train_rejection_sft.py
# Standard SFT on filtered dataset -- reuses train_standard_lora.py patterns

# Load base model (NOT the GRPO checkpoint -- fresh LoRA from base)
model, processor = load_model_and_processor()
model = apply_lora(model, rank=4, alpha=4, dropout=0.05, target_mlp=True)

# Use the filtered dataset as training data
train_dataset = ASRFairnessDataset("outputs/rejection-sampling/filtered/filtered_dataset.csv")
# ... standard Trainer setup identical to train_standard_lora.py
```

### Pattern 2: Best-of-N Per-Utterance Selection (Not Global Top-K)

**What:** For each utterance, select the single best candidate by WER. Do NOT use global top-K across all utterances.

**When to use:** Always for ASR. Unlike open-ended generation where global top-K finds the most reward-rich prompt-completion pairs, ASR requires maintaining coverage of ALL demographic groups and utterance types.

**Why:** Global top-K would bias toward easy utterances (short, clear audio) where WER is naturally low, dropping hard utterances that are disproportionately from minority demographics. Per-utterance selection preserves the demographic distribution of the original training set. [ASSUMED -- based on fairness reasoning, not empirically validated]

### Pattern 3: Fairness-Aware Filtering (Two-Pass)

**What:** First pass selects the best WER candidate per utterance. Second pass evaluates demographic gap on the resulting dataset and optionally rebalances by adjusting acceptance thresholds per group.

**When to use:** When the goal is explicitly fairness, not just accuracy distillation.

**Example:**
```python
def apply_fairness_filter(df, demographic_axis, max_gap_pct=10.0):
    """Two-pass fairness filter.
    
    Pass 1: Compute per-group WER on the candidate dataset
    Pass 2: If gap exceeds threshold, tighten WER threshold for 
            overperforming groups (accept slightly worse candidates 
            for groups that are already ahead)
    """
    group_wers = df.groupby(demographic_axis).apply(
        lambda g: jiwer.wer(g["reference"].tolist(), g["hypothesis"].tolist())
    )
    mean_wer = group_wers.mean()
    gap = group_wers.max() - group_wers.min()
    
    if (gap / mean_wer * 100) <= max_gap_pct:
        return df  # Already within fairness threshold
    
    # Tighten threshold for overperforming groups
    # Accept more "imperfect" candidates from underperforming groups
    # This balances the dataset without dropping utterances
    return rebalanced_df
```

### Anti-Patterns to Avoid

- **Anti-pattern: Training the SFT on the GRPO adapter** -- Do NOT continue fine-tuning the GRPO adapter. Create a fresh LoRA adapter on the base model. The GRPO adapter carries RL-specific weight distributions that may cause instability in SFT. The point is to distill the GRPO policy's outputs, not its weights. [ASSUMED]
- **Anti-pattern: Beam search for candidate generation** -- Beam search produces high-quality but low-diversity candidates. For rejection sampling, diversity matters more than individual quality. Use temperature sampling (0.7-1.0) with `do_sample=True`. [CITED: rlhfbook.com/c/09-rejection-sampling]
- **Anti-pattern: Global top-K selection** -- See Pattern 2 above. Global selection destroys demographic balance. [ASSUMED]
- **Anti-pattern: Generating too many candidates with large batch sizes** -- With T4 15GB VRAM, generating 10 candidates simultaneously via `num_return_sequences=10` will multiply KV cache memory. Process one utterance at a time (or small batches) with sequential candidate generation. [VERIFIED: T4 has 15360 MiB VRAM]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WER computation | Custom edit distance | `jiwer.wer()` | Already used everywhere in project, handles normalization edge cases |
| Text normalization | Custom normalizer | `EnglishTextNormalizer` from whisper-normalizer | Must match evaluation pipeline exactly (Pitfall from PITFALLS.md) |
| Audio loading + resampling | torchaudio.load | `soundfile.read()` + `torchaudio.functional.resample()` | torchaudio.load is broken in this env (documented in data_loader.py) |
| Model loading + LoRA | Custom PEFT setup | `load_model_with_adapter()` from evaluate_adapter.py | Battle-tested adapter loading pattern |
| Chat template formatting | Manual token building | `processor.apply_chat_template()` + manual assistant prefix | Pattern established in evaluate_adapter.py, handles special tokens |
| SFT training loop | Custom training loop | HuggingFace `Trainer` with `DataCollatorForQwen3ASR` | Already validated in train_standard_lora.py |
| Evaluation after training | Custom eval pipeline | `evaluate_adapter.py` as-is | Produces fairness metrics matching project standards |
| Speaker-disjoint splitting | Random split | `create_speaker_disjoint_split()` from train_standard_lora.py | Prevents leakage (Pitfall #9) |

**Key insight:** Phase 6 should reuse 90% of existing code. The only genuinely new code is (1) the candidate generation loop with `num_return_sequences`, (2) the fairness-aware filtering logic, and (3) a thin wrapper script that connects the filtered CSV to the existing SFT Trainer.

## Common Pitfalls

### Pitfall 1: Low Candidate Diversity Due to ASR Unique-Answer Problem

**What goes wrong:** Temperature sampling at 0.8 produces 10 candidates that are nearly identical for most utterances. Filtering becomes trivial (all candidates have similar WER) and the "rejection" step adds no value -- you end up with essentially the same dataset as direct GRPO inference.

**Why it happens:** ASR has one correct answer per utterance. Unlike reasoning tasks where multiple valid solution paths exist, transcription candidates vary only in minor word choices, insertions, or deletions. The GRPO checkpoint may have further reduced diversity by learning a peaked distribution.

**How to avoid:**
1. Measure candidate diversity BEFORE filtering: compute pairwise edit distance between the 10 candidates for a sample of 100 utterances. If median edit distance < 2 words, diversity is too low.
2. If diversity is low, increase temperature to 1.0 or even 1.2. Accept some quality degradation in individual candidates in exchange for meaningful selection signal.
3. Consider generating candidates from the base model (pre-GRPO) as well, to get structurally different errors.
4. If diversity remains insufficient, fall back to "best-of-N" distillation without fairness filtering -- still valuable as it selects the GRPO model's best output per utterance.

**Warning signs:** >80% of utterances have all 10 candidates producing identical WER. Filtered dataset is >95% the size of the original (nothing was rejected).

### Pitfall 2: Threshold Sensitivity -- Too Strict Drops Data, Too Lenient Adds Noise

**What goes wrong:** WER threshold of 0.05 (5%) rejects most candidates from hard utterances (accented speech, noisy audio), leaving a dataset biased toward easy, clear-speech utterances. WER threshold of 0.50 accepts nearly everything, including hallucinated transcriptions.

**Why it happens:** The WER distribution across utterances is highly skewed. Easy utterances may have 0% WER for all candidates, while hard ones may have 30-50% WER even for the best candidate. A single global threshold cannot handle this variance.

**How to avoid:**
1. Use per-utterance RELATIVE thresholds: accept the best candidate per utterance if it improves on the reference (or matches it within tolerance), rather than a global WER cutoff.
2. Start with a generous threshold (accept best candidate per utterance regardless of absolute WER), then tighten iteratively based on the resulting dataset's fairness metrics.
3. Log the acceptance rate and demographic distribution at each threshold level. Target 60-80% acceptance rate. [ASSUMED]
4. Consider a two-tier approach: (a) always accept if WER < 0.10, (b) conditionally accept if 0.10 < WER < 0.30 based on fairness need, (c) always reject if WER > 0.30.

**Warning signs:** Acceptance rate < 30% (too strict) or > 95% (too lenient). Demographic distribution of filtered set significantly skewed relative to original.

### Pitfall 3: VRAM Exhaustion During Multi-Candidate Generation

**What goes wrong:** Generating 10 candidates simultaneously with `num_return_sequences=10` on T4 (15GB) causes OOM. The KV cache for 10 parallel sequences on a 1.7B model exceeds available memory.

**Why it happens:** Each additional return sequence multiplies the KV cache memory. With Qwen3-ASR's audio encoder + decoder architecture, the audio features are shared but the decoder KV cache is per-sequence.

**How to avoid:**
1. Generate candidates in batches of 2-3 at a time, not all 10 simultaneously. Loop 4-5 times per utterance with `num_return_sequences=2`.
2. Use `torch.no_grad()` (no backprop needed during generation).
3. Clear KV cache between utterances: `torch.cuda.empty_cache()`.
4. Process utterances sequentially (batch_size=1 for the audio input).
5. Profile first 10 utterances with `torch.cuda.max_memory_allocated()` to calibrate safe batch size.

**Warning signs:** OOM errors. Memory climbing during generation loop.

### Pitfall 4: Fairness Filter Drops Minority-Group Utterances

**What goes wrong:** The WER threshold applied uniformly across groups disproportionately rejects utterances from demographic groups that the GRPO model struggles with (exactly the groups that need MORE data, not less).

**Why it happens:** If the GRPO model has higher WER on accented speech, even its best candidate for those utterances will have higher WER. A global threshold filters these out, making the SFT training data LESS diverse than the original.

**How to avoid:**
1. Apply per-group acceptance thresholds. For groups with higher baseline WER, use a more lenient threshold.
2. After filtering, verify that every demographic group retains at least MIN_GROUP_SIZE (50) utterances.
3. If any group drops below the minimum, relax its threshold until coverage is restored.
4. Log pre- and post-filter demographic distribution.

**Warning signs:** Any demographic group losing >50% of its utterances during filtering. Post-filter fairness metrics worse than pre-filter.

### Pitfall 5: SFT Overfitting on Small Filtered Dataset

**What goes wrong:** If the filtered dataset is significantly smaller than the original training set (e.g., 15K from 45K), the SFT round may overfit quickly, especially with the same number of epochs as Phase 3.

**Why it happens:** LoRA with a small dataset + multiple epochs leads to memorization. The filtered data may also be less diverse (easy utterances overrepresented).

**How to avoid:**
1. Reduce training to 1 epoch maximum for the SFT round.
2. Use early stopping based on validation loss.
3. Increase dropout slightly (0.1 instead of 0.05) if dataset is <50% of original size.
4. Monitor training loss curve -- if it drops to near-zero within first 20% of training, overfitting is likely.

**Warning signs:** Training loss near zero. Validation loss increasing while training loss decreasing.

## Code Examples

### Generating Multiple Candidates from GRPO Checkpoint

```python
# Source: adapted from evaluate_adapter.py inference pattern + HF model.generate docs
# [VERIFIED: model.generate supports num_return_sequences per HF transformers docs]

def generate_candidates(model, processor, audio_np, n_candidates=10,
                        temperature=0.8, max_new_tokens=256, batch_gen=2):
    """Generate N candidate transcriptions for one audio sample.
    
    Generates in sub-batches of batch_gen to avoid VRAM exhaustion.
    """
    candidates = []
    
    # Build chat-format input (same as evaluate_adapter.py)
    conversation = [{"role": "user",
                     "content": [{"type": "audio", "audio": audio_np}]}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    text += "language English<asr_text>"
    
    inputs = processor(text=text, audio=[audio_np],
                       return_tensors="pt", padding=False)
    inputs_gpu = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
    # Cast floating point tensors to model dtype
    model_dtype = next(model.parameters()).dtype
    inputs_gpu = {k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                  for k, v in inputs_gpu.items()}
    
    remaining = n_candidates
    while remaining > 0:
        n_this_batch = min(batch_gen, remaining)
        with torch.no_grad():
            gen_output = model.generate(
                **inputs_gpu,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=n_this_batch,
            )
        
        output_ids = gen_output.sequences if hasattr(gen_output, "sequences") else gen_output
        input_len = inputs_gpu.get("input_ids", torch.tensor([])).shape[-1]
        
        for seq_idx in range(n_this_batch):
            new_tokens = output_ids[seq_idx, input_len:]
            hypothesis = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            candidates.append(normalize_text(hypothesis))
        
        remaining -= n_this_batch
        torch.cuda.empty_cache()
    
    return candidates
```

### Fairness-Aware Filtering

```python
# Source: project-specific pattern combining jiwer WER + demographic grouping
# [VERIFIED: jiwer API from existing codebase usage]

def filter_candidates(candidates_df, demographic_axis, 
                      wer_threshold=0.30, min_group_size=50):
    """Select best candidate per utterance, apply WER + fairness filters.
    
    Args:
        candidates_df: DataFrame with columns [utterance_id, candidate_idx, 
                       hypothesis, reference, wer, demographic_group]
        demographic_axis: 'ethnicity' or 'accent'
        wer_threshold: Maximum WER for acceptance
        min_group_size: Minimum utterances per demographic group
    
    Returns:
        Filtered DataFrame with one row per accepted utterance
    """
    # Step 1: Best candidate per utterance (lowest WER)
    best = candidates_df.loc[
        candidates_df.groupby("utterance_id")["wer"].idxmin()
    ].copy()
    
    print(f"  Candidates generated for {len(best)} utterances")
    print(f"  Mean best-candidate WER: {best['wer'].mean():.4f}")
    
    # Step 2: Apply WER threshold
    accepted = best[best["wer"] <= wer_threshold].copy()
    print(f"  After WER filter (<={wer_threshold}): {len(accepted)} "
          f"({len(accepted)/len(best)*100:.1f}%)")
    
    # Step 3: Verify demographic coverage
    if demographic_axis in accepted.columns:
        group_counts = accepted.groupby(demographic_axis).size()
        small_groups = group_counts[group_counts < min_group_size]
        if len(small_groups) > 0:
            print(f"  WARNING: Groups below min_size={min_group_size}: "
                  f"{dict(small_groups)}")
            # Relax threshold for underrepresented groups
            for group_name in small_groups.index:
                group_mask = best[demographic_axis] == group_name
                group_candidates = best[group_mask].nsmallest(
                    min_group_size, "wer")
                accepted = pd.concat([
                    accepted[accepted[demographic_axis] != group_name],
                    group_candidates
                ])
    
    # Step 4: Log fairness metrics on filtered set
    if demographic_axis in accepted.columns:
        for group, gdf in accepted.groupby(demographic_axis):
            if len(gdf) >= min_group_size:
                gwer = jiwer.wer(gdf["reference"].tolist(), 
                                 gdf["hypothesis"].tolist())
                print(f"    {group}: WER={gwer:.4f}, N={len(gdf)}")
    
    return accepted.reset_index(drop=True)
```

### SFT Training on Filtered Data (Reusing Existing Infrastructure)

```python
# Source: adapted from train_standard_lora.py run_train()
# [VERIFIED: exact same Trainer + DataCollator + LoRA pattern as Phase 3]

def train_rejection_sft(filtered_csv, output_dir, locked_config_path):
    """SFT on rejection-sampled data using identical infrastructure to Phase 3."""
    
    # Load locked config (same LoRA hyperparameters)
    with open(locked_config_path) as f:
        locked = json.load(f)
    params = locked["params"]
    
    # Fresh LoRA on base model (NOT the GRPO adapter)
    model, processor = load_model_and_processor()  # from train_standard_lora.py
    model = apply_lora(
        model,
        rank=int(params["rank"]),
        alpha=int(params["rank"]) * int(params.get("alpha_ratio", 1)),
        dropout=float(params.get("dropout", 0.05)),
        target_mlp=bool(params.get("target_mlp", True)),
    )
    
    # Speaker-disjoint split on filtered data
    filtered_df = pd.read_csv(filtered_csv)
    train_df, eval_df = create_speaker_disjoint_split(filtered_df, test_size=0.1)
    
    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    eval_path = os.path.join(output_dir, "eval.csv")
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    
    # Create datasets and train -- identical to train_standard_lora.py
    axis = "ethnicity" if "ethnicity" in train_df.columns else "accent"
    train_dataset = ASRFairnessDataset(train_path, demographic_axis=axis)
    eval_dataset = ASRFairnessDataset(eval_path, demographic_axis=axis)
    collator = DataCollatorForQwen3ASR(processor)
    
    # Reduce epochs for smaller filtered dataset
    steps_per_epoch = max(1, len(train_df) // 4)  # effective_batch=4
    max_steps = steps_per_epoch * 1  # 1 epoch max to prevent overfitting
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=float(params["lr"]),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(1, max_steps // 5),
        save_strategy="steps",
        save_steps=max(1, max_steps // 3),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=42,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )
    
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    
    # Save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| RLHF -> deploy RL model | RLHF -> rejection sampling -> SFT distillation | LLaMA 2 (2023), DeepSeek-R1 (2025) | Removes RL overhead at inference, standard post-training pipeline now |
| Binary accept/reject threshold | Per-utterance best-of-N with adaptive thresholds | 2024-2025 (AdaSTaR, RSO) | Better data efficiency, reduces simplicity bias |
| Single reward model scoring | Multi-objective filtering (accuracy + fairness) | Novel for this project | Enables demographic-parity-aware distillation |
| Global top-K selection | Per-prompt best-of-N | LLaMA 2 (2023) | Maintains input distribution coverage |

**Deprecated/outdated:**
- TRL `BestOfNSampler`: Deprecated in TRL 0.25+. Do not use even if TRL were installed. [CITED: huggingface.co/docs/trl/best_of_n]
- PPO-based distillation: Replaced by simpler rejection sampling + SFT in modern pipelines. [CITED: rlhfbook.com/c/09-rejection-sampling]

## Key Design Decisions

### Decision 1: Number of Candidates Per Utterance

**Recommendation:** N=10 candidates per utterance. [ASSUMED -- based on literature guidance]

**Rationale:** The RLHF Book recommends 10-30 candidates for rejection sampling [CITED: rlhfbook.com/c/09-rejection-sampling]. LLaMA 2 used an unspecified but reportedly large number (possibly 25-70 for the 70B model). For ASR where output diversity is inherently lower than open-ended generation, 10 candidates is a practical starting point that balances compute cost (45K utterances x 10 = 450K inference passes) against selection quality.

**Compute estimate:** With ~3.5 sec/sample inference (from Phase 2 benchmarks), 450K passes would take ~437 hours. This is infeasible. Instead, we must reduce scope:
- Use a representative subset (e.g., 5K utterances x 10 candidates = 50K passes = ~48 hours)
- Or reduce to N=3-5 candidates per utterance on the full set (45K x 5 = 225K passes = ~218 hours)
- **Practical recommendation:** 5K representative subset x 10 candidates = ~48 hours. Feasible on T4.

### Decision 2: Fresh LoRA vs. Continuing GRPO Adapter

**Recommendation:** Train a fresh LoRA adapter on the base model using the rejection-sampled data. Do NOT continue fine-tuning the GRPO adapter. [ASSUMED]

**Rationale:** The purpose of rejection sampling is to distill the GRPO model's behavior into data, then train a clean SFT model on that data. DeepSeek-R1 follows this exact pattern: they retrain the base model (DeepSeek-V3-Base) from scratch on rejection-sampled data [CITED: arxiv.org/pdf/2501.12948]. Using the same LoRA config (rank=4, alpha=4, target_mlp=true) ensures comparability with the Phase 3 SFT baseline.

### Decision 3: Filtering Strategy

**Recommendation:** Per-utterance best-of-N selection (lowest WER) with a lenient WER ceiling (0.30), followed by a fairness rebalancing pass to ensure minimum demographic coverage.

**Rationale:** ASR has limited output diversity, so strict thresholds will reject very few candidates (most will have similar WER). The value comes from (a) selecting the best among small variations, and (b) ensuring the filtered dataset maintains demographic balance. The fairness filter is the novel contribution.

### Decision 4: Dataset Size for SFT Round

**Recommendation:** Aim for a filtered dataset of 3,000-5,000 utterances (one best candidate per utterance from the 5K-utterance generation subset). Train for 1 epoch only. [ASSUMED]

**Rationale:** DeepSeek-R1 used 800K samples for distillation and trained for 2 epochs [CITED: arxiv.org/html/2501.12948v1]. Scaling down proportionally to our 1.7B model and limited compute, 3-5K high-quality samples with 1 epoch should be sufficient for behavioral distillation. Phase 3 trained on ~41K samples, so a rejection-sampled set of 3-5K is significantly smaller -- the quality advantage must compensate for the quantity disadvantage.

## Assumptions Log

> List all claims tagged [ASSUMED] in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Fresh LoRA from base model is better than continuing GRPO adapter for SFT distillation | Architecture Patterns, Decision 2 | If wrong, could get better results by continuing GRPO adapter; easy to A/B test |
| A2 | Per-utterance best-of-N selection preserves demographic balance better than global top-K | Pattern 2, Anti-patterns | If wrong, global top-K might produce better accuracy; fairness filter is the safety net |
| A3 | 60-80% acceptance rate is the target for rejection filtering | Pitfall 2 | If wrong, need to adjust thresholds; log acceptance rate to detect |
| A4 | 3-5K filtered samples sufficient for 1-epoch SFT distillation | Decision 4 | If wrong, may need to generate more candidates or use full dataset; overfitting is the risk |
| A5 | Temperature 0.8 produces sufficient diversity for ASR rejection sampling | Pattern 1, Pitfall 1 | If wrong, increase to 1.0-1.2; measure diversity before committing |
| A6 | Candidate generation can process 5K utterances x 10 candidates in reasonable time | Decision 1 | If wrong, reduce N or subset size; compute estimate suggests ~48 hours which may be too long |

## Open Questions

1. **How much candidate diversity will temperature sampling actually produce for ASR?**
   - What we know: ASR has the unique-answer problem (documented in GRPO-ASR paper). Temperature helps but the improvement is uncertain for this specific model.
   - What's unclear: Will 10 candidates at temperature=0.8 produce meaningfully different transcriptions, or will 8 out of 10 be identical?
   - Recommendation: Run a pilot on 100 utterances FIRST. Measure candidate diversity (pairwise edit distance). Only proceed with full generation if diversity is sufficient (median edit distance >= 1 word).

2. **What is the optimal WER threshold for filtering?**
   - What we know: Should be lenient enough to retain ~60-80% of utterances but strict enough to exclude hallucinations.
   - What's unclear: The WER distribution of GRPO model candidates is unknown until generation runs.
   - Recommendation: Generate first, analyze the WER distribution, THEN set the threshold based on empirical data. Start with "accept best candidate per utterance" (no WER ceiling) and measure.

3. **Is the compute budget feasible for full rejection sampling on T4?**
   - What we know: Phase 2 inference was ~3.5 sec/sample. 45K x 10 = 450K passes = ~437 hours. This is NOT feasible.
   - What's unclear: Whether generating multiple candidates is faster than sequential single inference (shared KV cache prefix).
   - Recommendation: Scope to 5K utterance subset x 10 candidates. Even at 50K inference passes, this is ~48 hours. Consider reducing to N=5 (24 hours) for initial experiment.

4. **Does fairness-aware filtering actually improve fairness over standard best-of-N?**
   - What we know: The fairness filter is the novel contribution. Standard best-of-N just picks lowest WER.
   - What's unclear: Whether the demographic rebalancing in the filtering step produces measurably better fairness than simple best-of-N.
   - Recommendation: Run both versions (best-of-N only vs. best-of-N + fairness filter) and compare fairness metrics. This is a key ablation for the paper.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| T4 GPU (15GB) | Model inference + training | Yes | Tesla T4 | -- |
| transformers | model.generate() | Yes | 4.57.6 | -- |
| peft | LoRA adapters | Yes | 0.18.1 | -- |
| torch | Compute | Yes | 2.8.0+cu128 | -- |
| jiwer | WER computation | Yes | 4.0.0 | -- |
| qwen-asr | Model loading | Yes | 0.0.6 | -- |
| soundfile | Audio loading | Yes | 0.13.1 | -- |
| wandb | Experiment tracking | Yes | 0.25.1 | -- |
| vLLM | Faster batch inference | No | -- | Use HF model.generate() (slower but sufficient) |
| TRL | BestOfNSampler | No | -- | Custom implementation (TRL's is deprecated anyway) |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- vLLM: Not installed, but HF model.generate() with sequential processing is sufficient for the 5K-subset scope.

## Project Constraints (from CLAUDE.md)

| Directive | Compliance |
|-----------|------------|
| Script-oriented pipeline architecture | Phase 6 follows same pattern: three scripts in `scripts/training/` |
| Manifests as CSV contracts | Filtered dataset stored as CSV, same column schema as existing manifests |
| GPU Safety: check nvidia-smi before launch | Generation and training scripts must include VRAM checks |
| GPU Safety: no launch if >80% used | Include pre-flight VRAM check in all scripts |
| Documentation: log changes as separate files | Log files in `logs/`, experiment files in `experiments/` |
| Existing pipeline is read-only | No modifications to `run_inference.py`, `compute_fairness_metrics.py`, etc. |
| Audio loading via soundfile, not torchaudio.load | Already in all existing code; reuse same pattern |

## Sources

### Primary (HIGH confidence)
- [RLHF Book - Rejection Sampling](https://rlhfbook.com/c/09-rejection-sampling) -- 10-30 candidates per prompt, temperature 0.7-1.0, per-prompt vs global top-K selection
- [DeepSeek-R1 Paper](https://arxiv.org/html/2501.12948v1) -- 600K rejection-sampled + 200K supervised for distillation SFT, 2 epochs
- [HuggingFace model.generate docs](https://huggingface.co/docs/transformers/main_classes/text_generation) -- `num_return_sequences`, `do_sample`, `temperature` parameters
- [Existing codebase: evaluate_adapter.py](/home/zeus/project/scripts/training/evaluate_adapter.py) -- Inference pattern, model loading, chat template
- [Existing codebase: train_standard_lora.py](/home/zeus/project/scripts/training/train_standard_lora.py) -- SFT training pattern, LoRA config, data splitting

### Secondary (MEDIUM confidence)
- [LLaMA 2 Overview](https://www.interconnects.ai/p/llama-2-from-meta) -- Rejection sampling + PPO pipeline, offline generation + ranking
- [EmergentMind - RSFT](https://www.emergentmind.com/topics/rejection-sampling-fine-tuning-rsft) -- Failure modes: signal destruction, data inefficiency, overfitting to success
- [PEFT Issue #2264](https://github.com/huggingface/peft/issues/2264) -- Multi-stage LoRA fine-tuning guidance (save first adapter, load + retrain)
- [TRL BestOfNSampler docs](https://huggingface.co/docs/trl/best_of_n) -- Deprecated in TRL 0.25+; do not use

### Tertiary (LOW confidence)
- [ICML 2025 - RFT Secretly Performs PPO](https://icml.cc/virtual/2025/48054) -- Theoretical connection between rejection sampling FT and PPO
- [ICLR 2025 - Preserving Diversity in SFT](https://proceedings.iclr.cc/paper_files/paper/2025/file/a548ef984f30bca3abdc09f43743827f-Paper-Conference.pdf) -- Diversity collapse in fine-tuning
- Compute estimates for Phase 6 -- based on Phase 2 benchmarks (3.5 sec/sample), may vary with temperature sampling

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed, no new dependencies needed
- Architecture: MEDIUM-HIGH -- three-stage pipeline follows established RLHF distillation patterns; ASR-specific adaptations are novel
- Pitfalls: MEDIUM -- core failure modes identified from literature; ASR-specific diversity issue is the key uncertainty
- Compute feasibility: MEDIUM -- 48-hour generation run on T4 is possible but tight; pilot run essential

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable domain; main uncertainty is empirical not evolving)
