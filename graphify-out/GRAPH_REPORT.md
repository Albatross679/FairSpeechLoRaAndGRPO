# Graph Report - FairSpeechLoRaAndGRPO  (2026-04-29)

## Corpus Check
- 57 files · ~210,754 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1074 nodes · 2113 edges · 55 communities detected
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 588 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]

## God Nodes (most connected - your core abstractions)
1. `ASRFairnessDataset` - 154 edges
2. `DataCollatorForQwen3ASR` - 132 edges
3. `DemographicStratifiedSampler` - 99 edges
4. `read_csv()` - 51 edges
5. `parse_args()` - 28 edges
6. `RewardComputer` - 25 edges
7. `PackedTrainer` - 23 edges
8. `wer()` - 19 edges
9. `FrameBudgetBatchSampler` - 18 edges
10. `mountToolbar()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `test_fairspeech_has_speaker_id()` --calls--> `read_csv()`  [INFERRED]
  tests/test_phase02_validation.py → scripts/validation/validate_fairspeech_compression.py
- `test_iter_manifest_batches_maps_by_utterance_id()` --calls--> `iter_manifest_batches()`  [INFERRED]
  tests/test_fairspeech_compression_infra.py → scripts/inference/run_inference.py
- `TestPatchOuterForward` --uses--> `ASRFairnessDataset`  [INFERRED]
  tests/test_phase1_validation.py → scripts/training/data_loader.py
- `TestPatchOuterForward` --uses--> `DemographicStratifiedSampler`  [INFERRED]
  tests/test_phase1_validation.py → scripts/training/data_loader.py
- `TestPatchOuterForward` --uses--> `DataCollatorForQwen3ASR`  [INFERRED]
  tests/test_phase1_validation.py → scripts/training/data_collator.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (142): Dataset, Sampler, _make_manifest_csv(), _manifest_rows(), Phase 1 (1.0 + 1.1) Nyquist validation tests.  Tests exercise code-level logic W, DATA-02: collate_fn pads audio to batch-max length, returns correct shapes., Padded tensor width equals the longest sample in the batch., Padded positions contain zeros (silence). (+134 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (56): check_cv_manifest(), discover_speaker_column(), main(), parse_args(), print_distribution_stats(), Print per-axis demographic distribution comparison (train% vs eval%)., Verify Common Voice test manifest exists and print accent distribution., Discover the speaker/participant ID column from Fair-Speech metadata TSV.      T (+48 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (20): Phase 2 (02-standard-lora-baseline-evaluation-bridge) Nyquist validation tests., Validate train_standard_lora.py code structure without execution., Validate medium-scale validation results and locked config., Validate trained LoRA adapter files., Adapter for rank=4 LoRA should be in the MB range., Adapter config LoRA rank should match locked config., Validate training_config.json from final training run., Validate evaluate_adapter.py code structure without execution. (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (46): lambda_already_done(), load_existing_results(), main(), Full-scale GRPO lambda sweep orchestrator (Phase 5).  Iterates over a configurab, Load sweep results already on disk (for --resume)., Check if a lambda value has a completed training summary., Run full-scale GRPO training for each lambda value sequentially., run_lambda_sweep() (+38 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (50): classify_insertions(), compute_fairness_table(), compute_group_table(), compute_paired_delta(), error_counts(), group_rows(), jiwer_module(), load_all_predictions() (+42 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (46): AudioVariant, build_source_manifest(), build_summary(), ffmpeg_base(), generate_variant_audio(), generate_variant_audio_job(), main(), normalize_text() (+38 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (42): _extract_granite_transcription(), IncrementalCSVWriter, infer_granite(), infer_qwen3_asr(), infer_wav2vec2(), infer_whisper(), iter_manifest_batches(), load_audio() (+34 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (33): _apply_lora_with_rslora(), create_packed_model(), _create_scheduler(), CyclingIterator, PackedTrainer, Multi-adapter round-robin training loop for PLoRA packed sweep.  Manages N LoRA, Multi-adapter round-robin training loop.      Manages N LoRA adapters on one fro, Set up adapters, optimizers, schedulers, and data iterators.          The first (+25 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (41): append_status(), build_full_batch_plans(), build_run_matrix(), compute_metrics(), disk_audit(), disk_entry(), ensure_runtime_dirs(), from_args() (+33 more)

### Community 9 - "Community 9"
Cohesion: 0.07
Nodes (37): load_model_and_processor(), main(), parse_args(), patch_outer_forward(), print_gpu_memory(), LoRA prototype training for Qwen3-ASR-1.7B.  Attaches LoRA to decoder self-atten, Load Qwen3-ASR-1.7B via qwen-asr wrapper and extract HF model.      Uses Pattern, Apply LoRA adapters to decoder self-attention projections.      Freezes the audi (+29 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (34): build_cell_command(), compute_real_samples_per_step(), compute_verdict(), fmt_num(), get_gpu_mem_used_mib(), get_gpu_name_and_total(), iso_now(), main() (+26 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (33): actionBtn(), autoBoot(), btn(), clearAll(), clearGlyph(), closeGlyph(), disable(), drawStroke() (+25 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (25): analyze_model(), bootstrap_wer(), compute_fairness_metrics(), compute_group_wer(), compute_h1_gap(), generate_latex_table(), get_group_order(), main() (+17 more)

### Community 13 - "Community 13"
Cohesion: 0.16
Nodes (26): fig1_ethnicity_heatmap(), fig2_accent_heatmap(), fig3_whisper_scaling(), fig5_compression_fairness(), fig6_l1_analysis(), fig7_pareto(), fig8_scaling_curves(), get_group_wers() (+18 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (25): bootstrap_ci(), classify_insertions(), compute_absolute_gap(), compute_group_wer(), compute_mmr(), generate_fairness_amplification_heatmap(), generate_fairness_gap_amplification(), generate_hallucination_classification_table() (+17 more)

### Community 15 - "Community 15"
Cohesion: 0.12
Nodes (23): EnglishTextNormalizer, Lightweight fallback so registry/tools import without Whisper installed., check_adapter_roundtrip(), check_transcription_quality(), check_vram(), load_adapter_model(), load_base_model_and_processor(), load_eval_samples() (+15 more)

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (20): add_noise_at_snr(), apply_reverb(), generate_perturbed_manifest(), inject_silence(), load_audio(), main(), mask_audio_chunks(), NoisePool (+12 more)

### Community 17 - "Community 17"
Cohesion: 0.16
Nodes (20): analyze_model(), bootstrap_wer(), compute_fairness_metrics(), compute_group_wer(), generate_latex_table(), get_ordered_models(), main(), pairwise_significance() (+12 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (18): check_audio_existence(), check_cv_manifest(), check_demographic_completeness(), check_distribution_drift(), check_min_group_size(), check_speaker_leakage(), check_text_normalization(), generate_plots() (+10 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (3): Validate analysis JSON structure and content., Fairness metrics should have actual values for FS and CV (not 'error')., TestAnalysisJSON

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (9): Tests for previously-xfailed gaps that are now bridged., Common Voice manifest should include speaker_id from client_id., train_standard_lora.py should handle speaker_id for disjoint splits., Bootstrap CIs should be present for groups >= MIN_GROUP_SIZE., LibriSpeech should now be included in evaluation., LibriSpeech test-clean manifest should exist., LS test-clean WER should be reasonable (< 10%)., All three datasets (FS, CV, LS) should be in analysis. (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.33
Nodes (15): gen_fig1_ethnicity_heatmap(), gen_fig2_accent_heatmap(), gen_fig7_pareto(), gen_fig8_scaling_curves(), gen_hallucination_categories_cv(), gen_perturbation_amplification_heatmap(), gen_perturbation_hallucination_types_masking_fs(), gen_perturbation_pareto_ethnicity_fs() (+7 more)

### Community 22 - "Community 22"
Cohesion: 0.19
Nodes (9): bchDigit(), bchFormat(), bchVersion(), createBytes(), gexp(), mount(), pickVersion(), render() (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.36
Nodes (10): assert_gpu_safe(), build_plan(), main(), monitor_peak_vram(), parse_csv_numbers(), query_gpu_memory(), read_manifest_count_and_seconds(), read_plan_summary() (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.33
Nodes (10): default_hf_hub_cache(), directory_size_bytes(), import_registry(), main(), package_version(), parse_models(), Return the directory that snapshot_download should use for model snapshots., smoke_command() (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.36
Nodes (7): check_ci_overlap(), main(), parse_log(), print_summary(), Check Black/AA vs White CI overlap for all 9 models on Fair-Speech ethnicity., Print summary of extracted data., Parse the entire log file and return structured data.

### Community 26 - "Community 26"
Cohesion: 0.67
Nodes (3): main(), normalize_text(), Normalize text for WER using Whisper's English normalizer.

### Community 27 - "Community 27"
Cohesion: 0.67
Nodes (2): clean_coraal_text(), Normalize CORAAL sociolinguistic conventions for baseline WER computation.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Prepare LibriSpeech test-clean manifest for ASR fairness evaluation.  Loads from

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Fair-Speech manifest should have speaker_id for disjoint splits.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Convert a free-form condition label to a filename-safe suffix.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Yield (batch_df, row_positions, batch_id) using fixed or planned batches.      B

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Lightweight fallback so registry/tools import without Whisper installed.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Normalize text for WER using Whisper's English normalizer.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Writes full result rows to the output CSV incrementally during inference.      T

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Convert a raw prediction dict to a full result row.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Write a batch of predictions to CSV. Called after each inference batch.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Load audio file, resample to target sr, return numpy array.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Load wav2vec2 model and processor.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): CTC-based inference with wav2vec2.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): CTC-based inference with wav2vec2 (legacy single-call interface).

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Load Whisper model and processor.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Encoder-decoder inference with Whisper.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Whisper inference (legacy single-call interface).

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Load Qwen3-ASR model.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): LLM-based inference with Qwen3-ASR via qwen-asr package.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Qwen3-ASR inference (legacy single-call interface).

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Extract the actual transcription from Granite's potentially chatty output.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Load Granite-Speech model and processor.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): LLM-based inference with Granite-Speech.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Granite inference (legacy single-call interface).

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): SALM-based inference with Canary-Qwen-2.5B (FastConformer + Qwen3 LoRA).

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Normalize reference text using Whisper if available, otherwise a light fallback.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Read WAV header metadata with the stdlib wave module.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Create one derived WAV. Returns status: created/skipped/dry-run.

## Knowledge Gaps
- **231 isolated node(s):** `Phase 2 (02-standard-lora-baseline-evaluation-bridge) Nyquist validation tests.`, `Validate HP sweep outputs from 20 Optuna trials.`, `Validate train_standard_lora.py code structure without execution.`, `Validate medium-scale validation results and locked config.`, `Validate trained LoRA adapter files.` (+226 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 27`** (3 nodes): `clean_coraal_text()`, `Normalize CORAAL sociolinguistic conventions for baseline WER computation.`, `prepare_h1_datasets.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (2 nodes): `Prepare LibriSpeech test-clean manifest for ASR fairness evaluation.  Loads from`, `prepare_librispeech.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Fair-Speech manifest should have speaker_id for disjoint splits.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Convert a free-form condition label to a filename-safe suffix.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Yield (batch_df, row_positions, batch_id) using fixed or planned batches.      B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Lightweight fallback so registry/tools import without Whisper installed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Normalize text for WER using Whisper's English normalizer.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Writes full result rows to the output CSV incrementally during inference.      T`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Convert a raw prediction dict to a full result row.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Write a batch of predictions to CSV. Called after each inference batch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Load audio file, resample to target sr, return numpy array.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Load wav2vec2 model and processor.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `CTC-based inference with wav2vec2.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `CTC-based inference with wav2vec2 (legacy single-call interface).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Load Whisper model and processor.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Encoder-decoder inference with Whisper.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Whisper inference (legacy single-call interface).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Load Qwen3-ASR model.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `LLM-based inference with Qwen3-ASR via qwen-asr package.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Qwen3-ASR inference (legacy single-call interface).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Extract the actual transcription from Granite's potentially chatty output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Load Granite-Speech model and processor.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `LLM-based inference with Granite-Speech.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Granite inference (legacy single-call interface).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `SALM-based inference with Canary-Qwen-2.5B (FastConformer + Qwen3 LoRA).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Normalize reference text using Whisper if available, otherwise a light fallback.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Read WAV header metadata with the stdlib wave module.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Create one derived WAV. Returns status: created/skipped/dry-run.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `read_csv()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 9`, `Community 12`, `Community 14`, `Community 15`, `Community 16`, `Community 17`, `Community 18`, `Community 20`, `Community 21`?**
  _High betweenness centrality (0.407) - this node is a cross-community bridge._
- **Why does `parse_args()` connect `Community 1` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 9`, `Community 10`, `Community 14`, `Community 16`, `Community 23`, `Community 24`?**
  _High betweenness centrality (0.247) - this node is a cross-community bridge._
- **Why does `test_fairspeech_has_speaker_id()` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Are the 146 inferred relationships involving `ASRFairnessDataset` (e.g. with `TestASRFairnessDatasetColumnValidation` and `TestCollateFn`) actually correct?**
  _`ASRFairnessDataset` has 146 INFERRED edges - model-reasoned connections that need verification._
- **Are the 126 inferred relationships involving `DataCollatorForQwen3ASR` (e.g. with `TestASRFairnessDatasetColumnValidation` and `TestCollateFn`) actually correct?**
  _`DataCollatorForQwen3ASR` has 126 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `DemographicStratifiedSampler` (e.g. with `TestASRFairnessDatasetColumnValidation` and `TestCollateFn`) actually correct?**
  _`DemographicStratifiedSampler` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 49 inferred relationships involving `read_csv()` (e.g. with `.test_subset_sizes_correct()` and `.test_subset_preserves_groups()`) actually correct?**
  _`read_csv()` has 49 INFERRED edges - model-reasoned connections that need verification._