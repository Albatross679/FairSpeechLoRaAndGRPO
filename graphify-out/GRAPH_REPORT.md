# Graph Report - .  (2026-04-20)

## Corpus Check
- 139 files · ~219,010 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1464 nodes · 2646 edges · 50 communities detected
- Extraction: 74% EXTRACTED · 26% INFERRED · 0% AMBIGUOUS · INFERRED: 680 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Data Loader & Collator|Data Loader & Collator]]
- [[_COMMUNITY_Head Surgery Scripts|Head Surgery Scripts]]
- [[_COMMUNITY_Phase 2 Validation Tests|Phase 2 Validation Tests]]
- [[_COMMUNITY_ASR Fairness Concepts|ASR Fairness Concepts]]
- [[_COMMUNITY_Autoresearch Program|Autoresearch Program]]
- [[_COMMUNITY_Evaluation Metrics & COLM Paper|Evaluation Metrics & COLM Paper]]
- [[_COMMUNITY_Report & Plot Generation|Report & Plot Generation]]
- [[_COMMUNITY_CV25 Fixing-Set Analysis|CV25 Fixing-Set Analysis]]
- [[_COMMUNITY_Packed LoRA Trainer|Packed LoRA Trainer]]
- [[_COMMUNITY_Fairness Metrics Computation|Fairness Metrics Computation]]
- [[_COMMUNITY_VRAM Tuning Experiments|VRAM Tuning Experiments]]
- [[_COMMUNITY_GRPO Sweep & Reward|GRPO Sweep & Reward]]
- [[_COMMUNITY_Fixing-Set Set-Cover Impl|Fixing-Set Set-Cover Impl]]
- [[_COMMUNITY_Inference Dispatch|Inference Dispatch]]
- [[_COMMUNITY_VRAM Tune Gate Validation|VRAM Tune Gate Validation]]
- [[_COMMUNITY_Pencil.js Asset|Pencil.js Asset]]
- [[_COMMUNITY_LoRA Adapter Evaluation|LoRA Adapter Evaluation]]
- [[_COMMUNITY_Splits Validation|Splits Validation]]
- [[_COMMUNITY_Audio Perturbations|Audio Perturbations]]
- [[_COMMUNITY_Perturbation Metrics|Perturbation Metrics]]
- [[_COMMUNITY_LoRA Prototype Validation|LoRA Prototype Validation]]
- [[_COMMUNITY_LoRA Prototype Training|LoRA Prototype Training]]
- [[_COMMUNITY_LoRA & VRAM Explainers|LoRA & VRAM Explainers]]
- [[_COMMUNITY_Perturbation Plots|Perturbation Plots]]
- [[_COMMUNITY_Phase 2 Bridged-Gap Tests|Phase 2 Bridged-Gap Tests]]
- [[_COMMUNITY_Large-Font Figure Regen|Large-Font Figure Regen]]
- [[_COMMUNITY_QR Code Asset|QR Code Asset]]
- [[_COMMUNITY_Whisper Hallucination Analysis|Whisper Hallucination Analysis]]
- [[_COMMUNITY_Fair-Speech Split Prep|Fair-Speech Split Prep]]
- [[_COMMUNITY_Phase 2 HP Sweep|Phase 2 HP Sweep]]
- [[_COMMUNITY_WebDataset Pre-Decoding|WebDataset Pre-Decoding]]
- [[_COMMUNITY_Bootstrap CI Parsing|Bootstrap CI Parsing]]
- [[_COMMUNITY_Overleaf Preparation|Overleaf Preparation]]
- [[_COMMUNITY_CORAAL  H1 Data Prep|CORAAL / H1 Data Prep]]
- [[_COMMUNITY_Head-Surgery Heatmap Script|Head-Surgery Heatmap Script]]
- [[_COMMUNITY_Lightning AI Cloud Deploy|Lightning AI Cloud Deploy]]
- [[_COMMUNITY_Dataset Preparation|Dataset Preparation]]
- [[_COMMUNITY_LibriSpeech Prep|LibriSpeech Prep]]
- [[_COMMUNITY_FileClass Experiment  Log|FileClass: Experiment / Log]]
- [[_COMMUNITY_FileClass Issue  Task|FileClass: Issue / Task]]
- [[_COMMUNITY_FileClass Knowledge  Reference|FileClass: Knowledge / Reference]]
- [[_COMMUNITY_FS Speaker-ID Test|FS Speaker-ID Test]]
- [[_COMMUNITY_Tests Package Init|Tests Package Init]]
- [[_COMMUNITY_Head Surgery Package Init|Head Surgery Package Init]]
- [[_COMMUNITY_Test Clip Extractor|Test Clip Extractor]]
- [[_COMMUNITY_Head Surgery Stage G Gates|Head Surgery Stage G Gates]]
- [[_COMMUNITY_Silence Perturbation Generator|Silence Perturbation Generator]]
- [[_COMMUNITY_Reproducibility Config|Reproducibility Config]]
- [[_COMMUNITY_Phase 2 Test Module|Phase 2 Test Module]]
- [[_COMMUNITY_RTX 3090 Hardware Node|RTX 3090 Hardware Node]]

## God Nodes (most connected - your core abstractions)
1. `ASRFairnessDataset` - 154 edges
2. `DataCollatorForQwen3ASR` - 132 edges
3. `DemographicStratifiedSampler` - 99 edges
4. `BatchedHeadMaskHook` - 31 edges
5. `parse_args()` - 28 edges
6. `RewardComputer` - 25 edges
7. `PackedTrainer` - 23 edges
8. `SerialHeadMaskHook` - 20 edges
9. `Tokenizer` - 18 edges
10. `mountToolbar()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `MMR (max/min WER ratio) fairness metric` --semantically_similar_to--> `Fairness Metrics (MMR, gaps)`  [INFERRED] [semantically similar]
  colm2026_conference.pdf → docs/asr-evaluation-metrics.html
- `Silence-injection robustness (energy VAD)` --semantically_similar_to--> `12-condition perturbation suite (noise/reverb/silence/mask)`  [INFERRED] [semantically similar]
  docs/head_surgery_report.md → colm2026_conference.pdf
- `Do LLM Decoders Listen Fairly? (COLM 2026)` --semantically_similar_to--> `LLM ASR Fairness Midterm Report`  [INFERRED] [semantically similar]
  colm2026_conference.pdf → llm-asr-fairness-midterm.pdf
- `MODEL_REGISTRY dispatch (run_inference.py)` --references--> `Whisper small/medium/large-v3 (Gen 2)`  [INFERRED]
  CLAUDE.md → colm2026_conference.pdf
- `check_4_gradient_flow()` --calls--> `norm()`  [INFERRED]
  scripts/training/phase1_sanity.py → autoresearch/train.py

## Hyperedges (group relationships)
- **PLoRA + ASHA + RsLoRA HP Sweep Stack** — plora_explainer_plora, asha_rslora_explainer_asha_pruner, asha_rslora_explainer_rslora, asha_rslora_explainer_optuna [EXTRACTED 0.95]
- **Head-Surgery Diagnosis Pipeline (Stages A-G)** — head_surgery_report_whisper_large_v3, head_surgery_report_head_sweep, head_surgery_report_paired_bootstrap, head_surgery_report_keystone_head_l0h5, head_surgery_report_edited_fixing_set_analysis, head_surgery_heatmap_heatmap_figure [EXTRACTED 0.95]
- **Three-generation ASR fairness benchmark (CTC / Whisper / LLM-decoder)** — colm2026_conference_wav2vec2_large, colm2026_conference_whisper_models, colm2026_conference_gen3_llm_decoders, colm2026_conference_mmr_metric, colm2026_conference_perturbation_suite [EXTRACTED 0.95]
- **Phase 3 SFT LoRA pipeline (infra + VRAM tune + launch)** — phase3_full_sft_lora_infrastructure_log, phase3_vram_tune_infrastructure_log, phase3_vram_tune_grid_log, phase3_vram_preflight_cellc_nw2_log, phase3_full_sft_lora_launch_log, train_standard_lora, tune_vram [INFERRED 1.00]
- **Head Surgery diagnosis pipeline (Stages A-F + fixing set)** — head_surgery_stage_d_scoring_log, head_surgery_stage_f_energy_vad_log, head_surgery_diagnosis_complete_log, head_surgery_fixing_set_analysis_log, score_heads, energy_vad, fixing_set_analysis, whisper_large_v3, cv25 [INFERRED 1.00]
- **Data loader stack (dataset + sampler + factory + splits)** — asr_fairness_dataset_creation_log, data_loader_dataset_collate_log, demographic_stratified_sampler_creation_log, data_loader_sampler_factory_log, prepare_splits_creation_log, validate_splits_creation_log, data_loader, prepare_splits, validate_splits [INFERRED 1.00]
- **he_lora_stage_pipeline** — exp_plora_stage1, exp_plora_stage2, exp_plora_stage3 [INFERRED 1.00]
- **he_head_surgery_stages** — stage_a_baseline, stage_c_full_sweep, stage_d_scoring [INFERRED 1.00]
- **he_fixing_set_cover** — head_surgery_coverage_matrix, head_surgery_greedy_cover, head_surgery_ilp_cover [INFERRED 1.00]
- **he_three_filters** — head_surgery_filter_helps, head_surgery_filter_noharm, head_surgery_filter_regguard [INFERRED 1.00]
- **he_bottleneck_types** — io_bound, cpu_bound, vram_bound [INFERRED 1.00]
- **he_sft_rl_loop** — sft_method, grpo_algorithm, rejection_sampling [INFERRED 1.00]
- **he_vram_core_techniques** — mixed_precision_fp16, gradient_checkpointing, gradient_accumulation [INFERRED 1.00]
- **he_io_remediation_fix** — webdataset_option1, num_workers_2_option4, cpu_decode_ceiling [INFERRED 1.00]
- **he_audio_pipeline_pivot** — head_surgery_audio_fix_16k, head_surgery_audio_fix_librosa, head_surgery_audit_pivot [INFERRED 1.00]
- **he_prototyping_synthesis** — karpathy_recipe, google_tuning_playbook, mup_parameterization [INFERRED 1.00]
- **he_lora_hp_priority** — lora_learning_rate, lora_target_mlp, lora_rank_ladder [INFERRED 1.00]

## Communities

### Community 0 - "Data Loader & Collator"
Cohesion: 0.02
Nodes (158): DataCollatorForQwen3ASR, Data collator for Qwen3-ASR fine-tuning with prefix masking.  Implements DataCol, Find the token position where the actual transcript starts.          Searches fo, Pad all sequences in the batch to the same length.          Args:             al, Collates ASRFairnessDataset items into Qwen3-ASR chat-format training batches., Collate a list of dataset items into a training batch.          Args:, ASRFairnessDataset, collate_fn() (+150 more)

### Community 1 - "Head Surgery Scripts"
Cohesion: 0.04
Nodes (92): main(), One-shot follow-up: full 4-metric table for the §7.4 top-10 hallucination-drivin, main(), One-shot follow-up: compute non-Indian insertion rate for the 8 §8.2 keystone he, _cli(), _generate_with_config(), Stage E — decoding-strategy ablation (T5).  2 × 3 × 3 × 2 = 36 configs: beam ∈ {, run_decoding_grid() (+84 more)

### Community 2 - "Phase 2 Validation Tests"
Cohesion: 0.02
Nodes (22): Phase 2 (02-standard-lora-baseline-evaluation-bridge) Nyquist validation tests., Validate train_standard_lora.py code structure without execution., Validate medium-scale validation results and locked config., Validate trained LoRA adapter files., Adapter for rank=4 LoRA should be in the MB range., Adapter config LoRA rank should match locked config., Validate training_config.json from final training run., Validate evaluate_adapter.py code structure without execution. (+14 more)

### Community 3 - "ASR Fairness Concepts"
Cohesion: 0.04
Nodes (80): Log: ASRFairnessDataset + collate_fn, Common Voice (CV), Composite R_acc + R_fair reward, scripts/training/data_collator.py, scripts/training/data_loader.py, Log: ASRFairnessDataset and collate_fn creation (dup), Log: DemographicStratifiedSampler + DataLoader factory, DeepSeek-R1 4-Stage Pipeline (+72 more)

### Community 4 - "Autoresearch Program"
Cohesion: 0.04
Nodes (50): BPE tokenizer, Autonomous Experiment Loop, Muon + AdamW optimizer, nanochat, autoresearch program.md, autoresearch progress chart, autoresearch README, results.tsv log (+42 more)

### Community 5 - "Evaluation Metrics & COLM Paper"
Cohesion: 0.03
Nodes (71): DP Alignment (Edit Types), Fairness Metrics (MMR, gaps), Rationale: Why Not AUC-ROC for ASR, WER (Word Error Rate), WER Decomposition (Sub/Del/Ins), CV25 audio duration histogram (Indian vs non-Indian), Caveman Skill (token reduction), MODEL_REGISTRY dispatch (run_inference.py) (+63 more)

### Community 6 - "Report & Plot Generation"
Cohesion: 0.05
Nodes (52): build_report(), _cli(), _dataset_section(), _fmt_rate(), _md_table(), Stage G — assemble the final head-surgery report (T6).  Writes docs/head_surgery, Render df as a GitHub-flavored markdown table. Fallback if `tabulate` missing., _summary_section() (+44 more)

### Community 7 - "CV25 Fixing-Set Analysis"
Cohesion: 0.04
Nodes (64): Calm-Whisper (arxiv 2505.12969), Common Voice 24, Common Voice 25, Common Voice 25 test split, 81 GB CV25 tarball (truncated), scripts/head_surgery/energy_vad.py, scripts/head_surgery/fixing_set_analysis.py, Gate G1: CV25 baseline reproduction (+56 more)

### Community 8 - "Packed LoRA Trainer"
Cohesion: 0.06
Nodes (36): _apply_lora_with_rslora(), create_packed_model(), _create_scheduler(), CyclingIterator, PackedTrainer, Multi-adapter round-robin training loop for PLoRA packed sweep.  Manages N LoRA, Multi-adapter round-robin training loop.      Manages N LoRA adapters on one fro, Set up adapters, optimizers, schedulers, and data iterators.          The first (+28 more)

### Community 9 - "Fairness Metrics Computation"
Cohesion: 0.06
Nodes (45): analyze_model(), bootstrap_wer(), compute_fairness_metrics(), compute_group_wer(), analyze_model(), bootstrap_wer(), compute_fairness_metrics(), compute_group_wer() (+37 more)

### Community 10 - "VRAM Tuning Experiments"
Cohesion: 0.06
Nodes (48): Option 6: Accept Degraded ~45% Util, ASHA via Optuna HyperbandPruner, 8-bit Optimizer (bitsandbytes), Chinchilla Scaling Law, CPU-bound Bottleneck, CPU Audio Decode Ceiling (~6-8 samples/s), Common Voice Dev Set, DeepSpeed ZeRO (+40 more)

### Community 11 - "GRPO Sweep & Reward"
Cohesion: 0.07
Nodes (32): lambda_already_done(), load_existing_results(), main(), Full-scale GRPO lambda sweep orchestrator (Phase 5).  Iterates over a configurab, Load sweep results already on disk (for --resume)., Check if a lambda value has a completed training summary., Run full-scale GRPO training for each lambda value sequentially., run_lambda_sweep() (+24 more)

### Community 12 - "Fixing-Set Set-Cover Impl"
Cohesion: 0.09
Nodes (39): build_count_table(), build_coverage_matrix(), _cli(), count_insertions(), greedy_cover(), identify_affected(), ilp_cover(), Post-hoc fixing-set analysis on the 640-head × 484-utterance sweep.  For each In (+31 more)

### Community 13 - "Inference Dispatch"
Cohesion: 0.08
Nodes (36): _extract_granite_transcription(), IncrementalCSVWriter, infer_granite(), infer_qwen3_asr(), infer_wav2vec2(), infer_whisper(), load_audio(), load_granite() (+28 more)

### Community 14 - "VRAM Tune Gate Validation"
Cohesion: 0.09
Nodes (34): build_cell_command(), compute_real_samples_per_step(), compute_verdict(), fmt_num(), get_gpu_mem_used_mib(), get_gpu_name_and_total(), iso_now(), main() (+26 more)

### Community 15 - "Pencil.js Asset"
Cohesion: 0.14
Nodes (33): actionBtn(), autoBoot(), btn(), clearAll(), clearGlyph(), closeGlyph(), disable(), drawStroke() (+25 more)

### Community 16 - "LoRA Adapter Evaluation"
Cohesion: 0.09
Nodes (33): bootstrap_wer(), compute_fairness_metrics(), compute_group_wer(), evaluate_on_dataset(), load_audio(), load_model_with_adapter(), main(), normalize_text() (+25 more)

### Community 17 - "Splits Validation"
Cohesion: 0.1
Nodes (24): DATA-03: Validation checks detect issues in split data., check_speaker_leakage flags overlapping speakers., check_speaker_leakage passes when splits are disjoint., check_demographic_completeness flags >5% missing ethnicity labels., check_distribution_drift flags large proportion differences., TestValidateSplitsChecks, check_audio_existence(), check_cv_manifest() (+16 more)

### Community 18 - "Audio Perturbations"
Cohesion: 0.11
Nodes (20): add_noise_at_snr(), apply_reverb(), generate_perturbed_manifest(), inject_silence(), load_audio(), main(), mask_audio_chunks(), NoisePool (+12 more)

### Community 19 - "Perturbation Metrics"
Cohesion: 0.13
Nodes (25): bootstrap_ci(), classify_insertions(), compute_absolute_gap(), compute_group_wer(), compute_mmr(), generate_fairness_amplification_heatmap(), generate_fairness_gap_amplification(), generate_hallucination_classification_table() (+17 more)

### Community 20 - "LoRA Prototype Validation"
Cohesion: 0.12
Nodes (23): check_adapter_roundtrip(), check_loss_trend(), check_transcription_quality(), check_vram(), load_adapter_model(), load_base_model_and_processor(), load_eval_samples(), main() (+15 more)

### Community 21 - "LoRA Prototype Training"
Cohesion: 0.16
Nodes (18): create_stratified_subset(), load_model_and_processor(), main(), parse_args(), patch_outer_forward(), print_gpu_memory(), LoRA prototype training for Qwen3-ASR-1.7B.  Attaches LoRA to decoder self-atten, Create a 200-sample stratified subset from Fair-Speech + Common Voice.      Samp (+10 more)

### Community 22 - "LoRA & VRAM Explainers"
Cohesion: 0.12
Nodes (18): LoRA Alpha Normalization, ASHA Pruner, Optuna, RsLoRA (Rank-Stabilized LoRA), ML Training Phase Plan (infra/VRAM/launch), FlashAttention (FA2/FA3), fp8 / MX Format Training, FSDP / ZeRO-3 (+10 more)

### Community 23 - "Perturbation Plots"
Cohesion: 0.22
Nodes (16): main(), plot_amplification_by_type(), plot_fairness_amplification_heatmap(), plot_hallucination_types(), plot_insertion_rate_by_group(), plot_pareto_under_degradation(), plot_wer_degradation_curves(), plot_werd_by_generation() (+8 more)

### Community 24 - "Phase 2 Bridged-Gap Tests"
Cohesion: 0.12
Nodes (9): Tests for previously-xfailed gaps that are now bridged., Common Voice manifest should include speaker_id from client_id., train_standard_lora.py should handle speaker_id for disjoint splits., Bootstrap CIs should be present for groups >= MIN_GROUP_SIZE., LibriSpeech should now be included in evaluation., LibriSpeech test-clean manifest should exist., LS test-clean WER should be reasonable (< 10%)., All three datasets (FS, CV, LS) should be in analysis. (+1 more)

### Community 25 - "Large-Font Figure Regen"
Cohesion: 0.33
Nodes (15): gen_fig1_ethnicity_heatmap(), gen_fig2_accent_heatmap(), gen_fig7_pareto(), gen_fig8_scaling_curves(), gen_hallucination_categories_cv(), gen_perturbation_amplification_heatmap(), gen_perturbation_hallucination_types_masking_fs(), gen_perturbation_pareto_ethnicity_fs() (+7 more)

### Community 26 - "QR Code Asset"
Cohesion: 0.19
Nodes (9): bchDigit(), bchFormat(), bchVersion(), createBytes(), gexp(), mount(), pickVersion(), render() (+1 more)

### Community 27 - "Whisper Hallucination Analysis"
Cohesion: 0.23
Nodes (12): analyze_model_insertions(), cross_model_comparison(), extract_and_categorize_insertions(), main(), parse_args(), plot_hallucination_categories(), plot_insertion_rate_comparison(), Analyze insertions for a model across demographic groups. (+4 more)

### Community 28 - "Fair-Speech Split Prep"
Cohesion: 0.27
Nodes (10): check_cv_manifest(), discover_speaker_column(), main(), parse_args(), print_distribution_stats(), Print per-axis demographic distribution comparison (train% vs eval%)., Verify Common Voice test manifest exists and print accent distribution., Discover the speaker/participant ID column from Fair-Speech metadata TSV.      T (+2 more)

### Community 29 - "Phase 2 HP Sweep"
Cohesion: 0.25
Nodes (10): apply_lora(), create_objective(), create_stratified_subset(), load_model_and_processor(), main(), patch_outer_forward(), Phase 2: Small-Scale HP Exploration for LoRA fine-tuning.  Runs an Optuna TPE sw, Create stratified subset with proportional FS/CV split. (+2 more)

### Community 30 - "WebDataset Pre-Decoding"
Cohesion: 0.22
Nodes (9): I/O Binding Constraint, 6-Cell GPU Tuning Grid (Phase 03-02), Pre-decoded ASR Training Shards, Rejected: int8 audio / 8 kHz sample rate, WebDataset Packaging Layer, MP3 decompression CPU bottleneck, Tiny-file overlayfs throughput ceiling, Pre-Decode Shard Trick (+1 more)

### Community 31 - "Bootstrap CI Parsing"
Cohesion: 0.36
Nodes (7): check_ci_overlap(), main(), parse_log(), print_summary(), Check Black/AA vs White CI overlap for all 9 models on Fair-Speech ethnicity., Print summary of extracted data., Parse the entire log file and return structured data.

### Community 32 - "Overleaf Preparation"
Cohesion: 0.67
Nodes (0): 

### Community 33 - "CORAAL / H1 Data Prep"
Cohesion: 0.67
Nodes (2): clean_coraal_text(), Normalize CORAAL sociolinguistic conventions for baseline WER computation.

### Community 34 - "Head-Surgery Heatmap Script"
Cohesion: 0.67
Nodes (1): D7 stretch — per-head heatmap of Δ insertion rate on Indian-accent CV.

### Community 35 - "Lightning AI Cloud Deploy"
Cohesion: 0.67
Nodes (3): AI Hub One-Command Deploy, Lightning AI CLI, Spot Instances (Cloud GPU)

### Community 36 - "Dataset Preparation"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "LibriSpeech Prep"
Cohesion: 1.0
Nodes (1): Prepare LibriSpeech test-clean manifest for ASR fairness evaluation.  Loads from

### Community 38 - "FileClass: Experiment / Log"
Cohesion: 1.0
Nodes (2): FileClass: Experiment, FileClass: Log

### Community 39 - "FileClass: Issue / Task"
Cohesion: 1.0
Nodes (2): FileClass: Issue, FileClass: Task

### Community 40 - "FileClass: Knowledge / Reference"
Cohesion: 1.0
Nodes (2): FileClass: Knowledge, FileClass: Reference

### Community 41 - "FS Speaker-ID Test"
Cohesion: 1.0
Nodes (1): Fair-Speech manifest should have speaker_id for disjoint splits.

### Community 42 - "Tests Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "Head Surgery Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Test Clip Extractor"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "Head Surgery Stage G Gates"
Cohesion: 1.0
Nodes (1): Head Surgery Stage G: Gate checks

### Community 46 - "Silence Perturbation Generator"
Cohesion: 1.0
Nodes (1): scripts/head_surgery/_generate_silence_perturbations.py

### Community 47 - "Reproducibility Config"
Cohesion: 1.0
Nodes (1): scripts/head_surgery/repro_config.py

### Community 48 - "Phase 2 Test Module"
Cohesion: 1.0
Nodes (1): tests/test_phase02_validation.py

### Community 49 - "RTX 3090 Hardware Node"
Cohesion: 1.0
Nodes (1): NVIDIA RTX 3090 (24GB)

## Ambiguous Edges - Review These
- `GRPO Fairness-Aware ASR Fine-Tuning Project` → `Caveman Skill (token reduction)`  [AMBIGUOUS]
  docs/caveman-eval-report.html · relation: conceptually_related_to

## Knowledge Gaps
- **293 isolated node(s):** `One-time data preparation for autoresearch experiments. Downloads data shards an`, `Download one parquet shard with retries. Returns True on success.`, `Download training shards + pinned validation shard.`, `Return sorted list of parquet file paths in the data directory.`, `Yield documents from training split (all shards except pinned val shard).` (+288 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Dataset Preparation`** (2 nodes): `main()`, `prepare_dataset.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LibriSpeech Prep`** (2 nodes): `Prepare LibriSpeech test-clean manifest for ASR fairness evaluation.  Loads from`, `prepare_librispeech.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FileClass: Experiment / Log`** (2 nodes): `FileClass: Experiment`, `FileClass: Log`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FileClass: Issue / Task`** (2 nodes): `FileClass: Issue`, `FileClass: Task`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FileClass: Knowledge / Reference`** (2 nodes): `FileClass: Knowledge`, `FileClass: Reference`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FS Speaker-ID Test`** (1 nodes): `Fair-Speech manifest should have speaker_id for disjoint splits.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tests Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Head Surgery Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Clip Extractor`** (1 nodes): `extract_test_clips.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Head Surgery Stage G Gates`** (1 nodes): `Head Surgery Stage G: Gate checks`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Silence Perturbation Generator`** (1 nodes): `scripts/head_surgery/_generate_silence_perturbations.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reproducibility Config`** (1 nodes): `scripts/head_surgery/repro_config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase 2 Test Module`** (1 nodes): `tests/test_phase02_validation.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `RTX 3090 Hardware Node`** (1 nodes): `NVIDIA RTX 3090 (24GB)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `GRPO Fairness-Aware ASR Fine-Tuning Project` and `Caveman Skill (token reduction)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `parse_args()` connect `Report & Plot Generation` to `Data Loader & Collator`, `Head Surgery Scripts`, `Packed LoRA Trainer`, `GRPO Sweep & Reward`, `Fixing-Set Set-Cover Impl`, `VRAM Tune Gate Validation`, `LoRA Adapter Evaluation`, `Audio Perturbations`, `Perturbation Metrics`, `Perturbation Plots`, `Phase 2 HP Sweep`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Why does `ASRFairnessDataset` connect `Data Loader & Collator` to `Packed LoRA Trainer`, `GRPO Sweep & Reward`, `Splits Validation`, `LoRA Prototype Training`, `Phase 2 HP Sweep`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `DataCollatorForQwen3ASR` connect `Data Loader & Collator` to `Packed LoRA Trainer`, `Splits Validation`, `Audio Perturbations`, `LoRA Prototype Training`, `Phase 2 HP Sweep`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 146 inferred relationships involving `ASRFairnessDataset` (e.g. with `TestASRFairnessDatasetColumnValidation` and `TestCollateFn`) actually correct?**
  _`ASRFairnessDataset` has 146 INFERRED edges - model-reasoned connections that need verification._
- **Are the 126 inferred relationships involving `DataCollatorForQwen3ASR` (e.g. with `TestASRFairnessDatasetColumnValidation` and `TestCollateFn`) actually correct?**
  _`DataCollatorForQwen3ASR` has 126 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `DemographicStratifiedSampler` (e.g. with `TestASRFairnessDatasetColumnValidation` and `TestCollateFn`) actually correct?**
  _`DemographicStratifiedSampler` has 92 INFERRED edges - model-reasoned connections that need verification._