# Requirements: GRPO Fairness-Aware ASR Fine-Tuning

**Defined:** 2026-04-05
**Core Value:** Produce a working GRPO training pipeline that traces the accuracy-fairness Pareto frontier for Qwen3-ASR-1.7B

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Preparation

- [ ] **DATA-01**: Fair-Speech and Common Voice 24 train/eval splits created with demographic labels (ethnicity, accent) preserved
- [ ] **DATA-02**: Data loaders yield batches of (audio_tensor, reference_transcript, demographic_group) for training
- [ ] **DATA-03**: Each demographic group has sufficient representation in both train and eval splits for per-group metrics

### Infrastructure

- [ ] **INFRA-01**: LoRA adapters attach to Qwen3-ASR-1.7B decoder with frozen encoder via PEFT
- [ ] **INFRA-02**: Model loads from raw HuggingFace weights (not qwen-asr inference wrapper) for backpropagation
- [ ] **INFRA-03**: VRAM profiling confirms peak usage < 14GB on RTX A4000 with G=4 completions
- [ ] **INFRA-04**: Gradient checkpointing enabled to fit within 16GB VRAM budget
- [ ] **INFRA-05**: Adapter save/load/merge lifecycle works end-to-end (train -> save -> load -> evaluate)

### GRPO Training

- [ ] **GRPO-01**: Custom GRPO training loop generates G candidate outputs per input audio
- [ ] **GRPO-02**: WER-based accuracy reward computed per candidate using jiwer: R_acc = (1-lambda)(1-WER)
- [ ] **GRPO-03**: Fairness penalty computed per candidate: R_fair = lambda(-|WER_g - WER_mean|) over demographic groups
- [ ] **GRPO-04**: Composite reward R = R_acc + R_fair with configurable lambda
- [ ] **GRPO-05**: Group-relative advantage normalization across G candidates
- [ ] **GRPO-06**: Clipped surrogate policy loss (PPO-style) for stable updates
- [ ] **GRPO-07**: Dr. GRPO variant available: removes std normalization, adds length normalization
- [ ] **GRPO-08**: Stratified demographic batching ensures each batch contains multiple demographic groups
- [ ] **GRPO-09**: Per-group WER tracked and logged every N steps during training
- [ ] **GRPO-10**: Adaptive lambda scheduling option (fixed, linear warmup, cosine anneal)

### Baselines

- [x] **BASE-01**: Standard LoRA baseline (lambda=0, pure accuracy reward, no fairness term)
- [ ] **BASE-02**: FairLoRA baseline: L = L_standard + lambda * Var(L_g across groups)
- [ ] **BASE-03**: Group-DRO baseline with smoothed exponential group weight updates
- [ ] **BASE-04**: ICASSP 2026 fairness-prompted baseline: L = lambda_e*L_ERM + lambda_s*L_SD + lambda_d*L_DRO + lambda_i*L_IRM

### Evaluation

- [ ] **EVAL-01**: Trained adapters evaluated on Fair-Speech via existing inference pipeline
- [ ] **EVAL-02**: Trained adapters evaluated on Common Voice 24 via existing inference pipeline
- [ ] **EVAL-03**: Fairness metrics computed for all trained models: max-min ratio, gap%, std, bootstrap CIs
- [ ] **EVAL-04**: Evaluation covers both ethnicity and accent demographic axes independently
- [ ] **EVAL-05**: Perturbation robustness evaluation under existing 12 conditions for all trained adapters
- [ ] **EVAL-06**: Error decomposition (sub/ins/del rates) by demographic group for all models

### Experiments

- [ ] **EXPT-01**: Lambda sweep across multiple values traces Pareto frontier of accuracy vs. fairness
- [ ] **EXPT-02**: Pareto frontier visualization: scatter plot of (mean_WER, fairness_metric) for all methods/lambdas
- [ ] **EXPT-03**: Statistical significance testing (bootstrap pairwise) between GRPO and each baseline
- [ ] **EXPT-04**: Hyperparameter sensitivity analysis: ablation over G, learning rate, beta (KL coefficient)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Methods

- **EXT-01**: Multiple reward function ablation (WER, exact match, edit distance)
- **EXT-02**: Beam search vs. multinomial sampling comparison for GRPO generation
- **EXT-03**: Intersectional analysis (ethnicity x accent x gender) -- group sizes may be too small
- **EXT-04**: Demographic conditioning / group-specific LoRA adapters (MAS-LoRA style)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full fine-tuning (no LoRA) | 16GB VRAM makes this infeasible; unfair comparison since baselines also use param-efficient methods |
| PPO with separate value model | GRPO's advantage is eliminating the critic; PPO doubles memory |
| Attention head surgery / hallucination mitigation | Separate future work direction from Section 4.2 |
| Real-time / streaming inference | Research pipeline, not production |
| Automated HPO (Optuna, Ray Tune) | Compute budget too limited; manual selection + lambda sweep sufficient |
| Web UI or dashboard | CLI scripts + matplotlib, consistent with existing pipeline |
| Multi-task learning (ASR + fairness classification) | GRPO already encodes fairness via reward; auxiliary heads add complexity without clear benefit |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1.0 | Pending |
| DATA-02 | Phase 1.0 | Pending |
| DATA-03 | Phase 1.0 | Pending |
| INFRA-01 | Phase 1.1 | Pending |
| INFRA-02 | Phase 1.1 | Pending |
| INFRA-03 | Phase 1.1 | Pending |
| INFRA-04 | Phase 1.1 | Pending |
| INFRA-05 | Phase 1.1 | Pending |
| GRPO-01 | Phase 2 | Pending |
| GRPO-02 | Phase 2 | Pending |
| GRPO-03 | Phase 2 | Pending |
| GRPO-04 | Phase 2 | Pending |
| GRPO-05 | Phase 2 | Pending |
| GRPO-06 | Phase 2 | Pending |
| GRPO-07 | Phase 2 | Pending |
| GRPO-08 | Phase 2 | Pending |
| GRPO-09 | Phase 2 | Pending |
| GRPO-10 | Phase 2 | Pending |
| BASE-01 | Phase 1.2 | Complete |
| BASE-02 | Phase 3 | Pending |
| BASE-03 | Phase 3 | Pending |
| BASE-04 | Phase 3 | Pending |
| EVAL-01 | Phase 1.2 | Pending |
| EVAL-02 | Phase 1.2 | Pending |
| EVAL-03 | Phase 1.2 | Pending |
| EVAL-04 | Phase 1.2 | Pending |
| EVAL-05 | Phase 3 | Pending |
| EVAL-06 | Phase 3 | Pending |
| EXPT-01 | Phase 4 | Pending |
| EXPT-02 | Phase 4 | Pending |
| EXPT-03 | Phase 4 | Pending |
| EXPT-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 32
- Unmapped: 0

---
*Requirements defined: 2026-04-05*
*Last updated: 2026-04-05 after roadmap creation*
