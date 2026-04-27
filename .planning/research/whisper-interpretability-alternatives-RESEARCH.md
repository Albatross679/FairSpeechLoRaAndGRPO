---
fileClass: Reference
title: Whisper Interpretability Alternatives to Iterative Head Ablation
researched: 2026-04-21
domain: mechanistic interpretability / ASR fairness
confidence: medium-high
mode: survey
source: web research (arXiv, Neel Nanda, Mozilla Builders, NNsight, TransformerLens)
url: (see Sources table)
authors: (multi-source synthesis)
---

# Whisper Interpretability Alternatives to Iterative Head Ablation

## Summary

The user currently zero-masks each of Whisper-large-v3's 640 decoder heads (32 layers, 20 heads) individually and runs full inference on an accent subgroup (484 utterances). Cost scales as O(n_heads x eval_set x forward_pass). This is the most faithful method but also by far the most expensive, and it is not what the 2024 to 2026 mechanistic interpretability literature uses at scale.

**Primary recommendation:** use **attribution patching** (gradient-based linear approximation to activation patching, Nanda 2023 / Syed et al. 2023) as a cheap pre-screen to rank all 640 heads in ~2 forward + 1 backward pass per utterance, then confirm the top-K (say K=20) with your existing full-ablation pipeline. This is the exact workflow the AtP* paper (Kramar et al., DeepMind 2024) recommends and what "Attribution Patching Outperforms Automated Circuit Discovery" (Syed, Rager, Conmy, NeurIPS ATTRIB 2023) validates empirically.

**Honest caveats:** (1) attribution patching is a first-order Taylor approximation and it is unreliable when (a) you patch large activations (residual stream, not per-head), (b) effect sizes are small, or (c) you are near softmax saturation. In your CV25 Indian-accent baseline the insertion rate is already 1.27%, so the maximum possible Delta is ~1.27pp. Gradient noise at that scale may exceed signal. (2) No off-the-shelf mech-interp library (TransformerLens, nnsight, pyvene) ships with a turnkey Whisper wrapper, though nnsight and pyvene are architecture-agnostic and work via PyTorch hooks on the HuggingFace `WhisperForConditionalGeneration`. (3) For a 7-day class project with a working full-ablation pipeline already, the honest thing to do is keep iterative ablation as the ground truth, add attribution patching only as a sanity-check secondary score, and stop there.

## Landscape Map

### 1. Ablation-based (what the user does today)

| Method | Description | Notes |
|---|---|---|
| Zero ablation (user) | Set W_O head output to 0, run inference | Current pipeline |
| Mean ablation | Replace head output with dataset mean | Cleaner counterfactual, same cost |
| Resample ablation | Replace with activation from distractor input | Used in ACDC, Calm-Whisper |
| Optimal ablation (Li & Janson 2024) | Learn an optimal constant to substitute | Improves faithfulness over zero |

### 2. Gradient-based attribution (O(1) per forward-backward pair)

| Method | Description | Origin |
|---|---|---|
| Head importance score I_h | E[\|dL/d xi_h\|] at xi_h=1 (mask variable) | Michel, Levy, Neubig, NeurIPS 2019 |
| Attribution patching (AtP) | (clean_act - corr_act) * corr_grad, summed | Nanda 2023 blog |
| Attribution patching* (AtP*) | AtP with two failure-mode fixes: Q/K dropout handling and iterative recomputation | Kramar et al., DeepMind 2024 |
| Edge attribution patching (EAP) | AtP applied to all edges in the computation graph for circuit discovery | Syed, Rager, Conmy, NeurIPS ATTRIB 2023 |
| Integrated Gradients | Path integral from baseline to input | Sundararajan et al. 2017 |
| AttAttr (self-attention attribution) | Integrated gradients on attention weights | Hao et al. 2021 |

### 3. Activation patching / causal tracing (one forward pass per patch)

| Method | Description | Notes |
|---|---|---|
| Activation patching | Replace a single component's activation from clean to corrupted run; measure output delta | Meng et al. 2022, Heimersheim and Nanda 2024 |
| Path patching | Patch only paths between a specific upstream-downstream component pair | Goldowsky-Dill et al. 2023 |
| Causal tracing | Activation patching with Gaussian-noised counterfactual | Meng et al. (ROME) |
| ACDC | Automated circuit discovery via greedy edge ablation | Conmy et al. NeurIPS 2023 |

### 4. Probing / lens methods (cheap, correlational only)

| Method | Description | Notes |
|---|---|---|
| Logit lens | Project hidden states through unembed at every layer | nostalgebraist 2020; applied to Whisper by Hughes 2022 |
| Tuned lens | Train per-layer affine to improve logit lens | Belrose et al. 2023 |
| DecoderLens | Encoder-decoder variant: decode intermediate encoder states | Langedijk et al. 2023 (directly relevant to Whisper) |
| Linear probes | Train a linear classifier on frozen activations to predict an attribute | Alain and Bengio 2016; used in "Beyond Transcription" 2025 |

### 5. Decomposition-based (expensive to set up, cheap to query)

| Method | Description | Cost profile |
|---|---|---|
| Sparse Autoencoders (SAEs) | Overcomplete dictionary on residual stream finds interpretable features | High upfront cost (hours to days of training per layer), cheap at query time |
| AudioSAE | SAEs trained across all Whisper encoder layers | Sadov / Mozilla Builders 2024, AudioSAE paper 2026 |
| Cross-coders / transcoders | SAEs across layer pairs | Very recent, no Whisper implementation yet |

### 6. ASR-specific techniques

| Method | Description | Source |
|---|---|---|
| Cross-attention alignment (DTW over attention) | Use Whisper's cross-attention as time-alignment signal | OpenAI Whisper repo |
| "Crazy head" layer-wide masking | Mask head index h across ALL decoder layers simultaneously | Calm-Whisper (Wang et al. 2025) |
| Encoder layer skipping | Remove top k encoder layers to test redundancy | Hughes 2022 (logit lens hackathon); Beyond Transcription 2025 |
| Component-type patching | Patch all cross-attn vs all self-attn vs all FFN as groups, not per head | Beyond Transcription (Glazer et al. 2025) |

## Compute Cost Comparison

Assume: n_utt = 484 (your CV25 Indian subset), n_heads = 640, Whisper-large-v3, 1 forward pass ~ 0.5 GB activations, 1 backward pass ~ 3x forward cost.

| Method | Cost per utterance | Scales with n_heads? | Faithfulness vs ground-truth ablation | Whisper tooling ready? |
|---|---|---|---|---|
| Iterative single-head zero-ablation (current) | n_heads x 1 fwd = 640 fwd | Linear O(H) | **Ground truth (definitional)** | Yes, user has it |
| Mean / resample ablation | n_heads x 1 fwd = 640 fwd | Linear O(H) | Equivalent to zero-ablation, cleaner counterfactual | Drop-in, user has hooks |
| Attribution patching (AtP) | 2 fwd + 1 bwd = ~5 fwd-equivalents | **O(1)** (all heads scored in one bwd) | ~0.7 to 0.9 correlation for head outputs; poor for residual-stream patches | Needs hooks on Whisper attn_output (no wrapper) |
| AtP* (Kramar 2024) | 2 fwd + 1 bwd + iterative recompute for suspicious heads (~2-5x more) | O(1) + correction passes | Higher recall than plain AtP, fewer false negatives | Reference PyTorch/nnsight impl exists (koayon/atp_star) |
| Edge Attribution Patching (EAP) | 2 fwd + 1 bwd for per-edge scores | O(1) per step, O(K) greedy | Outperforms ACDC at circuit recovery | Code: Aaquib111/edge-attribution-patching (LLM-oriented, porting required) |
| Michel 2019 head importance I_h | 1 fwd + 1 bwd | **O(1)** | Correlation with true ablation ~0.4 for IG-family; Michel reports pruning 40-80% heads without major loss | Drop-in: add scalar masks then call .backward() |
| Activation patching (single circuit) | 1 fwd per patch | O(components being patched) | Ground-truth under the corruption chosen | Raw hooks; no Whisper-specific library |
| Path patching | 1 fwd per (src, dst) pair | O(H^2) pairs but usually O(H) targeted | Ground truth for direct effects | Raw hooks |
| ACDC | Hours per circuit, greedy edge sweep | O(E) edges | Ground-truth greedy; slower than EAP with no AUC gain | Conmy et al. codebase, LLM-oriented |
| Logit lens | 1 fwd, decode every layer | O(L) layers | Correlational, no causal claim | Trivial: `model.lm_head(hidden_state)` at each layer. Whisper-specific code: jplhughes/whisper_logit_lens |
| Linear probes | 1 fwd (collect activations) + probe training | O(1) per layer | Correlational | Trivial with HF hooks |
| SAE training (per layer) | Hours to days, millions of forward passes | O(L) but each SAE huge | High fidelity features but features != causal components | AudioSAE codebase (Mozilla) has pretrained Whisper SAEs |

**Back-of-envelope for the user's setup:**

- Current pipeline: 640 heads x 484 utts = 309,760 forward passes (roughly matches their reported `sweep.csv` row count). On a single H100 at Whisper-large batch 32 that is many hours.
- Attribution patching: 2 forward + 1 backward per utt x 484 utts = ~2,420 forward-equivalents. That is **~128x cheaper** than the current sweep while producing a score for every one of the 640 heads.
- Michel 2019 head importance: 1 forward + 1 backward per utt x 484 utts = ~1,936 forward-equivalents. Even cheaper, but no baseline / counterfactual contrast.

## Trends

**Growing (2024 to 2026):**
- Attribution patching and AtP* as the default pre-screen before any expensive intervention (Kramar 2024, Syed 2023, nnsight tutorials now feature it as a canonical tutorial).
- SAE-based feature discovery, including AudioSAE / Mozilla Insider-Whisper work (2024 to 2026).
- Circuit tracing / attribution graphs (Anthropic 2025) as a unified framework combining AtP, activation patching, and SAEs.
- Tuned-lens and DecoderLens variants adapted for encoder-decoder models.

**Declining:**
- Pure ACDC-style greedy ablation sweeps for large models: ACDC is ROC-AUC worse or equal to EAP while being orders of magnitude slower (Syed et al. 2023).
- Ad-hoc per-head ablation as a primary method in the interpretability literature. It is still used in ASR papers (Calm-Whisper, Beyond Transcription) but those papers mask a head INDEX across all layers, not per (layer, head) cell, which is ~32x fewer runs.

**Emerging:**
- Mechanistic interpretability for ASR as a distinct subfield: "Beyond Transcription" (Glazer et al., AAAI 2026 / arXiv 2508.15882) is the first systematic application of logit lens + activation patching + probing to Whisper.
- LoRA-adapter interpretability with NNsight (Wang et al. 2025, arXiv 2509.08454) uses NNsight on Whisper explicitly.
- Fairness-aware structured pruning that uses head importance proxies to locate bias-driving heads (AAAI 2024).

## Notable Projects and Papers

### Attribution Patching (Nanda 2023)
Blog post that introduced AtP as a linear approximation to activation patching using the identity: `patch_effect ~= sum((clean_act - corr_act) * corr_grad)`. Reports ~30,000x speedup over activation patching for GPT-2 XL when patching each head at each position. Caveat: works well for head outputs (correlation 0.7 to 0.9 with true ablation) and poorly for residual stream and MLP0 (correlation 0.2 to 0.4). Explicitly labels it "exploratory, hypothesis-generating, not confirmatory."

### AtP* (Kramar et al., DeepMind, 2024)
Identifies two failure modes of plain AtP: (a) failure near softmax saturation (dropout-style correction), (b) cancellation between query and key contributions. Adds two fixes while keeping O(1) scaling. Reference implementation at github.com/koayon/atp_star using PyTorch + NNsight.

### Edge Attribution Patching (Syed, Rager, Conmy, NeurIPS ATTRIB 2023)
Applies AtP to the full computational graph's edges for automated circuit discovery. Outperforms ACDC on ROC AUC for circuit recovery while using only 2 forward + 1 backward pass instead of O(edges) forward passes. Code: github.com/Aaquib111/edge-attribution-patching.

### Beyond Transcription (Glazer, Segal-Feldman, Segev, et al., 2025)
First systematic mech-interp study of ASR. Applies activation patching + linear probing + logit lens to Whisper decoder. Finds that cross-attention in layer 23 alone explains 76% of repetition hallucinations, and head 13 in layer 18 suppresses 78.1% of repetitions when targeted alone. Does NOT use TransformerLens or nnsight, just raw PyTorch hooks.

### Calm-Whisper (Wang et al., Interspeech 2025)
Iteratively masks each of the 20 decoder self-attention head INDICES (across all layers simultaneously, not per-layer) on UrbanSound8K. Finds heads 1, 6, 11 jointly cause 75% of non-speech hallucinations. Confirms that iterative ablation is still viable for Whisper when the search space is collapsed to n_heads_per_layer = 20 instead of n_heads_total = 640.

### AudioSAE / Mozilla Insider-Whisper (Sadov 2024, Zaiem et al. 2026)
Sparse autoencoders trained on all encoder layers of Whisper and HuBERT. Pretrained checkpoints on HuggingFace. Feature steering reduces false speech detections by 70% with negligible WER hit. Highest-fidelity Whisper interpretability tooling currently available, but setup cost is high (SAE training is days per layer) and features are not the same unit as "attention heads."

### LoRA-Whisper Interpretability (Wang et al., 2025, arXiv 2509.08454)
Uses NNsight on HuggingFace Whisper for layer-contribution probing, logit lens, and representational similarity of LoRA adapters. Confirms NNsight works on Whisper in practice. No activation patching or AtP in this paper though.

### Whisper Logit Lens Hackathon (Hughes 2022)
Alignment Jam 2022 project. Raw PyTorch + HuggingFace Transformers. Shows Whisper's decoder tolerates removing up to 3 of 24 layers without degradation and encoder tolerates up to 8. Beyond 15 encoder layers removed, repetition hallucinations emerge. Simplest Whisper-specific interpretability codebase to fork.

### Koenecke et al. 2020 and ASR fairness lineage
The foundational ASR fairness work (AAVE 2x higher WER than white speech) treats ASR as a **black box** throughout. The recent fairness ASR literature (FairLENS 2024, ASR Under the Stethoscope 2025, Pacific Northwest English Corpus 2025) continues to treat models as black boxes. Linking fairness metrics to model internals is still a research gap. Your project pushing this direction is novel.

## Gaps and Opportunities

1. **No Whisper-native mech-interp library.** TransformerLens supports T5 (encoder-decoder) via `t5_block` components but NOT Whisper (verified via model properties table as of April 2026). nnsight and pyvene both work architecture-agnostically through PyTorch hooks but ship no Whisper-specific wrapper, examples, or tutorial.
2. **No published attribution-patching study of Whisper.** The 2025 Whisper mech-interp papers all use raw activation patching or ablation; none apply AtP or EAP. Low-hanging fruit for a class project.
3. **Fairness x mech-interp is an open research area.** Koenecke-line ASR fairness is black-box; mech-interp ASR work (Beyond Transcription, Calm-Whisper) focuses on hallucinations not demographic bias. Applying AtP per-subgroup to find heads that matter more for accented speech than for native speech is genuinely novel.
4. **Small effect sizes break gradient methods.** Your CV25 Indian baseline is 1.27% insertion, best ablation gives -0.08pp. The approximation error of AtP may be larger than this signal. Use AtP for ranking, verify top-K with your existing full-ablation pipeline.
5. **Layer-vs-(layer,head) collapse.** Calm-Whisper's trick of masking head-INDEX across all layers simultaneously (20 configs instead of 640) is a worthwhile middle ground if AtP is too noisy.

## Recommendation for This Project

The user is a student finishing a 7-day milestone on a 24-hour implementation budget with an already-working full-ablation pipeline. The pragmatic path is:

### Phase 1: Add attribution patching as a secondary score (1 to 2 days)

Do this in the existing `scripts/head_surgery/` module. No new library required.

```python
# Pseudocode. Core idea: forward+backward on a batch, read grad at each head's
# attention output, multiply by the (zero-mean-ablation - clean) activation delta.

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Cache activations and gradients on attn output of every decoder self-attn
acts, grads = {}, {}
for layer_idx, layer in enumerate(model.model.decoder.layers):
    def save_act(m, i, o, L=layer_idx): acts[L] = o
    def save_grad(m, gi, go, L=layer_idx): grads[L] = go[0]
    layer.self_attn.out_proj.register_forward_hook(save_act)
    layer.self_attn.out_proj.register_full_backward_hook(save_grad)

# Clean + corrupted forward, then backward on the metric
clean_out = model(**batch_clean); clean_loss = metric(clean_out)
# For mean ablation the "corrupted" activation is the batch mean per head
clean_loss.backward()

# For each (layer, head) compute attribution
# Whisper head_dim = d_model // num_heads; out_proj weight is (d_model, d_model)
# but conceptually split per head.
for L, act in acts.items():
    for h in range(num_heads):
        head_slice = slice(h * head_dim, (h+1) * head_dim)
        attr[L, h] = ((act[..., head_slice] - act[..., head_slice].mean(0))
                      * grads[L][..., head_slice]).sum().item()
```

Run this **once** over the 484 Indian-accent utterances (one forward + one backward per utterance). Compare the resulting rank-ordering of 640 heads against your existing full-ablation `head_scores.csv` via Spearman correlation. If rho > 0.7 on the top-K heads, the method is trustworthy for this model and you have a 128x speedup for future subgroup audits.

Library choice: **just use HuggingFace Transformers + raw PyTorch hooks**. Skip TransformerLens (no Whisper support as of April 2026), skip nnsight for the initial proof-of-concept (adds a dependency for a one-file script). If you want a cleaner API later, nnsight is the right choice and the LoRA-Whisper 2025 paper confirms it works on Whisper. Pyvene also works but the docs are thinner.

### Phase 2: Apply to every subgroup (1 day)

Re-run Phase 1 for each demographic slice (Indian accent, non-Indian accent, gender, age if available). Compute per-head subgroup **attribution gap**: `attr_subgroup[L, h] - attr_reference[L, h]`. Large positive values are heads that matter disproportionately for the disadvantaged subgroup, which is the fairness-relevant mechanistic signal that nobody has published yet.

### Phase 3: Confirm top-K with full ablation (existing pipeline)

Take the top 20 heads by attribution gap, run your existing zero-mask pipeline on only those 20 heads x 484 utts = 9,680 forward passes (~30x cheaper than the full sweep). Report both the full-ablation and attribution-patching scores in the paper. This is honest, cheap, and novel.

### What NOT to do

- Do not train SAEs. Setup cost is days per layer, does not answer the (layer, head) ablation question, and AudioSAE already has pretrained checkpoints if you ever want to go that route.
- Do not try to port TransformerLens to Whisper. This is a multi-week project.
- Do not run ACDC. Slower than both EAP and your current pipeline, no AUC gain, and designed for algorithmic tasks not ASR subgroups.
- Do not rely on logit lens or probes as the PRIMARY fairness attribution. They are correlational only, and the fairness community will (rightly) ask for causal evidence.

### Effort estimate

- Phase 1 (attribution patching + correlation check): **1 to 2 days**, half a day if you already have hooks on attn outputs.
- Phase 2 (subgroup attribution gap): **1 day**, mostly bookkeeping.
- Phase 3 (confirm top-K with full ablation): **half a day** at current pipeline throughput.

Total: **~3 days** for a story that is both faster AND more novel than the pure ablation sweep. The existing `sweep.csv` becomes your "gold" for validating AtP on this model, which is itself a contribution.

## Sources

| # | Source | URL | Accessed |
|---|---|---|---|
| 1 | Attribution Patching: Activation Patching At Industrial Scale (Nanda 2023 blog) | https://www.neelnanda.io/mechanistic-interpretability/attribution-patching | 2026-04-21 |
| 2 | AtP*: Efficient and scalable methods for localizing LLM behaviour (Kramar et al., DeepMind 2024) | https://deepmind.google/research/publications/68553/ | 2026-04-21 |
| 3 | Attribution Patching Outperforms Automated Circuit Discovery (Syed, Rager, Conmy, NeurIPS ATTRIB 2023) | https://arxiv.org/abs/2310.10348 | 2026-04-21 |
| 4 | How to use and interpret activation patching (Heimersheim and Nanda 2024) | https://arxiv.org/html/2404.15255v1 | 2026-04-21 |
| 5 | Beyond Transcription: Mechanistic Interpretability in ASR (Glazer et al. 2025) | https://arxiv.org/html/2508.15882 | 2026-04-21 |
| 6 | Calm-Whisper: Reduce Whisper Hallucination On Non-Speech (Wang et al. 2025) | https://arxiv.org/html/2505.12969v1 | 2026-04-21 |
| 7 | Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper (Wang et al. 2025) | https://arxiv.org/html/2509.08454 | 2026-04-21 |
| 8 | Whisper Logit Lens hackathon (Hughes 2022) | https://github.com/jplhughes/whisper_logit_lens | 2026-04-21 |
| 9 | Feature Discovery in Audio Models (Sadov, Mozilla Builders 2024) | https://builders.mozilla.org/insider-whisper/ | 2026-04-21 |
| 10 | AudioSAE paper (EACL 2026) | https://arxiv.org/html/2602.05027 | 2026-04-21 |
| 11 | Are Sixteen Heads Really Better than One? (Michel, Levy, Neubig, NeurIPS 2019) | https://arxiv.org/abs/1905.10650 | 2026-04-21 |
| 12 | Sixteen Heads CMU blog (explains I_h formula) | https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/ | 2026-04-21 |
| 13 | Localizing Model Behavior with Path Patching (Goldowsky-Dill et al. 2023) | https://arxiv.org/abs/2304.05969 | 2026-04-21 |
| 14 | Towards Automated Circuit Discovery (ACDC, Conmy et al. NeurIPS 2023) | https://arxiv.org/abs/2304.14997 | 2026-04-21 |
| 15 | TransformerLens supported models table | https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html | 2026-04-21 |
| 16 | nnsight documentation | https://nnsight.net/documentation/ | 2026-04-21 |
| 17 | nnsight Attribution Patching tutorial | https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/attribution_patching/ | 2026-04-21 |
| 18 | pyvene (Stanford NLP) | https://github.com/stanfordnlp/pyvene | 2026-04-21 |
| 19 | AtP* reference implementation | https://github.com/koayon/atp_star | 2026-04-21 |
| 20 | Edge Attribution Patching code | https://github.com/Aaquib111/edge-attribution-patching | 2026-04-21 |
| 21 | DecoderLens (Langedijk et al. 2023) | https://arxiv.org/pdf/2310.03686 | 2026-04-21 |
| 22 | Optimal ablation for interpretability (Li and Janson, NeurIPS 2024) | https://proceedings.neurips.cc/paper_files/paper/2024/file/c55e6792923cc16fd6ed5c3f672420a5-Paper-Conference.pdf | 2026-04-21 |
| 23 | Investigation of Whisper ASR Hallucinations (non-speech, black-box) | https://arxiv.org/html/2501.11378v1 | 2026-04-21 |
| 24 | LessWrong: Interpreting OpenAI's Whisper (behind 429, referenced only) | https://www.lesswrong.com/posts/thePw6qdyabD8XR4y/interpreting-openai-s-whisper | 2026-04-21 |
