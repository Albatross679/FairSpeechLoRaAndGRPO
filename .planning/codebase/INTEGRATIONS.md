# External Integrations

**Analysis Date:** 2026-04-05

## APIs & External Services

**HuggingFace Model Hub:**
- Service: HuggingFace Hub (huggingface.co)
- What it's used for: Model downloads and inference for wav2vec2, Whisper, Granite, Qwen3-ASR
  - SDK/Client: `transformers` library
  - Auth: Implicit (public models; no auth required in code)
  - Usage locations: `scripts/run_inference.py` lines 242, 314, 476
  - Model loading: `from_pretrained()` calls with model IDs like `facebook/wav2vec2-large-960h-lv60-self`, `openai/whisper-large-v3`, `Qwen/Qwen3-ASR-1.7B`, `ibm-granite/granite-speech-3.3-8b`, `nvidia/canary-qwen-2.5b`

## Data Storage

**Databases:**
- Not applicable (No SQL/NoSQL database)

**File Storage:**
- Local filesystem only - All data stored on HPC cluster
  - Audio datasets: `/users/PAS2030/srishti/asr_fairness/data/`
  - Results: `/users/PAS2030/srishti/asr_fairness/results/commonvoice/`, `/results/fairspeech/`
  - Perturbed audio: `/users/PAS2030/srishti/asr_fairness/perturbed_audio/`
  - Metadata: `/users/PAS2030/srishti/asr_fairness/perturbed_audio/metadata/`

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- None required - All models are public and accessed via HuggingFace Hub without authentication

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, error tracking service)

**Logs:**
- stdout/stderr only
  - Console output during inference (`print()` statements at lines like 294, 369, 430, 564)
  - Progress reporting every 20 batches
  - Summary statistics after inference completion

**Metadata Output:**
- JSON metadata written per model (`meta_{model_key}.json`) containing:
  - Model name and configuration
  - Perturbation type
  - Total utterances processed
  - Overall WER
  - Elapsed time
  - Device used (cuda/cpu)
  - Batch size
  - Timestamp

## CI/CD & Deployment

**Hosting:**
- HPC cluster (Ohio Supercomputer Center)
- Job submission via Bash scripts: `scripts/launch_priority_batch.sh`

**CI Pipeline:**
- None detected (no GitHub Actions, GitLab CI, Jenkins, etc.)

## Environment Configuration

**Required env vars:**
- None explicitly required (all paths hardcoded in scripts)

**Secrets location:**
- Not applicable (no API keys, secrets, or authentication required)

## Data Inputs

**Datasets:**
- CommonVoice v24.0 (English test split)
  - Source: Mozilla CommonVoice project
  - Location: `/users/PAS2030/srishti/bootcamp/data/commonvoice/cv-corpus-24.0-2025-12-05/en`
  - Metadata: `test.tsv` with demographics (gender, accent, age)
  - Manifest: `data/cv_test_manifest.csv` (created by `scripts/prepare_dataset.py`)

- Fair-Speech dataset
  - Source: Fair-Speech project
  - Location: `/users/PAS2030/srishti/asr_fairness/data/fairspeech/`
  - Metadata: `fairspeech_metadata.tsv` with demographics (ethnicity, gender, age, first_language, socioeconomic)
  - Manifest: `data/fs_manifest.csv` (created by `scripts/prepare_fairspeech.py`)

- MUSAN corpus (noise)
  - Purpose: Additive noise for SNR perturbations
  - Download script: `scripts/download_musan.sh`
  - Stored at: `/users/PAS2030/srishti/asr_fairness/data/MUSAN/`

- RIRs (Room Impulse Responses)
  - Purpose: Reverberation perturbations
  - Download script: `scripts/download_rirs.sh`
  - Stored at: `/users/PAS2030/srishti/asr_fairness/data/RIRS/`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## External Tool Dependencies

**Audio Format Support:**
- WAV files (16 kHz mono, float32)
- MP3 files (CommonVoice source format, converted to WAV)

**Text Normalization:**
- Whisper's English text normalizer (from `whisper` or `whisper-normalizer` package)
  - Location: `scripts/run_inference.py` lines 136-139
  - Handles: Case normalization, punctuation removal, number words, contractions

**Model Inference Frameworks:**
- PyTorch (CPU and CUDA 11.x/12.x)
- NVIDIA NeMo (for Canary-Qwen SALM)
- HuggingFace Transformers (model loading and inference)

---

*Integration audit: 2026-04-05*
