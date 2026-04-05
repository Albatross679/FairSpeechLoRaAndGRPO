# Technology Stack

**Analysis Date:** 2026-04-05

## Languages

**Primary:**
- Python 3.7+ - All source code, scripts, and data processing pipelines

**Secondary:**
- Bash - HPC job submission and data pipeline orchestration (`scripts/download_musan.sh`, `scripts/download_rirs.sh`, `scripts/launch_priority_batch.sh`, `scripts/setup_nemo.sh`)

## Runtime

**Environment:**
- Python 3.x (unspecified, commonly 3.9+)

**Package Manager:**
- pip - Primary dependency management
- Lockfile: Not detected (no requirements.txt or Pipfile.lock found)

## Frameworks

**Core ASR/Speech ML:**
- Transformers (huggingface) - Model loading and inference for wav2vec2, Whisper, Granite
- PyTorch (torch) - Neural network computation and model loading
- torchaudio - Audio loading, resampling, feature extraction (16kHz standardization)
- NeMo - NVIDIA Speech AI toolkit for Canary-Qwen SALM model (`nemo.collections.speechlm2.models.SALM`)
- qwen_asr - Qwen3-ASR model wrapper (`Qwen3ASRModel.from_pretrained()`)

**Data Processing:**
- Pandas - DataFrame operations for manifests, predictions, analysis
- NumPy - Numerical computations for signal processing
- soundfile (sf) - Audio file I/O for reading/writing WAV files
- scipy - Signal processing (convolution for reverberation via `scipy.signal.fftconvolve`)
- librosa (optional) - Audio resampling alternative if torchaudio unavailable

**Testing & Analysis:**
- jiwer - Word Error Rate (WER) computation and error decomposition (substitutions, insertions, deletions)
- scipy.stats - Mann-Whitney U tests, bootstrap confidence intervals

**Visualization:**
- Matplotlib - Low-level plotting (non-interactive Agg backend on HPC)
- Seaborn - Statistical visualization for fairness metrics

**Utilities:**
- tqdm - Progress bars for batch processing
- datasets (huggingface) - Dataset loading for CommonVoice, FairSpeech

## Key Dependencies

**Critical (Model Inference):**
- Transformers [version unspecified] - Loads wav2vec2, Whisper, Granite models from HuggingFace Hub
- PyTorch [version unspecified] - Tensor computation, GPU support via `torch.cuda` or `torch.device`
- torchaudio [version unspecified] - Audio preprocessing and resampling
- jiwer [version unspecified] - WER computation (corpus-level via `jiwer.wer()`, error breakdown via `jiwer.process_words()`)

**Infrastructure (HPC/Data):**
- NeMo >= 2.5.0 - Required for Canary-Qwen model support (optional installation: `pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'`)
- qwen-asr - Qwen3-ASR package (model-specific wrapper)
- Whisper normalizer - Text normalization (`whisper.normalizers.EnglishTextNormalizer` or fallback `whisper_normalizer.english`)

**Data Processing:**
- Pandas - CSV reading/writing for manifests and results
- NumPy - Numerical array operations
- soundfile - WAV file I/O
- scipy - Signal processing (FFT convolution)
- datasets - HuggingFace dataset API

**Analysis:**
- scipy.stats - Statistical significance testing
- Matplotlib/Seaborn - Publication-ready figures

## Configuration

**Environment:**
- HPC cluster paths hardcoded in scripts (e.g., `/users/PAS2030/srishti/asr_fairness/`)
- No .env file detected
- Dataset and results directories defined as constants in each script

**Build:**
- No explicit build system (pure Python scripts)
- Bash scripts for HPC job submission and data downloads

## Platform Requirements

**Development:**
- Python 3.7+
- CUDA-capable GPU (optional but recommended)
- 16GB+ RAM for model loading
- Linux/HPC environment

**Production (Inference):**
- NVIDIA GPU (CUDA 11.x or 12.x) for model inference
- 8GB+ VRAM (varies by model size, 0.6B–8B parameters)
- 16GB+ system RAM
- Local filesystem for audio data and results storage

**Data Storage:**
- Local filesystem for audio datasets (CommonVoice v24.0, Fair-Speech)
- CSV manifests for metadata and predictions
- JSON metadata for analysis results

---

*Stack analysis: 2026-04-05*
