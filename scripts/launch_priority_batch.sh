#!/bin/bash
# ── Priority batch launcher for perturbation experiments ─────────────────
# Follows the README's priority order:
#   1. Noise (SNR) on Fair-Speech
#   2. Silence injection on Fair-Speech
#   3. Noise (SNR) on Common Voice
#   4. Silence injection on Common Voice
#   5. Remaining conditions
#
# Usage:
#   bash scripts/launch_priority_batch.sh         # Priority 1+2 (54 jobs)
#   bash scripts/launch_priority_batch.sh --all    # All 216 jobs

set -uo pipefail

PROJECT_DIR="/users/PAS2030/srishti/asr_fairness"

MODELS=(
    "wav2vec2-large"
    "whisper-small"
    "whisper-medium"
    "whisper-large-v3"
    "qwen3-asr-0.6b"
    "qwen3-asr-1.7b"
    "canary-qwen-2.5b"
    "granite-speech-3.3-2b"
    "granite-speech-3.3-8b"
)

# Priority 1: SNR on Fair-Speech  (indices 0-2)
# Priority 2: Silence on Fair-Speech (indices 6-8)
# Combined as array indices for slurm
PRIORITY_FS_INDICES="0-2,6-8"

# Priority 3: SNR on Common Voice (indices 0-2)
# Priority 4: Silence on Common Voice (indices 6-8)
PRIORITY_CV_INDICES="0-2,6-8"

# Remaining: reverb (3-5) + masking (9-11)
REMAINING_INDICES="3-5,9-11"

echo "═══════════════════════════════════════════════════"
echo "  Perturbation Experiment Batch Launcher"
echo "═══════════════════════════════════════════════════"

if [ "${1:-}" = "--all" ]; then
    echo "Mode: ALL conditions (216 total jobs)"
    echo ""

    for MODEL in "${MODELS[@]}"; do
        echo "Submitting: ${MODEL} × all perturbations × Fair-Speech"
        DATASET=fs MODEL="${MODEL}" sbatch --array=0-11 \
            "${PROJECT_DIR}/slurm/run_perturbed_inference.sh"
    done
    echo ""
    for MODEL in "${MODELS[@]}"; do
        echo "Submitting: ${MODEL} × all perturbations × Common Voice"
        DATASET=cv MODEL="${MODEL}" sbatch --array=0-11 \
            "${PROJECT_DIR}/slurm/run_perturbed_inference.sh"
    done
else
    echo "Mode: Priority conditions only (108 jobs)"
    echo "  Priority 1: SNR (20/10/0 dB) on Fair-Speech"
    echo "  Priority 2: Silence (25/50/75%) on Fair-Speech"
    echo "  Priority 3: SNR (20/10/0 dB) on Common Voice"
    echo "  Priority 4: Silence (25/50/75%) on Common Voice"
    echo ""

    for MODEL in "${MODELS[@]}"; do
        echo "  [FS] Submitting: ${MODEL}"
        DATASET=fs MODEL="${MODEL}" sbatch --array="${PRIORITY_FS_INDICES}" \
            "${PROJECT_DIR}/slurm/run_perturbed_inference.sh"
    done
    echo ""
    for MODEL in "${MODELS[@]}"; do
        echo "  [CV] Submitting: ${MODEL}"
        DATASET=cv MODEL="${MODEL}" sbatch --array="${PRIORITY_CV_INDICES}" \
            "${PROJECT_DIR}/slurm/run_perturbed_inference.sh"
    done
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Jobs submitted! Monitor with: squeue -u \$USER"
echo "═══════════════════════════════════════════════════"
