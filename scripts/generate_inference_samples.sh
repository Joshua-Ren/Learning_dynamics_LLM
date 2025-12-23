#!/bin/bash
# ============================================================================
# Inference Sample Generation Script
# ============================================================================
# This script generates inference samples for evaluation from trained models.
# It generates responses for prob_train_gen, prob_test_gen, prob_train_selfr,
# and prob_test_selfr datasets.
# ============================================================================

set -e

# Default model to generate from - can be overridden by command line argument
MODEL=${1:-"qwen18"}
CHECKPOINT=${2:-"qwen18_dpo_base_ep6"}
EPOCH=${3:-""}  # Optional: specify epoch like "epoch_2", "epoch_4", "epoch_6"

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0

# Navigate to source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src"

# Determine checkpoint path
if [ -n "${EPOCH}" ]; then
    ARCHIVE="${CHECKPOINT}/${EPOCH}"
    EXP_NAME="gen_${CHECKPOINT}_${EPOCH}"
else
    ARCHIVE="${CHECKPOINT}"
    EXP_NAME="gen_${CHECKPOINT}"
fi

echo "=============================================="
echo "Inference Sample Generation"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Checkpoint: ${ARCHIVE}"
echo "Output: exp_results/${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "exp_results/${ARCHIVE}/policy.pt" ]; then
    echo "ERROR: Checkpoint not found at exp_results/${ARCHIVE}/policy.pt"
    exit 1
fi

# Create output directory
mkdir -p "exp_results/${EXP_NAME}"

# Run inference generation
python -u gen_inference_samples.py \
    model=${MODEL} \
    model.archive=${ARCHIVE} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    max_length=400 \
    wandb.enabled=false

echo ""
echo "=============================================="
echo "Inference Generation Complete!"
echo "=============================================="
echo "Generated files saved to: exp_results/${EXP_NAME}/"
echo "  - prob_train_gen_response.jsonl"
echo "  - prob_test_gen_response.jsonl"
echo "  - prob_train_selfr_response.jsonl"
echo "  - prob_test_selfr_response.jsonl"
echo ""
