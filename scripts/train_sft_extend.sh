#!/bin/bash
# ============================================================================
# SFT Extend Training Script for Qwen 1.8B
# ============================================================================
# This script trains the extended SFT model using the train_sft_extend split
# Configuration: 2 epochs, 10000 examples
# ============================================================================

set -e

# Configuration
MODEL="qwen18"
EXP_NAME="qwen18_sft_extend_ep2"
N_EPOCHS=2
N_EXAMPLES=10000
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1  # Effective batch = BATCH_SIZE * GRAD_ACCUM = 32
EVAL_EVERY=500
LR="5e-7"

# GPU Configuration - Use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Navigate to source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src"

echo "=============================================="
echo "SFT Extend Training - Qwen 1.8B"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Experiment: ${EXP_NAME}"
echo "Train Split: train_sft_extend"
echo "Epochs: ${N_EPOCHS}"
echo "Examples: ${N_EXAMPLES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Run training
python -u train.py \
    model=${MODEL} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    train_split=train_sft_extend \
    n_epochs=${N_EPOCHS} \
    n_examples=${N_EXAMPLES} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    eval_every=${EVAL_EVERY} \
    lr=${LR} \
    save_ckp=true \
    fine_evaluation=true \
    wandb.enabled=true \
    wandb.project="learntune_replication"

echo ""
echo "=============================================="
echo "SFT Extend Training Complete!"
echo "=============================================="
echo "Checkpoint saved to: exp_results/${EXP_NAME}"
echo ""
