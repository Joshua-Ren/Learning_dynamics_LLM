#!/bin/bash
# ============================================================================
# DPO Base Training Script for Qwen 1.8B
# ============================================================================
# This script trains DPO on top of the base SFT model
# Configuration: 6 epochs with checkpoints at epochs 2, 4, 6
# ============================================================================

set -e

# Configuration
MODEL="qwen18"
SFT_CHECKPOINT="qwen18_sft_base_ep2"  # The SFT checkpoint to load
EXP_NAME="qwen18_dpo_base_ep6"
N_EPOCHS=6
N_EXAMPLES=30000  # 5000 examples per epoch
BATCH_SIZE=32  # 97GB VRAM easily handles both policy + reference model
GRADIENT_ACCUMULATION_STEPS=1  # Effective batch = 32
EVAL_EVERY=1000
LR="5e-7"
BETA=0.1  # DPO beta parameter
SAVE_EPOCHS="2,4,6"  # Epochs at which to save checkpoints

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0

# Navigate to source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../src"

echo "=============================================="
echo "DPO Base Training - Qwen 1.8B"
echo "=============================================="
echo "Model: ${MODEL}"
echo "SFT Checkpoint: ${SFT_CHECKPOINT}"
echo "Experiment: ${EXP_NAME}"
echo "Epochs: ${N_EPOCHS}"
echo "Examples: ${N_EXAMPLES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "DPO Beta: ${BETA}"
echo "Save at epochs: ${SAVE_EPOCHS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Check if SFT checkpoint exists
if [ ! -f "exp_results/${SFT_CHECKPOINT}/policy.pt" ]; then
    echo "ERROR: SFT checkpoint not found at exp_results/${SFT_CHECKPOINT}/policy.pt"
    echo "Please run train_sft_base.sh first."
    exit 1
fi

# Run training
python -u train.py \
    loss=dpo \
    loss.beta=${BETA} \
    model=${MODEL} \
    model.archive=${SFT_CHECKPOINT} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    train_split=train_dpo \
    n_epochs=${N_EPOCHS} \
    n_examples=${N_EXAMPLES} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    eval_every=${EVAL_EVERY} \
    lr=${LR} \
    save_ckp=true \
    save_epochs="${SAVE_EPOCHS}" \
    fine_evaluation=true \
    wandb.enabled=true \
    wandb.project="learntune_replication"

echo ""
echo "=============================================="
echo "DPO Base Training Complete!"
echo "=============================================="
echo "Checkpoints saved to:"
echo "  - exp_results/${EXP_NAME}/epoch_2/"
echo "  - exp_results/${EXP_NAME}/epoch_4/"
echo "  - exp_results/${EXP_NAME}/epoch_6/"
echo "  - exp_results/${EXP_NAME}/ (final)"
echo ""
