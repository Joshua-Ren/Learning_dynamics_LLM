#!/bin/bash
# ============================================================================
# Master Experiment Runner Script
# ============================================================================
# This script runs all experiments in sequence:
# 1. SFT Base (2 epochs, 10000 examples)
# 2. SFT Extend (2 epochs, 10000 examples)
# 3. DPO Base (6 epochs with checkpoints at 2, 4, 6)
# 4. DPO Extend (6 epochs with checkpoints at 2, 4, 6)
# 5. Generate inference samples for all models
# 6. (Optional) Run evaluation
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Learning Dynamics LLM - Full Experiment Run"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p "${SCRIPT_DIR}/../logs"

# Step 1: Train SFT Base
echo "=============================================="
echo "Step 1/5: Training SFT Base..."
echo "=============================================="
bash "${SCRIPT_DIR}/train_sft_base.sh" 2>&1 | tee "${SCRIPT_DIR}/../logs/sft_base.log"
echo "SFT Base completed at $(date)"
echo ""

# Step 2: Train SFT Extend
echo "=============================================="
echo "Step 2/5: Training SFT Extend..."
echo "=============================================="
bash "${SCRIPT_DIR}/train_sft_extend.sh" 2>&1 | tee "${SCRIPT_DIR}/../logs/sft_extend.log"
echo "SFT Extend completed at $(date)"
echo ""

# Step 3: Train DPO Base
echo "=============================================="
echo "Step 3/5: Training DPO Base..."
echo "=============================================="
bash "${SCRIPT_DIR}/train_dpo_base.sh" 2>&1 | tee "${SCRIPT_DIR}/../logs/dpo_base.log"
echo "DPO Base completed at $(date)"
echo ""

# Step 4: Train DPO Extend
echo "=============================================="
echo "Step 4/5: Training DPO Extend..."
echo "=============================================="
bash "${SCRIPT_DIR}/train_dpo_extend.sh" 2>&1 | tee "${SCRIPT_DIR}/../logs/dpo_extend.log"
echo "DPO Extend completed at $(date)"
echo ""

# Step 5: Generate inference samples
echo "=============================================="
echo "Step 5/5: Generating Inference Samples..."
echo "=============================================="

# Generate for SFT models
bash "${SCRIPT_DIR}/generate_inference_samples.sh" qwen18 qwen18_sft_base_ep2 2>&1 | tee -a "${SCRIPT_DIR}/../logs/inference_gen.log"
bash "${SCRIPT_DIR}/generate_inference_samples.sh" qwen18 qwen18_sft_extend_ep2 2>&1 | tee -a "${SCRIPT_DIR}/../logs/inference_gen.log"

# Generate for DPO models at different epochs
for EPOCH in epoch_2 epoch_4 epoch_6; do
    bash "${SCRIPT_DIR}/generate_inference_samples.sh" qwen18 qwen18_dpo_base_ep6 ${EPOCH} 2>&1 | tee -a "${SCRIPT_DIR}/../logs/inference_gen.log"
    bash "${SCRIPT_DIR}/generate_inference_samples.sh" qwen18 qwen18_dpo_extend_ep6 ${EPOCH} 2>&1 | tee -a "${SCRIPT_DIR}/../logs/inference_gen.log"
done

# Generate for final DPO models
bash "${SCRIPT_DIR}/generate_inference_samples.sh" qwen18 qwen18_dpo_base_ep6 2>&1 | tee -a "${SCRIPT_DIR}/../logs/inference_gen.log"
bash "${SCRIPT_DIR}/generate_inference_samples.sh" qwen18 qwen18_dpo_extend_ep6 2>&1 | tee -a "${SCRIPT_DIR}/../logs/inference_gen.log"

echo ""
echo "=============================================="
echo "All Experiments Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Summary of generated checkpoints:"
echo "  - exp_results/qwen18_sft_base_ep2/"
echo "  - exp_results/qwen18_sft_extend_ep2/"
echo "  - exp_results/qwen18_dpo_base_ep6/epoch_2/"
echo "  - exp_results/qwen18_dpo_base_ep6/epoch_4/"
echo "  - exp_results/qwen18_dpo_base_ep6/epoch_6/"
echo "  - exp_results/qwen18_dpo_base_ep6/"
echo "  - exp_results/qwen18_dpo_extend_ep6/epoch_2/"
echo "  - exp_results/qwen18_dpo_extend_ep6/epoch_4/"
echo "  - exp_results/qwen18_dpo_extend_ep6/epoch_6/"
echo "  - exp_results/qwen18_dpo_extend_ep6/"
echo ""
echo "To run evaluation, use:"
echo "  python ${SCRIPT_DIR}/evaluate_models.py --model_a qwen18_dpo_base_ep6 --model_b qwen18_dpo_extend_ep6"
echo ""
