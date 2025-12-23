# Learning Dynamics of LLM Finetuning - Experiment Scripts

This directory contains scripts for replicating the experiments from the paper "Learning Dynamics of LLM Finetuning".

## Quick Start

```bash
# 1. Set up the environment
bash scripts/setup_environment.sh

# 2. Activate the environment
conda activate learntune

# 3. Run all experiments
bash scripts/run_all_experiments.sh
```

## System Requirements

- **GPU**: NVIDIA RTX PRO 6000 (97GB VRAM) or equivalent
- **CUDA**: 12.x or 13.x
- **RAM**: 32GB minimum
- **Storage**: 50GB for models and checkpoints

## Scripts Overview

### Environment Setup
- `setup_environment.sh` - Creates conda environment with all dependencies

### Training Scripts
- `train_sft_base.sh` - SFT training with train_dpo split (2 epochs, 10000 examples)
- `train_sft_extend.sh` - SFT training with train_sft_extend split (2 epochs, 10000 examples)
- `train_dpo_base.sh` - DPO training on SFT base (6 epochs, saves at 2,4,6)
- `train_dpo_extend.sh` - DPO training on SFT extend (6 epochs, saves at 2,4,6)

### Inference & Evaluation
- `generate_inference_samples.sh` - Generate model responses for evaluation
- `evaluate_models.py` - Compare two models using GPT/Claude as judge

### Master Script
- `run_all_experiments.sh` - Runs the complete experiment pipeline

## Detailed Instructions

### Step 1: Environment Setup

```bash
cd /home/llm/arp/learntune/Learning_dynamics_LLM
bash scripts/setup_environment.sh
conda activate learntune
```

### Step 2: Verify GPU Setup

```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Step 3: Run SFT Training

```bash
# Run base SFT (uses train_dpo split)
bash scripts/train_sft_base.sh

# Run extend SFT (uses train_sft_extend split)
bash scripts/train_sft_extend.sh
```

### Step 4: Run DPO Training

```bash
# Run DPO on base SFT (saves checkpoints at epochs 2, 4, 6)
bash scripts/train_dpo_base.sh

# Run DPO on extend SFT (saves checkpoints at epochs 2, 4, 6)
bash scripts/train_dpo_extend.sh
```

### Step 5: Generate Inference Samples

```bash
# Generate samples from a specific checkpoint
bash scripts/generate_inference_samples.sh qwen18 qwen18_dpo_base_ep6

# Generate samples from a specific epoch
bash scripts/generate_inference_samples.sh qwen18 qwen18_dpo_base_ep6 epoch_4
```

### Step 6: Run Evaluation

```bash
# Set your API key (choose one)
export ANTHROPIC_API_KEY="your-key-here"
# OR
export OPENAI_API_KEY="your-key-here"

# Run evaluation
cd scripts
python evaluate_models.py \
    --model_a gen_qwen18_dpo_base_ep6 \
    --model_b gen_qwen18_dpo_extend_ep6 \
    --evaluator claude
```

## Experiment Configuration

### SFT Training
| Parameter | Value |
|-----------|-------|
| Model | Qwen 1.8B |
| Epochs | 2 |
| Examples | 10,000 |
| Batch Size | 4 |
| Learning Rate | 5e-7 |

### DPO Training
| Parameter | Value |
|-----------|-------|
| Model | Qwen 1.8B |
| Epochs | 6 |
| Examples | 30,000 |
| Batch Size | 2 |
| Learning Rate | 5e-7 |
| Beta | 0.1 |
| Save Epochs | 2, 4, 6 |

## Output Structure

After running all experiments, you'll have:

```
exp_results/
├── qwen18_sft_base_ep2/
│   ├── policy.pt
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── prob_train_metrics.json
├── qwen18_sft_extend_ep2/
│   └── ...
├── qwen18_dpo_base_ep6/
│   ├── epoch_2/
│   │   └── policy.pt
│   ├── epoch_4/
│   │   └── policy.pt
│   ├── epoch_6/
│   │   └── policy.pt
│   └── policy.pt (final)
├── qwen18_dpo_extend_ep6/
│   └── ...
├── gen_qwen18_dpo_base_ep6/
│   ├── prob_test_gen_response.jsonl
│   └── ...
└── gen_qwen18_dpo_extend_ep6/
    └── ...
```

## GPU Usage

The scripts are configured to use specific GPUs:
- GPU 0: SFT Base, DPO Base
- GPU 1: SFT Extend, DPO Extend

To run training in parallel on different GPUs:

```bash
# Terminal 1
bash scripts/train_sft_base.sh &

# Terminal 2
bash scripts/train_sft_extend.sh &
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in the training scripts
- For DPO, batch_size=2 is recommended

### Missing Checkpoint Error
- Ensure SFT training completed before running DPO
- Check `exp_results/<exp_name>/policy.pt` exists

### API Rate Limiting (Evaluation)
- The evaluation script includes 0.5s delay between requests
- Increase delay if you hit rate limits

## Weights & Biases

Training logs are automatically sent to W&B. To disable:
```bash
# Add to training command
wandb.enabled=false
```

To change project name:
```bash
wandb.project="your_project_name"
```
