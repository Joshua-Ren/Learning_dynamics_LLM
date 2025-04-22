#!/bin/bash
#SBATCH --account=rrg-dsuth
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-20:00            # time (DD-HH:MM)
#SBATCH --output=./logs/stage1.txt 
#SBATCH --job-name=eval

# 1. Load the required modules
module load python/3.10 StdEnv/2023 cudacore/.12.2.2 arrow/14.0.0

# 2. Load your environment
source /home/joshua52/projects/def-dsuth/joshua52/env_llm/bin/activate

# 3. Go to the correct path
cd /home/joshua52/projects/def-dsuth/joshua52/finetuning_dynamics

#python -u gen_inference_samples.py model=qwen18 model.archive=baseline_sft_qwen18 exp_name=eval_baseline_sft_qwen18
#python -u gen_inference_samples.py model=qwen18 model.archive=extend_sft_qwen18 exp_name=eval_extend_sft_qwen18
# python -u gen_inference_samples.py model=qwen18 model.archive=baseline_dpo_qwen18_ep2 exp_name=eval_baseline_dpo_qwen18_ep2
# python -u gen_inference_samples.py model=qwen18 model.archive=extend_dpo_qwen18_ep2 exp_name=eval_extend_dpo_qwen18_ep2
# python -u gen_inference_samples.py model=qwen18 model.archive=baseline_dpo_qwen18_ep6 exp_name=eval_baseline_dpo_qwen18_ep6
# python -u gen_inference_samples.py model=qwen18 model.archive=extend_dpo_qwen18_ep6 exp_name=eval_extend_dpo_qwen18_ep6
