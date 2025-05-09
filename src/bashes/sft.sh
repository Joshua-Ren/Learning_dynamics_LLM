#!/bin/bash
#SBATCH --account=rrg-dsuth
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=2-10:00            # time (DD-HH:MM)
#SBATCH --output=./logs/stage1.txt 
#SBATCH --job-name=sft

# 1. Load the required modules
module load python/3.11 StdEnv/2023 cudacore/.12.2.2 arrow/14.0.0

# 2. Load your environment
source /home/joshua52/projects/def-dsuth/joshua52/env_llm/bin/activate

# 3. Go to the correct path
cd /home/joshua52/projects/def-dsuth/joshua52/Learning_dynamics_LLM/src

#python -u train.py model=pythia28 exp_name=sft_pythia28_ep6 trainer=BasicTrainer n_epochs=6 n_examples=30000
#python -u train.py model=qwen exp_name=sft_qwen05_ep10 trainer=BasicTrainer n_epochs=6 n_examples=30000
#python -u train.py model=pythia14 exp_name=pythia14_supreject_20240419 trainer=BasicTrainer n_epochs=4 train_supervise=rejected
#python -u train.py model=pythia410m exp_name=sft_pythia410m_save_ep4 trainer=BasicTrainer n_epochs=4 n_examples=20000
#python -u train.py model=pythia1b exp_name=sft_pythia1b_save_ep4 trainer=BasicTrainer n_epochs=2 n_examples=10000

# python -u train.py model=qwen18 exp_name=extend_sft_qwen18_observe trainer=BasicTrainer train_split=train_sft_extend n_epochs=4 n_examples=40000 save_ckp=false eval_every=1000
#python -u train.py model=qwen18 exp_name=baseline_sft_qwen18_ep8 trainer=BasicTrainer train_split=train_dpo n_epochs=8 n_examples=40000 save_ckp=true eval_every=1000

#python -u train.py model=pythia14 exp_name=ultrafb_sft_pythia14 datasets=ultrafb trainer=BasicTrainer n_epochs=6 n_examples=30000 save_ckp=false eval_every=500
# python -u train.py model=pythia410m exp_name=ultrafb_sft_pythia410m datasets=ultrafb trainer=BasicTrainer n_epochs=6 n_examples=30000 save_ckp=false eval_every=500
# python -u train.py model=pythia1b exp_name=ultrafb_sft_pythia1b datasets=ultrafb trainer=BasicTrainer n_epochs=6 n_examples=30000 save_ckp=false eval_every=500
#python -u train.py model=pythia28 exp_name=ultrafb_sft_pythia28 datasets=ultrafb trainer=BasicTrainer n_epochs=6 n_examples=30000 save_ckp=false eval_every=500

#python -u train.py model=qwen exp_name=baseline_sft_qwen05_ep8 trainer=BasicTrainer train_split=train_dpo n_epochs=8 n_examples=40000 save_ckp=true eval_every=1000
python -u train.py model=qwen exp_name=qwen05_baseline_sft_checkbug trainer=BasicTrainer train_split=train_dpo n_epochs=8 n_examples=40000 save_ckp=false eval_every=1000 fine_evaluation=true
python -u train.py model=pythia410m exp_name=pythia410_baseline_sft_verify_argmax_bug trainer=BasicTrainer train_split=train_dpo n_epochs=8 n_examples=40000 save_ckp=false eval_every=1000 fine_evaluation=true

