#!/bin/bash
# ============================================================================
# Environment Setup Script for Learning Dynamics of LLM Finetuning
# ============================================================================
# Target System: NVIDIA RTX PRO 6000 Blackwell (sm_120) with CUDA 13.0
# This script creates a conda environment optimized for Blackwell GPUs.
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Learning Dynamics LLM - Environment Setup"
echo "=============================================="
echo "Detected: NVIDIA RTX PRO 6000 Blackwell (sm_120)"
echo "=============================================="

# Configuration
ENV_NAME="learntune"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. GPU support may not work."
fi

# Remove existing environment if it exists
echo ""
echo "Checking for existing environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create conda environment
echo ""
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# CRITICAL: Install NumPy < 2.0 first to avoid compatibility issues
echo ""
echo "Installing NumPy < 2.0 (required for PyTorch compatibility)..."
pip install "numpy<2.0"

# Install PyTorch with CUDA 12.8 support for Blackwell (sm_120)
# Blackwell GPUs require PyTorch nightly with CUDA 12.8+
echo ""
echo "Installing PyTorch nightly with CUDA 12.8 support (for Blackwell sm_120)..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch CUDA support for Blackwell
echo ""
echo "Verifying PyTorch Blackwell support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Compute capability: {props.major}.{props.minor}')
        print(f'  Memory: {props.total_memory / 1e9:.1f} GB')
"

# Install core dependencies (latest versions for PyTorch nightly compatibility)
echo ""
echo "Installing core dependencies..."
pip install transformers
pip install datasets
pip install tokenizers
pip install accelerate

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install tqdm
pip install wandb
pip install hydra-core==1.3.2
pip install omegaconf
pip install beautifulsoup4
pip install fsspec
pip install ipykernel
pip install jupyter
pip install matplotlib
pip install seaborn
pip install pandas

# Install tensor-parallel (optional, for multi-GPU tensor parallelism)
echo ""
echo "Installing tensor-parallel..."
pip install tensor-parallel

# Install flash-attention for faster training (optional but recommended for Blackwell)
echo ""
echo "Installing flash-attention (this may take a while)..."
pip install ninja packaging
pip install flash-attn --no-build-isolation || echo "Warning: flash-attn installation failed. Continuing without it."

# Verify installation
echo ""
echo "=============================================="
echo "Verifying Installation..."
echo "=============================================="

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}')
        print(f'    Compute capability: {props.major}.{props.minor}')
        print(f'    Memory: {props.total_memory / 1e9:.1f} GB')
    # Test CUDA tensor allocation
    try:
        x = torch.zeros(100, 100).cuda()
        print('  CUDA tensor test: PASSED')
    except Exception as e:
        print(f'  CUDA tensor test: FAILED - {e}')
"

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
python -c "import hydra; print(f'Hydra version: {hydra.__version__}')"
python -c "import wandb; print(f'Wandb version: {wandb.__version__}')"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "IMPORTANT: To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test GPU memory allocation, run:"
echo "  python -c \"import torch; x = torch.zeros(1000, 1000, 1000).cuda(); print('GPU memory test passed!')\""
echo ""
