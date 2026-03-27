#!/bin/bash
# remote_setup.sh - Sets up the ML environment on OSCAR scratch

PROJECT_DIR="$HOME/scratch/arxiv-recommender"
VENV_DIR="$PROJECT_DIR/venv"
MODEL_DIR="$PROJECT_DIR/models"

mkdir -p "$PROJECT_DIR"
mkdir -p "$MODEL_DIR"

# Ensure module command is available
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

module purge
module load python/3.11.11-5e66
module load cuda/12.9.0
module load cudnn/9.8.0.87-12-y7fu

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    # Install PyTorch with CUDA 12.1 support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers adapters numpy
else
    source "$VENV_DIR/bin/activate"
fi

echo "Verifying GPU availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU found')"

echo "Environment ready."
