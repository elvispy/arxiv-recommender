#!/bin/bash
#SBATCH -J dailyarxiv-infer
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/infer_%j.out
#SBATCH -e logs/infer_%j.err

# Ensure module command is available
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# 1. Load Environment (via remote_setup.sh)
PROJECT_DIR="$HOME/scratch/arxiv-recommender"
VENV_DIR="$PROJECT_DIR/venv"
MODEL_DIR="$PROJECT_DIR/models"

# Ensure remote_setup.sh is run to set up environment and activate venv
# This also handles module purge and loads
bash "$PROJECT_DIR/remote_setup.sh"

# Re-activate venv in the current shell
source "$VENV_DIR/bin/activate"

# 2. Run Inference
# $1: input.json, $2: output.json
mkdir -p "$PROJECT_DIR/logs"
"$VENV_DIR/bin/python3" "$PROJECT_DIR/oscar_infer.py" "$1" "$2" --model_cache "$MODEL_DIR"
