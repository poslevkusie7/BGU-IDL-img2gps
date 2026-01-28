#!/usr/bin/env bash

set -e  # Exit immediately if a command fails

# -------------------- CONFIG --------------------
ENV_NAME="img2gps"
PYTHON_VERSION="3.12"     # Use the Python version you want
CUDA_VERSION="12.1"       # Adjust to your GPU / driver (12.1, 12.4, 12.8)
# ------------------------------------------------

echo "============================================================="
echo " Setting up environment: $ENV_NAME"
echo " Python $PYTHON_VERSION  •  PyTorch + CUDA $CUDA_VERSION"
echo "============================================================="

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Install Anaconda or Miniconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Removing existing environment: $ENV_NAME"
    conda remove -n "$ENV_NAME" --all -y
fi

# Create new environment
echo "Creating environment $ENV_NAME ..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate environment (works in non-interactive scripts)
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install PyTorch + CUDA
echo "Installing PyTorch + torchvision + torchaudio ..."
conda install pytorch torchvision torchaudio pytorch-cuda="$CUDA_VERSION" \
    -c pytorch -c nvidia -y

# Install remaining Python packages
echo "Installing additional packages ..."
pip install --upgrade pip
pip install timm pandas pillow utm folium

# Optional: install UTM if needed
# pip install utm

echo ""
echo "============================================================="
echo "       Environment $ENV_NAME setup complete!"
echo " To activate: conda activate $ENV_NAME"
echo "============================================================="
