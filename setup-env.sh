#!/bin/bash
# Python/PyTorch Development Environment Setup Script
# Using uv for fast package management
# Author: Nipun Batra
# Last updated: 2025-12-24

set -e  # Exit on error

echo "=========================================="
echo "Python Development Environment Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ------------------------------
# 1. Install uv if not present
# ------------------------------
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}Installing uv...${NC}"
    # Use SHELL=bash to avoid fish config issues
    curl -LsSf https://astral.sh/uv/install.sh | env SHELL=/bin/bash sh
    echo -e "${GREEN}✓ uv installed${NC}"
else
    echo -e "${GREEN}✓ uv already installed $(uv --version)${NC}"
fi

# ------------------------------
# 2. Create virtual environment
# ------------------------------
VENV_NAME="${1:-.venv}"  # Default to .venv, or use first argument

echo -e "${BLUE}Creating virtual environment: ${VENV_NAME}...${NC}"
uv venv "$VENV_NAME" --python 3.11

# Activate for the rest of this script
source "$VENV_NAME/bin/activate"
echo -e "${GREEN}✓ Virtual environment created and activated${NC}"

# ------------------------------
# 3. Core Scientific Python
# ------------------------------
echo -e "${BLUE}Installing core scientific packages...${NC}"
uv pip install numpy scipy pandas matplotlib seaborn

# ------------------------------
# 4. PyTorch with CUDA
# ------------------------------
echo -e "${BLUE}Installing PyTorch with CUDA support...${NC}"
# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA version..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected, installing CPU version..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"

# ------------------------------
# 5. Deep Learning / ML Libraries
# ------------------------------
echo -e "${BLUE}Installing ML/DL libraries...${NC}"
uv pip install \
    transformers \
    datasets \
    accelerate \
    einops \
    timm \
    safetensors \
    huggingface_hub

echo -e "${GREEN}✓ ML libraries installed${NC}"

# ------------------------------
# 6. Jupyter / Notebooks
# ------------------------------
echo -e "${BLUE}Installing Jupyter...${NC}"
uv pip install \
    jupyter \
    jupyterlab \
    ipykernel \
    ipywidgets \
    nbformat

# Register kernel
python -m ipykernel install --user --name="$VENV_NAME" --display-name="Python ($VENV_NAME)"
echo -e "${GREEN}✓ Jupyter installed${NC}"

# ------------------------------
# 7. Computer Vision
# ------------------------------
echo -e "${BLUE}Installing CV libraries...${NC}"
uv pip install \
    opencv-python \
    pillow \
    albumentations \
    imageio \
    imageio-ffmpeg

echo -e "${GREEN}✓ CV libraries installed${NC}"

# ------------------------------
# 8. Utilities
# ------------------------------
echo -e "${BLUE}Installing utilities...${NC}"
uv pip install \
    tqdm \
    rich \
    python-dotenv \
    requests \
    httpx \
    pyyaml

echo -e "${GREEN}✓ Utilities installed${NC}"

# ------------------------------
# 9. Development Tools
# ------------------------------
echo -e "${BLUE}Installing dev tools...${NC}"
uv pip install \
    ruff \
    pytest \
    black \
    isort

echo -e "${GREEN}✓ Dev tools installed${NC}"

# ------------------------------
# 10. Optional: Quarto/Blog dependencies
# ------------------------------
echo -e "${BLUE}Installing blog/notebook dependencies...${NC}"
uv pip install \
    nbdev \
    quarto-cli 2>/dev/null || true  # May not be available via pip

echo -e "${GREEN}✓ Blog dependencies installed${NC}"

# ------------------------------
# Summary
# ------------------------------
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Installed packages:"
uv pip list | head -20
echo "... (truncated, run 'uv pip list' for full list)"
