#!/bin/bash
# Python/PyTorch Development Environment Setup Script
# Using uv for fast package management - GLOBAL environment
# Author: Nipun Batra
# Last updated: 2025-12-26

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
    curl -LsSf https://astral.sh/uv/install.sh | env SHELL=/bin/bash sh
    echo -e "${GREEN}✓ uv installed${NC}"
else
    echo -e "${GREEN}✓ uv already installed $(uv --version)${NC}"
fi

# ------------------------------
# 2. Create GLOBAL virtual environment
# ------------------------------
VENV_NAME="${1:-nb-base}"  # Default to nb-base
VENV_PATH="$HOME/.uv/$VENV_NAME"

echo -e "${BLUE}Creating global virtual environment: ${VENV_PATH}...${NC}"

if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}✓ Environment already exists at ${VENV_PATH}${NC}"
else
    uv venv "$VENV_PATH" --python 3.12
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate for the rest of this script
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Activated${NC}"

# ------------------------------
# 3. Core Scientific Python
# ------------------------------
echo -e "${BLUE}Installing core scientific packages...${NC}"
uv pip install numpy scipy pandas matplotlib seaborn

# ------------------------------
# 4. PyTorch with CUDA (2.6+ required for transformers security)
# ------------------------------
echo -e "${BLUE}Installing PyTorch with CUDA support...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA version..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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

# Register kernel globally
python -m ipykernel install --user --name="$VENV_NAME" --display-name="Python ($VENV_NAME)"
echo -e "${GREEN}✓ Jupyter installed and kernel registered${NC}"

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
    quarto-cli 2>/dev/null || true

echo -e "${GREEN}✓ Blog dependencies installed${NC}"

# ------------------------------
# Summary
# ------------------------------
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Global environment location: $VENV_PATH"
echo ""
echo "To activate the environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Or add this alias to your ~/.bashrc:"
echo "  alias nb='source $VENV_PATH/bin/activate'"
echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Jupyter kernel '$VENV_NAME' is registered globally."
echo "Select it in VSCode/JupyterLab to use this environment."
