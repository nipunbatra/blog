---
aliases:
- /setup/2025/01/15/setup-linux-remote
author: Nipun Batra
badges: true
categories:
- setup
- linux
- remote
- development-environment
- gpu
- cuda
date: '2025-01-15'
description: Setting up Python Environment on Linux Remote Servers with GPU Support
layout: post
title: Linux Remote Server Setup with uv and CUDA
toc: true
---

# Setting up Python Environment on Linux Remote Servers

This guide covers setting up a Python development environment on Linux remote servers (e.g., Ubuntu-based GPU servers) using `uv`, a fast Python package installer and resolver.

## Prerequisites

- SSH access to a Linux remote server
- Ubuntu 20.04+ or similar Debian-based distribution
- (Optional) NVIDIA GPU with CUDA support for deep learning workloads

## Initial Server Setup

### Update system packages

```bash
sudo apt update
sudo apt upgrade -y
```

### Install essential build tools

```bash
sudo apt install -y build-essential curl wget git vim
```

### Install NVIDIA drivers and CUDA (if using GPU)

Check if NVIDIA driver is installed:

```bash
nvidia-smi
```

If not installed, install NVIDIA drivers and CUDA toolkit:

```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
# Should show GPU information and CUDA version
```

## Python Environment Setup with uv

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Add to shell configuration:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Check installation:

```bash
uv --version
```

### 2. Install Python 3.12

PyTorch GPU wheels are available only up to Python 3.12.

```bash
uv python install 3.12
```

### 3. Create a global environment

```bash
uv venv ~/.uv/nb-base --python 3.12
source ~/.uv/nb-base/bin/activate
```

Optional auto-activation:

```bash
echo 'source ~/.uv/nb-base/bin/activate' >> ~/.bashrc
```

### 4. Install core packages

```bash
uv pip install -U pip setuptools wheel
uv pip install -U jupyterlab ipykernel numpy pandas scipy matplotlib seaborn scikit-learn tqdm requests
```

### 5. Install PyTorch + CUDA 12.4

For GPU-enabled servers with CUDA support:

```bash
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only servers:

```bash
uv pip install torch torchvision torchaudio
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output (GPU server):
```
PyTorch version: 2.5.0
CUDA available: True
CUDA version: 12.4
```

Test GPU:

```bash
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 6. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name nb-base --display-name "Python (nb-base)"
```

### 7. Verify installation

```bash
which python
python --version
jupyter kernelspec list
```

Expected output:

```
/home/<user>/.uv/nb-base/bin/python
Python 3.12.x
nb-base   /home/<user>/.local/share/jupyter/kernels/nb-base
```

## Remote Access with JupyterLab

### Start JupyterLab on remote server

```bash
jupyter lab --no-browser --port=8888
```

Note the token from the output.

### SSH tunnel from local machine

On your local machine:

```bash
ssh -N -L 8888:localhost:8888 user@remote-server
```

Then open in browser: `http://localhost:8888`

### Alternative: Use VSCode Remote SSH

Install "Remote - SSH" extension in VSCode, then:

1. Press `Cmd/Ctrl + Shift + P`
2. Select "Remote-SSH: Connect to Host"
3. Enter: `user@remote-server`
4. Open any `.ipynb` file
5. Select kernel: Python (nb-base)

Optional workspace config `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "/home/<user>/.uv/nb-base/bin/python",
  "jupyter.jupyterServerType": "local"
}
```

## Project-Specific Environments

For individual projects, create separate environments:

```bash
cd ~/projects/my-project
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Register project-specific kernel:

```bash
python -m ipykernel install --user --name my-project --display-name "Python (my-project)"
```

## Installing Additional Deep Learning Frameworks

### JAX with CUDA

```bash
uv pip install -U "jax[cuda12]"
```

### TensorFlow with CUDA

```bash
uv pip install tensorflow[and-cuda]
```

### Hugging Face Transformers

```bash
uv pip install transformers datasets accelerate
```

### Other common ML packages

```bash
uv pip install lightning tensorboard wandb optuna
```

## Package Management

### List installed packages

```bash
uv pip list
```

### Update a specific package

```bash
uv pip install -U <package-name>
```

### Check outdated packages

```bash
uv pip list --outdated
```

### Export environment

```bash
uv pip freeze > requirements.txt
```

### Install from requirements

```bash
uv pip install -r requirements.txt
```

## Troubleshooting

### CUDA out of memory

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

Clear PyTorch cache:

```python
import torch
torch.cuda.empty_cache()
```

### Permission denied errors

Check file permissions and ownership:

```bash
ls -la ~/.uv/
```

### Kernel not found in Jupyter

Reinstall kernel:

```bash
python -m ipykernel install --user --name nb-base --display-name "Python (nb-base)" --force
```

## Useful Aliases

Add to `~/.bashrc`:

```bash
alias activate='source ~/.uv/nb-base/bin/activate'
alias jlab='jupyter lab --no-browser --port=8888'
alias gpustat='watch -n 1 nvidia-smi'
```

Apply changes:

```bash
source ~/.bashrc
```

## Keeping System Updated

Regularly update packages:

```bash
# System packages
sudo apt update && sudo apt upgrade -y

# Python packages
source ~/.uv/nb-base/bin/activate
uv pip list --outdated
uv pip install -U <package-names>
```

## LaTeX Installation (Optional)

For document generation and scientific writing:

### Install TinyTeX via Quarto

First, install Quarto:

```bash
# Download latest Quarto for Linux
wget https://quarto.org/download/latest/quarto-linux-amd64.deb
sudo dpkg -i quarto-linux-amd64.deb
```

Install TinyTeX:

```bash
quarto install tool tinytex
```

### Install comprehensive LaTeX packages

Install a comprehensive set of packages to avoid missing dependencies before deadlines:

```bash
# Core collections and fonts
tlmgr install collection-fontsrecommended collection-latexrecommended \
  collection-fontsextra collection-latexextra

# Graphics and plotting
tlmgr install pgfplots tikz-cd pgf pgfgantt tikzscale tikz-3dplot

# Fonts
tlmgr install psnfss type1cm cm-super sourcesanspro sourcecodespro \
  lato roboto fira libertine

# Beamer themes and presentation
tlmgr install beamertheme-metropolis beamer-verona pgfopts \
  appendixnumberbeamer

# Bibliography and references
tlmgr install biblatex biber logreq natbib

# Math and symbols
tlmgr install amsmath amscls amsfonts mathtools unicode-math \
  stmaryrd bbm-macros dsfont

# Tables and formatting
tlmgr install booktabs multirow longtable array tabularx \
  threeparttable siunitx

# Figures and floats
tlmgr install adjustbox collectbox subcaption wrapfig float \
  caption placeins

# Code listings
tlmgr install listings minted fancyvrb

# Miscellaneous utilities
tlmgr install underscore ucs xstring etoolbox xifthen \
  enumitem parskip geometry fancyhdr hyperref cleveref \
  todonotes comment csquotes microtype

# PDF and graphics
tlmgr install pdfpages dvipng epstopdf graphicx xcolor

# Algorithm packages
tlmgr install algorithm2e algorithms algorithmicx

# Symbols and icons
tlmgr install fontawesome5 academicons

# Additional useful packages
tlmgr install appendix blindtext lipsum standalone
```

Add TinyTeX to PATH:

```bash
echo 'export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Searching for missing packages

If you need a specific `.sty` file:

```bash
tlmgr search --global --file "/packagename.sty"
```

## Additional Resources

- [uv documentation](https://github.com/astral-sh/uv)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [CUDA compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [JupyterLab documentation](https://jupyterlab.readthedocs.io/)
- [Quarto documentation](https://quarto.org/docs/guide/)
- [TinyTeX documentation](https://yihui.org/tinytex/)
