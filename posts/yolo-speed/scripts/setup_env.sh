#!/usr/bin/env bash
# One-time env setup on Bhaskar: uv venv with torch cu126 + export/runtime deps.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/yolo-speed

[ -d .venv ] || uv venv --python 3.11 .venv
source .venv/bin/activate
# tensorrt-cu12, NOT the bare `tensorrt` metapackage: that now resolves to
# TensorRT 11 / CUDA 13 wheels, which the CUDA 12.4 driver on Bhaskar can't load
uv pip install ultralytics onnx onnxslim "onnxruntime-gpu<1.23" tensorrt-cu12 openvino pycocotools
# pip: uv venvs ship without it, and ultralytics' auto-installer shells out to
# `python -m pip`. modelopt: TensorRT 11 removed the implicit-quantization API,
# so ultralytics builds FP16/INT8 engines through nvidia-modelopt.
uv pip install pip "nvidia-modelopt[onnx]>=0.44"
# torch LAST and pinned to the cu126 index: modelopt requires torch>=2.8 and
# will otherwise drag in a +cu130 torch that the CUDA 12.4 driver can't load.
# 2.8.0 <-> torchvision 0.23.0 is the matching pair.
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

python - <<'EOF'
import torch, onnxruntime, tensorrt, openvino, ultralytics
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("tensorrt", tensorrt.__version__)
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
print("openvino", openvino.__version__)
print("ultralytics", ultralytics.__version__)
EOF
echo SETUP_DONE
