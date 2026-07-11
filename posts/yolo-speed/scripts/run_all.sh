#!/usr/bin/env bash
# Run every benchmark in sequence on Bhaskar GPU 0. Idempotent: exports are
# skipped if the artifact already exists.
set -euo pipefail
cd ~/yolo-speed
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
# Make torch's pip-installed CUDA libs (cuDNN 9 etc.) visible to onnxruntime
NVLIBS=$(python - <<'EOF'
import glob, os, nvidia
# nvidia is a namespace package: no __file__, use __path__
print(":".join(sorted(glob.glob(os.path.join(nvidia.__path__[0], "*", "lib")))))
EOF
)
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p logs

python scripts/export_all.py     2>&1 | tee logs/export.log
python scripts/bench_latency.py  2>&1 | tee logs/latency.log
python scripts/bench_batch.py    2>&1 | tee logs/batch.log
python scripts/bench_accuracy.py 2>&1 | tee logs/accuracy.log
python scripts/bench_cpu.py      2>&1 | tee logs/cpu.log
echo ALL_DONE
