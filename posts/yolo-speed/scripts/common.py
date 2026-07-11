"""Shared helpers for the YOLO inference-speed benchmarks (run on Bhaskar)."""
import json
import platform
from pathlib import Path

import numpy as np

BASE = Path.home() / "yolo-speed"
MODELS = BASE / "work" / "models"
RESULTS = BASE / "results"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

IMGSZ = 640


def stats(ms):
    """Summary stats (ms) for a list of per-iteration times."""
    a = np.asarray(ms, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "n": int(a.size),
    }


def env_info():
    import torch

    info = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    for mod in ("ultralytics", "tensorrt", "onnxruntime", "openvino"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def save_json(name, obj):
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2))
    print(f"wrote {p}")
