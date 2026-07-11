"""Experiment 2: pure GPU forward-pass throughput vs batch size for yolo11s.

We bypass the predictor (no image decode, no NMS) and time AutoBackend
forward passes on a resident CUDA tensor, with torch.cuda.synchronize()
around the timed region. This isolates what the runtime (PyTorch / ONNX
Runtime / TensorRT) actually accelerates.

Also includes a static batch=1 FP16 engine as a single point, to measure the
cost of building the TensorRT engine with a dynamic batch profile instead of
a fixed one.
"""
import time

import torch
from ultralytics.nn.autobackend import AutoBackend

from common import MODELS, IMGSZ, env_info, save_json

BATCHES = [1, 2, 4, 8, 16, 32]
DEVICE = torch.device("cuda:0")

CONFIGS = [  # (name, weights, fp16, batches)
    ("pt-fp32", "yolo11s.pt", False, BATCHES),
    ("pt-fp16", "yolo11s.pt", True, BATCHES),
    ("onnx-gpu", "yolo11s_dyn.onnx", False, BATCHES),
    ("trt-fp16-dyn", "yolo11s_fp16_dyn.engine", True, BATCHES),
    ("trt-fp16-static-b1", "yolo11s_fp16_b1.engine", True, [1]),
]

runs = []
for name, fname, fp16, batches in CONFIGS:
    w = MODELS / fname
    if not w.exists():
        print(f"MISSING {w}, skipping")
        continue
    ab = AutoBackend(str(w), device=DEVICE, fp16=fp16)
    dtype = torch.float16 if ab.fp16 else torch.float32
    for n in batches:
        im = torch.rand(n, 3, IMGSZ, IMGSZ, device=DEVICE, dtype=dtype)
        for _ in range(15):
            ab(im)
        torch.cuda.synchronize()
        iters = max(30, 600 // n)
        t0 = time.perf_counter()
        for _ in range(iters):
            ab(im)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ips = n * iters / dt
        runs.append({"backend": name, "batch": n, "iters": iters,
                     "ms_per_batch": dt / iters * 1e3,
                     "images_per_s": ips})
        print(f"{name:20s} batch {n:3d}: {ips:8.1f} img/s "
              f"({dt / iters * 1e3:6.2f} ms/batch)")
    del ab
    torch.cuda.empty_cache()

save_json("batch.json", {"env": env_info(), "runs": runs})
print("BATCH_DONE")
