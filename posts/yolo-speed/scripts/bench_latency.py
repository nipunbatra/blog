"""Experiment 1: single-image (batch=1) end-to-end predict() latency, per
backend and model size, on one RTX A5000.

For each (model, backend) we run 30 warmup + 200 timed calls of
`model.predict(img)` on a pre-loaded bus.jpg (so disk I/O is excluded) and
record ultralytics' own per-stage speeds (preprocess / inference /
postprocess, ms) plus our wall-clock per call. Detection counts are kept as a
cross-backend sanity check.
"""
import time

import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from common import MODELS, IMGSZ, stats, env_info, save_json

WARMUP, ITERS = 30, 200
IMG = cv2.imread(str(ASSETS / "bus.jpg"))

BACKENDS = [  # (name, weights-pattern, predict-kwargs)
    ("pt-fp32", "yolo11{s}.pt", {"half": False}),
    ("pt-fp16", "yolo11{s}.pt", {"half": True}),
    ("onnx-gpu", "yolo11{s}_b1.onnx", {}),
    ("trt-fp32", "yolo11{s}_fp32_b1.engine", {}),
    ("trt-fp16", "yolo11{s}_fp16_b1.engine", {}),
    ("trt-int8", "yolo11{s}_int8_b1.engine", {}),
]

runs = []
for size in ["n", "s", "m"]:
    for name, pattern, kw in BACKENDS:
        w = MODELS / pattern.format(s=size)
        if not w.exists():
            print(f"MISSING {w}, skipping")
            continue
        model = YOLO(str(w), task="detect")
        args = dict(imgsz=IMGSZ, device=0, verbose=False, **kw)
        for _ in range(WARMUP):
            model.predict(IMG, **args)
        pre, inf, post, wall = [], [], [], []
        n_det = None
        for _ in range(ITERS):
            t0 = time.perf_counter()
            r = model.predict(IMG, **args)[0]
            wall.append((time.perf_counter() - t0) * 1e3)
            pre.append(r.speed["preprocess"])
            inf.append(r.speed["inference"])
            post.append(r.speed["postprocess"])
            n_det = len(r.boxes)
        rec = {
            "model": f"yolo11{size}", "backend": name, "n_det": n_det,
            "preprocess": stats(pre), "inference": stats(inf),
            "postprocess": stats(post), "wall": stats(wall),
        }
        runs.append(rec)
        print(f"yolo11{size} {name:9s} wall {rec['wall']['median']:7.2f} ms "
              f"(inf {rec['inference']['median']:6.2f}) {n_det} dets")
        del model

save_json("latency.json", {"env": env_info(), "warmup": WARMUP,
                           "iters": ITERS, "runs": runs})
print("LATENCY_DONE")
