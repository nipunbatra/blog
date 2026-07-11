"""Experiment 4 (bonus): no GPU? yolo11n end-to-end predict() latency on the
host CPU (Xeon Gold 6326, 16C/32T) for PyTorch, ONNX Runtime and OpenVINO.

Same protocol as bench_latency.py but fewer iterations (CPU is slow).
"""
import time

import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from common import MODELS, IMGSZ, stats, env_info, save_json

WARMUP, ITERS = 10, 100
IMG = cv2.imread(str(ASSETS / "bus.jpg"))

BACKENDS = [
    ("pt-cpu", "yolo11n.pt"),
    ("onnx-cpu", "yolo11n_b1.onnx"),
    ("openvino-cpu", "yolo11n_openvino_model"),
]

runs = []
for name, fname in BACKENDS:
    w = MODELS / fname
    if not w.exists():
        print(f"MISSING {w}, skipping")
        continue
    model = YOLO(str(w), task="detect")
    args = dict(imgsz=IMGSZ, device="cpu", verbose=False)
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
    rec = {"model": "yolo11n", "backend": name, "n_det": n_det,
           "preprocess": stats(pre), "inference": stats(inf),
           "postprocess": stats(post), "wall": stats(wall)}
    runs.append(rec)
    print(f"{name:13s} wall {rec['wall']['median']:7.2f} ms "
          f"(inf {rec['inference']['median']:6.2f}) {n_det} dets")
    del model

save_json("cpu.json", {"env": env_info(), "warmup": WARMUP, "iters": ITERS,
                       "runs": runs})
print("CPU_DONE")
