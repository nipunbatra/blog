"""Experiment 3: does the fast path cost accuracy? mAP50-95 on COCO128 for
every backend and model size.

COCO128 is a slice of the COCO *train* set, so absolute mAP is optimistic —
we only care about the *relative* drop across backends of the same model
(FP32 -> FP16 -> INT8), which is what quantisation affects.
"""
from ultralytics import YOLO

from common import MODELS, IMGSZ, env_info, save_json

BACKENDS = [  # (name, weights-pattern, val-kwargs)
    ("pt-fp32", "yolo11{s}.pt", {"half": False, "batch": 16}),
    ("pt-fp16", "yolo11{s}.pt", {"half": True, "batch": 16}),
    ("onnx-gpu", "yolo11{s}_b1.onnx", {"batch": 1}),
    ("trt-fp32", "yolo11{s}_fp32_b1.engine", {"batch": 1}),
    ("trt-fp16", "yolo11{s}_fp16_b1.engine", {"batch": 1}),
    ("trt-int8", "yolo11{s}_int8_b1.engine", {"batch": 1}),
]

runs = []
for size in ["n", "s", "m"]:
    for name, pattern, kw in BACKENDS:
        w = MODELS / pattern.format(s=size)
        if not w.exists():
            print(f"MISSING {w}, skipping")
            continue
        model = YOLO(str(w), task="detect")
        m = model.val(data="coco128.yaml", imgsz=IMGSZ, device=0,
                      verbose=False, plots=False, **kw)
        runs.append({
            "model": f"yolo11{size}", "backend": name,
            "map50_95": round(float(m.box.map), 4),
            "map50": round(float(m.box.map50), 4),
            "val_inference_ms": round(float(m.speed["inference"]), 2),
        })
        print(f"yolo11{size} {name:9s} mAP50-95 {m.box.map:.4f} "
              f"mAP50 {m.box.map50:.4f}")
        del model

save_json("accuracy.json", {"env": env_info(), "data": "coco128",
                            "runs": runs})
print("ACCURACY_DONE")
