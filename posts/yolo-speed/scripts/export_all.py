"""Export every YOLO11 artifact the benchmarks need, once, with unambiguous
names (ultralytics reuses `model.onnx` / `model.engine` for every precision,
so we rename each artifact right after export).

Artifacts per model size (n/s/m), all imgsz=640:
  - {m}.pt                    (downloaded)
  - {m}_b1.onnx               static batch=1 ONNX
  - {m}_fp32_b1.engine        TensorRT FP32, static batch=1
  - {m}_fp16_b1.engine        TensorRT FP16, static batch=1
  - {m}_int8_b1.engine        TensorRT INT8, static batch=1, COCO128 calibration
Extra for yolo11s (batch experiments):
  - s_dyn.onnx                dynamic-shape ONNX
  - s_fp16_dyn.engine         TensorRT FP16, dynamic batch 1..32
Extra for yolo11n (CPU experiments):
  - yolo11n_openvino_model/   OpenVINO FP32
Build wall-time and artifact size go to results/exports.json.
"""
import os
import time
from pathlib import Path

from ultralytics import YOLO

from common import MODELS, IMGSZ, env_info, save_json

os.chdir(MODELS)

SIZES = ["n", "s", "m"]
records = []


def do_export(pt, target, **kwargs):
    """Export `pt` with kwargs, rename the artifact to `target`, log timing."""
    target = Path(target)
    if target.exists():
        print(f"skip (exists): {target}")
        return
    model = YOLO(pt)
    t0 = time.perf_counter()
    out = Path(model.export(imgsz=IMGSZ, device=0, **kwargs))
    dt = time.perf_counter() - t0
    if out != target:
        out.rename(target)
    size_mb = target.stat().st_size / 1e6 if target.is_file() else sum(
        f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
    records.append({"artifact": target.name, "format": kwargs.get("format"),
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                    "export_s": round(dt, 1), "size_mb": round(size_mb, 1)})
    print(f"exported {target.name} in {dt:.1f}s ({size_mb:.1f} MB)")


for s in SIZES:
    pt = f"yolo11{s}.pt"
    YOLO(pt)  # download if needed
    records.append({"artifact": pt, "format": "pytorch", "export_s": 0,
                    "size_mb": round(Path(pt).stat().st_size / 1e6, 1)})
    do_export(pt, f"yolo11{s}_b1.onnx", format="onnx", batch=1, simplify=True)
    do_export(pt, f"yolo11{s}_fp32_b1.engine", format="engine", half=False, batch=1)
    do_export(pt, f"yolo11{s}_fp16_b1.engine", format="engine", half=True, batch=1)
    do_export(pt, f"yolo11{s}_int8_b1.engine", format="engine", int8=True,
              batch=1, data="coco128.yaml")

do_export("yolo11s.pt", "yolo11s_dyn.onnx", format="onnx", dynamic=True, simplify=True)
do_export("yolo11s.pt", "yolo11s_fp16_dyn.engine", format="engine", half=True,
          dynamic=True, batch=32)
do_export("yolo11n.pt", "yolo11n_openvino_model", format="openvino", batch=1)

save_json("exports.json", {"env": env_info(), "exports": records})
print("EXPORTS_DONE")
