"""Figures for the YOLO inference-speed post. Runs locally on the JSONs
pulled back from Bhaskar into posts/yolo-speed/outputs/.

Color system: hue = runtime family (PyTorch blue, ONNX Runtime green,
OpenVINO purple, TensorRT red; palette validated for CVD + contrast),
lightness = precision within a family. Identity is always also carried by
axis tick labels, so color is never the only encoding.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent / "outputs"

INK, MUTED, GRID = "#333333", "#666666", "#d9d9d9"
FAMILY = {"pt": "#4c72b0", "onnx": "#358a5a", "openvino": "#7a5fc0",
          "trt": "#c44e52"}


def tint(hex_color, f):
    """Mix a color with white; f=0 -> color, f=1 -> white."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


BACKEND_ORDER = ["pt-fp32", "pt-fp16", "onnx-gpu", "trt-fp32", "trt-fp16",
                 "trt-int8"]
BACKEND_LABEL = {
    "pt-fp32": "PyTorch FP32", "pt-fp16": "PyTorch FP16",
    "onnx-gpu": "ONNX Runtime", "trt-fp32": "TensorRT FP32",
    "trt-fp16": "TensorRT FP16", "trt-int8": "TensorRT INT8",
    "pt-cpu": "PyTorch", "onnx-cpu": "ONNX Runtime",
    "openvino-cpu": "OpenVINO",
}
BACKEND_COLOR = {
    "pt-fp32": FAMILY["pt"], "pt-fp16": tint(FAMILY["pt"], 0.35),
    "onnx-gpu": FAMILY["onnx"], "trt-fp32": FAMILY["trt"],
    "trt-fp16": tint(FAMILY["trt"], 0.30), "trt-int8": tint(FAMILY["trt"], 0.55),
    "pt-cpu": FAMILY["pt"], "onnx-cpu": FAMILY["onnx"],
    "openvino-cpu": FAMILY["openvino"],
}


def style(ax, xgrid=False, ygrid=False):
    ax.spines[["top", "right"]].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.7)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ax.spines.values():
        s.set_color(GRID)


def load(name):
    return json.loads((OUT / name).read_text())


def fig_latency():
    """Small multiples: batch-1 end-to-end latency per backend, per model."""
    data = load("latency.json")["runs"]
    models = ["yolo11n", "yolo11s", "yolo11m"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=220, sharey=True)
    for ax, m in zip(axes, models):
        runs = {r["backend"]: r for r in data if r["model"] == m}
        order = [b for b in BACKEND_ORDER if b in runs][::-1]
        vals = [runs[b]["wall"]["median"] for b in order]
        base = runs["pt-fp32"]["wall"]["median"]
        y = np.arange(len(order))
        ax.barh(y, vals, height=0.62,
                color=[BACKEND_COLOR[b] for b in order],
                edgecolor="white", linewidth=1.5)
        for yi, v, b in zip(y, vals, order):
            note = f"{v:.1f}" if b == "pt-fp32" else f"{v:.1f} ({base / v:.1f}x)"
            ax.text(v + max(vals) * 0.02, yi, note, va="center",
                    fontsize=7.8, color=INK)
        ax.set_yticks(y, [BACKEND_LABEL[b] for b in order], fontsize=8.5,
                      color=INK)
        ax.set_xlim(0, max(vals) * 1.38)
        ax.set_title(m, fontsize=10, color=INK)
        ax.set_xlabel("end-to-end latency, batch=1 (ms)", fontsize=8.5,
                      color=MUTED)
        style(ax, xgrid=True)
    fig.suptitle("Same model, same GPU - only the runtime changes",
                 fontsize=11, color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "latency_backends.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote latency_backends.png")


def fig_breakdown(model="yolo11s"):
    """Stacked bars: where the milliseconds go (pre / inference / post)."""
    data = load("latency.json")["runs"]
    runs = {r["backend"]: r for r in data if r["model"] == model}
    order = [b for b in BACKEND_ORDER if b in runs][::-1]
    stages = [("preprocess", "#9199a1", "preprocess"),
              ("inference", "#4c72b0", "inference"),
              ("postprocess", "#c44e52", "postprocess (NMS)")]
    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=220)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for key, color, label in stages:
        vals = np.array([runs[b][key]["median"] for b in order])
        ax.barh(y, vals, left=left, height=0.6, color=color, label=label,
                edgecolor="white", linewidth=1.5)
        left += vals
    for yi, tot in zip(y, left):
        ax.text(tot + left.max() * 0.015, yi, f"{tot:.1f}", va="center",
                fontsize=8, color=INK)
    ax.set_yticks(y, [BACKEND_LABEL[b] for b in order], fontsize=8.5,
                  color=INK)
    ax.set_xlabel(f"median time per image (ms), {model}, batch=1",
                  fontsize=8.5, color=MUTED)
    ax.set_xlim(0, left.max() * 1.12)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncols=1)
    style(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(OUT / "latency_breakdown.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote latency_breakdown.png")


def fig_batch():
    """Throughput vs batch size (pure GPU forward, yolo11s)."""
    data = load("batch.json")["runs"]
    series = [("pt-fp32", FAMILY["pt"], "--"),
              ("pt-fp16", FAMILY["pt"], "-"),
              ("onnx-gpu", FAMILY["onnx"], "-"),
              ("trt-fp16-dyn", FAMILY["trt"], "-")]
    labels = {"pt-fp32": "PyTorch FP32", "pt-fp16": "PyTorch FP16",
              "onnx-gpu": "ONNX Runtime", "trt-fp16-dyn": "TensorRT FP16"}
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=220)
    for name, color, ls in series:
        pts = sorted([(r["batch"], r["images_per_s"]) for r in data
                      if r["backend"] == name])
        if not pts:
            continue
        x, ys = zip(*pts)
        ax.plot(x, ys, ls, color=color, marker="o", markersize=4.5,
                linewidth=2, label=labels[name])
        ax.annotate(labels[name], (x[-1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=color, va="center")
    stat = [r for r in data if r["backend"] == "trt-fp16-static-b1"]
    if stat:
        ax.plot(stat[0]["batch"], stat[0]["images_per_s"], "D",
                color=FAMILY["trt"], markersize=6, markerfacecolor="white",
                markeredgewidth=1.6, label="TensorRT FP16 (static b=1)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
    ax.set_xlabel("batch size", fontsize=9, color=MUTED)
    ax.set_ylabel("throughput (images / s)", fontsize=9, color=MUTED)
    ax.set_xlim(0.9, 90)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("yolo11s pure GPU forward pass (no NMS, no preprocessing)",
                 fontsize=10, color=INK)
    style(ax, ygrid=True)
    fig.tight_layout()
    fig.savefig(OUT / "throughput_batch.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote throughput_batch.png")


def fig_accuracy():
    """Accuracy retention: mAP50-95 per backend, per model (dot plot)."""
    acc = load("accuracy.json")["runs"]
    models = ["yolo11n", "yolo11s", "yolo11m"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=220, sharey=True)
    for ax, m in zip(axes, models):
        runs = {r["backend"]: r for r in acc if r["model"] == m}
        order = [b for b in BACKEND_ORDER if b in runs][::-1]
        vals = [runs[b]["map50_95"] for b in order]
        base = runs["pt-fp32"]["map50_95"]
        y = np.arange(len(order))
        ax.axvline(base, color=GRID, linewidth=1)
        ax.scatter(vals, y, s=42, color=[BACKEND_COLOR[b] for b in order],
                   zorder=3)
        for yi, v in zip(y, vals):
            d = v - base
            ax.text(v, yi + 0.33, f"{v:.3f}" + (f" ({d:+.3f})" if abs(d) > 5e-4 else ""),
                    ha="center", fontsize=7.2, color=INK)
        ax.set_yticks(y, [BACKEND_LABEL[b] for b in order], fontsize=8.5,
                      color=INK)
        lo, hi = min(vals), max(vals)
        pad = max(0.01, (hi - lo) * 0.6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, len(order) - 0.2)
        ax.set_title(m, fontsize=10, color=INK)
        ax.set_xlabel("mAP50-95 (COCO128)", fontsize=8.5, color=MUTED)
        style(ax, xgrid=True)
    fig.suptitle("Accuracy is (almost) free: mAP by backend, gray line = FP32 baseline",
                 fontsize=11, color=INK, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_backends.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote accuracy_backends.png")


def fig_cpu():
    """CPU (yolo11n) vs the best GPU number, for scale."""
    cpu = load("cpu.json")["runs"]
    lat = load("latency.json")["runs"]
    gpu_best = min(r["wall"]["median"] for r in lat
                   if r["model"] == "yolo11n")
    order = ["pt-cpu", "onnx-cpu", "openvino-cpu"]
    runs = {r["backend"]: r for r in cpu}
    order = [b for b in order if b in runs][::-1]
    vals = [runs[b]["wall"]["median"] for b in order]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=220)
    y = np.arange(len(order))
    ax.barh(y, vals, height=0.6, color=[BACKEND_COLOR[b] for b in order],
            edgecolor="white", linewidth=1.5)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yi, f"{v:.0f} ms", va="center",
                fontsize=8.5, color=INK)
    ax.axvline(gpu_best, color=FAMILY["trt"], linewidth=1.2, linestyle=":")
    ax.text(gpu_best, len(order) - 0.25,
            f"  best GPU backend: {gpu_best:.1f} ms", fontsize=8,
            color=FAMILY["trt"], va="bottom")
    ax.set_yticks(y, [BACKEND_LABEL[b] for b in order], fontsize=8.5,
                  color=INK)
    ax.set_xlabel("yolo11n end-to-end latency on Xeon Gold 6326 16C/32T (ms), batch=1",
                  fontsize=8.5, color=MUTED)
    style(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(OUT / "cpu_backends.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote cpu_backends.png")


if __name__ == "__main__":
    fig_latency()
    fig_breakdown()
    fig_batch()
    fig_accuracy()
    fig_cpu()
