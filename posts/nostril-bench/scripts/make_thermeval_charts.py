"""Charts for the ThermEval-D bake-off."""
import json, statistics
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np

OUT = Path("/Users/nipun/git/blog/posts/nostril-bench/outputs")
RUN = OUT / "thermeval_50"

d = json.load(open(RUN / "summary.json"))
total_gt = sum(r["n_gt"] for r in d)

MODELS = ["sapiens2_0.4b", "dwpose", "mediapipe_facemesh"]
LABEL = {"sapiens2_0.4b": "Sapiens2-0.4b",
         "dwpose": "DWPose",
         "mediapipe_facemesh": "MediaPipe FaceMesh"}
COLOR = {"sapiens2_0.4b": "#55a868", "dwpose": "#4c72b0",
         "mediapipe_facemesh": "#c44e52"}


def aggregate():
    out = {}
    for m in MODELS:
        n_pred = sum(r["per_model"][m]["n_pred"] for r in d)
        n_match = sum(r["per_model"][m]["n_matched"] for r in d)
        errs = []
        for r in d: errs.extend(r["per_model"][m].get("errs", []))
        times = [r["per_model"][m]["elapsed_s"] * 1000 for r in d
                 if r["per_model"][m].get("elapsed_s")]
        out[m] = {
            "n_pred": n_pred, "n_match": n_match,
            "det_rate": n_match / total_gt,
            "mean_err": np.mean(errs) if errs else float("nan"),
            "median_err": float(np.median(errs)) if errs else float("nan"),
            "pck5": sum(1 for e in errs if e <= 5) / total_gt,
            "pck10": sum(1 for e in errs if e <= 10) / total_gt,
            "pck20": sum(1 for e in errs if e <= 20) / total_gt,
            "time_ms": np.mean(times) if times else 0,
        }
    return out


def bar(metric, ylabel, title, fname, ylim=None, fmt="{:.1%}"):
    s = aggregate()
    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=220)
    vals = [s[m][metric] for m in MODELS]
    bars = ax.bar(range(len(MODELS)), vals, color=[COLOR[m] for m in MODELS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] if ylim else max(vals)) * 0.02,
                fmt.format(v), ha="center", fontsize=10)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


bar("det_rate", "detection rate (matched preds / GT noses)",
    f"Detection rate on ThermEval-D ({total_gt} GT noses across {len(d)} frames)",
    "te_detection.png", ylim=(0, 1.1), fmt="{:.0%}")

bar("pck10", "PCK@10px (over all GT noses, not just matched)",
    "PCK@10px on ThermEval-D — strict, counts missed detections as 0",
    "te_pck10.png", ylim=(0, 1.1), fmt="{:.0%}")

bar("median_err", "median nostril error (px, on MATCHED only)",
    "Median error on matched predictions (lower=better)",
    "te_median.png", fmt="{:.1f}")

bar("time_ms", "per-image time (ms)",
    "Per-image inference time (RTX A5000)",
    "te_speed.png", fmt="{:.0f}ms")


# Combined: scatter detection rate vs median accuracy
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=220)
s = aggregate()
for m in MODELS:
    ax.scatter(s[m]["det_rate"], s[m]["median_err"], s=400, color=COLOR[m],
               edgecolors="black", linewidth=1)
    ax.annotate(LABEL[m], (s[m]["det_rate"], s[m]["median_err"]),
                xytext=(8, 8), textcoords="offset points", fontsize=11)
ax.set_xlabel("detection rate (higher better)")
ax.set_ylabel("median nostril error in px (lower better)")
ax.set_title("ThermEval-D: detection rate × accuracy (upper-right = win)")
ax.set_xlim(-0.05, 1.1)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "te_scatter.png", bbox_inches="tight")
plt.close(fig)


# Multi-frame grid: 6 frames × full panel each (the viz dir already has these)
viz_dir = RUN / "viz"
files = sorted(viz_dir.glob("img*.png"))[:6]
images = [cv2.imread(str(f)) for f in files if cv2.imread(str(f)) is not None]
if images:
    # all have shape (height x width) — stack in 3x2 grid
    h, w = images[0].shape[:2]
    rows = []
    for i in range(0, 6, 2):
        if i + 1 < len(images):
            rows.append(np.hstack([images[i], images[i + 1]]))
        else:
            rows.append(np.hstack([images[i], np.zeros_like(images[i])]))
    grid = np.vstack(rows)
    cv2.imwrite(str(OUT / "te_six_frames.png"), grid)
    print(f"wrote te_six_frames.png  {grid.shape}")

print("aggregate:", aggregate())
