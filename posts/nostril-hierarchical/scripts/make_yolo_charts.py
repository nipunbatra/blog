"""Charts for the YOLO-nostril Blog 3."""
import json, statistics
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np

OUT = Path("/Users/nipun/git/blog/posts/nostril-hierarchical/outputs")
RUN = OUT / "hier_yolo_thermeval"

rows = json.load(open(RUN / "summary.json"))
total_gt = sum(r["n_gt"] for r in rows)
print(f"N {len(rows)} GT {total_gt}")

PIPELINES = [
    ("a", "(A) Single-stage\nMediaPipe", "#c44e52"),
    ("b", "(B) Hierarchical\nBlazeFace + MediaPipe", "#dd8452"),
    ("c", "(C) Hierarchical\nBlazeFace + YOLO", "#9467bd"),
    ("d", "(D) Hierarchical\nGT bbox + YOLO", "#4c72b0"),
    ("e", "(E) Raw YOLO\non whole frame", "#55a868"),
]


def agg():
    out = {}
    for k, _, _ in PIPELINES:
        n_pred = sum(r[k]["n"] for r in rows)
        errs = []
        for r in rows: errs.extend(r[k]["errs"])
        n_match = len(errs)
        times = [r[k]["t_ms"] for r in rows]
        out[k] = {
            "n_pred": n_pred, "n_match": n_match,
            "det_rate": n_match / total_gt,
            "false_pos": (n_pred - n_match) / max(n_pred, 1),
            "precision": n_match / max(n_pred, 1),
            "median_err": float(np.median(errs)) if errs else float("nan"),
            "mean_err": float(np.mean(errs)) if errs else float("nan"),
            "pck5": sum(1 for e in errs if e <= 5) / total_gt,
            "pck10": sum(1 for e in errs if e <= 10) / total_gt,
            "time_ms": float(np.mean(times)) if times else 0,
        }
    return out


s = agg()


def bar(metric, ylabel, title, fname, ylim=None, fmt="{:.0%}"):
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=220)
    vals = [s[k][metric] for k, _, _ in PIPELINES]
    cols = [c for _, _, c in PIPELINES]
    bars = ax.bar(range(len(PIPELINES)), vals, color=cols)
    for b, v in zip(bars, vals):
        if v != v:
            ax.text(b.get_x() + b.get_width() / 2, 0.02, "n/a", ha="center", fontsize=9)
            continue
        ymax = ylim[1] if ylim else max(vals) if max(vals) > 0 else 1
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, fmt.format(v),
                ha="center", fontsize=10)
    ax.set_xticks(range(len(PIPELINES)))
    ax.set_xticklabels([lbl for _, lbl, _ in PIPELINES], fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


bar("det_rate", "detection rate (matched preds / GT noses)",
    f"Detection rate on ThermEval-D ({total_gt} GT noses, {len(rows)} frames)",
    "te_yolo_detection.png", ylim=(0, 1.05))
bar("pck10", "PCK@10px (strict, over all GT)",
    "Strict PCK@10px (missed detections count as 0)",
    "te_yolo_pck10.png", ylim=(0, 1.05))
bar("median_err", "median nostril error (px, matched only)",
    "Median error on matched predictions (lower is better)",
    "te_yolo_median.png", fmt="{:.1f}")
bar("precision", "precision (matched / total preds)",
    "Precision — fraction of predictions that hit a real nostril",
    "te_yolo_precision.png", ylim=(0, 1.05))
bar("time_ms", "per-frame inference time (ms)",
    "Per-frame inference time on bhaskar (RTX A5000)",
    "te_yolo_speed.png", fmt="{:.0f}ms")


# Combined scatter: detection rate vs median error
fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
for k, label, color in PIPELINES:
    ax.scatter(s[k]["det_rate"], s[k]["median_err"], s=400, color=color,
               edgecolors="black", linewidth=1)
    ax.annotate(label.replace("\n", " "),
                (s[k]["det_rate"], s[k]["median_err"]),
                xytext=(10, 10), textcoords="offset points", fontsize=9)
ax.set_xlabel("detection rate (higher better)")
ax.set_ylabel("median nostril error (px, lower better)")
ax.set_title("ThermEval-D: 5-pipeline comparison (upper-right = best)")
ax.set_xlim(-0.05, 1.1)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "te_yolo_scatter.png", bbox_inches="tight")
plt.close(fig)


# Multi-frame visualization grid — 4 frames in a 2x2
viz_dir = RUN / "viz"
files = sorted(viz_dir.glob("img*.png"))[:4]
imgs = [cv2.imread(str(f)) for f in files if cv2.imread(str(f)) is not None]
if imgs:
    # Each viz frame is 2 rows x 3 cols = 5 panels + 1 blank.
    # We want to show 2 frames stacked vertically (using just 2 from imgs).
    grid = np.vstack(imgs[:2])
    cv2.imwrite(str(OUT / "te_yolo_two_frames.png"), grid)
    print("wrote te_yolo_two_frames.png", grid.shape)

print(s)
