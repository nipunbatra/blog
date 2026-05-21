"""Charts for the ThermEval hierarchical post."""
import json, statistics
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np

OUT = Path("/Users/nipun/git/blog/posts/nostril-hierarchical/outputs")
RUN = OUT / "hier_thermeval"

rows = json.load(open(RUN / "summary.json"))
total_gt = sum(r["n_gt"] for r in rows)
print(f"N frames {len(rows)}, total GT {total_gt}")

PIPELINES = [
    ("single", "Single-stage\nMediaPipe FaceMesh", "#c44e52"),
    ("hier_blaze", "Hierarchical\n(BlazeFace detector)", "#dd8452"),
    ("hier_gt", "Hierarchical\n(GT Person bbox)", "#4c72b0"),
]


def aggregate():
    out = {}
    for key, _, _ in PIPELINES:
        n_pred = sum(r[f"{key}_n"] for r in rows)
        errs = []
        for r in rows: errs.extend(r[f"{key}_errs"])
        n_match = len(errs)
        times = [r[f"{key}_t_ms"] for r in rows]
        out[key] = {
            "n_pred": n_pred,
            "n_match": n_match,
            "det_rate": n_match / total_gt,
            "median_err": float(np.median(errs)) if errs else float("nan"),
            "mean_err": float(np.mean(errs)) if errs else float("nan"),
            "pck5": sum(1 for e in errs if e <= 5) / total_gt,
            "pck10": sum(1 for e in errs if e <= 10) / total_gt,
            "time_ms": float(np.mean(times)) if times else 0,
        }
    return out


s = aggregate()


def bar(metric, ylabel, title, fname, ylim=None, fmt="{:.0%}"):
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=220)
    vals = [s[k][metric] for k, _, _ in PIPELINES]
    cols = [c for _, _, c in PIPELINES]
    bars = ax.bar(range(len(PIPELINES)), vals, color=cols)
    for b, v in zip(bars, vals):
        if v != v:  # nan
            ax.text(b.get_x() + b.get_width() / 2, 0.02, "n/a", ha="center",
                    fontsize=9)
            continue
        ymax = ylim[1] if ylim else max(vals) if max(vals) > 0 else 1
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, fmt.format(v),
                ha="center", fontsize=10)
    ax.set_xticks(range(len(PIPELINES)))
    ax.set_xticklabels([lbl for _, lbl, _ in PIPELINES], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


bar("det_rate", "detection rate (matched preds / GT noses)",
    f"Detection rate on ThermEval-D ({total_gt} GT noses, {len(rows)} frames)",
    "te_hier_detection.png", ylim=(0, 0.6))
bar("pck10", "PCK@10px (over all GT, strict)",
    "PCK@10px — strict (missed detections count as 0)",
    "te_hier_pck10.png", ylim=(0, 0.6))
bar("median_err", "median nostril error (px, matched only)",
    "Median error on matched predictions (lower = better)",
    "te_hier_median.png", fmt="{:.1f}")
bar("time_ms", "per-frame inference time (ms)",
    "Per-frame inference time (CPU MediaPipe)",
    "te_hier_speed.png", fmt="{:.0f}ms")


# Multi-frame visualization grid - pull 4 viz panels and stack
viz_dir = RUN / "viz"
files = sorted(viz_dir.glob("img*.png"))[:4]
images = [cv2.imread(str(f)) for f in files if cv2.imread(str(f)) is not None]
if images:
    rows_img = [images[0], images[1]] if len(images) >= 2 else [images[0]]
    if len(images) >= 4:
        grid = np.vstack([images[0], images[1], images[2], images[3]])
    else:
        grid = np.vstack(images)
    cv2.imwrite(str(OUT / "te_hier_four_frames.png"), grid)
    print(f"wrote te_hier_four_frames.png  {grid.shape}")


print("aggregate:", s)
