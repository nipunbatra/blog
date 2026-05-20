"""Make summary charts + close-ups for the bake-off post.

  - bar_error.png       : mean nostril error (px) per model x modality
  - bar_pck.png         : PCK@{5,10,20}px per model x modality
  - bar_speed.png       : per-image time (ms) per model
  - fail_sapiens.png    : Sapiens2 wholebody output on thermal, showing why it
                         confuses ears with nose
  - closeup_grid.png    : 4 cropped (nose region) panels across models
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/Users/nipun/git/blog/posts/nostril-bench/outputs")
SUBSET = OUT / "subset_0.4b_v5"


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def aggregate(summary):
    per = {}  # (model, modality) -> list of errors
    times = {}
    for item in summary["items"]:
        mod = item["modality"]
        for m, stats in item["per"].items():
            key = (m, mod)
            per.setdefault(key, [])
            times.setdefault(key, [])
            if stats.get("err_l") is not None: per[key].append(stats["err_l"])
            if stats.get("err_r") is not None: per[key].append(stats["err_r"])
            times[key].append(stats["time_ms"])
    return per, times


MODELS = ["sapiens2_0.4b", "dwpose", "mediapipe_facemesh"]
LABELS = {"sapiens2_0.4b": "Sapiens2-0.4b\n(308 wholebody)",
          "dwpose": "DWPose\n(133 wholebody)",
          "mediapipe_facemesh": "MediaPipe\nFaceMesh (478)"}
MODALITIES = ["gray", "iron"]
COL = {"gray": "#4c72b0", "iron": "#dd8452"}


def bar_error(per):
    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=220)
    width = 0.38
    x = np.arange(len(MODELS))
    for i, mod in enumerate(MODALITIES):
        vals = [np.mean(per[(m, mod)]) if (m, mod) in per and per[(m, mod)]
                else 0 for m in MODELS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=mod.upper(),
               color=COL[mod])
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 0.5) * width, v + 1, f"{v:.0f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS],
                                         fontsize=9)
    ax.set_ylabel("mean nostril localisation error (px)")
    ax.set_title("Lower is better — thermal face, 10 subjects")
    ax.legend(title="thermal palette", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "bar_error.png", bbox_inches="tight")
    plt.close(fig)


def bar_pck(per):
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=220)
    width = 0.18
    x = np.arange(len(MODELS))
    radii = [5, 10, 20, 40]
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#999999"]
    for i, r in enumerate(radii):
        vals = [sum(1 for e in per.get((m, "gray"), []) if e <= r)
                / max(1, len(per.get((m, "gray"), []))) for m in MODELS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=f"≤{r}px",
               color=colors[i])
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1.5) * width, v + 0.02, f"{v:.0%}",
                    ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS],
                                         fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("fraction of nostril preds within R px")
    ax.set_title("PCK on GRAY thermal — higher = better")
    ax.legend(title="radius", fontsize=8, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "bar_pck.png", bbox_inches="tight")
    plt.close(fig)


def bar_speed(times):
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=220)
    x = np.arange(len(MODELS))
    vals = [np.mean(times[(m, "gray")]) if (m, "gray") in times else 0
            for m in MODELS]
    bars = ax.bar(x, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 5, f"{v:.0f} ms",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS],
                                         fontsize=9)
    ax.set_ylabel("per-image time (ms, single RTX A5000)")
    ax.set_title("Speed — including model load (DWPose excludes its YOLOX call)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "bar_speed.png", bbox_inches="tight")
    plt.close(fig)


def fail_closeup():
    """For one image, show all-model nostril predictions zoomed in around GT."""
    fname = "100_1_2_1_1134_36_1"
    img = cv2.imread(str(SUBSET / "gray" / f"{fname}_grid.png"))
    if img is not None:
        cv2.imwrite(str(OUT / "fail_grid.png"), img)
    # we'll do a custom close-up overlay:
    with open(SUBSET / "gray" / f"{fname}_results.json") as f:
        d = json.load(f)
    local_src = Path("/Users/nipun/git/blog/posts/nostril-bench/inputs/sample_gray.png")
    src = cv2.imread(str(local_src))
    if src is None:
        print("missing local source image", local_src)
        return
    gt = d["gt_nose_tip"]
    if gt is None: return
    gt_l, gt_r = gt[1], gt[3]
    cx, cy = (gt_l[0] + gt_r[0]) / 2, (gt_l[1] + gt_r[1]) / 2
    r = 80  # crop half-size
    x0 = int(max(0, cx - r)); y0 = int(max(0, cy - r))
    x1 = int(min(src.shape[1], cx + r)); y1 = int(min(src.shape[0], cy + r))

    palette = {"sapiens2_0.4b": (0, 255, 0),
               "dwpose": (255, 128, 0),
               "mediapipe_facemesh": (255, 0, 255)}
    label_map = {"sapiens2_0.4b": "Sapiens2-0.4b",
                 "dwpose": "DWPose",
                 "mediapipe_facemesh": "MediaPipe"}
    idx = {"sapiens2_0.4b": (55, 58), "dwpose": (55, 58),
           "mediapipe_facemesh": (48, 278)}

    panels = []
    for m in ["sapiens2_0.4b", "dwpose", "mediapipe_facemesh"]:
        sub = src.copy()
        kp = d["models"][m].get("kp", [])
        # draw GT
        cv2.drawMarker(sub, (int(gt_l[0]), int(gt_l[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(sub, (int(gt_r[0]), int(gt_r[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        li, ri = idx[m]
        if kp and li < len(kp) and ri < len(kp):
            cv2.circle(sub, (int(kp[li][0]), int(kp[li][1])), 7,
                       palette[m], -1)
            cv2.circle(sub, (int(kp[ri][0]), int(kp[ri][1])), 7,
                       palette[m], -1)
            cv2.line(sub, (int(kp[li][0]), int(kp[li][1])),
                     (int(gt_l[0]), int(gt_l[1])), (255, 255, 255), 1)
            cv2.line(sub, (int(kp[ri][0]), int(kp[ri][1])),
                     (int(gt_r[0]), int(gt_r[1])), (255, 255, 255), 1)
        crop = sub[y0:y1, x0:x1]
        crop = cv2.resize(crop, (320, 320), cv2.INTER_NEAREST)
        cv2.rectangle(crop, (0, 0), (320, 22), (0, 0, 0), -1)
        cv2.putText(crop, label_map[m], (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(crop)
    stacked = np.hstack(panels)
    cv2.imwrite(str(OUT / "closeup_grid.png"), stacked)


if __name__ == "__main__":
    summary = load_summary(SUBSET / "summary.json")
    per, times = aggregate(summary)
    bar_error(per); bar_pck(per); bar_speed(times)
    fail_closeup()
    print("ok")
