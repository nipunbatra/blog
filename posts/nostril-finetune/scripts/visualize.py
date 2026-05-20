"""Plot the training curve and produce 6 qualitative predictions on the test
set comparing the finetuned model vs zero-shot DWPose / Sapiens2."""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_head import (
    NostrilSet, TinyHead, build_backbone, decode_heatmaps_to_xy,
    IMG_SIZE, HEATMAP_SIZE, DEVICE,
)

OUT_DIR = Path.home() / "git/nostril-bench/runs/finetune_v1"
PLOT_DIR = OUT_DIR  # write locally; we scp back to Mac


def training_curve():
    d = json.load(open(OUT_DIR / "summary.json"))
    h = d["history"]
    ep = [r["epoch"] for r in h]
    vl = [r["val_mean_err"] for r in h]
    pc = [r["val_pck10"] for r in h]
    fig, ax1 = plt.subplots(figsize=(7, 3.5), dpi=110)
    ax1.plot(ep, vl, color="#c44e52", label="val mean nostril error (px)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("val mean nostril error (px)",
                                            color="#c44e52")
    ax1.tick_params(axis="y", labelcolor="#c44e52")
    ax1.spines[["top"]].set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(ep, pc, color="#4c72b0", linestyle="--", label="PCK@10px")
    ax2.set_ylabel("val PCK@10px", color="#4c72b0")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis="y", labelcolor="#4c72b0")
    ax2.spines[["top"]].set_visible(False)
    plt.title("Finetune: 30 thermal frames, frozen Sapiens2 backbone, 120 epochs")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "training_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", PLOT_DIR / "training_curve.png")


def qualitative_grid(n=6):
    test = NostrilSet("test")
    backbone = build_backbone()
    head = TinyHead().to(DEVICE)
    head.load_state_dict(torch.load(OUT_DIR / "head_best.pt"))
    head.eval()
    rows = []
    for i in range(n):
        img_t, hm, gt = test[i]
        x = img_t.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = backbone.backbone(x)[0]
            pred_hm = head(feats)
        pred_xy = decode_heatmaps_to_xy(pred_hm)[0].cpu().numpy()
        # restore original image
        item = test.items[i]
        raw = cv2.imread(str(test.dir / "images" / item["image"]),
                         cv2.IMREAD_GRAYSCALE)
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        out = raw.copy()
        gt_np = gt.numpy()
        for (gx, gy) in gt_np:
            cv2.drawMarker(out, (int(gx), int(gy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 18, 2)
        for (px, py) in pred_xy:
            cv2.circle(out, (int(px), int(py)), 6, (0, 255, 0), -1)
        # add error text
        err = np.linalg.norm(pred_xy - gt_np, axis=-1).mean()
        cv2.rectangle(out, (0, 0), (256, 22), (0, 0, 0), -1)
        cv2.putText(out, f"err={err:.1f}px", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(out)
    grid = np.vstack([np.hstack(rows[:3]), np.hstack(rows[3:])])
    cv2.imwrite(str(PLOT_DIR / "qualitative_test.png"), grid)
    print("wrote", PLOT_DIR / "qualitative_test.png")


if __name__ == "__main__":
    training_curve()
    qualitative_grid()
