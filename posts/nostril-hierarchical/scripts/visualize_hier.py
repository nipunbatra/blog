"""Build the visualisations for the hierarchical-vs-single-stage post.

  - face_size_curve.png  : line chart of mean nostril error vs face-side
  - detection_rate.png   : line chart of detection rate vs face-side
  - panel_compact.png    : 4 rows (4 shrink factors) x 2 cols (single, hier)
                           cropped to the upper-left where the face is
"""
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

ROOT = Path("/Users/nipun/git/blog/posts/nostril-hierarchical/outputs")
RUN = ROOT / "hier_v2"

rows = json.load(open(RUN / "rows.json"))


def per_factor():
    g = defaultdict(lambda: {"single": [], "hier": [], "face_px": [],
                             "n_total": 0, "n_single_det": 0, "n_hier_det": 0})
    for r in rows:
        f = r["factor"]
        g[f]["n_total"] += 2
        g[f]["face_px"].append(r["face_side_px"])
        for k in ("single_err_l", "single_err_r"):
            if r[k] is not None:
                g[f]["single"].append(r[k]); g[f]["n_single_det"] += 1
        for k in ("hier_err_l", "hier_err_r"):
            if r[k] is not None:
                g[f]["hier"].append(r[k]); g[f]["n_hier_det"] += 1
    return g


def line_chart(g, kind):
    factors = sorted(g.keys())
    fpx = [np.mean(g[f]["face_px"]) for f in factors]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=110)
    if kind == "error":
        s_y = [np.mean(g[f]["single"]) if g[f]["single"] else np.nan
               for f in factors]
        h_y = [np.mean(g[f]["hier"]) if g[f]["hier"] else np.nan
               for f in factors]
        ax.plot(fpx, s_y, "o-", color="#c44e52", label="single-stage MediaPipe")
        ax.plot(fpx, h_y, "s-", color="#4c72b0",
                label="hierarchical (BlazeFace full-range → crop → FaceMesh)")
        ax.set_ylabel("mean nostril error (px, pseudo-GT)")
        ax.set_title("Localisation accuracy vs face size in image")
    else:
        s_y = [g[f]["n_single_det"] / g[f]["n_total"] for f in factors]
        h_y = [g[f]["n_hier_det"] / g[f]["n_total"] for f in factors]
        ax.plot(fpx, s_y, "o-", color="#c44e52", label="single-stage MediaPipe")
        ax.plot(fpx, h_y, "s-", color="#4c72b0", label="hierarchical")
        ax.set_ylabel("detection rate (fraction of keypoints found)")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title("Detection rate vs face size in image")
    ax.set_xlabel("face inter-nostril width in image (pixels)")
    ax.invert_xaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    name = "face_size_error.png" if kind == "error" else "detection_rate.png"
    fig.savefig(ROOT / name, bbox_inches="tight")
    plt.close(fig)


def crop_face_corner(img, face_px_in_shrunk, factor, margin=40):
    """Top-left corner of img contains the shrunk face. Crop a tight window
    around it."""
    H, W = img.shape[:2]
    sh = H // factor; sw = W // factor
    side = max(face_px_in_shrunk * 2.4, 80)
    x0 = 0; y0 = 0
    x1 = int(min(W, max(sw, int(side))))
    y1 = int(min(H, max(sh, int(side))))
    return img[y0:y1, x0:x1]


def compact_panel():
    """Pick ONE image and show 4 shrink-factor rows × 2 method columns,
    cropped to face region."""
    pick_image = "100_1_2_3_1070_27_3.png"
    # use the panel.png (which already has shrink=max) — but we want all
    # shrink factors. Rerun cropping on rows of the SAME image.
    rows_pick = [r for r in rows if pick_image in r["image"]]
    if not rows_pick:
        return
    # Read the original image (we don't have it locally, so use the inputs we copied)
    # we'll just re-construct the panel by reading the bhaskar panel images...
    # but easier: read the panel.png that already has 8x crops for all 8 imgs.
    full = cv2.imread(str(RUN / "panel.png"))
    if full is None: return
    H, W = full.shape[:2]
    # Each block in panel.png is 348 high x 928 wide (2 panels horizontal).
    # 8 images = 8 blocks vertically.
    rows_h = H // 8; col_w = W // 2
    # Pick the 3rd row (index 2)
    idx = 2
    single_block = full[idx*rows_h:(idx+1)*rows_h, 0:col_w]
    hier_block = full[idx*rows_h:(idx+1)*rows_h, col_w:]
    # crop to upper-left
    crop = 150
    single_c = single_block[:crop, :crop]
    hier_c = hier_block[:crop, :crop]
    # upscale x4 for readability
    single_c = cv2.resize(single_c, (single_c.shape[1]*4, single_c.shape[0]*4),
                          interpolation=cv2.INTER_NEAREST)
    hier_c = cv2.resize(hier_c, (hier_c.shape[1]*4, hier_c.shape[0]*4),
                        interpolation=cv2.INTER_NEAREST)
    grid = np.hstack([single_c, hier_c])
    cv2.imwrite(str(ROOT / "panel_compact.png"), grid)
    print(f"wrote {ROOT/'panel_compact.png'}  shape={grid.shape}")


if __name__ == "__main__":
    g = per_factor()
    line_chart(g, "error")
    line_chart(g, "detection")
    compact_panel()
    print("ok")
