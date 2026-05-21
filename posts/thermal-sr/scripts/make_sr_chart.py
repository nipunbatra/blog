"""Build the comparison chart + side-by-side grid for the thermal SR post."""
import json
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np
from PIL import Image

OUT = Path("/Users/nipun/git/blog/posts/thermal-sr/outputs")
d = json.load(open(OUT / "summary.json"))

METHODS = [
    ("bicubic", "Bicubic"),
    ("edsr_x4", "EDSR x4"),
    ("msrn_x4", "MSRN x4"),
    ("a2n_x4", "A2N x4"),
    ("drln_x4", "DRLN x4"),
    ("sd_x4", "SD x4 upscaler"),
]

COLORS = {"bicubic": "#888888",
          "edsr_x4": "#4c72b0", "msrn_x4": "#55a868",
          "a2n_x4": "#c44e52", "drln_x4": "#8172b3",
          "sd_x4": "#dd8452"}

# 2x2 chart grid
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5), dpi=220)
labels = [t for _, t in METHODS]
ks = [k for k, _ in METHODS]
psnr = [d["methods"][k]["psnr"] for k in ks]
ssim = [d["methods"][k]["ssim"] for k in ks]
lpips = [d["methods"][k]["lpips"] for k in ks]
nose_err = [d["methods"][k]["dwpose_err_px"] for k in ks]
cols = [COLORS[k] for k in ks]

ax[0].bar(labels, psnr, color=cols)
for i, v in enumerate(psnr): ax[0].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
ax[0].set_title("PSNR (higher = closer to HR)")
ax[0].set_ylim(0, max(psnr) * 1.15)

ax[1].bar(labels, ssim, color=cols)
for i, v in enumerate(ssim): ax[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax[1].set_title("SSIM (higher = closer)")
ax[1].set_ylim(0, 1.1)

ax[2].bar(labels, lpips, color=cols)
for i, v in enumerate(lpips): ax[2].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax[2].set_title("LPIPS (lower = closer perceptually)")

ax[3].bar(labels, nose_err, color=cols)
for i, v in enumerate(nose_err): ax[3].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
ax[3].set_title("DWPose nostril error (px, lower = better)")
ax[3].set_ylim(0, max(nose_err) * 1.15)

for a in ax:
    a.tick_params(axis="x", rotation=20, labelsize=8)
    a.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Thermal super-resolution: 4× from {d['lr_size'][0]}×{d['lr_size'][1]} to {d['hr_size'][0]}×{d['hr_size'][1]}",
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "sr_metrics_chart.png", bbox_inches="tight")
plt.close(fig)
print("wrote sr_metrics_chart.png")


# Side-by-side visual grid (HR, LR, each method)
hr = cv2.imread(str(OUT / "hr_target.png"))
lr = cv2.imread(str(OUT / "lr_input.png"))
H, W = hr.shape[:2]
# upsample LR to HR-size with nearest-neighbour for display
lr_nn = cv2.resize(lr, (W, H), interpolation=cv2.INTER_NEAREST)

panels = []
def label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img

panels.append(label(hr, "HR target (real thermal)"))
panels.append(label(lr_nn, f"LR input (NN to display) {d['lr_size'][0]}x{d['lr_size'][1]}"))
for k, t in METHODS:
    img = cv2.imread(str(OUT / f"restored_{k}.png"))
    if img is None: continue
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    psnr_v = d["methods"][k]["psnr"]; lpips_v = d["methods"][k]["lpips"]
    nose_v = d["methods"][k]["dwpose_err_px"] or float("nan")
    txt = f"{t}  PSNR={psnr_v:.1f}  LPIPS={lpips_v:.2f}  nose_err={nose_v:.1f}px"
    panels.append(label(img, txt))

# stack 2 cols
half = (len(panels) + 1) // 2
rows = []
for i in range(0, len(panels), 2):
    if i + 1 < len(panels):
        rows.append(np.hstack([panels[i], panels[i + 1]]))
    else:
        blank = np.zeros_like(panels[i])
        rows.append(np.hstack([panels[i], blank]))
grid = np.vstack(rows)
cv2.imwrite(str(OUT / "sr_side_by_side.png"), grid)
print(f"wrote sr_side_by_side.png  shape={grid.shape}")
