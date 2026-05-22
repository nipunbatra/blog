"""3-way comparison chart: zero-shot / v1 L1-only / v2 L1+LPIPS+EMA."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/Users/nipun/git/blog/posts/thermal-sr/outputs")
s = json.load(open(OUT / "ft_v2_test_summary.json"))

METHODS = [("zero_shot", "zero-shot", "#888"),
           ("ft_v1", "v1 L1-only (60ep)", "#dd8452"),
           ("ft_v2", "v2 L1+LPIPS+EMA (200ep)", "#c44e52")]
SPLITS = ["all", "sftl54", "thermeval"]


def metric_chart(key, ylabel, title, fname, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=220)
    width = 0.25
    x = np.arange(len(SPLITS))
    for i, (mkey, label, color) in enumerate(METHODS):
        vals = [s[mkey][sp][key] for sp in SPLITS]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * width, v + (max(vals) * 0.01),
                    fmt.format(v), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(SPLITS)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


metric_chart("psnr_mean", "PSNR (dB) ↑",
              "Test PSNR — zero-shot vs v1 (L1) vs v2 (L1+LPIPS+EMA)",
              "v2_psnr.png", "{:.2f}")
metric_chart("lpips_mean", "LPIPS ↓",
              "Test LPIPS — v2 perceptual loss closes the gap",
              "v2_lpips.png", "{:.3f}")
metric_chart("nose_err_median", "DWPose nostril median err (px) ↓",
              "Downstream nostril localisation — v2 recovers what v1 lost",
              "v2_nose.png", "{:.2f}")
print("wrote v2_*.png")


# Single 4-panel combined chart
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5), dpi=220)
keys = [("psnr_mean", "PSNR (dB) ↑", "{:.2f}"),
        ("ssim_mean", "SSIM ↑", "{:.3f}"),
        ("lpips_mean", "LPIPS ↓", "{:.3f}"),
        ("nose_err_median", "Nose err median (px) ↓", "{:.2f}")]
for col, (key, ylabel, fmt) in enumerate(keys):
    width = 0.25
    x = np.arange(len(SPLITS))
    for i, (mkey, label, color) in enumerate(METHODS):
        vals = [s[mkey][sp][key] for sp in SPLITS]
        ax[col].bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for j, v in enumerate(vals):
            ax[col].text(x[j] + (i - 1) * width, v + (max(vals) * 0.015),
                        fmt.format(v), ha="center", fontsize=7)
    ax[col].set_xticks(x); ax[col].set_xticklabels(SPLITS, fontsize=9)
    ax[col].set_ylabel(ylabel); ax[col].set_title(ylabel.split("(")[0].strip())
    ax[col].spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Thermal SR finetune v2: LPIPS + EMA beats L1-only on every "
              "metric except raw PSNR — and on the downstream task too",
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "v2_combined4.png", bbox_inches="tight")
plt.close(fig)
print("wrote v2_combined4.png")
