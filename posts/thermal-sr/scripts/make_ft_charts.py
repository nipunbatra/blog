"""Build charts for the finetune-vs-zero-shot post."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/Users/nipun/git/blog/posts/thermal-sr/outputs")
summary = json.load(open(OUT / "ft_test_summary.json"))
history = json.load(open(OUT / "ft_history.json"))


# Chart 1 — per-dataset PSNR before/after
fig, ax = plt.subplots(1, 4, figsize=(20, 4.2), dpi=220)
splits = ["all", "sftl54", "thermeval"]
zs = [summary["zero_shot"][s]["psnr_mean"] for s in splits]
ft = [summary["finetuned"][s]["psnr_mean"] for s in splits]
x = np.arange(len(splits))
ax[0].bar(x - 0.2, zs, 0.4, label="zero-shot", color="#888")
ax[0].bar(x + 0.2, ft, 0.4, label="finetuned", color="#c44e52")
for i, (z, f) in enumerate(zip(zs, ft)):
    ax[0].text(i - 0.2, z + 0.1, f"{z:.2f}", ha="center", fontsize=9)
    ax[0].text(i + 0.2, f + 0.1, f"{f:.2f}", ha="center", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(splits)
ax[0].set_title("PSNR ↑")
ax[0].legend()

zs = [summary["zero_shot"][s]["ssim_mean"] for s in splits]
ft = [summary["finetuned"][s]["ssim_mean"] for s in splits]
ax[1].bar(x - 0.2, zs, 0.4, label="zero-shot", color="#888")
ax[1].bar(x + 0.2, ft, 0.4, label="finetuned", color="#c44e52")
for i, (z, f) in enumerate(zip(zs, ft)):
    ax[1].text(i - 0.2, z + 0.003, f"{z:.3f}", ha="center", fontsize=8)
    ax[1].text(i + 0.2, f + 0.003, f"{f:.3f}", ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(splits)
ax[1].set_title("SSIM ↑")
ax[1].set_ylim(0.92, 1.0)
ax[1].legend()

zs = [summary["zero_shot"][s]["lpips_mean"] for s in splits]
ft = [summary["finetuned"][s]["lpips_mean"] for s in splits]
ax[2].bar(x - 0.2, zs, 0.4, label="zero-shot", color="#888")
ax[2].bar(x + 0.2, ft, 0.4, label="finetuned", color="#c44e52")
for i, (z, f) in enumerate(zip(zs, ft)):
    ax[2].text(i - 0.2, z + 0.001, f"{z:.3f}", ha="center", fontsize=8)
    ax[2].text(i + 0.2, f + 0.001, f"{f:.3f}", ha="center", fontsize=8)
ax[2].set_xticks(x); ax[2].set_xticklabels(splits)
ax[2].set_title("LPIPS ↓")
ax[2].legend()

zs = [summary["zero_shot"][s]["nose_err_median"] for s in splits]
ft = [summary["finetuned"][s]["nose_err_median"] for s in splits]
ax[3].bar(x - 0.2, zs, 0.4, label="zero-shot", color="#888")
ax[3].bar(x + 0.2, ft, 0.4, label="finetuned", color="#c44e52")
for i, (z, f) in enumerate(zip(zs, ft)):
    ax[3].text(i - 0.2, z + 0.05, f"{z:.2f}", ha="center", fontsize=8)
    ax[3].text(i + 0.2, f + 0.05, f"{f:.2f}", ha="center", fontsize=8)
ax[3].set_xticks(x); ax[3].set_xticklabels(splits)
ax[3].set_title("DWPose nostril err median (px) ↓")
ax[3].legend()

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("DRLN x4 thermal SR: zero-shot vs finetuned (60 epochs, multi-dataset)",
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "ft_vs_zs.png", bbox_inches="tight")
plt.close(fig)
print("wrote ft_vs_zs.png")


# Chart 2 — training history (PSNR over epochs, per split)
fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
epochs = [h["epoch"] for h in history if h["epoch"] >= 0]
psnr_all = [h["metrics"]["all"]["psnr_mean"] for h in history if h["epoch"] >= 0]
psnr_sftl = [h["metrics"]["sftl54"]["psnr_mean"] for h in history if h["epoch"] >= 0]
psnr_te = [h["metrics"]["thermeval"]["psnr_mean"] for h in history if h["epoch"] >= 0]

# zero-shot baselines (from epoch -1)
zs_all = history[0]["metrics"]["all"]["psnr_mean"]
zs_sftl = history[0]["metrics"]["sftl54"]["psnr_mean"]
zs_te = history[0]["metrics"]["thermeval"]["psnr_mean"]

ax.axhline(zs_all, color="#888", linestyle=":", lw=1, label=f"zero-shot (all) = {zs_all:.2f}")
ax.plot(epochs, psnr_all, "-", color="black", lw=2, label="finetuned (all)")
ax.plot(epochs, psnr_sftl, "-", color="#4c72b0", lw=1.5, label="finetuned (SF-TL54)")
ax.plot(epochs, psnr_te, "-", color="#c44e52", lw=1.5, label="finetuned (ThermEval-D)")
ax.set_xlabel("epoch")
ax.set_ylabel("val PSNR (dB)")
ax.set_title("DRLN x4 finetune on thermal — val PSNR over epochs")
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "ft_history_curve.png", bbox_inches="tight")
plt.close(fig)
print("wrote ft_history_curve.png")
