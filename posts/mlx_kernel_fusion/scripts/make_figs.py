"""Generate the post's figures from CSV-like literal results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGS = Path(__file__).resolve().parent.parent / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

# ---- compile (GELU) ----
gelu = {
    "shapes":   ["1k²", "2k²", "4k²", "8k²"],
    "eager":    [0.644, 2.439, 11.877, 32.505],
    "compiled": [0.272, 1.244, 0.711, 7.368],
}

# ---- LayerNorm ----
ln = {
    "shapes":   ["1×1k×1k", "1×2k×2k", "1×4k×4k", "1×8k×4k", "4×4k×4k"],
    "composite": [0.360, 0.554, 3.070, 5.968, 11.893],
    "compiled":  [0.288, 0.505, 1.694, 3.195, 6.347],
    "fast":      [0.168, 0.237, 0.557, 0.945, 1.683],
}

# ---- Attention ----
attn = {
    "T":        [256, 512, 1024, 2048, 4096, 8192],
    "composite": [0.429, 0.406, 1.151, 3.837, 13.521, 53.819],
    "compiled":  [0.294, 0.422, 1.089, 3.791, 13.612, 53.819],
    "fast":      [0.431, 0.413, 0.834, 2.652, 9.541, 36.572],
    "attn_mb":   [4.2, 16.8, 67.1, 268.4, 1073.7, 4295.0],
}

# ---- Train step ----
train = {
    "labels":   ["256/512/8/2k", "512/512/8/2k", "1k/768/12/3k", "2k/768/12/3k"],
    "eager":    [3.227, 5.173, 19.527, 47.360],
    "compiled": [2.624, 4.205, 16.525, 40.386],
}


def bar(ax, groups, series, colors, labels, title, ylabel, log=False):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, (s, c, lab) in enumerate(zip(series, colors, labels)):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, s, w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)


# Fig 1 — GELU compile speedup
fig, ax = plt.subplots(figsize=(6, 3.4))
bar(ax, gelu["shapes"], [gelu["eager"], gelu["compiled"]],
    ["#888888", "#c44536"], ["eager", "mx.compile"],
    "GELU element-wise composite", "ms / call (lower is better)")
for i, (e, c) in enumerate(zip(gelu["eager"], gelu["compiled"])):
    ax.text(i, max(e, c) * 1.04, f"{e/c:.1f}×", ha="center", fontsize=8.5,
            color="#c44536")
fig.tight_layout()
fig.savefig(FIGS / "fig_compile_gelu.png")
plt.close(fig)

# Fig 2 — LayerNorm
fig, ax = plt.subplots(figsize=(7.2, 3.6))
bar(ax, ln["shapes"], [ln["composite"], ln["compiled"], ln["fast"]],
    ["#888888", "#3a86ff", "#c44536"],
    ["composite (eager)", "mx.compile", "mx.fast.layer_norm"],
    "LayerNorm: three implementations", "ms / call")
for i, (e, f) in enumerate(zip(ln["composite"], ln["fast"])):
    ax.text(i, e * 1.04, f"{e/f:.1f}×", ha="center", fontsize=8.5,
            color="#c44536")
fig.tight_layout()
fig.savefig(FIGS / "fig_layernorm.png")
plt.close(fig)

# Fig 3 — Attention time
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
ax.plot(attn["T"], attn["composite"], "o-", color="#888888", label="composite (eager)")
ax.plot(attn["T"], attn["compiled"], "s-", color="#3a86ff", label="mx.compile")
ax.plot(attn["T"], attn["fast"], "D-", color="#c44536",
        label="mx.fast.scaled_dot_product_attention")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(attn["T"]); ax.set_xticklabels(attn["T"])
ax.set_xlabel("sequence length T"); ax.set_ylabel("ms / call")
ax.set_title("Attention time (B=1, H=16, D=64)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.bar(np.arange(len(attn["T"])), attn["attn_mb"], color="#c44536")
ax.set_xticks(np.arange(len(attn["T"])))
ax.set_xticklabels(attn["T"])
ax.set_yscale("log")
ax.set_xlabel("sequence length T")
ax.set_ylabel("attention matrix size (MB)")
ax.set_title("Memory the composite materialises (fast SDPA materialises 0)")
for i, v in enumerate(attn["attn_mb"]):
    ax.text(i, v * 1.18, f"{v:.0f}MB", ha="center", fontsize=8.5)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(FIGS / "fig_attention.png")
plt.close(fig)

# Fig 4 — Train step
fig, ax = plt.subplots(figsize=(6.5, 3.4))
bar(ax, train["labels"], [train["eager"], train["compiled"]],
    ["#888888", "#c44536"], ["eager", "mx.compile (capturing optimizer state)"],
    "Training step (Block + AdamW)", "ms / step\n(B=2, T/D/H/FF labels)")
for i, (e, c) in enumerate(zip(train["eager"], train["compiled"])):
    ax.text(i, e * 1.04, f"{e/c:.2f}×", ha="center", fontsize=8.5, color="#c44536")
fig.tight_layout()
fig.savefig(FIGS / "fig_train.png")
plt.close(fig)

print(f"Wrote figures to {FIGS}")
