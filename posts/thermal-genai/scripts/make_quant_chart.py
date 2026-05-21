"""Build paper-grade comparison chart + summary table from quant_metrics.json."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs"
d = json.load(open(OUT / "quant_metrics.json"))

# Real-thermal reference targets (from any candidate's "_real" entries)
sample = next(iter(d.values()))
TARGETS = {
    "eye_to_orbital": sample.get("eye_to_orbital_real"),
    "hair_to_face":   sample.get("hair_to_face_real"),
    "nose_to_cheek":  sample.get("nose_to_cheek_real"),
}
print("real-thermal target ratios:", TARGETS)


def deviation(c, target):
    if c is None or target is None: return None
    return abs(c - target) / max(target, 1e-6)


# Build a unified deviation score: smaller is better
def score_row(m):
    row = {
        "ssim_y": m.get("ssim_y"),
        "lpips": m.get("lpips"),
        "hist_inter_rgb": m.get("hist_inter_rgb"),
        "eye_to_orb_dev": deviation(m.get("eye_to_orbital_cand"),
                                      TARGETS["eye_to_orbital"]),
        "hair_to_face_dev": deviation(m.get("hair_to_face_cand"),
                                        TARGETS["hair_to_face"]),
        "nose_to_cheek_dev": deviation(m.get("nose_to_cheek_cand"),
                                         TARGETS["nose_to_cheek"]),
    }
    return row


# Rename for readability
SHORT = {
    "flash_exp1a_generic": "Flash 1a generic",
    "flash_exp1b_rgb_derived": "Flash 1b RGB-cap",
    "flash_exp1c_thermal_physics": "Flash 1c phys",
    "flash_exp1d_gemini_thermcap": "Flash 1d therm-cap",
    "flash_exp3_rgb_plus_thermcap": "Flash 2 RGB+cap",
    "flash_exp4_refined_iter1": "Flash iter1",
    "flash_exp4_refined_iter2": "Flash iter2",
    "pro_exp1a_generic": "Pro 1a generic",
    "pro_exp1b_rgb_derived": "Pro 1b RGB-cap",
    "pro_exp1c_thermal_physics": "Pro 1c phys",
    "pro_exp1d_gemini_thermcap": "Pro 1d therm-cap",
    "pro_exp3_rgb_plus_thermcap": "Pro 2 RGB+cap",
    "pro_exp4_refined_iter1": "Pro iter1",
    "pro_exp4_refined_iter2": "Pro iter2",
    "tg_ThermalGen-L-2-concat_ds7_cfg1.0": "ThermalGen L (iron)",
    "tg_ThermalGen-B-2_ds21_cfg1.0": "ThermalGen B (iron)",
}

rows = []
for k, m in d.items():
    label = SHORT.get(k, k)
    rows.append((label, score_row(m)))

# --- print summary table ---
print()
print(f"{'Method':<24} {'SSIM↑':>7} {'LPIPS↓':>8} {'HIST↑':>7} "
      f"{'eye/orb_dev':>11} {'hair/face_dev':>14} {'nose/cheek_dev':>15}")
print("-" * 95)
for lbl, r in rows:
    def fmt(v, ndp=3):
        return f"{v:>.{ndp}f}" if v is not None else "n/a"
    print(f"{lbl:<24} {fmt(r['ssim_y']):>7} {fmt(r['lpips']):>8} "
          f"{fmt(r['hist_inter_rgb'], 2):>7} {fmt(r['eye_to_orb_dev'], 2):>11} "
          f"{fmt(r['hair_to_face_dev'], 2):>14} {fmt(r['nose_to_cheek_dev'], 2):>15}")


# --- bar chart: SSIM, hist_inter, and inverse LPIPS ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=220,
                          sharey=False)
labels = [lbl for lbl, _ in rows]
ssim_vals = [r["ssim_y"] or 0 for _, r in rows]
lpips_vals = [r["lpips"] or 1.0 for _, r in rows]
hist_vals = [r["hist_inter_rgb"] or 0 for _, r in rows]
eye_dev = [r["eye_to_orb_dev"] or 1.0 for _, r in rows]


def color_of(lbl):
    if lbl.startswith("Flash"): return "#4c72b0"
    if lbl.startswith("Pro"): return "#c44e52"
    if "ThermalGen" in lbl: return "#55a868"
    return "#888"

cols = [color_of(l) for l in labels]

axes[0].barh(range(len(labels)), ssim_vals, color=cols)
axes[0].set_yticks(range(len(labels)))
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].set_title("SSIM (higher = closer)\n[full image vs real thermal]")
axes[0].invert_yaxis()

axes[1].barh(range(len(labels)), lpips_vals, color=cols)
axes[1].set_yticks(range(len(labels))); axes[1].set_yticklabels([])
axes[1].set_title("LPIPS (lower = closer)\n[perceptual distance, AlexNet]")
axes[1].invert_yaxis()

axes[2].barh(range(len(labels)), hist_vals, color=cols)
axes[2].set_yticks(range(len(labels))); axes[2].set_yticklabels([])
axes[2].set_title("Histogram intersection\n(RGB, max=3)")
axes[2].invert_yaxis()

axes[3].barh(range(len(labels)), eye_dev, color=cols)
axes[3].set_yticks(range(len(labels))); axes[3].set_yticklabels([])
axes[3].set_title("Eye/orbital ratio deviation\nfrom real (lower better)")
axes[3].invert_yaxis()

fig.suptitle(
    f"Paper-grade thermal-physics metrics (target eye/orb={TARGETS['eye_to_orbital']:.2f}, "
    f"hair/face={TARGETS['hair_to_face']:.2f}, nose/cheek={TARGETS['nose_to_cheek']:.2f})",
    fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "quant_metrics_chart.png", bbox_inches="tight")
plt.close(fig)
print("wrote quant_metrics_chart.png")
