"""Build side-by-side Flash vs Pro panels + a scores chart."""
import json, re
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLASH = ROOT / "outputs"
PRO = ROOT / "outputs_pro"

f_sum = json.load(open(FLASH / "summary.json"))
p_sum = json.load(open(PRO / "summary.json"))

ATTEMPTS = [
    ("exp1a_generic", "1a generic caption"),
    ("exp1b_rgb_derived", "1b RGB-caption"),
    ("exp1c_thermal_physics", "1c thermal-physics caption"),
    ("exp1d_gemini_thermcap", "1d Gemini-therm-caption"),
    ("exp3_rgb_plus_thermcap", "2 RGB + caption"),
    ("exp4_refined_iter1", "3 self-critique iter 1"),
    ("exp4_refined_iter2", "3 self-critique iter 2"),
]


def score_of(summary, name):
    s = summary["judge_scores"].get(name)
    if not s: return None
    m = re.match(r"SCORE:\s*(\d+)/10", s)
    return int(m.group(1)) if m else None


# --- Score bar chart ---
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=220)
x = np.arange(len(ATTEMPTS))
flash_scores = [score_of(f_sum, k) or 0 for k, _ in ATTEMPTS]
pro_scores = [score_of(p_sum, k) or 0 for k, _ in ATTEMPTS]
ax.bar(x - 0.2, flash_scores, 0.4, label="Nano Banana 2 (Flash)", color="#4c72b0")
ax.bar(x + 0.2, pro_scores, 0.4, label="Nano Banana Pro", color="#c44e52")
for i, (f, p) in enumerate(zip(flash_scores, pro_scores)):
    ax.text(i - 0.2, f + 0.1, str(f), ha="center", fontsize=9)
    ax.text(i + 0.2, p + 0.1, str(p), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([t for _, t in ATTEMPTS], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Gemini-judge thermal-physics score (1-10)")
ax.set_title("Nano Banana Pro vs Flash on 7 thermal-generation attempts")
ax.set_ylim(0, 10)
ax.legend(loc="lower left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FLASH / "pro_vs_flash_scores.png", bbox_inches="tight")
plt.close(fig)
print("wrote pro_vs_flash_scores.png")


# --- Side-by-side panels: for each attempt, real | flash | pro ---
def label(img, txt):
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img

real_path = FLASH / "source_thermal_real.png"
real = cv2.imread(str(real_path))

target_h = 320
rows = []
for k, t in ATTEMPTS:
    fpath = FLASH / f"{k}.png"
    ppath = PRO / f"{k}.png"
    if not (fpath.exists() and ppath.exists()):
        continue
    f_img = cv2.imread(str(fpath))
    p_img = cv2.imread(str(ppath))
    r_img = cv2.imread(str(real_path))

    def fit(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h))

    f_img = label(fit(f_img).copy(), f"Flash: {t}  ({score_of(f_sum, k)}/10)")
    p_img = label(fit(p_img).copy(), f"Pro:   {t}  ({score_of(p_sum, k)}/10)")
    r_img = label(fit(r_img).copy(), "REAL thermal (target)")

    max_w = max(f_img.shape[1], p_img.shape[1], r_img.shape[1])
    def pad(img):
        h, w = img.shape[:2]
        if w < max_w:
            return np.hstack([img, np.zeros((h, max_w - w, 3), np.uint8)])
        return img
    rows.append(np.hstack([pad(r_img), pad(f_img), pad(p_img)]))

# pad rows to the widest row's width
max_row_w = max(r.shape[1] for r in rows)
padded_rows = []
for r in rows:
    h, w, _ = r.shape
    if w < max_row_w:
        r = np.hstack([r, np.zeros((h, max_row_w - w, 3), np.uint8)])
    padded_rows.append(r)
grid = np.vstack(padded_rows)
cv2.imwrite(str(FLASH / "pro_vs_flash_grid.png"), grid)
print(f"wrote pro_vs_flash_grid.png  shape={grid.shape}")
