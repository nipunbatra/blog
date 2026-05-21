"""Build the side-by-side: real thermal | Gemini Pro best | ThermalGen best."""
import json, re
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

real = cv2.imread(str(OUT / "source_thermal_real.png"))
gem_flash_best = cv2.imread(str(OUT / "exp4_refined_iter1.png"))     # Flash best 7/10
gem_pro_best = cv2.imread(str(ROOT / "outputs_pro/exp4_refined_iter1.png"))  # Pro best 7/10
tg_iron = cv2.imread(str(OUT / "thermalgen_iron/ThermalGen-L-2-concat_ds7_cfg1.0.png"))
tg_gray = cv2.imread(str(OUT / "thermalgen/ThermalGen-L-2-concat_ds7_cfg1.0.png"))


def fit(img, h=320):
    if img is None: return None
    H, W = img.shape[:2]
    return cv2.resize(img, (int(W * h / H), h))


def label(img, txt):
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


panels = [
    label(fit(real).copy(), "REAL thermal (SF-TL54)"),
    label(fit(gem_flash_best).copy(), "Gemini Flash best (7/10 judge)"),
    label(fit(gem_pro_best).copy(), "Gemini Pro best (7/10 judge)"),
    label(fit(tg_iron).copy(), "ThermalGen-L iron (4/10 judge!)"),
    label(fit(tg_gray).copy(), "ThermalGen-L native grayscale"),
]
# pad widths
max_w = max(p.shape[1] for p in panels)
padded = []
for p in panels:
    h, w, _ = p.shape
    if w < max_w:
        p = np.hstack([p, np.zeros((h, max_w - w, 3), np.uint8)])
    padded.append(p)
grid = np.hstack(padded)
cv2.imwrite(str(OUT / "thermalgen_vs_gemini.png"), grid)
print(f"wrote thermalgen_vs_gemini.png  {grid.shape}")
