"""More downstream applications composed from Sapiens2 normals + body-part seg.

All of these reuse the per-image cache produced by `run_normal.py` and
`run_seg.py` — no new model calls.

Apps:
  - background_removal (alpha matte from seg.fg)
  - hair_recolour      (HSV hue swap on the hair seg class)
  - skin_smooth        (selective bilateral blur on Face_Neck class)
  - toon_shade         (quantised Lambertian shading from normals + mask)
  - normal_outline     (edge map from per-channel normal gradients)
  - face_swap_tint     (clothing-tint preserving face/skin colour)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent
OUT = POST_DIR / "outputs"

# Class IDs from sapiens.dense.datasets.seg_utils.DOME_CLASSES_29
CLS = {
    "Background": 0, "Apparel": 1, "Eyeglass": 2, "Face_Neck": 3, "Hair": 4,
    "Left_Foot": 5, "Left_Hand": 6, "Left_Lower_Arm": 7, "Left_Lower_Leg": 8,
    "Left_Shoe": 9, "Left_Sock": 10, "Left_Upper_Arm": 11, "Left_Upper_Leg": 12,
    "Lower_Clothing": 13, "Right_Foot": 14, "Right_Hand": 15,
    "Right_Lower_Arm": 16, "Right_Lower_Leg": 17, "Right_Shoe": 18,
    "Right_Sock": 19, "Right_Upper_Arm": 20, "Right_Upper_Leg": 21,
    "Torso": 22, "Upper_Clothing": 23, "Lower_Lip": 24, "Upper_Lip": 25,
    "Lower_Teeth": 26, "Upper_Teeth": 27, "Tongue": 28,
}


def load_cache(stem: str):
    img = cv2.imread(str(POST_DIR / f"inputs/{stem}.jpg"))
    normal = np.load(OUT / f"{stem}_normal.npy")  # (3, H, W)
    seg = np.load(OUT / f"{stem}_seg.npy")        # (H, W) int
    h, w = img.shape[:2]
    if normal.shape[1:] != (h, w):
        normal = np.stack([cv2.resize(normal[i], (w, h),
                                       interpolation=cv2.INTER_LINEAR)
                           for i in range(3)], axis=0)
        normal /= np.clip(np.linalg.norm(normal, axis=0, keepdims=True), 1e-6, None)
    if seg.shape != (h, w):
        seg = cv2.resize(seg.astype(np.int16), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(np.int16)
    return img, normal, seg


# ----------------------------------------------------------------------
# 1. Background removal — clean alpha from seg
def background_removal(img, seg):
    fg = (seg > 0).astype(np.uint8) * 255
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
    fg = cv2.GaussianBlur(fg, (5, 5), 0)
    bgra = np.dstack([img, fg])
    # Composite onto a chequerboard so the matte is visible in JPG
    h, w = img.shape[:2]
    sq = 24
    yy, xx = np.indices((h, w))
    chk = (((yy // sq) + (xx // sq)) % 2).astype(np.float32)
    chk = (180 + 35 * chk)[..., None].astype(np.float32)
    chk_bgr = np.repeat(chk, 3, axis=2).astype(np.uint8)
    a = (fg.astype(np.float32) / 255.0)[..., None]
    composite = (img.astype(np.float32) * a +
                 chk_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)
    return composite, bgra


# ----------------------------------------------------------------------
# 2. Hair recolour — HSV hue rotation only on the Hair class
def hair_recolour(img, seg, target_hue: int = 145):
    """target_hue in OpenCV HSV (0..180). 145 ≈ deep magenta/violet."""
    hair = (seg == CLS["Hair"])
    if not hair.any():
        return img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    # rotate hue so the median hair hue moves to target
    cur = int(np.median(hsv[..., 0][hair]))
    delta = (target_hue - cur) % 180
    hsv[..., 0] = np.where(hair, (hsv[..., 0] + delta) % 180, hsv[..., 0])
    # boost saturation a bit on hair so the recolour reads
    hsv[..., 1] = np.where(hair, np.clip(hsv[..., 1] * 1.4, 0, 255).astype(np.int32),
                           hsv[..., 1])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ----------------------------------------------------------------------
# 3. Skin smoothing — bilateral filter on Face_Neck only
def skin_smooth(img, seg, strength: int = 35):
    face = (seg == CLS["Face_Neck"]).astype(np.uint8)
    if not face.any():
        return img.copy()
    smoothed = cv2.bilateralFilter(img, d=15, sigmaColor=strength, sigmaSpace=strength)
    a = cv2.GaussianBlur(face.astype(np.float32) * 1.0, (9, 9), 0)[..., None]
    out = img.astype(np.float32) * (1 - a) + smoothed.astype(np.float32) * a
    return out.astype(np.uint8)


# ----------------------------------------------------------------------
# 4. Toon-shade — quantise Lambertian shading from normals + mask
def toon_shade(img, normal, seg, light=(0.5, -0.4, 1.0), levels: int = 4):
    L = np.array(light, dtype=np.float32)
    L /= np.linalg.norm(L) + 1e-8
    nx, ny, nz = normal
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
    bands = np.linspace(0.20, 1.0, levels)
    quant = bands[np.minimum((diff * levels).astype(np.int32), levels - 1)]

    fg = (seg > 0)
    base = img.astype(np.float32) / 255.0
    flat = np.zeros_like(base)
    # Per-class flat colour: pick the median colour of pixels in the class
    for cid in np.unique(seg):
        if cid == 0:
            continue
        m = (seg == cid)
        if m.sum() < 50:
            continue
        med = np.median(base[m], axis=0)
        flat[m] = med
    shade_rgb = flat * quant[..., None]

    # Outline: edges in the seg label map
    edges = cv2.Canny(seg.astype(np.uint8), 0, 1)
    edges_bg = cv2.Canny(((seg > 0).astype(np.uint8) * 255), 50, 150)
    outline = np.maximum(edges, edges_bg)

    out = np.where(fg[..., None], shade_rgb, base)
    out = (out * 255).astype(np.uint8)
    out[outline > 0] = (20, 20, 20)
    return out


# ----------------------------------------------------------------------
# 5. Outline drawing from surface normals (curvature edges)
def normal_outline(normal):
    """High response where the surface normal direction changes fast."""
    nx, ny, nz = normal
    gx = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3) ** 2 \
       + cv2.Sobel(ny, cv2.CV_32F, 1, 0, ksize=3) ** 2 \
       + cv2.Sobel(nz, cv2.CV_32F, 1, 0, ksize=3) ** 2
    gy = cv2.Sobel(nx, cv2.CV_32F, 0, 1, ksize=3) ** 2 \
       + cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3) ** 2 \
       + cv2.Sobel(nz, cv2.CV_32F, 0, 1, ksize=3) ** 2
    mag = np.sqrt(gx + gy)
    mag = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
    inv = 255 - mag
    return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------
def panel(images, labels, max_w=2400):
    h = max(im.shape[0] for im in images)
    rs = [cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h)) for im in images]
    strip = np.concatenate(rs, axis=1)
    if strip.shape[1] > max_w:
        s = max_w / strip.shape[1]
        strip = cv2.resize(strip, (max_w, int(strip.shape[0] * s)))
    cap = np.full((44, strip.shape[1], 3), 255, np.uint8)
    cw = strip.shape[1] // len(images)
    for i, lab in enumerate(labels):
        cv2.putText(cap, lab, (i * cw + 12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return np.concatenate([strip, cap], axis=0)


def main(stem: str):
    img, normal, seg = load_cache(stem)

    bg_comp, bgra = background_removal(img, seg)
    cv2.imwrite(str(OUT / f"{stem}_bgremove.jpg"), bg_comp)
    cv2.imwrite(str(OUT / f"{stem}_alpha.png"), bgra)

    recoloured = [hair_recolour(img, seg, h) for h in (145, 90, 15, 60)]
    strip = panel([img] + recoloured,
                  ["original", "magenta hue 145", "green 90",
                   "warm orange 15", "yellow 60"])
    cv2.imwrite(str(OUT / f"{stem}_hair_recolour.jpg"), strip)

    smoothed = skin_smooth(img, seg, strength=45)
    cv2.imwrite(str(OUT / f"{stem}_skin_smooth.jpg"),
                panel([img, smoothed],
                      ["original", "Face_Neck-only bilateral smooth"]))

    toon = toon_shade(img, normal, seg, levels=4)
    outline = normal_outline(normal)
    cv2.imwrite(str(OUT / f"{stem}_toon.jpg"),
                panel([img, toon, outline],
                      ["original", "toon (4-level Lambertian on flat seg colour)",
                       "outline from normal-gradient magnitude"]))
    print(f"[done] apps for {stem} written to {OUT}")


if __name__ == "__main__":
    stems = sys.argv[1:] or ["person1", "person2"]
    for s in stems:
        main(s)
