"""Composite task: surface normals + body-part segmentation -> Lambertian relighting.

The point of the demo is that the *outputs of two Sapiens2 heads compose into a
useful downstream task*. We use the seg mask to confine relighting to the
foreground, and the normals to compute Lambertian shading L = max(0, n · l).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent


def lambertian_shade(normal_chw: np.ndarray, light_dir: np.ndarray,
                     ambient: float = 0.25) -> np.ndarray:
    """normal: (3,H,W) unit; light_dir: (3,); returns (H,W) shading in [0, 1]."""
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
    nx, ny, nz = normal_chw
    # Sapiens2 normals: x right, y down, z toward camera (-z away).
    # We treat the light as coming *toward* the surface; standard Lambertian.
    diff = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    diff = np.clip(diff, 0.0, 1.0)
    return ambient + (1.0 - ambient) * diff


def relight(img_bgr: np.ndarray, normal_chw: np.ndarray, mask: np.ndarray,
            light_dir: np.ndarray, tint=(245, 230, 215)) -> np.ndarray:
    """Multiplicative relight on the foreground; identity on background."""
    h, w = img_bgr.shape[:2]
    if normal_chw.shape[1:] != (h, w):
        # bilinear-resize each channel
        normal_chw = np.stack([
            cv2.resize(normal_chw[i], (w, h), interpolation=cv2.INTER_LINEAR)
            for i in range(3)
        ], axis=0)
        norm = np.linalg.norm(normal_chw, axis=0, keepdims=True)
        normal_chw = normal_chw / np.clip(norm, 1e-6, None)
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    shade = lambertian_shade(normal_chw, light_dir)              # (H, W) in [0, 1]
    tint_bgr = np.array(tint[::-1], dtype=np.float32) / 255.0   # OpenCV is BGR

    out = img_bgr.astype(np.float32) / 255.0
    relit = out * (shade[..., None] * tint_bgr * 1.6)            # warm tint, bright
    out = np.where(mask[..., None], relit, out)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def panel(images, labels, max_w=2400):
    """Stack images horizontally with captions underneath."""
    h = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        scale = h / im.shape[0]
        w = int(round(im.shape[1] * scale))
        resized.append(cv2.resize(im, (w, h)))
    strip = np.concatenate(resized, axis=1)
    if strip.shape[1] > max_w:
        scale = max_w / strip.shape[1]
        strip = cv2.resize(strip, (max_w, int(strip.shape[0] * scale)))
    # caption row
    caption = np.full((44, strip.shape[1], 3), 255, np.uint8)
    cell_w = strip.shape[1] // len(images)
    for i, lab in enumerate(labels):
        cv2.putText(caption, lab, (i * cell_w + 12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return np.concatenate([strip, caption], axis=0)


def main(stem: str = "person1"):
    out = POST_DIR / "outputs"
    img = cv2.imread(str(POST_DIR / f"inputs/{stem}.jpg"))
    normal = np.load(out / f"{stem}_normal.npy")              # (3, H, W)
    seg = np.load(out / f"{stem}_seg.npy")                    # (H, W) int
    foreground = seg > 0                                      # everything that is a body part

    # Encode the normal map as RGB for display
    normal_rgb = ((normal.transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)

    # Encode mask as overlay
    mask_vis = img.copy()
    mask_vis[~foreground] = (mask_vis[~foreground] * 0.25).astype(np.uint8)

    # Three relit variants under different light directions
    lights = {
        "key (front-right)": np.array([0.6,  -0.2, 1.0]),
        "key (rim, top-left)": np.array([-0.7, -0.7, 0.4]),
        "key (under-light)":  np.array([0.0,  0.9, 0.4]),
    }
    relit = []
    for name, L in lights.items():
        relit.append(relight(img, normal, foreground, L))

    strip1 = panel([img, normal_bgr, mask_vis],
                   ["original", "Sapiens2 normals", "Sapiens2 seg (fg in colour)"])
    strip2 = panel([img] + relit,
                   ["original"] + list(lights.keys()))
    cv2.imwrite(str(out / f"{stem}_pipeline.jpg"), strip1)
    cv2.imwrite(str(out / f"{stem}_relit.jpg"), strip2)
    print(f"[done] {out / f'{stem}_pipeline.jpg'}")
    print(f"[done] {out / f'{stem}_relit.jpg'}")


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else "person1"
    main(stem)
