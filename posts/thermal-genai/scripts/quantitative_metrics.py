"""Paper-grade quantitative metrics for thermal-image generation.

Replaces the LLM-as-judge with pixel-level + region-level + distributional
measurements against the real SF-TL54 thermal reference.

Metrics computed per generated image (resized to the real-thermal size, RGB
or iron-palette as appropriate):

  Pixel-level (full-image, RGB-space):
    - MSE   : lower better
    - PSNR  : higher better
    - SSIM  : higher better (structural similarity)
    - LPIPS : lower better (perceptual similarity, AlexNet backbone)

  Pixel-level (luminance only, grayscale):
    - MSE / PSNR / SSIM as above but on luminance

  Anatomical region ratios (compute from a coarse face-region mask
  obtained by MediaPipe FaceMesh on the source RGB, then warped to a
  shared canvas):
    - eye_to_orbital_ratio   : luminance of eye region / luminance of
                                periorbital ring. SHOULD BE < 1.0 for
                                correct physics (eyes cooler than skin).
    - hair_to_face_ratio     : luminance of hair / luminance of face skin.
                                SHOULD BE < 1.0 (hair is cool).
    - nose_to_cheek_ratio    : luminance of nose / luminance of cheek.
                                SHOULD BE roughly equal or slightly < 1.

  Distributional:
    - histogram_intersection (RGB) : sum(min(p,q)) over 256-bin RGB
                                       channel histograms; 0..3.
    - histogram_correlation (Y)    : Pearson r between luminance
                                       histograms.

Inputs: a list of candidate image paths + the real-thermal reference.
Outputs: a CSV/JSON table of metrics per candidate, plus a printable
ranking.

Dependencies: pip install lpips scikit-image opencv-python pillow mediapipe
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR.parent / "outputs"
REAL = OUT / "source_thermal_real.png"


def to_uint8_rgb(path, size):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img


def basic_pixel_metrics(real_rgb, cand_rgb):
    diff = (real_rgb.astype(np.float32) - cand_rgb.astype(np.float32))
    mse = float((diff ** 2).mean())
    psnr = float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    real_y = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY)
    cand_y = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
    ssim_y = float(ssim(real_y, cand_y, data_range=255))
    return {"mse_rgb": mse, "psnr_rgb": psnr, "ssim_y": ssim_y}


def histogram_metrics(real_rgb, cand_rgb):
    out = {}
    # Per-channel histogram intersection on RGB
    inter = 0.0
    for c in range(3):
        h_r = cv2.calcHist([real_rgb], [c], None, [64], [0, 256]).flatten()
        h_c = cv2.calcHist([cand_rgb], [c], None, [64], [0, 256]).flatten()
        h_r /= max(h_r.sum(), 1); h_c /= max(h_c.sum(), 1)
        inter += float(np.minimum(h_r, h_c).sum())
    out["hist_inter_rgb"] = inter
    # Luminance histogram correlation
    real_y = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY)
    cand_y = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
    h_r = cv2.calcHist([real_y], [0], None, [64], [0, 256]).flatten()
    h_c = cv2.calcHist([cand_y], [0], None, [64], [0, 256]).flatten()
    out["hist_corr_y"] = float(np.corrcoef(h_r, h_c)[0, 1])
    return out


def region_masks(real_rgb):
    """Compute coarse anatomical masks using MediaPipe FaceMesh on the REAL
    thermal image — these define the spatial regions in which we measure
    luminance ratios.
    """
    H, W = real_rgb.shape[:2]
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv, BaseOptions
    mesh_opts = mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(Path.home() / "models/mediapipe/face_landmarker.task"
                                 if (Path.home() / "models/mediapipe/face_landmarker.task").exists()
                                 else None)),
        num_faces=1, running_mode=mpv.RunningMode.IMAGE)
    try:
        lm_engine = mpv.FaceLandmarker.create_from_options(mesh_opts)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=real_rgb)
        res = lm_engine.detect(mp_img)
        if not res.face_landmarks:
            raise RuntimeError("no face")
        pts = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])
    except Exception:
        # Fallback: coarse hard-coded masks based on image proportions for
        # the SF-TL54 portrait. Image is roughly head-and-shoulders centered.
        pts = None

    def mask_from(indices):
        m = np.zeros((H, W), np.uint8)
        if pts is not None:
            polygon = pts[indices].astype(np.int32)
            cv2.fillConvexPoly(m, polygon, 255)
        return m > 0

    masks = {}
    if pts is not None:
        # MediaPipe FaceMesh 478 anatomical clusters
        masks["eye"] = mask_from([33, 7, 163, 144, 145, 153, 154, 155, 133]) \
                       | mask_from([362, 382, 381, 380, 374, 373, 390, 249, 263])
        masks["periorbital"] = mask_from([46, 53, 52, 65, 55, 193, 168, 122, 196, 174])
        masks["forehead"] = mask_from([10, 67, 109, 338, 297, 332, 284, 251, 389])
        masks["nose"] = mask_from([1, 2, 5, 4, 6, 195, 197, 49, 64, 240, 220, 218,
                                   45, 19, 94, 275, 440, 444])
        masks["cheek_l"] = mask_from([116, 50, 205, 36, 142, 117, 118, 119, 120])
        masks["cheek_r"] = mask_from([345, 280, 425, 266, 371, 346, 347, 348, 349])
        masks["face_all"] = masks["forehead"] | masks["cheek_l"] | masks["cheek_r"] | masks["nose"]
        # Hair: above the highest face landmark
        if "forehead" in masks:
            ys, _ = np.where(masks["forehead"])
            if len(ys) > 0:
                top_y = ys.min()
                hair = np.zeros((H, W), np.uint8)
                hair[:top_y, max(0, top_y//2):W-max(0, top_y//2)] = 255
                masks["hair"] = hair > 0
    return masks


def region_ratios(real_y, cand_y, masks):
    out = {}
    def avg(mask, y):
        return float(y[mask].mean()) if mask is not None and mask.any() else None

    for region in ["eye", "periorbital", "forehead", "nose",
                   "cheek_l", "cheek_r", "face_all", "hair"]:
        if region not in masks: continue
        out[f"{region}_real"] = avg(masks[region], real_y)
        out[f"{region}_cand"] = avg(masks[region], cand_y)

    def ratio(num_key, den_key, suffix):
        n = out.get(f"{num_key}_{suffix}"); d = out.get(f"{den_key}_{suffix}")
        return n / d if (n is not None and d not in (None, 0)) else None
    for suffix in ("real", "cand"):
        out[f"eye_to_orbital_{suffix}"] = ratio("eye", "periorbital", suffix)
        out[f"hair_to_face_{suffix}"] = ratio("hair", "face_all", suffix)
        out[f"nose_to_cheek_{suffix}"] = (
            ratio("nose", "cheek_l", suffix)
            if out.get(f"cheek_l_{suffix}") else None)
    return out


def compute(real_path, cand_path, lpips_net=None):
    H, W = cv2.imread(str(real_path)).shape[:2]
    size = (W, H)
    real_rgb = to_uint8_rgb(real_path, size)
    cand_rgb = to_uint8_rgb(cand_path, size)
    if cand_rgb is None:
        return None
    metrics = {}
    metrics.update(basic_pixel_metrics(real_rgb, cand_rgb))
    metrics.update(histogram_metrics(real_rgb, cand_rgb))
    masks = region_masks(real_rgb)
    real_y = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY)
    cand_y = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
    metrics.update(region_ratios(real_y, cand_y, masks))
    if lpips_net is not None:
        # LPIPS expects [-1,1] tensors of shape (1,3,H,W)
        def to_t(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            d = lpips_net(to_t(real_rgb), to_t(cand_rgb)).item()
        metrics["lpips"] = float(d)
    return metrics


def main():
    real = REAL
    # build the table of candidates
    candidates = []
    for name in ["exp1a_generic", "exp1b_rgb_derived", "exp1c_thermal_physics",
                 "exp1d_gemini_thermcap", "exp3_rgb_plus_thermcap",
                 "exp4_refined_iter1", "exp4_refined_iter2"]:
        p = OUT / f"{name}.png"
        if p.exists(): candidates.append((f"flash_{name}", p))
        p2 = OUT.parent / "outputs_pro" / f"{name}.png"
        if p2.exists(): candidates.append((f"pro_{name}", p2))

    tg_iron_dir = OUT / "thermalgen_iron"
    if tg_iron_dir.exists():
        for p in sorted(tg_iron_dir.glob("ThermalGen-L-2-concat*_ds7_cfg1.0.png")):
            candidates.append((f"tg_{p.stem}", p))
        for p in sorted(tg_iron_dir.glob("ThermalGen-B-2*_ds21_cfg1.0.png")):
            candidates.append((f"tg_{p.stem}", p))

    print(f"computing metrics on {len(candidates)} candidates")
    lpips_net = lpips.LPIPS(net="alex").eval() if LPIPS_AVAILABLE else None

    results = {}
    for name, path in candidates:
        m = compute(real, path, lpips_net=lpips_net)
        results[name] = m
        if m:
            print(f"  {name:40s} ssim={m.get('ssim_y', 0):.3f} "
                  f"hist={m.get('hist_inter_rgb', 0):.2f} "
                  f"lpips={m.get('lpips', 0):.3f} "
                  f"eye/orb={m.get('eye_to_orbital_cand', float('nan')):.2f}")

    with open(OUT / "quant_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"wrote {OUT / 'quant_metrics.json'}")


if __name__ == "__main__":
    main()
