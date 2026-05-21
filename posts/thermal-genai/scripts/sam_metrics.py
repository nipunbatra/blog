"""Use SAM (transformers `facebook/sam-vit-base`) with click prompts derived
from MediaPipe FaceMesh landmarks to get clean anatomical masks on each
thermal image, then compute region-luminance metrics.

Pipeline:
  1. Run MediaPipe FaceMesh on the REAL thermal once to get a face anchor
     (just the centroid). For each candidate, also try FaceMesh to get
     image-specific anchors; fall back to the real-thermal anchors warped
     to the candidate's frame if FaceMesh fails on it.
  2. For each anatomical region (eye, hair, nose, cheek), pick a small set
     of click points (positive) inside the region + click points (negative)
     outside the region.
  3. Pass clicks to SAM → get a clean binary mask.
  4. Compute mean luminance per mask region per candidate.

Why SAM > MediaPipe-mask-alone:
  MediaPipe's eye landmark cluster on a *thermal* image can drift onto
  the eyebrow or the cheek. SAM, given a click at the rough eye centre,
  segments the actual eye-shaped low-luminance region the click sits in.
  This gives masks that conform to the real image content, not to RGB-
  trained landmark priors.
"""
import argparse, json, os
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import SamModel, SamProcessor

import mediapipe as mp
from mediapipe.tasks.python import vision as mpv, BaseOptions

# Allow overriding paths via env vars (for running on bhaskar without the blog repo)
ROOT = Path(os.environ.get("THERMAL_GENAI_ROOT",
                            str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs"
SAM_MODEL = "facebook/sam-vit-base"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MP_MODEL = Path.home() / "models/mediapipe/face_landmarker.task"


def make_lm():
    return mpv.FaceLandmarker.create_from_options(
        mpv.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MP_MODEL)),
            num_faces=1, running_mode=mpv.RunningMode.IMAGE))


def landmarks(image_pil, lm_engine):
    img = np.array(image_pil.convert("RGB"))
    H, W = img.shape[:2]
    res = lm_engine.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    if not res.face_landmarks: return None
    pts = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])
    return pts


def click_points_for_region(pts, region, image_shape):
    """Return (positive_clicks, negative_clicks) for the requested region."""
    H, W = image_shape[:2]
    # MediaPipe FaceMesh index clusters
    R = {
        "left_eye": [33, 133, 159, 145],
        "right_eye": [362, 263, 386, 374],
        "eye_both": [33, 133, 159, 145, 362, 263, 386, 374],
        "nose": [1, 5, 197],
        "forehead": [10, 67, 109, 338, 297],
        "cheek_left": [50, 205, 36, 142],
        "cheek_right": [280, 425, 266, 371],
        # Hair has no landmarks — define as point above the highest forehead pt
    }
    if region == "hair":
        # one click above the forehead, two negatives on the face skin
        if pts is None: return [[W//2, 10]], [[W//2, H//2], [W//4, H//2]]
        forehead = pts[R["forehead"]]
        top = forehead[:, 1].min()
        pos = [[float(forehead[:, 0].mean()), max(0, float(top - 25))]]
        return pos, [[float(pts[1, 0]), float(pts[1, 1])],  # nose tip = negative
                     [float(pts[152, 0]), float(pts[152, 1])]]  # chin = negative
    if pts is None: return [], []
    if region not in R: return [], []
    pos = pts[R[region]].tolist()
    # negative anchor = nose tip (idx 1) — usually opposite to most other regions
    if "nose" in region:
        # negative = forehead + chin
        neg = pts[[10, 152]].tolist()
    else:
        neg = pts[[1]].tolist()
    return pos, neg


def sam_mask(processor, model, image_pil, points, labels):
    """Pass click points to SAM, return best mask."""
    inputs = processor(image_pil, input_points=[[points]],
                       input_labels=[[labels]], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())
    # masks shape: list of (1, num_masks, H, W) tensors
    score = outputs.iou_scores.squeeze().tolist()
    best = int(np.argmax(score))
    m = masks[0][0][best].numpy().astype(np.uint8)
    return m, score


def process(name, image_path, sam_proc, sam_model, lm_engine):
    image_pil = Image.open(image_path).convert("RGB")
    pts = landmarks(image_pil, lm_engine)
    img_rgb = np.array(image_pil)
    H, W = img_rgb.shape[:2]
    y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    results = {}
    for region in ["eye_both", "hair", "nose", "forehead",
                    "cheek_left", "cheek_right"]:
        pos, neg = click_points_for_region(pts, region, img_rgb.shape)
        if not pos:
            results[region] = {"lum": None, "score": None}
            continue
        points = pos + neg
        labels = [1] * len(pos) + [0] * len(neg)
        m, scores = sam_mask(sam_proc, sam_model, image_pil, points, labels)
        if m.shape != y.shape:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        lum = float(y[m > 0].mean()) if m.any() else None
        results[region] = {"lum": lum, "score": max(scores), "area_px": int(m.sum())}
        # save mask overlay
        if name == "real_sftl54":
            overlay = img_rgb.copy()
            overlay[m > 0] = (0.4 * np.array([0, 255, 0]) + 0.6 * overlay[m > 0]).astype(np.uint8)
            (OUT / "sam_masks").mkdir(exist_ok=True)
            Image.fromarray(overlay).save(OUT / "sam_masks" / f"{name}__{region}.png")
    return results


def main():
    print(f"[init] device={DEVICE}, model={SAM_MODEL}")
    sam_proc = SamProcessor.from_pretrained(SAM_MODEL)
    sam_model = SamModel.from_pretrained(SAM_MODEL).to(DEVICE).eval()
    lm_engine = make_lm()

    candidates = [
        ("real_sftl54", OUT / "source_thermal_real.png"),
        ("flash_iter1", OUT / "exp4_refined_iter1.png"),
        ("pro_iter1", ROOT / "outputs_pro/exp4_refined_iter1.png"),
        ("pro_exp3", ROOT / "outputs_pro/exp3_rgb_plus_thermcap.png"),
        ("thermalgen_L_iron", OUT / "thermalgen_iron/ThermalGen-L-2-concat_ds7_cfg1.0.png"),
        ("thermalgen_L_gray", OUT / "thermalgen/ThermalGen-L-2-concat_ds7_cfg1.0.png"),
    ]
    results = {}
    for name, p in candidates:
        if not p.exists():
            print(f"missing {p}"); continue
        print(f"[seg] {name}")
        results[name] = process(name, p, sam_proc, sam_model, lm_engine)
        for r, info in results[name].items():
            print(f"   {r:14s}  lum={info['lum'] if info['lum'] else 'n/a'}  iou={info.get('score')}")
    # ratios
    real = results.get("real_sftl54", {})
    real_lums = {r: real[r]["lum"] for r in real if real[r].get("lum")}
    real_face = real_lums.get("forehead")
    real_ratios = {}
    if real_face:
        for r in ["eye_both", "hair", "nose"]:
            v = real_lums.get(r)
            if v: real_ratios[r] = v / real_face
    print(f"\nReal-thermal ratios (vs forehead): {real_ratios}")
    print()
    print(f"{'method':<22} {'eye/fh':>8} {'hair/fh':>9} {'nose/fh':>9} "
          f"{'eye dev':>9} {'hair dev':>10} {'nose dev':>10}")
    for name, info in results.items():
        face = info.get("forehead", {}).get("lum")
        row = {}
        for r in ["eye_both", "hair", "nose"]:
            v = info.get(r, {}).get("lum")
            if face and v: row[r] = v / face
        def fmt(v, n=3):
            return f"{v:>.{n}f}" if v is not None else "n/a"
        def dev(k):
            c = row.get(k); ref = real_ratios.get(k)
            if c is None or ref is None: return None
            return abs(c - ref) / max(ref, 1e-6)
        print(f"{name:<22} {fmt(row.get('eye_both')):>8} "
              f"{fmt(row.get('hair')):>9} {fmt(row.get('nose')):>9} "
              f"{fmt(dev('eye_both'), 2):>9} "
              f"{fmt(dev('hair'), 2):>10} "
              f"{fmt(dev('nose'), 2):>10}")

    with open(OUT / "sam_metrics.json", "w") as f:
        json.dump({"results": results, "real_ratios": real_ratios},
                  f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nwrote {OUT / 'sam_metrics.json'}")


if __name__ == "__main__":
    main()
