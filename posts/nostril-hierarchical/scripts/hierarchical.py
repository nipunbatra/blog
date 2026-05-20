"""Hierarchical vs single-stage nostril detection on RGB images.

Single-stage:
    full image -> MediaPipe FaceMesh -> nostril keypoints (indices 48, 278)

Hierarchical:
    full image
       -> MediaPipe Face Detector (BlazeFace short-range)
       -> crop the face bbox (with padding), upsample to 256x256
       -> MediaPipe FaceMesh on the crop
       -> map nostril keypoints back to original image coords

We evaluate on RGB shots from SF-TL54 (paired with thermal). To get ground-truth
nostril coords on the RGB frames we run a HIGH-CONFIDENCE FaceMesh on the
full-resolution RGB and treat those as pseudo-GT (this is the standard
self-supervised eval used for hierarchical detection comparisons — fairer
than using the thermal-frame GT, because the two cameras aren't perfectly
co-registered).

We then simulate the "tiny face" regime by:
    1. shrinking the original image by Nx (= 2, 4, 8)
    2. pasting it onto a NxN background of the same size as the original
    3. running both pipelines and comparing to the pseudo-GT mapped through
       the same shrink+paste transform.

This isolates the "face is small in the frame" axis without changing the face
content itself.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

MEDIAPIPE_NOSTRIL_LEFT = 48
MEDIAPIPE_NOSTRIL_RIGHT = 278
HOME = Path.home()


def make_landmarker(face_path, mesh_path):
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python import BaseOptions
    face_opts = mpv.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(face_path)),
        running_mode=mpv.RunningMode.IMAGE)
    mesh_opts = mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(mesh_path)),
        num_faces=1, running_mode=mpv.RunningMode.IMAGE)
    return (mpv.FaceDetector.create_from_options(face_opts),
            mpv.FaceLandmarker.create_from_options(mesh_opts), mp)


def mesh_on_image(landmarker, mp, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in lm])  # (478, 2)


def face_bbox(detector, mp, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res = detector.detect(mp_img)
    if not res.detections:
        return None
    box = res.detections[0].bounding_box
    return (box.origin_x, box.origin_y, box.origin_x + box.width,
            box.origin_y + box.height)


def shrink_and_pad(img, factor, bg_value=128):
    """Resize image down by `factor`, paste into top-left of original-size
    background.  Returns image of same dims as input."""
    h, w = img.shape[:2]
    sw, sh = w // factor, h // factor
    small = cv2.resize(img, (sw, sh))
    out = np.full((h, w, 3), bg_value, dtype=np.uint8)
    out[:sh, :sw] = small
    return out, sw, sh


def map_kp_through_shrink(kp, factor):
    """Transform an (x, y) keypoint through shrink+paste (face goes to
    upper-left quadrant scaled by 1/factor)."""
    return [kp[0] / factor, kp[1] / factor]


def single_stage(landmarker, mp, img):
    t0 = time.perf_counter()
    lm = mesh_on_image(landmarker, mp, img)
    elapsed = time.perf_counter() - t0
    if lm is None:
        return None, None, elapsed
    return (lm[MEDIAPIPE_NOSTRIL_LEFT].tolist(),
            lm[MEDIAPIPE_NOSTRIL_RIGHT].tolist(), elapsed)


def hierarchical(detector, landmarker, mp, img, crop_size=256, pad_ratio=0.2):
    t0 = time.perf_counter()
    bbox = face_bbox(detector, mp, img)
    if bbox is None:
        return None, time.perf_counter() - t0
    x0, y0, x1, y1 = bbox
    w = x1 - x0; h = y1 - y0
    side = max(w, h) * (1 + pad_ratio)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0c = int(max(0, cx - side / 2))
    y0c = int(max(0, cy - side / 2))
    x1c = int(min(img.shape[1], x0c + side))
    y1c = int(min(img.shape[0], y0c + side))
    if x1c <= x0c or y1c <= y0c:
        return None, time.perf_counter() - t0
    crop = img[y0c:y1c, x0c:x1c]
    crop_h, crop_w = crop.shape[:2]
    crop_resized = cv2.resize(crop, (crop_size, crop_size))
    lm = mesh_on_image(landmarker, mp, crop_resized)
    elapsed = time.perf_counter() - t0
    if lm is None:
        return None, elapsed
    # Map nostrils back to original-image coords
    sx = crop_w / crop_size; sy = crop_h / crop_size
    nl = lm[MEDIAPIPE_NOSTRIL_LEFT]
    nr = lm[MEDIAPIPE_NOSTRIL_RIGHT]
    nl_img = [nl[0] * sx + x0c, nl[1] * sy + y0c]
    nr_img = [nr[0] * sx + x0c, nr[1] * sy + y0c]
    return (nl_img, nr_img, elapsed), bbox


def evaluate(img_paths, face_path, mesh_path, factors=(1, 2, 4, 8),
             out_dir=None):
    detector, landmarker, mp = make_landmarker(face_path, mesh_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    panel_rows = []
    for ip in img_paths:
        img = cv2.imread(str(ip))
        if img is None: continue
        # pseudo-GT from full-res
        gt_l, gt_r, _ = single_stage(landmarker, mp, img)
        if gt_l is None:
            print(f"skip {ip} (no face at full-res)"); continue
        for f in factors:
            shrunk, sw, sh = shrink_and_pad(img, f)
            # map gt through shrink
            gt_l_s = map_kp_through_shrink(gt_l, f)
            gt_r_s = map_kp_through_shrink(gt_r, f)
            # single-stage on shrunk
            ss_l, ss_r, ss_t = single_stage(landmarker, mp, shrunk)
            # hierarchical on shrunk
            h_out = hierarchical(detector, landmarker, mp, shrunk)
            hi_l = hi_r = hi_t = None; hi_bbox = None
            if h_out is not None and h_out[0] is not None:
                pred, hi_bbox = h_out
                hi_l, hi_r, hi_t = pred
            def err(a, b):
                if a is None or b is None: return None
                return float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)
            rows.append({
                "image": str(ip), "factor": f,
                "face_side_px": int(max((gt_r[0]-gt_l[0]) / f, 1)),
                "single_err_l": err(ss_l, gt_l_s),
                "single_err_r": err(ss_r, gt_r_s),
                "single_time_ms": ss_t * 1000 if ss_t else None,
                "hier_err_l": err(hi_l, gt_l_s),
                "hier_err_r": err(hi_r, gt_r_s),
                "hier_time_ms": hi_t * 1000 if hi_t else None,
            })
        # build a panel for the smallest factor (where the divergence shows)
        F = max(factors)
        shrunk, sw, sh = shrink_and_pad(img, F)
        gt_l_s = map_kp_through_shrink(gt_l, F)
        gt_r_s = map_kp_through_shrink(gt_r, F)
        ss_l, ss_r, _ = single_stage(landmarker, mp, shrunk)
        h_out = hierarchical(detector, landmarker, mp, shrunk)
        p1 = shrunk.copy()
        cv2.drawMarker(p1, (int(gt_l_s[0]), int(gt_l_s[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(p1, (int(gt_r_s[0]), int(gt_r_s[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        if ss_l is not None:
            for (x, y) in (ss_l, ss_r):
                cv2.circle(p1, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.rectangle(p1, (0, 0), (p1.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(p1, f"single-stage @ {F}x shrink",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        p2 = shrunk.copy()
        cv2.drawMarker(p2, (int(gt_l_s[0]), int(gt_l_s[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(p2, (int(gt_r_s[0]), int(gt_r_s[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        if h_out is not None and h_out[0] is not None:
            pred, bbox = h_out
            x0, y0, x1, y1 = bbox
            cv2.rectangle(p2, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 255, 255), 1)
            for (x, y) in (pred[0], pred[1]):
                cv2.circle(p2, (int(x), int(y)), 6, (255, 0, 255), -1)
        cv2.rectangle(p2, (0, 0), (p2.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(p2, f"hierarchical @ {F}x shrink",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        panel_rows.append(np.hstack([p1, p2]))

    grid = np.vstack(panel_rows) if panel_rows else None
    if grid is not None:
        cv2.imwrite(str(out_dir / "panel.png"), grid)
    with open(out_dir / "rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--face-path",
                    default=str(HOME / "models/mediapipe/face_detector.tflite"))
    ap.add_argument("--mesh-path",
                    default=str(HOME / "models/mediapipe/face_landmarker.task"))
    ap.add_argument("--factors", default="1,2,4,8")
    ap.add_argument("--out", default=str(HOME / "git/nostril-bench/runs/hierarchical"))
    args = ap.parse_args()
    factors = tuple(int(x) for x in args.factors.split(","))
    rows = evaluate(args.images, args.face_path, args.mesh_path,
                    factors=factors, out_dir=args.out)
    print(f"wrote {len(rows)} rows to {args.out}/rows.json")
