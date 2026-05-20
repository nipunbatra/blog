"""Run four pose / face-landmark models on SF-TL54 thermal + RGB images.

Outputs per-image, per-model JSON with the predicted nose-region keypoints and
a side-by-side annotated PNG. The four models:

  1. Sapiens2-pose-0.4b   (Meta, 308 wholebody kpts)
  2. ViTPose+ base        (HF transformers, 133 COCO-WholeBody kpts)
  3. DWPose (RTMPose-x)   (rtmlib ONNX, 133 COCO-WholeBody kpts)
  4. MediaPipe FaceMesh   (Google, 478 face kpts)

For thermal images (1 channel) we replicate to 3 channels before model input.

Saves to /home/nipun.batra/git/nostril-bench/runs/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.modules["mmpretrain"] = None  # avoid sapiens import chain

ROOT = Path.home() / "git/nostril-bench"
RUNS_DIR = ROOT / "runs"
SAPIENS_REPO = Path.home() / "git/sapiens2"
MODELS = Path.home() / "models"
DATA = Path.home() / "data/SFTL54/sftl/sftl"

# Sapiens2 308-kpt indices for the nose tip (COCO-WholeBody body+face mapping).
# In the 308-kpt set: 0-16 = body, 17-22 = feet, 23-90 = face (68-pt), then
# hands. Face kpts 23-90 use dlib-style 68 indices, so nose-bottom is 31..35
# inside the face block — absolute indices 54..58 in the 308 set.
SAPIENS_NOSE_TIP_RANGE = list(range(54, 59))  # 5 nose-bottom kpts
SAPIENS_NOSTRIL_LEFT = 55   # dlib 32 -> 55
SAPIENS_NOSTRIL_RIGHT = 58  # dlib 35 -> 58

# COCO-WholeBody 133-kpt set used by ViTPose+ and DWPose:
# 0-16 body, 17-22 feet, 23-90 face (dlib 68), 91-132 hands.
COCOWB_NOSE_TIP_RANGE = list(range(54, 59))
COCOWB_NOSTRIL_LEFT = 55
COCOWB_NOSTRIL_RIGHT = 58

# MediaPipe FaceMesh 478 indices: nostril alae centre are 48 (left) and 278 (right).
# Tip = 4. We'll report all three.
MEDIAPIPE_NOSE_TIP = 4
MEDIAPIPE_NOSTRIL_LEFT = 48
MEDIAPIPE_NOSTRIL_RIGHT = 278


# ---------------------------------------------------------------------------
# Sapiens2
# ---------------------------------------------------------------------------
def run_sapiens(image_bgr, size="0.4b", device="cuda:0"):
    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo

    cfg = (f"{SAPIENS_REPO}/sapiens/pose/configs/keypoints308/"
           f"shutterstock_goliath_3po/sapiens2_{size}_keypoints308_"
           f"shutterstock_goliath_3po-1024x768.py")
    ckpt = f"{MODELS}/sapiens2_pose_{size}/sapiens2_{size}_pose.safetensors"

    model = init_model(cfg, ckpt, device=device)
    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file=f"{SAPIENS_REPO}/sapiens/pose/configs/_base_/keypoints308.py"))
    codec_cfg = dict(model.cfg.codec); codec_cfg.pop("type")
    model.codec = UDPHeatmap(**codec_cfg)

    H, W = image_bgr.shape[:2]
    bbox = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)
    data_info = dict(img=image_bgr, bbox=bbox,
                     bbox_score=np.ones(1, dtype=np.float32))
    data = model.pipeline(data_info)
    data = model.data_preprocessor(data)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(data["inputs"]).cpu().numpy()
    elapsed = time.perf_counter() - t0
    keypoints, scores = model.codec.decode(pred[0])
    input_size = data["data_samples"]["meta"]["input_size"]
    bc = data["data_samples"]["meta"]["bbox_center"]
    bs = data["data_samples"]["meta"]["bbox_scale"]
    keypoints = keypoints / input_size * bs + bc - 0.5 * bs
    return {
        "kp": keypoints[0].tolist(),         # 308x2
        "score": scores[0].tolist(),
        "elapsed_s": elapsed,
        "n_above_thresh": int((scores[0] > 0.3).sum()),
    }


# ---------------------------------------------------------------------------
# ViTPose+
# ---------------------------------------------------------------------------
def run_vitpose(image_bgr, device="cuda:0"):
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    model_path = f"{MODELS}/vitpose/vitpose-plus-base"
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = VitPoseForPoseEstimation.from_pretrained(model_path).to(device).eval()

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = [[[0, 0, W, H]]]  # one box covering image; one image -> [[box]]
    inputs = processor(image_rgb, boxes=boxes, return_tensors="pt").to(device)
    # ViTPose+ MoE indices: 0=COCO, 1=AiC, 2=MPII, 3=AP-10K, 4=APT-36K, 5=COCO-WholeBody
    inputs["dataset_index"] = torch.tensor([5], device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.perf_counter() - t0
    results = processor.post_process_pose_estimation(outputs, boxes=boxes)
    poses = results[0]
    if not poses:
        return {"kp": [], "score": [], "elapsed_s": elapsed, "n_above_thresh": 0}
    pred = poses[0]
    kp = pred["keypoints"].cpu().numpy()      # (K,2)
    scores = pred["scores"].cpu().numpy()     # (K,)
    return {
        "kp": kp.tolist(),
        "score": scores.tolist(),
        "elapsed_s": elapsed,
        "n_above_thresh": int((scores > 0.3).sum()),
    }


# ---------------------------------------------------------------------------
# DWPose / RTMPose wholebody (rtmlib ONNX)
# ---------------------------------------------------------------------------
def run_dwpose(image_bgr):
    from rtmlib import Wholebody

    # rtmlib auto-downloads weights to ~/.cache/rtmlib on first call
    wb = Wholebody(to_openpose=False, mode="balanced",
                   backend="onnxruntime", device="cuda")
    t0 = time.perf_counter()
    keypoints, scores = wb(image_bgr)   # K=133 wholebody
    elapsed = time.perf_counter() - t0
    if keypoints is None or len(keypoints) == 0:
        return {"kp": [], "score": [], "elapsed_s": elapsed, "n_above_thresh": 0}
    return {
        "kp": keypoints[0].tolist(),
        "score": scores[0].tolist(),
        "elapsed_s": elapsed,
        "n_above_thresh": int((scores[0] > 0.3).sum()),
    }


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh
# ---------------------------------------------------------------------------
def run_mediapipe(image_bgr):
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python import BaseOptions

    opts = mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=f"{MODELS}/mediapipe/face_landmarker.task"),
        num_faces=1,
        running_mode=mpv.RunningMode.IMAGE,
    )
    landmarker = mpv.FaceLandmarker.create_from_options(opts)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    t0 = time.perf_counter()
    res = landmarker.detect(mp_image)
    elapsed = time.perf_counter() - t0
    if not res.face_landmarks:
        return {"kp": [], "score": [], "elapsed_s": elapsed, "n_above_thresh": 0}
    H, W = image_bgr.shape[:2]
    lm = res.face_landmarks[0]
    kp = [[p.x * W, p.y * H] for p in lm]
    return {
        "kp": kp,
        "score": [1.0] * len(kp),
        "elapsed_s": elapsed,
        "n_above_thresh": len(kp),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def draw_dots(image, kpts, scores=None, color=(0, 255, 0), thresh=0.3, r=2):
    out = image.copy()
    if not kpts:
        return out
    for i, (x, y) in enumerate(kpts):
        if scores is not None and scores[i] < thresh:
            continue
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            continue
        cv2.circle(out, (int(x), int(y)), r, color, -1)
    return out


def annotate_panel(image, model_name, kpts, scores, nostril_left_idx=None,
                   nostril_right_idx=None, time_s=None, nose_color=(0, 255, 0)):
    out = draw_dots(image, kpts, scores, color=(120, 120, 120), r=1)
    # highlight the nostrils
    if nostril_left_idx is not None and nostril_left_idx < len(kpts):
        x, y = kpts[nostril_left_idx]
        cv2.circle(out, (int(x), int(y)), 6, nose_color, -1)
        cv2.circle(out, (int(x), int(y)), 8, (255, 255, 255), 1)
    if nostril_right_idx is not None and nostril_right_idx < len(kpts):
        x, y = kpts[nostril_right_idx]
        cv2.circle(out, (int(x), int(y)), 6, nose_color, -1)
        cv2.circle(out, (int(x), int(y)), 8, (255, 255, 255), 1)
    label = model_name
    if time_s is not None:
        label = f"{model_name}  {time_s*1000:.0f}ms"
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def overlay_gt_nostrils(image, gt_nose_tip):
    """gt_nose_tip = list of 5 (x,y) tuples in image coords, indices 32..36
    (1-indexed SFTL54). For our purposes: nostrils at indices 1 and 3 (i.e. the
    points immediately to the left/right of the central tip)."""
    out = image.copy()
    if gt_nose_tip is None or len(gt_nose_tip) < 5:
        return out
    nostril_left, nostril_right = gt_nose_tip[1], gt_nose_tip[3]
    for (x, y) in (nostril_left, nostril_right):
        cv2.drawMarker(out, (int(x), int(y)), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    return out


def stack_panels(panels, titles, ncols=2):
    """panels: list of HxWx3 BGR arrays of identical size."""
    rows = []
    n = len(panels)
    for i in range(0, n, ncols):
        row = panels[i:i + ncols]
        while len(row) < ncols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_image_3ch(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def main(image_path, out_dir, gt_nose_tip=None, models=None,
         label_prefix="img"):
    image = load_image_3ch(image_path)
    H, W = image.shape[:2]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    panels = []

    if models is None:
        models = ["sapiens", "vitpose", "dwpose", "mediapipe"]

    if "sapiens" in models:
        print(f"[sapiens] running on {image_path}")
        r = run_sapiens(image)
        results["sapiens2_0.4b"] = r
        panels.append(annotate_panel(
            image, "Sapiens2-0.4b", r["kp"], r["score"],
            nostril_left_idx=SAPIENS_NOSTRIL_LEFT,
            nostril_right_idx=SAPIENS_NOSTRIL_RIGHT,
            time_s=r["elapsed_s"], nose_color=(0, 255, 0)))

    if "vitpose" in models:
        print(f"[vitpose] running on {image_path}")
        try:
            r = run_vitpose(image)
        except Exception as e:
            print(f"  vitpose failed: {e}")
            r = {"kp": [], "score": [], "elapsed_s": 0, "n_above_thresh": 0,
                 "error": str(e)}
        results["vitpose+"] = r
        panels.append(annotate_panel(
            image, "ViTPose+ base", r.get("kp", []), r.get("score", []),
            nostril_left_idx=COCOWB_NOSTRIL_LEFT,
            nostril_right_idx=COCOWB_NOSTRIL_RIGHT,
            time_s=r["elapsed_s"], nose_color=(0, 255, 255)))

    if "dwpose" in models:
        print(f"[dwpose] running on {image_path}")
        try:
            r = run_dwpose(image)
        except Exception as e:
            print(f"  dwpose failed: {e}")
            r = {"kp": [], "score": [], "elapsed_s": 0, "n_above_thresh": 0,
                 "error": str(e)}
        results["dwpose"] = r
        panels.append(annotate_panel(
            image, "DWPose", r.get("kp", []), r.get("score", []),
            nostril_left_idx=COCOWB_NOSTRIL_LEFT,
            nostril_right_idx=COCOWB_NOSTRIL_RIGHT,
            time_s=r["elapsed_s"], nose_color=(255, 128, 0)))

    if "mediapipe" in models:
        print(f"[mediapipe] running on {image_path}")
        try:
            r = run_mediapipe(image)
        except Exception as e:
            print(f"  mediapipe failed: {e}")
            r = {"kp": [], "score": [], "elapsed_s": 0, "n_above_thresh": 0,
                 "error": str(e)}
        results["mediapipe_facemesh"] = r
        panels.append(annotate_panel(
            image, "MediaPipe FaceMesh", r.get("kp", []), r.get("score", []),
            nostril_left_idx=MEDIAPIPE_NOSTRIL_LEFT,
            nostril_right_idx=MEDIAPIPE_NOSTRIL_RIGHT,
            time_s=r["elapsed_s"], nose_color=(255, 0, 255)))

    # overlay GT crosses onto each panel
    if gt_nose_tip is not None:
        panels = [overlay_gt_nostrils(p, gt_nose_tip) for p in panels]

    grid = stack_panels(panels, [], ncols=2)
    out_png = out_dir / f"{label_prefix}_grid.png"
    cv2.imwrite(str(out_png), grid)
    out_json = out_dir / f"{label_prefix}_results.json"
    with open(out_json, "w") as f:
        json.dump({"image": str(image_path), "wh": [W, H],
                   "gt_nose_tip": gt_nose_tip,
                   "models": results}, f, indent=2)
    print(f"[done] wrote {out_png} and {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--gt-jsonl", help="SFTL54-style labels.jsonl to look up GT")
    ap.add_argument("--gt-name", help="image basename to look up in --gt-jsonl")
    ap.add_argument("--label", default="img", help="output filename prefix")
    ap.add_argument("--out", default=str(RUNS_DIR / "single"))
    ap.add_argument("--models", default="sapiens,vitpose,dwpose,mediapipe")
    args = ap.parse_args()

    gt = None
    if args.gt_jsonl and args.gt_name:
        with open(args.gt_jsonl) as f:
            for line in f:
                row = json.loads(line)
                if Path(row["image"]).name == args.gt_name:
                    gt = row.get("nose_tip")
                    break

    main(args.image, args.out, gt_nose_tip=gt,
         models=args.models.split(","),
         label_prefix=args.label)
