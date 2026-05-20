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

# All four schemes annotate the nose-bottom region differently. To compare
# like-for-like, we target the "alae centre" — the point roughly *below*
# each nostril opening, between the outer alae corner and the central tip:
#
#   Sapiens2 (308 Goliath): average of inner_corner + outer_corner per side
#                            (no single landmark sits at the alae centre).
#   DWPose (133 COCO-WB)  : dlib face-32 / face-34 = COCO-WB idx 55 / 57.
#                            (face-31 / face-35 would be the OUTER alae;
#                             face-32 / face-34 are the inner-row points
#                             that sit right below each nostril opening.)
#   MediaPipe FaceMesh 478: 48 (subject's left alae centre),
#                            278 (subject's right alae centre).
#   SFTL54 ground truth   : nose_tip[1] (subject's right) and
#                            nose_tip[3] (subject's left) — same anatomical
#                            point: below the alae opening, not the outer
#                            alae corner.

# COCO-WholeBody alae-centre (dlib face 32 and 34):
COCOWB_NOSTRIL_RIGHT = 55  # subject's right (image left)  = dlib face-32
COCOWB_NOSTRIL_LEFT = 57   # subject's left  (image right) = dlib face-34

# Sapiens2 nostril targets resolved at runtime via name2id:
SAPIENS_NOSTRIL_NAMES_LEFT = ["inner_corner_of_l_nostril",
                              "outer_corner_of_l_nostril"]
SAPIENS_NOSTRIL_NAMES_RIGHT = ["inner_corner_of_r_nostril",
                               "outer_corner_of_r_nostril"]
# Filled in by run_sapiens after init_model:
SAPIENS_NOSTRIL_LEFT_IDS = None
SAPIENS_NOSTRIL_RIGHT_IDS = None

# MediaPipe FaceMesh 478: nostril alae clusters per the published mesh:
#   subject's left  alae cluster: [102, 49, 48, 115]
#   subject's right alae cluster: [331, 279, 278, 344]
# We average each cluster to get a stable alae-centre. Tip = 4.
MEDIAPIPE_NOSE_TIP = 4
MEDIAPIPE_NOSTRIL_LEFT_IDS = [102, 49, 48, 115]
MEDIAPIPE_NOSTRIL_RIGHT_IDS = [331, 279, 278, 344]
# Legacy single-index aliases for back-compat (point at the cluster centre):
MEDIAPIPE_NOSTRIL_LEFT = 48
MEDIAPIPE_NOSTRIL_RIGHT = 278


# ---------------------------------------------------------------------------
# Sapiens2
# ---------------------------------------------------------------------------
def run_sapiens(image_bgr, size="0.4b", device="cuda:0"):
    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
    global SAPIENS_NOSTRIL_LEFT_IDS, SAPIENS_NOSTRIL_RIGHT_IDS

    cfg = (f"{SAPIENS_REPO}/sapiens/pose/configs/keypoints308/"
           f"shutterstock_goliath_3po/sapiens2_{size}_keypoints308_"
           f"shutterstock_goliath_3po-1024x768.py")
    ckpt = f"{MODELS}/sapiens2_pose_{size}/sapiens2_{size}_pose.safetensors"

    model = init_model(cfg, ckpt, device=device)
    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file=f"{SAPIENS_REPO}/sapiens/pose/configs/_base_/keypoints308.py"))
    n2i = model.pose_metainfo["keypoint_name2id"]
    SAPIENS_NOSTRIL_LEFT_IDS = [n2i[n] for n in SAPIENS_NOSTRIL_NAMES_LEFT]
    SAPIENS_NOSTRIL_RIGHT_IDS = [n2i[n] for n in SAPIENS_NOSTRIL_NAMES_RIGHT]
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
    # Append derived "alae centre" points at the end of the keypoint array,
    # so a single index pair can be used across all 3 models.
    kp = keypoints[0]
    sc = scores[0]
    left_pts = kp[SAPIENS_NOSTRIL_LEFT_IDS]
    right_pts = kp[SAPIENS_NOSTRIL_RIGHT_IDS]
    alae_left = left_pts.mean(axis=0)
    alae_right = right_pts.mean(axis=0)
    kp_extended = np.vstack([kp, alae_left, alae_right])  # (310, 2)
    sc_extended = np.concatenate([sc, [sc[SAPIENS_NOSTRIL_LEFT_IDS].mean(),
                                       sc[SAPIENS_NOSTRIL_RIGHT_IDS].mean()]])
    return {
        "kp": kp_extended.tolist(),
        "score": sc_extended.tolist(),
        "elapsed_s": elapsed,
        "n_above_thresh": int((sc > 0.3).sum()),
        "alae_left_idx": kp_extended.shape[0] - 2,
        "alae_right_idx": kp_extended.shape[0] - 1,
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
    # append cluster-averaged alae centres so a single index pair works
    kp_np = np.array(kp)
    alae_left = kp_np[MEDIAPIPE_NOSTRIL_LEFT_IDS].mean(axis=0).tolist()
    alae_right = kp_np[MEDIAPIPE_NOSTRIL_RIGHT_IDS].mean(axis=0).tolist()
    kp_ext = kp + [alae_left, alae_right]
    return {
        "kp": kp_ext,
        "score": [1.0] * len(kp_ext),
        "elapsed_s": elapsed,
        "n_above_thresh": len(kp),
        "alae_left_idx": len(kp_ext) - 2,
        "alae_right_idx": len(kp_ext) - 1,
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
    """gt_nose_tip = SFTL54 5-pt nose row (idx 32..36).
    Use the inner-but-not-central pair (idx 1, 3) as "nostril center" GT —
    these sit roughly below each nostril opening, the most useful anchor for
    breath-rate measurement and the closest match to MediaPipe's alae-centre
    indices."""
    out = image.copy()
    if gt_nose_tip is None or len(gt_nose_tip) < 5:
        return out
    nostril_right = gt_nose_tip[1]   # subject's right nostril centre
    nostril_left = gt_nose_tip[3]    # subject's left nostril centre
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
            nostril_left_idx=r["alae_left_idx"],
            nostril_right_idx=r["alae_right_idx"],
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
