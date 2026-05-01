"""Inference + visualization helpers for the Sapiens2 try-it app.

Each task (pose / seg / normal / pointmap) has:
    - a `_load_<task>` function that builds the model on demand
    - a `predict_<task>` function that takes a BGR uint8 image and returns
      a BGR uint8 visualization plus a small dict of metadata.

All models load lazily; callers should wrap with @st.cache_resource so each
model is loaded once and kept on the GPU between requests.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Bhaskar's NVIDIA driver (550.144) cannot init the cuDNN bundled with
# torch 2.6+cu124 (CUDNN_STATUS_NOT_INITIALIZED on the first conv2d).
# Sapiens2 is attention-dominated; native conv fallback is fast enough.
torch.backends.cudnn.enabled = False

# Some sapiens modules transitively touch mmpretrain through mmdet utils;
# zero it out so import never tries to resolve mmpretrain.
sys.modules["mmpretrain"] = None

REPO_ROOT = Path("/DATA/nipun.batra/sapiens2")
SRC_ROOT = REPO_ROOT / "src"
CKPT_ROOT = REPO_ROOT / "checkpoints"

os.environ.setdefault("SAPIENS_CHECKPOINT_ROOT", str(CKPT_ROOT))

CONFIG = {
    "pose":     SRC_ROOT / "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "seg":      SRC_ROOT / "sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_1b_seg_shutterstock_goliath-1024x768.py",
    "normal":   SRC_ROOT / "sapiens/dense/configs/normal/metasim_render_people/sapiens2_1b_normal_metasim_render_people-1024x768.py",
    "pointmap": SRC_ROOT / "sapiens/dense/configs/pointmap/render_people/sapiens2_1b_pointmap_render_people-1024x768.py",
}
CHECKPOINT = {
    "pose":     CKPT_ROOT / "pose/sapiens2_1b_pose.safetensors",
    "seg":      CKPT_ROOT / "seg/sapiens2_1b_seg.safetensors",
    "normal":   CKPT_ROOT / "normal/sapiens2_1b_normal.safetensors",
    "pointmap": CKPT_ROOT / "pointmap/sapiens2_1b_pointmap.safetensors",
}

# Body skeleton on the first 17 of the 308 COCO-WholeBody keypoints.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Force any input to a 3-channel BGR uint8 array. Single-channel
    thermal images get replicated across channels."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------- POSE -------------------------------------------------------------
def load_pose():
    from sapiens.pose.models import init_model
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo

    model = init_model(str(CONFIG["pose"]), str(CHECKPOINT["pose"]),
                       device=device())
    model.pose_metainfo = parse_pose_metainfo(dict(
        from_file=str(SRC_ROOT / "sapiens/pose/configs/_base_/keypoints308.py")))
    cfg = dict(model.cfg.codec); cfg.pop("type")
    model.codec = UDPHeatmap(**cfg)
    return model


def predict_pose(model, image: np.ndarray, kpt_thr: float = 0.30):
    image = to_bgr(image)
    H, W = image.shape[:2]
    bbox = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)
    data = model.pipeline(dict(img=image, bbox=bbox,
                               bbox_score=np.ones(1, dtype=np.float32)))
    data = model.data_preprocessor(data)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(data["inputs"]).cpu().numpy()
    dt = time.perf_counter() - t0

    kp, sc = model.codec.decode(pred[0])
    isz = data["data_samples"]["meta"]["input_size"]
    bc = data["data_samples"]["meta"]["bbox_center"]
    bs = data["data_samples"]["meta"]["bbox_scale"]
    kp = (kp / isz * bs + bc - 0.5 * bs)[0]
    sc = sc[0]

    out = image.copy()
    palette = (np.random.RandomState(7).rand(308, 3) * 255).astype(np.uint8).tolist()
    for k, ((x, y), s) in enumerate(zip(kp, sc)):
        if s < kpt_thr or x < 0 or y < 0 or x >= W or y >= H:
            continue
        col = tuple(int(c) for c in palette[k])
        cv2.circle(out, (int(x), int(y)), 3 if k >= 17 else 6, col, -1)
    for a, b in SKELETON:
        if sc[a] < kpt_thr or sc[b] < kpt_thr:
            continue
        cv2.line(out, (int(kp[a][0]), int(kp[a][1])),
                 (int(kp[b][0]), int(kp[b][1])), (255, 255, 255), 3,
                 cv2.LINE_AA)

    n_above = int((sc >= kpt_thr).sum())
    id2name = model.pose_metainfo.get("keypoint_id2name", {})
    names = [id2name.get(k, f"kp_{k}") for k in range(len(kp))]
    return out, dict(forward_s=dt, kpts_above_thr=n_above, total_kpts=308,
                     keypoints=kp.tolist(), scores=sc.tolist(), names=names,
                     kpt_thr=kpt_thr)


# ---------- SEG --------------------------------------------------------------
def load_seg():
    from sapiens.dense.models import init_model
    return init_model(str(CONFIG["seg"]), str(CHECKPOINT["seg"]),
                      device=device())


def predict_seg(model, image: np.ndarray, overlay: float = 0.55):
    from sapiens.dense.visualizers import SegVisualizer
    image = to_bgr(image)
    data = model.pipeline(dict(img=image))
    data = model.data_preprocessor(data)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(data["inputs"])
    dt = time.perf_counter() - t0
    logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear")
    labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    vis = SegVisualizer(class_palette_type="dome29", with_labels=False,
                        overlay_opacity=overlay)
    seg_vis = vis._visualize_segmentation(image, labels)

    from sapiens.dense.datasets import DOME_CLASSES_29
    counts = np.bincount(labels.flatten(), minlength=30)
    nonzero = sorted(
        [(int(c), int(n)) for c, n in enumerate(counts) if n > 0 and c != 0],
        key=lambda x: -x[1])[:6]
    class_names = {int(cid): meta.get("name", f"class_{cid}")
                   for cid, meta in DOME_CLASSES_29.items()}
    return seg_vis, dict(forward_s=dt, top_classes=nonzero,
                         fg_pct=100 * (labels > 0).mean(),
                         labels=labels, class_names=class_names)


# ---------- NORMAL -----------------------------------------------------------
def load_normal():
    from sapiens.dense.models import init_model
    return init_model(str(CONFIG["normal"]), str(CHECKPOINT["normal"]),
                      device=device())


def predict_normal(model, image: np.ndarray):
    image = to_bgr(image)
    data = model.pipeline(dict(img=image))
    data = model.data_preprocessor(data)
    inputs, ds = data["inputs"], data["data_samples"]
    t0 = time.perf_counter()
    with torch.no_grad():
        n = model(inputs)
        n = n / torch.norm(n, dim=1, keepdim=True).clamp(1e-8)
    dt = time.perf_counter() - t0
    pl, pr, pt, pb = ds["meta"]["padding_size"]
    n = n[:, :, pt:inputs.shape[2] - pb, pl:inputs.shape[3] - pr]
    n = F.interpolate(n, size=(image.shape[0], image.shape[1]),
                      mode="bilinear", align_corners=False)
    n = n.squeeze(0).float().cpu().numpy()
    rgb = ((n.transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, dict(forward_s=dt, mean_z=float(n[2].mean()))


# ---------- POINTMAP ---------------------------------------------------------
def load_pointmap():
    from sapiens.dense.models import init_model
    return init_model(str(CONFIG["pointmap"]), str(CHECKPOINT["pointmap"]),
                      device=device())


def predict_pointmap(model, image: np.ndarray, seg_mask: np.ndarray = None):
    import matplotlib.pyplot as plt
    image = to_bgr(image)
    data = model.pipeline(dict(img=image))
    data = model.data_preprocessor(data)
    inputs, ds = data["inputs"], data["data_samples"]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(inputs)
    dt = time.perf_counter() - t0
    pm = out[0] if isinstance(out, tuple) else out  # head returns (pointmap, scale)
    scale = float(out[1].mean()) if isinstance(out, tuple) else 1.0
    if "padding_size" in ds["meta"]:
        pl, pr, pt, pb = ds["meta"]["padding_size"]
        pm = pm[:, :, pt:inputs.shape[2] - pb, pl:inputs.shape[3] - pr]
    pm = F.interpolate(pm, size=(image.shape[0], image.shape[1]),
                       mode="bilinear", align_corners=False)
    pm = pm.squeeze(0).float().cpu().numpy().transpose(1, 2, 0)  # H, W, 3

    depth = pm[:, :, 2]
    # The model outputs Z in a sign-mixed convention (camera-facing surfaces
    # can be negative). Closer = smaller Z. Work directly on Z, percentile-
    # normalised, so the colormap shows a clean near→far gradient.
    if seg_mask is not None:
        mask = seg_mask > 0
    else:
        mask = np.isfinite(depth)
    bg = np.full((depth.shape[0], depth.shape[1], 3), 30, np.uint8)
    if mask.sum() > 0:
        z = depth[mask]
        lo, hi = np.percentile(z, 2), np.percentile(z, 98)
        norm = 1.0 - np.clip((z - lo) / (hi - lo + 1e-8), 0, 1)  # near = red
        cmap = plt.get_cmap("turbo")
        col = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        bg[mask] = col[..., ::-1]  # RGB -> BGR
    return bg, dict(forward_s=dt, scale=scale,
                    depth_min=float(depth.min()),
                    depth_max=float(depth.max()),
                    fg_frac=float(mask.mean()))


PREDICTORS = {
    "pose":     (load_pose,     predict_pose),
    "seg":      (load_seg,      predict_seg),
    "normal":   (load_normal,   predict_normal),
    "pointmap": (load_pointmap, predict_pointmap),
}
