"""Run Sapiens2 pose (308 keypoints) on a single image, no mmdet detector.

We bypass the bundled person detector by passing the full image as a single
"detection" — every test photo here is one centred subject, so a fake
full-frame bbox is good enough. The model still does the standard top-down
crop+affine via its `pipeline`.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# mmpretrain has a buggy import chain that bleeds into anything that touches
# mmdet utilities; force a clean ImportError.
sys.modules["mmpretrain"] = None

from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
from sapiens.pose.models import init_model

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent

# Body-only keypoint indices (COCO-WholeBody 17 body kpts come first in the
# 308-keypoint set). Skeleton is the standard COCO body topology.
BODY = list(range(17))
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def main(image_path: str, out_dir: str, model_size: str = "0.4b",
         device: str = None):
    config = (f"/tmp/sapiens2/sapiens/pose/configs/keypoints308/"
              f"shutterstock_goliath_3po/sapiens2_{model_size}_keypoints308_"
              f"shutterstock_goliath_3po-1024x768.py")
    checkpoint = os.path.expanduser(
        f"~/sapiens2_host/pose/sapiens2_{model_size}_pose.safetensors")
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[init] device={device}  size={model_size}")
    t = time.perf_counter()
    model = init_model(config, checkpoint, device=device)
    print(f"[init] done in {time.perf_counter() - t:.1f} s")

    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file="/tmp/sapiens2/sapiens/pose/configs/_base_/keypoints308.py"))
    codec_cfg = dict(model.cfg.codec)
    codec_cfg.pop("type")
    model.codec = UDPHeatmap(**codec_cfg)

    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    bbox = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)
    print(f"[image] {image_path}  {W}x{H}  (full-frame bbox)")

    data_info = dict(img=image, bbox=bbox, bbox_score=np.ones(1, dtype=np.float32))
    data = model.pipeline(data_info)
    data = model.data_preprocessor(data)
    inputs = data["inputs"]

    t = time.perf_counter()
    with torch.no_grad():
        pred = model(inputs).cpu().numpy()
    print(f"[infer] forward: {time.perf_counter() - t:.2f} s "
          f"(heatmap shape {pred.shape})")

    keypoints, scores = model.codec.decode(pred[0])
    input_size = data["data_samples"]["meta"]["input_size"]
    bc = data["data_samples"]["meta"]["bbox_center"]
    bs = data["data_samples"]["meta"]["bbox_scale"]
    keypoints = keypoints / input_size * bs + bc - 0.5 * bs       # 1, K, 2
    keypoints = keypoints[0]
    scores = scores[0]

    # body skeleton + face keypoints
    out_img = image.copy()
    palette = (np.random.RandomState(7).rand(308, 3) * 255).astype(np.uint8).tolist()
    for k, ((x, y), s) in enumerate(zip(keypoints, scores)):
        if s < 0.30 or x < 0 or y < 0 or x >= W or y >= H:
            continue
        col = tuple(int(c) for c in palette[k])
        cv2.circle(out_img, (int(x), int(y)),
                   3 if k >= 17 else 6, col, -1)

    for a, b in SKELETON:
        if scores[a] < 0.3 or scores[b] < 0.3:
            continue
        p1, p2 = keypoints[a], keypoints[b]
        cv2.line(out_img, (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])), (255, 255, 255), 3)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(str(out / f"{stem}_pose.jpg"), out_img)

    id2name = model.pose_metainfo["keypoint_id2name"]
    data = {
        "image": str(image_path),
        "image_size_wh": [W, H],
        "keypoints": [
            {"id": int(k), "name": id2name.get(k, f"kp_{k}"),
             "x": float(x), "y": float(y), "score": float(s)}
            for k, ((x, y), s) in enumerate(zip(keypoints, scores))
        ],
    }
    with open(out / f"{stem}_pose.json", "w") as f:
        json.dump(data, f)
    n = sum(1 for _, _, s in zip(keypoints[:, 0], keypoints[:, 1], scores) if s >= 0.3)
    print(f"[done] {n}/308 keypoints above threshold; wrote "
          f"{out / f'{stem}_pose.jpg'}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(POST_DIR / "inputs/person2.jpg")
    out = sys.argv[2] if len(sys.argv) > 2 else str(POST_DIR / "outputs")
    main(img, out)
