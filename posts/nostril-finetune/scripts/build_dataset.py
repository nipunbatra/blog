"""Slice SFTL54 into a tiny few-shot dataset for nostril keypoint training.

Outputs:
  - train/  N=30 thermal-gray frames + nostril GT (left, right) as JSONL
  - val/    N=10  thermal-gray frames + GT
  - test/   N=60  thermal-gray frames + GT  (larger for PCK measurement)

Uses the gray-thermal palette only. Each sample is a 256x256 face crop around
the GT face bbox, with the nostril keypoints in the crop's coordinate frame.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / "data/SFTL54/sftl/sftl"
OUT = Path.home() / "data/nostril-few-shot"
random.seed(7); np.random.seed(7)


def face_bbox_from_landmarks(row):
    """Approximate face bbox from chin (idx 1..17)."""
    chin = np.array(row["chin"])
    leyebrow = np.array(row["left_eyebrow"])
    reyebrow = np.array(row["right_eyebrow"])
    all_pts = np.concatenate([chin, leyebrow, reyebrow], axis=0)
    x0, y0 = all_pts.min(axis=0)
    x1, y1 = all_pts.max(axis=0)
    return x0, y0, x1, y1


def crop_around_face(img, row, out_size=256, padding=0.25):
    x0, y0, x1, y1 = face_bbox_from_landmarks(row)
    w = x1 - x0; h = y1 - y0
    side = max(w, h) * (1 + padding)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0_c = max(0, int(cx - side / 2))
    y0_c = max(0, int(cy - side / 2))
    x1_c = min(img.shape[1], x0_c + int(side))
    y1_c = min(img.shape[0], y0_c + int(side))
    crop = img[y0_c:y1_c, x0_c:x1_c]
    if crop.size == 0:
        return None, None, None
    scale = out_size / crop.shape[1]
    crop_resized = cv2.resize(crop, (out_size, out_size))
    # transform GT nostrils into the crop coordinate frame
    nose_tip = np.array(row["nose_tip"])  # 5 pts
    nostril_left = nose_tip[1] - np.array([x0_c, y0_c])
    nostril_right = nose_tip[3] - np.array([x0_c, y0_c])
    nostril_left *= scale
    nostril_right *= scale
    return crop_resized, nostril_left.tolist(), nostril_right.tolist()


def build_split(split_in, split_out, n, seen_subjects=None):
    rows = []
    labels_path = ROOT / "gray" / split_in / "labels.jsonl"
    with open(labels_path) as f:
        for line in f:
            rows.append(json.loads(line))
    random.shuffle(rows)
    seen_subjects = seen_subjects or set()
    out_dir = OUT / split_out
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    out_lines = []
    for r in rows:
        if len(out_lines) >= n:
            break
        full_path = ROOT / "gray" / split_in / r["image"]
        if not full_path.exists():
            continue
        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        crop, nl, nr = crop_around_face(img, r)
        if crop is None or any(x < 0 or x > 255 or y < 0 or y > 255
                               for x, y in (nl, nr)):
            continue
        subject = int(Path(r["image"]).name.split("_")[0])
        if split_out == "train" and subject in seen_subjects:
            continue
        out_name = Path(r["image"]).name
        cv2.imwrite(str(out_dir / "images" / out_name), crop)
        out_lines.append({
            "image": out_name,
            "subject": subject,
            "nostril_left": nl,
            "nostril_right": nr,
        })
        seen_subjects.add(subject)
    with open(out_dir / "labels.jsonl", "w") as f:
        for r in out_lines:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(out_lines)} samples to {out_dir}")
    return seen_subjects


def main():
    print("Building train/val/test few-shot splits in", OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    train_subjects = build_split("train", "train", 30)
    build_split("train", "val", 10, seen_subjects=train_subjects)
    build_split("test", "test", 60)
    print("done.")


if __name__ == "__main__":
    main()
