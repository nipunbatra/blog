"""Build a YOLO-format dataset for "nostril as object" detection on ThermEval-D.

For each ThermEval frame:
  - For each Person whose bbox contains a Nose bbox:
      - Crop the Person bbox (+25% padding), resize to 256x256.
      - Map the Nose bbox into the crop's coordinate frame.
  - Write image + a YOLO-style txt label: "0 cx cy w h" in normalised coords.

Layout follows the ultralytics standard:
  ~/data/nostril-yolo/
      train/images/*.png  + train/labels/*.txt
      val/images/*.png    + val/labels/*.txt
      test/images/*.png   + test/labels/*.txt
      data.yaml           (top-level config for yolo train ...)
"""
import json
import random
from pathlib import Path
import cv2
import numpy as np

random.seed(7)
THERMEVAL = Path.home() / "data/thermeval/ThermEval_KDD"
OUT = Path.home() / "data/nostril-yolo"
IMG_SIZE = 256
PAD = 0.25


def load_pairs(annot_file):
    a = json.load(open(THERMEVAL / annot_file))
    id2file = {im["id"]: im["file_name"].replace(".jpg", ".png")
               for im in a["images"]}
    by_img = {}
    for ann in a["annotations"]:
        by_img.setdefault(ann["image_id"], {}).setdefault(
            ann["category_id"], []).append(ann)
    out = []
    for iid, cats in by_img.items():
        if 0 not in cats or 3 not in cats:
            continue
        for nose in cats[3]:
            nx = nose["bbox"][0] + nose["bbox"][2] / 2
            ny = nose["bbox"][1] + nose["bbox"][3] / 2
            container = None
            for p in sorted(cats[0],
                            key=lambda x: x["bbox"][2] * x["bbox"][3]):
                px, py, pw, ph = p["bbox"]
                if px <= nx <= px + pw and py <= ny <= py + ph:
                    container = p; break
            if container is None:
                continue
            file_path = THERMEVAL / "images" / id2file[iid]
            if not file_path.exists():
                continue
            out.append((str(file_path), container["bbox"], nose["bbox"], iid))
    return out


def crop_and_label(file_path, person_bbox, nose_bbox):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    H, W = img.shape
    px, py, pw, ph = person_bbox
    side = max(pw, ph) * (1 + PAD)
    cx = px + pw / 2; cy = py + ph / 2
    x0 = max(0, int(cx - side / 2))
    y0 = max(0, int(cy - side / 2))
    x1 = min(W, int(x0 + side))
    y1 = min(H, int(y0 + side))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    sx = IMG_SIZE / crop.shape[1]
    sy = IMG_SIZE / crop.shape[0]
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    # nose bbox in crop coords
    nx, ny, nw, nh = nose_bbox
    cnx = (nx + nw / 2 - x0) * sx
    cny = (ny + nh / 2 - y0) * sy
    cnw = nw * sx; cnh = nh * sy
    if not (0 <= cnx <= IMG_SIZE and 0 <= cny <= IMG_SIZE):
        return None, None
    # ensure bbox doesn't exceed crop bounds
    if cnw < 2 or cnh < 2 or cnw > IMG_SIZE or cnh > IMG_SIZE:
        return None, None
    return crop_resized, (cnx, cny, cnw, cnh)


def write_split(pairs, n, name, exclude_iids=None):
    exclude = exclude_iids or set()
    out_imgs = OUT / name / "images"
    out_lbls = OUT / name / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    written = []; iids = set()
    for file_path, person, nose, iid in pairs:
        if iid in exclude: continue
        crop, bbox = crop_and_label(file_path, person, nose)
        if crop is None: continue
        stem = f"img_{iid}_{len(written)}"
        cv2.imwrite(str(out_imgs / f"{stem}.png"), crop)
        cx, cy, w, h = bbox
        # YOLO label: class cx cy w h  (all normalised 0..1)
        label_str = f"0 {cx/IMG_SIZE:.6f} {cy/IMG_SIZE:.6f} {w/IMG_SIZE:.6f} {h/IMG_SIZE:.6f}\n"
        (out_lbls / f"{stem}.txt").write_text(label_str)
        written.append(stem); iids.add(iid)
        if len(written) >= n: break
    print(f"  {name}: {len(written)} samples from {len(iids)} unique images")
    return iids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs1 = list(load_pairs("Annotations/annotations_1.json"))
    pairs2 = list(load_pairs("Annotations/annotations_2.json"))
    random.shuffle(pairs1); random.shuffle(pairs2)
    print(f"split 1: {len(pairs1)} pairs;  split 2: {len(pairs2)} pairs")
    train_iids = write_split(pairs1, 60, "train")
    val_iids = write_split(pairs1, 20, "val", exclude_iids=train_iids)
    write_split(pairs2, 120, "test")
    # data.yaml for ultralytics
    yaml = (
        f"path: {OUT}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: 1\n"
        f"names: ['nostril']\n"
    )
    (OUT / "data.yaml").write_text(yaml)
    print(f"wrote data.yaml at {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()
