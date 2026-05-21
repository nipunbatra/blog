"""Build a small few-shot finetune dataset from ThermEval-D.

For each frame with at least one (Person + Nose) annotation pair:
  1. For each Person bbox that contains a Nose centroid:
     - Crop the Person bbox (+25% padding) from the 192x256 thermal frame
     - Resize the crop to 256x256
     - Map the Nose centroid into the crop's coordinate frame

Train: 40 samples, Val: 15, Test: 80.  Crops are written to:
  ~/data/nostril-thermeval/{train,val,test}/images/*.png
  ~/data/nostril-thermeval/{train,val,test}/labels.jsonl

Each label record: {"image": "<basename>.png", "nostril": [x, y]}
"""
import json, random
from pathlib import Path
import cv2, numpy as np

THERMEVAL = Path.home() / "data/thermeval/ThermEval_KDD"
OUT = Path.home() / "data/nostril-thermeval"
random.seed(7)

IMG_SIZE = 256
PAD = 0.25  # padding fraction around person bbox

def load_pairs(annot_file):
    """Yield (file_path, person_bbox, nose_centroid)."""
    a = json.load(open(THERMEVAL / annot_file))
    # Split 2 annotations reference .jpg names but files on disk are .png.
    id2file = {im["id"]: im["file_name"].replace(".jpg", ".png")
               for im in a["images"]}
    by_img = {}
    for ann in a["annotations"]:
        by_img.setdefault(ann["image_id"], {}).setdefault(ann["category_id"], []).append(ann)
    for iid, cats in by_img.items():
        if 0 not in cats or 3 not in cats: continue
        persons = cats[0]; noses = cats[3]
        file_path = THERMEVAL / "images" / id2file[iid]
        if not file_path.exists(): continue
        # match each nose to the smallest-area Person bbox that contains it
        for nose in noses:
            nx, ny = nose["bbox"][0] + nose["bbox"][2]/2, nose["bbox"][1] + nose["bbox"][3]/2
            containing = []
            for p in persons:
                px, py, pw, ph = p["bbox"]
                if px <= nx <= px + pw and py <= ny <= py + ph:
                    containing.append((pw * ph, p))
            if not containing: continue
            containing.sort()  # smallest first
            person = containing[0][1]
            yield (str(file_path), person["bbox"], (nx, ny), iid)


def crop_and_label(file_path, person_bbox, nose_centroid):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None
    H, W = img.shape
    px, py, pw, ph = person_bbox
    side = max(pw, ph) * (1 + PAD)
    cx = px + pw / 2; cy = py + ph / 2
    x0 = max(0, int(cx - side / 2))
    y0 = max(0, int(cy - side / 2))
    x1 = min(W, int(x0 + side))
    y1 = min(H, int(y0 + side))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0: return None, None
    sx = IMG_SIZE / crop.shape[1]
    sy = IMG_SIZE / crop.shape[0]
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    nx = (nose_centroid[0] - x0) * sx
    ny = (nose_centroid[1] - y0) * sy
    if not (0 <= nx < IMG_SIZE and 0 <= ny < IMG_SIZE):
        return None, None
    return crop_resized, (nx, ny)


def build_split(pairs, n, name, exclude_iids=None):
    exclude_iids = exclude_iids or set()
    out_dir = OUT / name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    items = []
    iids = set()
    for file_path, person, nose, iid in pairs:
        if iid in exclude_iids: continue
        crop, kp = crop_and_label(file_path, person, nose)
        if crop is None: continue
        # write
        name_out = f"img_{iid}_{len(items)}.png"
        cv2.imwrite(str(out_dir / "images" / name_out), crop)
        items.append({"image": name_out, "nostril": [float(kp[0]), float(kp[1])],
                      "source_iid": iid})
        iids.add(iid)
        if len(items) >= n: break
    with open(out_dir / "labels.jsonl", "w") as f:
        for it in items: f.write(json.dumps(it) + "\n")
    print(f"  {name}: wrote {len(items)} samples from {len(iids)} unique images")
    return iids


def main():
    pairs = list(load_pairs("Annotations/annotations_1.json"))
    pairs2 = list(load_pairs("Annotations/annotations_2.json"))
    random.shuffle(pairs); random.shuffle(pairs2)
    print(f"split 1: {len(pairs)} (person, nose) pairs")
    print(f"split 2: {len(pairs2)} (person, nose) pairs")
    OUT.mkdir(parents=True, exist_ok=True)
    train_iids = build_split(pairs, 40, "train")
    val_iids = build_split(pairs, 15, "val", exclude_iids=train_iids)
    test_iids = build_split(pairs2, 80, "test")
    print("done")


if __name__ == "__main__":
    main()
