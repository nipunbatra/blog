"""Build a multi-dataset thermal SR training set: HR thermal images cropped
to a uniform 192x192 face region, then paired with their 4x-downsampled LR
versions on the fly during training.

Sources combined:
  - SF-TL54 (gray-thermal portraits, 464x348). Crop to a centred 192x192 face
    region using the per-image GT face bbox.
  - ThermEval-D (192x256 indoor scenes). Crop a 192x192 face region using the
    Person bbox + Forehead/Nose annotations (one crop per detected face).

We DON'T pre-downsample to LR — we generate the LR pair at dataloader time
via cv2.resize(INTER_AREA) so we can apply random degradations later if
needed.

Outputs:
  ~/data/thermal-sr/{train,val,test}/<dataset>/<id>.png  -- HR crops
  ~/data/thermal-sr/manifest.json                        -- split lists

Splits are image-disjoint:
  train: SF-TL54-train + ThermEval-D-split-1 (first 70%)
  val:   SF-TL54-val   + ThermEval-D-split-1 (last 30%)
  test:  SF-TL54-test  + ThermEval-D-split-2 (separate file)
"""
import json, random, sys
from pathlib import Path
import cv2, numpy as np

random.seed(7); np.random.seed(7)

SFTL = Path.home() / "data/SFTL54/sftl/sftl/gray"
THERMEVAL = Path.home() / "data/thermeval/ThermEval_KDD"
OUT = Path.home() / "data/thermal-sr"
HR_SIZE = 192
PAD = 0.15


def crop_centered(img, cx, cy, side, out_size=HR_SIZE):
    H, W = img.shape[:2]
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, x0 + side))
    y1 = int(min(H, y0 + side))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    return cv2.resize(crop, (out_size, out_size))


def crops_from_sftl(split):
    """Yield (id, hr_array, dataset) from SF-TL54."""
    labels_path = SFTL / split / "labels.jsonl"
    with open(labels_path) as f:
        for line in f:
            r = json.loads(line)
            full_path = SFTL / split / r["image"]
            if not full_path.exists(): continue
            img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.merge([img, img, img])
            chin = np.array(r["chin"])
            leyebrow = np.array(r["left_eyebrow"])
            reyebrow = np.array(r["right_eyebrow"])
            all_pts = np.concatenate([chin, leyebrow, reyebrow], axis=0)
            x0, y0 = all_pts.min(axis=0)
            x1, y1 = all_pts.max(axis=0)
            side = max(x1 - x0, y1 - y0) * (1 + PAD)
            cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
            crop = crop_centered(img, cx, cy, side)
            if crop is None: continue
            base = Path(r["image"]).name.replace(".png", "")
            yield (base, crop, "sftl54")


def crops_from_thermeval(annot_file):
    """Yield (id, hr_array, dataset) from ThermEval-D."""
    a = json.load(open(THERMEVAL / annot_file))
    id2file = {im["id"]: im["file_name"].replace(".jpg", ".png")
               for im in a["images"]}
    by_img = {}
    for ann in a["annotations"]:
        by_img.setdefault(ann["image_id"], {}).setdefault(
            ann["category_id"], []).append(ann)
    for iid, cats in by_img.items():
        if 0 not in cats or 3 not in cats: continue
        file_path = THERMEVAL / "images" / id2file[iid]
        if not file_path.exists(): continue
        img = cv2.imread(str(file_path))
        if img is None: continue
        for j, nose in enumerate(cats[3]):
            nx = nose["bbox"][0] + nose["bbox"][2] / 2
            ny = nose["bbox"][1] + nose["bbox"][3] / 2
            container = None
            for p in sorted(cats[0],
                            key=lambda x: x["bbox"][2] * x["bbox"][3]):
                px, py, pw, ph = p["bbox"]
                if px <= nx <= px + pw and py <= ny <= py + ph:
                    container = p; break
            if container is None: continue
            px, py, pw, ph = container["bbox"]
            side = max(pw, ph) * (1 + PAD)
            cx = px + pw / 2; cy = py + ph / 2
            crop = crop_centered(img, cx, cy, side)
            if crop is None: continue
            yield (f"{iid}_{j}", crop, "thermeval")


def write_split(items, split_name):
    out_dir = OUT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ds in {it[2] for it in items}:
        (out_dir / ds).mkdir(exist_ok=True)
    for base, crop, ds in items:
        path = out_dir / ds / f"{base}.png"
        cv2.imwrite(str(path), crop)
        written.append({"image": str(path.relative_to(OUT)),
                        "dataset": ds, "id": base})
    print(f"  {split_name}: {len(written)} crops "
          f"({sum(1 for w in written if w['dataset']=='sftl54')} SFTL54, "
          f"{sum(1 for w in written if w['dataset']=='thermeval')} ThermEval)")
    return written


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Reading SF-TL54...")
    sftl_train = list(crops_from_sftl("train"))
    sftl_val = list(crops_from_sftl("val"))
    sftl_test = list(crops_from_sftl("test"))
    print(f"  SFTL54: {len(sftl_train)} train, {len(sftl_val)} val, "
          f"{len(sftl_test)} test")

    print("Reading ThermEval-D...")
    te1 = list(crops_from_thermeval("Annotations/annotations_1.json"))
    te2 = list(crops_from_thermeval("Annotations/annotations_2.json"))
    random.shuffle(te1)
    te1_train = te1[:int(0.7 * len(te1))]
    te1_val = te1[int(0.7 * len(te1)):]
    print(f"  ThermEval: {len(te1_train)} train, {len(te1_val)} val (from split 1), "
          f"{len(te2)} test (split 2)")

    # Cap SF-TL54 to balance with ThermEval — too many SF-TL54 dominates
    sftl_train = sftl_train[:600]
    sftl_val = sftl_val[:80]
    sftl_test = sftl_test[:200]

    manifest = {
        "train": write_split(sftl_train + te1_train, "train"),
        "val": write_split(sftl_val + te1_val, "val"),
        "test": write_split(sftl_test + te2, "test"),
    }
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
