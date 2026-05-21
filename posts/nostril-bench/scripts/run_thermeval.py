"""Run the 4 pose / face models on ThermEval-D real-world thermal scenes.

ThermEval images contain 0-3 people per frame at small face sizes (~20-30 px
face width). GT is the centroid of the "Nose" polygon for each annotated
person (we use it as our 1-point GT instead of SF-TL54's nose-tip-row).

For each predicted nostril per model:
   - Find the nearest GT nose centroid (greedy 1-to-1 matching, capped at
     a max distance).
   - Report per-person error.
The Nose polygon is small enough (~5×7 px) that picking left-vs-right alae
is meaningless at this resolution; we collapse the per-person model output
to a single "nose centre" point: average of `alae_left`/`alae_right` for
each model.

Outputs the same JSON structure as run_batch.py, plus visualisations.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_all import (
    run_sapiens, run_vitpose, run_dwpose, run_mediapipe,
    load_image_3ch,
    COCOWB_NOSTRIL_LEFT, COCOWB_NOSTRIL_RIGHT,
    MEDIAPIPE_NOSTRIL_LEFT, MEDIAPIPE_NOSTRIL_RIGHT,
)

ROOT = Path.home() / "git/nostril-bench"
THERMEVAL = Path.home() / "data/thermeval/ThermEval_KDD"


def load_thermeval(split_json="Annotations/annotations_1.json"):
    """Returns: list of (file_path, [{nose_centroid, person_bbox, ...}])"""
    a = json.load(open(THERMEVAL / split_json))
    id2file = {im["id"]: im["file_name"] for im in a["images"]}
    by_img = {}
    for ann in a["annotations"]:
        by_img.setdefault(ann["image_id"], {}).setdefault(
            ann["category_id"], []).append(ann)
    out = []
    for iid, file_name in id2file.items():
        if iid not in by_img:
            continue
        cats = by_img[iid]
        if 3 not in cats:   # no Nose annotation
            continue
        nose_pts = []
        for ann in cats[3]:
            x, y, w, h = ann["bbox"]
            nose_pts.append({
                "centroid": (x + w / 2, y + h / 2),
                "bbox": ann["bbox"],
                "polygon": ann.get("segmentation"),
            })
        persons = []
        for ann in cats.get(0, []):
            persons.append({"bbox": ann["bbox"]})
        out.append({
            "file": str(THERMEVAL / "images" / file_name),
            "image_id": iid,
            "noses": nose_pts,
            "persons": persons,
        })
    return out


def nose_centre_of(info):
    """Average alae_left and alae_right per model, return single (x, y) or None."""
    kp = info.get("kp", [])
    li = info.get("alae_left_idx")
    ri = info.get("alae_right_idx")
    if not kp or li is None or ri is None or li >= len(kp) or ri >= len(kp):
        return None
    l = kp[li]; r = kp[ri]
    return ((l[0] + r[0]) / 2, (l[1] + r[1]) / 2)


def match_predictions(preds, gts, max_dist=100):
    """Greedy 1-to-1 matching by Euclidean distance, capped at max_dist.
    preds, gts: lists of (x, y). Returns list of (pred_idx, gt_idx, dist) and
    list of unmatched_pred_idx, list of unmatched_gt_idx."""
    pairs = []
    used_p = set(); used_g = set()
    candidates = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            d = ((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2) ** 0.5
            candidates.append((d, i, j))
    candidates.sort()
    for d, i, j in candidates:
        if i in used_p or j in used_g or d > max_dist:
            continue
        pairs.append((i, j, d)); used_p.add(i); used_g.add(j)
    return pairs


def run_one(image_bgr, item):
    """Run all four models on one ThermEval image; return per-model results
    with per-prediction nose centre."""
    results = {}
    # Sapiens2
    try:
        r = run_sapiens(image_bgr)
        results["sapiens2_0.4b"] = {
            "elapsed_s": r["elapsed_s"],
            "centres": [nose_centre_of(r)] if nose_centre_of(r) else [],
            "all_kp_count": len(r["kp"]),
        }
    except Exception as e:
        results["sapiens2_0.4b"] = {"error": str(e), "centres": [],
                                    "elapsed_s": 0}

    # DWPose (multi-person: rtmlib returns one set per detected person)
    try:
        t0 = time.perf_counter()
        from rtmlib import Wholebody
        wb = Wholebody(to_openpose=False, mode="balanced",
                       backend="onnxruntime", device="cuda")
        keypoints, scores = wb(image_bgr)
        elapsed = time.perf_counter() - t0
        centres = []
        if keypoints is not None and len(keypoints):
            for person_kps in keypoints:
                kp = person_kps  # (133, 2) per person
                if COCOWB_NOSTRIL_LEFT < len(kp) and COCOWB_NOSTRIL_RIGHT < len(kp):
                    l = kp[COCOWB_NOSTRIL_LEFT]
                    rr = kp[COCOWB_NOSTRIL_RIGHT]
                    centres.append(((l[0] + rr[0]) / 2, (l[1] + rr[1]) / 2))
        results["dwpose"] = {"elapsed_s": elapsed, "centres": centres,
                             "all_kp_count": (keypoints.shape if hasattr(keypoints, 'shape') else None)}
    except Exception as e:
        results["dwpose"] = {"error": str(e), "centres": [], "elapsed_s": 0}

    # MediaPipe FaceMesh (single face)
    try:
        r = run_mediapipe(image_bgr)
        c = nose_centre_of(r)
        results["mediapipe_facemesh"] = {
            "elapsed_s": r["elapsed_s"],
            "centres": [c] if c else [],
        }
    except Exception as e:
        results["mediapipe_facemesh"] = {"error": str(e), "centres": [],
                                         "elapsed_s": 0}

    # ViTPose+ (body only - skip)
    results["vitpose+"] = {"elapsed_s": 0, "centres": [],
                           "note": "17 body kpts only"}

    return results


def main(items, out_dir, viz_every=20):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)
    summary = []

    for idx, item in enumerate(items):
        image = load_image_3ch(item["file"])
        H, W = image.shape[:2]
        gts = [n["centroid"] for n in item["noses"]]
        r = run_one(image, item)

        per_model = {}
        for model, info in r.items():
            preds = info.get("centres", [])
            pairs = match_predictions(preds, gts, max_dist=80)
            errs = [p[2] for p in pairs]
            n_matched = len(pairs)
            n_pred = len(preds)
            n_gt = len(gts)
            per_model[model] = {
                "n_pred": n_pred, "n_gt": n_gt, "n_matched": n_matched,
                "errs": errs,
                "elapsed_s": info.get("elapsed_s"),
                "pairs": pairs,    # (pred_idx, gt_idx, dist)
            }
        summary.append({
            "file": item["file"],
            "image_id": item["image_id"],
            "image_wh": [W, H],
            "n_gt": len(gts),
            "per_model": {k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
                          for k, v in per_model.items()},
        })

        if idx % viz_every == 0 or idx < 6:
            viz_image_grid(image, item, r, per_model,
                           out_dir / "viz" / f"img{item['image_id']:04d}.png")

        if idx % 25 == 24:
            print(f"  {idx + 1}/{len(items)} done")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir / 'summary.json'} ({len(summary)} rows)")
    return summary


def viz_image_grid(image, item, r, per_model, out_path, upscale=4):
    """Build a 2x2 grid of model predictions on the same image, with GT
    crosses and predicted dots. Upscale because images are tiny (192x256)."""
    panels = []
    models = ["sapiens2_0.4b", "dwpose", "mediapipe_facemesh", "vitpose+"]
    color = {"sapiens2_0.4b": (0, 255, 0), "dwpose": (255, 128, 0),
             "mediapipe_facemesh": (255, 0, 255), "vitpose+": (200, 200, 200)}
    label = {"sapiens2_0.4b": "Sapiens2-0.4b", "dwpose": "DWPose",
             "mediapipe_facemesh": "MediaPipe", "vitpose+": "ViTPose+ (body-only)"}
    for m in models:
        p = image.copy()
        # GT nose centroids as red crosses
        for n in item["noses"]:
            cx, cy = n["centroid"]
            cv2.drawMarker(p, (int(cx), int(cy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 8, 1)
        # predictions
        for c in r[m].get("centres", []):
            if c is None: continue
            cv2.circle(p, (int(c[0]), int(c[1])), 3, color[m], -1)
        # upscale
        p_big = cv2.resize(p, (p.shape[1] * upscale, p.shape[0] * upscale),
                           interpolation=cv2.INTER_NEAREST)
        # label
        cv2.rectangle(p_big, (0, 0), (p_big.shape[1], 24), (0, 0, 0), -1)
        info = per_model[m]
        n_match = info.get("n_matched", 0); n_gt = info.get("n_gt", 0)
        n_pred = info.get("n_pred", 0)
        et = info.get("elapsed_s") or 0
        cv2.putText(p_big, f"{label[m]}  {n_match}/{n_gt} match, {n_pred} preds, {et*1000:.0f}ms",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(p_big)
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.imwrite(str(out_path), grid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Annotations/annotations_1.json")
    ap.add_argument("--n", type=int, default=50, help="how many images")
    ap.add_argument("--out", default=str(ROOT / "runs/thermeval"))
    args = ap.parse_args()
    items = load_thermeval(args.split)[:args.n]
    print(f"running on {len(items)} ThermEval images from {args.split}")
    main(items, args.out)
