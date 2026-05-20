"""Run the four models on multiple SF-TL54 images (thermal gray, thermal iron,
RGB pair) and aggregate per-image, per-model results into a single JSON.

Usage:
    python run_batch.py --image-list image_list.json --out runs/batch
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from run_all import (
    run_sapiens, run_vitpose, run_dwpose, run_mediapipe,
    annotate_panel, overlay_gt_nostrils, stack_panels, load_image_3ch,
    COCOWB_NOSTRIL_LEFT, COCOWB_NOSTRIL_RIGHT,
    MEDIAPIPE_NOSTRIL_LEFT, MEDIAPIPE_NOSTRIL_RIGHT,
)

ROOT = Path.home() / "git/nostril-bench"


def gt_for(jsonl_path, basename):
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if Path(row["image"]).name == basename:
                return row.get("nose_tip")
    return None


def euclid(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def run_one_image(path, gt, sapiens_size, label_dir):
    label_dir = Path(label_dir); label_dir.mkdir(parents=True, exist_ok=True)
    image = load_image_3ch(path)
    results = {"image": str(path), "wh": list(image.shape[:2][::-1]),
               "gt_nose_tip": gt, "models": {}}
    panels = []

    # Sapiens2 (chosen size)
    r = run_sapiens(image, size=sapiens_size)
    results["models"][f"sapiens2_{sapiens_size}"] = r
    panels.append(annotate_panel(
        image, f"Sapiens2-{sapiens_size}", r["kp"], r["score"],
        nostril_left_idx=r["alae_left_idx"],
        nostril_right_idx=r["alae_right_idx"],
        time_s=r["elapsed_s"], nose_color=(0, 255, 0)))

    # ViTPose+ (body-only baseline; no face kpts)
    try:
        r = run_vitpose(image)
    except Exception as e:
        r = {"kp": [], "score": [], "elapsed_s": 0, "n_above_thresh": 0,
             "error": str(e)}
    results["models"]["vitpose+"] = r
    panels.append(annotate_panel(
        image, "ViTPose+ base (17 body)", r.get("kp", []), r.get("score", []),
        nostril_left_idx=None, nostril_right_idx=None,
        time_s=r["elapsed_s"], nose_color=(0, 255, 255)))

    # DWPose
    try:
        r = run_dwpose(image)
    except Exception as e:
        r = {"kp": [], "score": [], "elapsed_s": 0, "n_above_thresh": 0,
             "error": str(e)}
    results["models"]["dwpose"] = r
    panels.append(annotate_panel(
        image, "DWPose (133 wholebody)", r.get("kp", []), r.get("score", []),
        nostril_left_idx=COCOWB_NOSTRIL_LEFT,
        nostril_right_idx=COCOWB_NOSTRIL_RIGHT,
        time_s=r["elapsed_s"], nose_color=(255, 128, 0)))

    # MediaPipe FaceMesh
    try:
        r = run_mediapipe(image)
    except Exception as e:
        r = {"kp": [], "score": [], "elapsed_s": 0, "n_above_thresh": 0,
             "error": str(e), "alae_left_idx": -2, "alae_right_idx": -1}
    results["models"]["mediapipe_facemesh"] = r
    panels.append(annotate_panel(
        image, "MediaPipe FaceMesh (478)", r.get("kp", []), r.get("score", []),
        nostril_left_idx=r.get("alae_left_idx", MEDIAPIPE_NOSTRIL_LEFT),
        nostril_right_idx=r.get("alae_right_idx", MEDIAPIPE_NOSTRIL_RIGHT),
        time_s=r["elapsed_s"], nose_color=(255, 0, 255)))

    # GT overlay
    if gt is not None:
        panels = [overlay_gt_nostrils(p, gt) for p in panels]

    grid = stack_panels(panels, [], ncols=2)
    stem = Path(path).stem
    cv2.imwrite(str(label_dir / f"{stem}_grid.png"), grid)
    with open(label_dir / f"{stem}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main(image_list, sapiens_size, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"sapiens_size": sapiens_size, "items": []}
    for item in image_list:
        path = item["path"]
        gt = gt_for(item["jsonl"], Path(path).name) if item.get("jsonl") else None
        modality = item.get("modality", "rgb")
        print(f"--- {modality}  {Path(path).name} ---")
        r = run_one_image(path, gt, sapiens_size, out_dir / modality)
        # compute per-model nostril error
        per = {}
        if gt is not None:
            # SFTL54 nose_tip[1] = subject's right nostril centre,
            # nose_tip[3] = subject's left  nostril centre (between outer alae and tip).
            gt_r, gt_l = gt[1], gt[3]
            for name, info in r["models"].items():
                kp = info.get("kp", [])
                if not kp:
                    per[name] = {"err_l": None, "err_r": None,
                                 "time_ms": info["elapsed_s"] * 1000}
                    continue
                if "mediapipe" in name or "sapiens" in name:
                    # use the appended alae-centre indices
                    l = info.get("alae_left_idx", -2)
                    rr = info.get("alae_right_idx", -1)
                else:  # dwpose / vitpose+ COCO-WholeBody
                    l, rr = COCOWB_NOSTRIL_LEFT, COCOWB_NOSTRIL_RIGHT
                if l < len(kp) and rr < len(kp):
                    per[name] = {"err_l": euclid(kp[l], gt_l),
                                 "err_r": euclid(kp[rr], gt_r),
                                 "time_ms": info["elapsed_s"] * 1000}
                else:
                    per[name] = {"err_l": None, "err_r": None,
                                 "time_ms": info["elapsed_s"] * 1000}
        summary["items"].append({"path": path, "modality": modality, "per": per})
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-list", required=True,
                    help="JSON file with list of {path,jsonl,modality}")
    ap.add_argument("--sapiens-size", default="0.4b",
                    choices=["0.4b", "0.8b"])
    ap.add_argument("--out", default=str(ROOT / "runs/batch"))
    args = ap.parse_args()
    with open(args.image_list) as f:
        image_list = json.load(f)
    main(image_list, args.sapiens_size, args.out)
