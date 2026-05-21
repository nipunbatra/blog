"""Hierarchical face -> YOLO-nostril detection on ThermEval-D.

Five pipelines per frame:
  (A) single MediaPipe         frame -> MediaPipe FaceMesh -> nostril
  (B) hier MediaPipe (Blaze)   frame -> BlazeFace -> crop -> FaceMesh -> nostril
  (C) hier YOLO (Blaze)        frame -> BlazeFace -> crop -> YOLO-nostril -> nostril
  (D) hier YOLO (GT person)    frame -> GT Person bbox -> crop -> YOLO-nostril
  (E) raw YOLO (no crop)       frame -> YOLO-nostril on full 192x256 frame

Compare detection rate (matched preds / GT noses) and median accuracy.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np

HOME = Path.home()
THERMEVAL = HOME / "data/thermeval/ThermEval_KDD"
MEDIAPIPE_NOSTRIL_LEFT_IDS = [102, 49, 48, 115]
MEDIAPIPE_NOSTRIL_RIGHT_IDS = [331, 279, 278, 344]
FACE_PATH = HOME / "models/mediapipe/face_detector_full_range.tflite"
MESH_PATH = HOME / "models/mediapipe/face_landmarker.task"
YOLO_PATH = HOME / "git/nostril-bench/runs/detect/runs/yolo_nostril/v1/weights/best.pt"
CROP_SIZE = 256
PAD = 0.25


def make_mp():
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python import BaseOptions
    fd = mpv.FaceDetector.create_from_options(
        mpv.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_PATH)),
            running_mode=mpv.RunningMode.IMAGE))
    fm = mpv.FaceLandmarker.create_from_options(
        mpv.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MESH_PATH)),
            num_faces=4, running_mode=mpv.RunningMode.IMAGE))
    return fd, fm, mp


def mp_mesh_nose(landmarker, mp, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=img_rgb))
    out = []
    if not res.face_landmarks: return out
    for lm in res.face_landmarks:
        pts = np.array([[p.x * W, p.y * H] for p in lm])
        l = pts[MEDIAPIPE_NOSTRIL_LEFT_IDS].mean(axis=0)
        r = pts[MEDIAPIPE_NOSTRIL_RIGHT_IDS].mean(axis=0)
        out.append(((l[0] + r[0]) / 2, (l[1] + r[1]) / 2))
    return out


def mp_face_bboxes(detector, mp, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=img_rgb))
    boxes = []
    if not res.detections: return boxes
    for d in res.detections:
        b = d.bounding_box
        boxes.append((b.origin_x, b.origin_y, b.width, b.height))
    return boxes


def crop_resize(img, bbox, pad=PAD, size=CROP_SIZE):
    x, y, w, h = bbox
    side = max(w, h) * (1 + pad)
    cx = x + w / 2; cy = y + h / 2
    H, W = img.shape[:2]
    x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, x0 + side)); y1 = int(min(H, y0 + side))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0: return None, None, None
    return cv2.resize(crop, (size, size)), (x0, y0), (x1 - x0, y1 - y0)


def yolo_nose(model, img_bgr, conf=0.05):
    """Run YOLO and return list of (cx, cy, score) for predicted nostrils."""
    res = model(img_bgr, imgsz=CROP_SIZE, conf=conf, verbose=False)
    out = []
    for r in res:
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            out.append((float(cx), float(cy), float(b.conf.item())))
    return out


def hier_yolo(model, img_bgr, bboxes, conf=0.05):
    out = []
    for bb in bboxes:
        crop, origin, csize = crop_resize(img_bgr, bb)
        if crop is None: continue
        preds = yolo_nose(model, crop, conf=conf)
        sx = csize[0] / CROP_SIZE; sy = csize[1] / CROP_SIZE
        for cx, cy, sc in preds:
            ox = cx * sx + origin[0]; oy = cy * sy + origin[1]
            out.append((ox, oy, sc, bb))
    return out


def match(preds, gts, max_dist=80):
    used_p, used_g, pairs = set(), set(), []
    candidates = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            d = ((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2) ** 0.5
            candidates.append((d, i, j))
    candidates.sort()
    for d, i, j in candidates:
        if i in used_p or j in used_g or d > max_dist: continue
        pairs.append((i, j, d)); used_p.add(i); used_g.add(j)
    return pairs


def main(n, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)
    fd, fm, mp = make_mp()
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_PATH))

    a = json.load(open(THERMEVAL / "Annotations/annotations_2.json"))
    id2file = {im["id"]: im["file_name"].replace(".jpg", ".png")
               for im in a["images"]}
    by_img = {}
    for ann in a["annotations"]:
        by_img.setdefault(ann["image_id"], {}).setdefault(
            ann["category_id"], []).append(ann)
    pickable = [iid for iid, cats in by_img.items() if 3 in cats and 0 in cats]
    pickable.sort()

    rows = []
    for k, iid in enumerate(pickable[:n]):
        fname = id2file[iid]
        img = cv2.imread(str(THERMEVAL / "images" / fname))
        if img is None: continue
        cats = by_img[iid]
        gts = [(n_["bbox"][0] + n_["bbox"][2] / 2,
                n_["bbox"][1] + n_["bbox"][3] / 2) for n_ in cats[3]]
        person_bboxes = [tuple(p["bbox"]) for p in cats.get(0, [])]

        t0 = time.perf_counter(); a_preds = mp_mesh_nose(fm, mp, img); a_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        blaze = mp_face_bboxes(fd, mp, img)
        b_out = []
        for bb in blaze:
            crop, origin, csize = crop_resize(img, bb)
            if crop is None: continue
            nps = mp_mesh_nose(fm, mp, crop)
            sx = csize[0] / CROP_SIZE; sy = csize[1] / CROP_SIZE
            for nx, ny in nps:
                b_out.append((nx * sx + origin[0], ny * sy + origin[1], bb))
        b_t = time.perf_counter() - t0
        b_preds = [(x, y) for x, y, _ in b_out]

        t0 = time.perf_counter()
        c_out = hier_yolo(yolo, img, blaze)
        c_t = time.perf_counter() - t0
        c_preds = [(x, y) for x, y, _, _ in c_out]

        t0 = time.perf_counter()
        d_out = hier_yolo(yolo, img, person_bboxes)
        d_t = time.perf_counter() - t0
        d_preds = [(x, y) for x, y, _, _ in d_out]

        t0 = time.perf_counter()
        e_preds_raw = yolo_nose(yolo, img, conf=0.05)
        e_t = time.perf_counter() - t0
        e_preds = [(x, y) for x, y, _ in e_preds_raw]

        def per(preds): return [pair[2] for pair in match(preds, gts)]
        rows.append({
            "iid": iid, "n_gt": len(gts),
            "a": {"n": len(a_preds), "errs": per(a_preds), "t_ms": a_t * 1000},
            "b": {"n": len(b_preds), "errs": per(b_preds), "t_ms": b_t * 1000},
            "c": {"n": len(c_preds), "errs": per(c_preds), "t_ms": c_t * 1000},
            "d": {"n": len(d_preds), "errs": per(d_preds), "t_ms": d_t * 1000},
            "e": {"n": len(e_preds), "errs": per(e_preds), "t_ms": e_t * 1000},
        })

        if k < 8:
            viz_frame(img, gts, a_preds, b_preds, c_preds, d_preds, e_preds,
                      blaze, person_bboxes,
                      out_dir / "viz" / f"img{iid:04d}.png")

        if k % 20 == 19: print(f"  {k+1}/{n} done")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out_dir/'summary.json'} ({len(rows)} rows)")


def viz_frame(img, gts, a, b, c, d, e, blaze, person_bb, out_path, upscale=4):
    H, W = img.shape[:2]
    rows = []
    for label, preds, color, boxes in [
        ("(A) single MP", a, (0, 255, 0), []),
        ("(B) hier-Blaze MP", b, (255, 128, 0), blaze),
        ("(C) hier-Blaze YOLO", c, (0, 255, 255), blaze),
        ("(D) hier-GTbbox YOLO", d, (255, 0, 255), person_bb),
        ("(E) raw YOLO no crop", e, (255, 255, 255), []),
    ]:
        p = img.copy()
        for g in gts:
            cv2.drawMarker(p, (int(g[0]), int(g[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 8, 1)
        for bb in boxes:
            cv2.rectangle(p, (int(bb[0]), int(bb[1])),
                          (int(bb[0]+bb[2]), int(bb[1]+bb[3])),
                          (180, 180, 180), 1)
        for c_ in preds:
            cv2.circle(p, (int(c_[0]), int(c_[1])), 3, color, -1)
        p_big = cv2.resize(p, (W*upscale, H*upscale), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(p_big, (0, 0), (p_big.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(p_big, f"{label}  preds={len(preds)}  GT={len(gts)}",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(p_big)
    grid = np.vstack([np.hstack([rows[0], rows[1], rows[2]]),
                       np.hstack([rows[3], rows[4], np.zeros_like(rows[0])])])
    cv2.imwrite(str(out_path), grid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=str(HOME / "git/nostril-bench/runs/hier_yolo_thermeval"))
    args = ap.parse_args()
    main(args.n, args.out)
