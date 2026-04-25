"""Health-oriented downstream tasks built on Sapiens2 outputs.

Quantitative primitives that map onto common screening / rehab questions:

  - joint_angles      :  knee, hip, elbow, neck flexion / shoulder elevation
  - body_symmetry     :  left/right shoulder + hip height delta (% of torso)
  - skin_roi          :  Face_Neck + hand crops for tele-dermatology pipelines
  - posture_summary   :  per-image card with all three rolled together

These are *primitives*, not clinical claims. They produce the numeric
inputs that a screening protocol or a physiotherapist would interpret.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent
OUT = POST_DIR / "outputs"

# COCO-WholeBody body indices (first 17 of the 308 keypoints in Sapiens2)
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}


def load_pose(stem: str):
    with open(OUT / f"{stem}_pose.json") as f:
        data = json.load(f)
    K = max(k["id"] for k in data["keypoints"]) + 1
    pts = np.full((K, 2), np.nan, dtype=np.float32)
    sc = np.zeros((K,), dtype=np.float32)
    for k in data["keypoints"]:
        pts[k["id"]] = (k["x"], k["y"])
        sc[k["id"]] = k["score"]
    return pts, sc, data["image_size_wh"]


def angle_deg(a, b, c) -> float:
    """Interior angle at vertex b, with rays to a and c. NaN if any vertex missing."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return float("nan")
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def joint_angles(pts, sc, thr=0.3) -> dict:
    def p(name):
        i = KP[name]
        return pts[i] if sc[i] >= thr else np.array([np.nan, np.nan])

    out = {
        "left_knee_deg":   angle_deg(p("left_hip"),   p("left_knee"),   p("left_ankle")),
        "right_knee_deg":  angle_deg(p("right_hip"),  p("right_knee"),  p("right_ankle")),
        "left_hip_deg":    angle_deg(p("left_shoulder"),  p("left_hip"),  p("left_knee")),
        "right_hip_deg":   angle_deg(p("right_shoulder"), p("right_hip"), p("right_knee")),
        "left_elbow_deg":  angle_deg(p("left_shoulder"),  p("left_elbow"),  p("left_wrist")),
        "right_elbow_deg": angle_deg(p("right_shoulder"), p("right_elbow"), p("right_wrist")),
    }

    # Neck flexion: angle between (mid-shoulders -> nose) and the vertical.
    ls, rs = p("left_shoulder"), p("right_shoulder")
    nose = p("nose")
    if not (np.isnan(ls).any() or np.isnan(rs).any() or np.isnan(nose).any()):
        mid = (ls + rs) / 2
        v = nose - mid                # screen y grows downward
        # Up direction is (0, -1); angle from upright
        cos = np.clip(-v[1] / (np.linalg.norm(v) + 1e-8), -1.0, 1.0)
        out["neck_flexion_deg"] = float(np.degrees(np.arccos(cos)))
    else:
        out["neck_flexion_deg"] = float("nan")
    return out


def body_symmetry(pts, sc, thr=0.3) -> dict:
    ls, rs = pts[KP["left_shoulder"]], pts[KP["right_shoulder"]]
    lh, rh = pts[KP["left_hip"]],      pts[KP["right_hip"]]
    out = {}
    if sc[KP["left_shoulder"]] >= thr and sc[KP["right_shoulder"]] >= thr:
        out["shoulder_tilt_deg"] = float(np.degrees(
            np.arctan2(rs[1] - ls[1], rs[0] - ls[0])))
    if sc[KP["left_hip"]] >= thr and sc[KP["right_hip"]] >= thr:
        out["hip_tilt_deg"] = float(np.degrees(
            np.arctan2(rh[1] - lh[1], rh[0] - lh[0])))
    # Trunk lean: angle between mid-shoulder→mid-hip and the vertical
    if all(sc[KP[n]] >= thr for n in
           ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
        ms = (ls + rs) / 2
        mh = (lh + rh) / 2
        v = mh - ms
        cos = np.clip(v[1] / (np.linalg.norm(v) + 1e-8), -1.0, 1.0)
        out["trunk_lean_deg"] = float(np.degrees(np.arccos(cos)))
    return out


def skin_roi(img, seg) -> dict:
    """Cropped Face_Neck and Hand boxes — typical tele-derm pre-processing."""
    rois = {}
    for name, ids in [("face_neck", [3]), ("hands", [6, 15])]:
        m = np.isin(seg, ids)
        if m.sum() < 200:
            continue
        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        # 8 % padding
        py = int(0.08 * (y1 - y0)); px = int(0.08 * (x1 - x0))
        y0, y1 = max(0, y0 - py), min(img.shape[0], y1 + py)
        x0, x1 = max(0, x0 - px), min(img.shape[1], x1 + px)
        crop = img[y0:y1, x0:x1].copy()
        # Knock background pixels inside the crop down to 30 % brightness
        crop_mask = m[y0:y1, x0:x1]
        crop = np.where(crop_mask[..., None], crop,
                        (crop * 0.30).astype(np.uint8))
        rois[name] = (crop, (x0, y0, x1, y1))
    return rois


def render_card(img, angles, sym, pts, sc, kpt_thr=0.3):
    out = img.copy()
    H, W = out.shape[:2]
    # Skeleton
    sk = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
          (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
    for a, b in sk:
        if sc[a] >= kpt_thr and sc[b] >= kpt_thr:
            cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     (255, 255, 255), 3, cv2.LINE_AA)
    for i in range(17):
        if sc[i] < kpt_thr:
            continue
        cv2.circle(out, tuple(pts[i].astype(int)), 6, (0, 200, 255), -1, cv2.LINE_AA)

    # Annotate the key joint angles directly on the image
    for joint, side, kp_b in [("knee", "left", 13), ("knee", "right", 14),
                              ("elbow", "left", 7), ("elbow", "right", 8)]:
        v = angles.get(f"{side}_{joint}_deg", float("nan"))
        if np.isnan(v) or sc[kp_b] < kpt_thr:
            continue
        x, y = int(pts[kp_b][0]), int(pts[kp_b][1])
        label = f"{int(round(v))}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x - 4, y - th - 8), (x + tw + 6, y - 2),
                      (0, 0, 0), -1)
        cv2.putText(out, label, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (x + tw + 4, y - th + 2), 3,
                   (0, 255, 255), 1, cv2.LINE_AA)  # tiny degree mark

    # Side info card on the right
    card_w = 360
    info = np.full((H, card_w, 3), 240, np.uint8)
    y = 32
    cv2.putText(info, "Sapiens2  posture readout", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    y += 24
    rows = [
        ("Neck flexion",   angles.get("neck_flexion_deg")),
        ("L knee angle",   angles.get("left_knee_deg")),
        ("R knee angle",   angles.get("right_knee_deg")),
        ("L hip angle",    angles.get("left_hip_deg")),
        ("R hip angle",    angles.get("right_hip_deg")),
        ("L elbow angle",  angles.get("left_elbow_deg")),
        ("R elbow angle",  angles.get("right_elbow_deg")),
        ("Shoulder tilt",  sym.get("shoulder_tilt_deg")),
        ("Hip tilt",       sym.get("hip_tilt_deg")),
        ("Trunk lean",     sym.get("trunk_lean_deg")),
    ]
    for label, val in rows:
        y += 26
        s = "n/a" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val:6.1f} deg"
        cv2.putText(info, label, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        cv2.putText(info, s, (220, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 130), 2)
    return np.concatenate([out, info], axis=1)


def main(stem: str):
    img = cv2.imread(str(POST_DIR / f"inputs/{stem}.jpg"))
    pts, sc, _ = load_pose(stem)
    seg = np.load(OUT / f"{stem}_seg.npy")

    angles = joint_angles(pts, sc)
    sym = body_symmetry(pts, sc)
    metrics = {**angles, **sym}
    print(f"\n[{stem}] metrics:")
    for k, v in metrics.items():
        v_str = "nan" if isinstance(v, float) and np.isnan(v) else f"{v:.1f}"
        print(f"  {k:24s} = {v_str}")

    card = render_card(img, angles, sym, pts, sc)
    cv2.imwrite(str(OUT / f"{stem}_posture.jpg"), card)

    rois = skin_roi(img, seg)
    if rois:
        crops = [c for c, _ in rois.values()]
        labels = list(rois.keys())
        from apps import panel
        cv2.imwrite(str(OUT / f"{stem}_skin_rois.jpg"), panel(crops, labels))

    with open(OUT / f"{stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] wrote {OUT / f'{stem}_posture.jpg'}")


if __name__ == "__main__":
    stems = sys.argv[1:] or ["person2", "person4"]
    for s in stems:
        main(s)
