"""Gait analysis on a walking video using Sapiens2 pose.

For every sampled frame:
  - run Sapiens2-pose-0.4b (no detector — full-frame bbox; one subject in frame)
  - extract body keypoints
  - compute trunk lean, head/torso lateral position, ankle vertical position
Then plot the time-series and render an annotated overlay video.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.modules["mmpretrain"] = None
from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
from sapiens.pose.models import init_model

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent
VIDEO = POST_DIR / "video" / "elder_walking.mp4"
OUT = POST_DIR / "outputs"
FRAMES = POST_DIR / "video" / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)

KP = {"nose": 0, "left_shoulder": 5, "right_shoulder": 6, "left_hip": 11,
      "right_hip": 12, "left_knee": 13, "right_knee": 14,
      "left_ankle": 15, "right_ankle": 16}


def load_model():
    config = ("/tmp/sapiens2/sapiens/pose/configs/keypoints308/"
              "shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_"
              "shutterstock_goliath_3po-1024x768.py")
    checkpoint = os.path.expanduser(
        "~/sapiens2_host/pose/sapiens2_0.4b_pose.safetensors")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = init_model(config, checkpoint, device=device)
    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file="/tmp/sapiens2/sapiens/pose/configs/_base_/keypoints308.py"))
    cfg = dict(model.cfg.codec); cfg.pop("type")
    model.codec = UDPHeatmap(**cfg)
    return model, device


def pose_one_frame(model, img):
    H, W = img.shape[:2]
    bbox = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)
    data = model.pipeline(dict(img=img, bbox=bbox,
                               bbox_score=np.ones(1, dtype=np.float32)))
    data = model.data_preprocessor(data)
    with torch.no_grad():
        pred = model(data["inputs"]).cpu().numpy()
    keypoints, scores = model.codec.decode(pred[0])
    input_size = data["data_samples"]["meta"]["input_size"]
    bc = data["data_samples"]["meta"]["bbox_center"]
    bs = data["data_samples"]["meta"]["bbox_scale"]
    keypoints = keypoints / input_size * bs + bc - 0.5 * bs
    return keypoints[0], scores[0]


def extract_frames(every_sec: float = 0.4, max_frames: int = 16):
    """Sample frames; crop to a person-centred strip so the subject
    fills enough of the input that pose lands. The subject in this clip
    walks down the centre-left of a vertical-orientation video."""
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    every_n = max(1, int(round(every_sec * fps)))
    frame_idxs = list(range(0, total, every_n))[:max_frames]
    frames = []
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        # Crop a vertical strip around the walker (subject is centre-left,
        # roughly x ∈ [0.10, 0.55] of the original 2160-wide frame).
        H0, W0 = fr.shape[:2]
        x0, x1 = int(0.10 * W0), int(0.60 * W0)
        y0, y1 = int(0.20 * H0), int(0.95 * H0)
        crop = fr[y0:y1, x0:x1]
        # Resize so width = 768 (the input width Sapiens2 expects)
        scale = 768 / crop.shape[1]
        crop = cv2.resize(crop, (768, int(crop.shape[0] * scale)))
        frames.append((fi / fps, crop))
    cap.release()
    return frames, duration, fps


def annotate(img, kp, sc, metrics, thr=0.3):
    out = img.copy()
    sk = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
          (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
    for a, b in sk:
        if sc[a] >= thr and sc[b] >= thr:
            cv2.line(out, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     (255, 255, 255), 3, cv2.LINE_AA)
    for i in range(17):
        if sc[i] >= thr:
            cv2.circle(out, tuple(kp[i].astype(int)), 6,
                       (0, 200, 255), -1, cv2.LINE_AA)

    # Sidebar
    H = out.shape[0]
    sb_w = 360
    sb = np.full((H, sb_w, 3), 240, np.uint8)
    cv2.putText(sb, "Gait readout", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    rows = [
        ("t (s)",          f"{metrics['t']:.2f}"),
        ("Trunk lean",     f"{metrics['trunk_lean_deg']:5.1f} deg"),
        ("Head x (norm)",  f"{metrics['head_x_norm']:+.3f}"),
        ("L ankle y",      f"{metrics['left_ankle_y_norm']:+.3f}"),
        ("R ankle y",      f"{metrics['right_ankle_y_norm']:+.3f}"),
        ("Stride asym",    f"{metrics['stride_asym']:+.3f}"),
    ]
    y = 60
    for label, val in rows:
        y += 30
        cv2.putText(sb, label, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        cv2.putText(sb, val, (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 130), 2)
    return np.concatenate([out, sb], axis=1)


def compute_metrics(kp, sc, frame_h, frame_w, thr=0.3):
    def get(name):
        i = KP[name]
        return kp[i] if sc[i] >= thr else np.array([np.nan, np.nan])
    ls, rs = get("left_shoulder"), get("right_shoulder")
    lh, rh = get("left_hip"), get("right_hip")
    nose = get("nose")
    la, ra = get("left_ankle"), get("right_ankle")
    if not (np.isnan(ls).any() or np.isnan(rs).any()
            or np.isnan(lh).any() or np.isnan(rh).any()):
        ms = (ls + rs) / 2
        mh = (lh + rh) / 2
        v = mh - ms
        cos = np.clip(v[1] / (np.linalg.norm(v) + 1e-8), -1.0, 1.0)
        trunk = float(np.degrees(np.arccos(cos)))
        # signed sway = horizontal component of trunk vector
        trunk_signed = float(np.degrees(np.arctan2(v[0], v[1])))
        head_x = (nose[0] - mh[0]) / frame_w if not np.isnan(nose).any() else float("nan")
    else:
        trunk = trunk_signed = head_x = float("nan")
    la_y = la[1] / frame_h if not np.isnan(la).any() else float("nan")
    ra_y = ra[1] / frame_h if not np.isnan(ra).any() else float("nan")
    return dict(trunk_lean_deg=trunk_signed, head_x_norm=head_x,
                left_ankle_y_norm=la_y, right_ankle_y_norm=ra_y,
                stride_asym=la_y - ra_y if not (np.isnan(la_y) or np.isnan(ra_y))
                            else float("nan"))


def make_strip(frames_with_overlay, n=4):
    pick = np.linspace(0, len(frames_with_overlay) - 1, n).astype(int)
    selected = [frames_with_overlay[i] for i in pick]
    h = max(im.shape[0] for im in selected)
    rs = [cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h))
          for im in selected]
    return np.concatenate(rs, axis=1)


def main():
    print("[gait] loading model …")
    t = time.perf_counter()
    model, dev = load_model()
    print(f"[gait] model loaded in {time.perf_counter() - t:.1f} s on {dev}")

    print("[gait] sampling frames …")
    frames, dur, fps = extract_frames(every_sec=0.6, max_frames=12)
    print(f"[gait] video {dur:.1f} s @ {fps:.1f} fps -> "
          f"{len(frames)} frames sampled")

    rows = []
    overlays = []
    for k, (t_sec, fr) in enumerate(frames):
        t = time.perf_counter()
        kp, sc = pose_one_frame(model, fr)
        H, W = fr.shape[:2]
        m = compute_metrics(kp, sc, H, W)
        m["t"] = t_sec
        rows.append(m)
        overlays.append(annotate(fr, kp, sc, m))
        print(f"  frame {k:02d}  t={t_sec:5.2f}s  trunk={m['trunk_lean_deg']:6.2f}  "
              f"asym={m['stride_asym']:+.3f}  ({time.perf_counter() - t:.2f} s)")

    # Save overlays as a video
    h, w = overlays[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_raw = OUT / "gait_overlay.mp4"
    vw = cv2.VideoWriter(str(out_video_raw), fourcc, 2.0, (w, h))
    for fr in overlays:
        vw.write(fr)
    vw.release()

    # Also a 4-frame strip for the post
    strip = make_strip(overlays, n=4)
    cv2.imwrite(str(OUT / "gait_strip.jpg"), strip)

    # Plot time-series
    ts = [r["t"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    for ax, key, lab in zip(
        axes,
        ["trunk_lean_deg", "head_x_norm", "stride_asym"],
        ["trunk lean (deg, +ve = forward)",
         "head x deviation from mid-hip (normalised)",
         "stride asymmetry: L ankle y − R ankle y (normalised)"],
    ):
        vs = [r[key] for r in rows]
        ax.plot(ts, vs, "-o", color="#c44536", linewidth=1.6, markersize=4)
        ax.axhline(0, color="#888", linewidth=0.7)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Gait time-series from Sapiens2 pose (12 sampled frames)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gait_timeseries.png", dpi=130)
    plt.close(fig)

    with open(OUT / "gait_metrics.json", "w") as f:
        json.dump(rows, f, indent=2, default=lambda o: None)

    print(f"[done] {out_video_raw}, gait_strip.jpg, gait_timeseries.png")


if __name__ == "__main__":
    main()
