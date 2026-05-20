"""Build a 4-row panel: one image, four shrink factors.

Each row shows the (cropped to face region) frame at one shrink factor.
Cols: original (full-resolution reference) | single-stage | hierarchical.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hierarchical import (
    make_landmarker, mesh_on_image, face_bbox, single_stage,
    hierarchical as run_hier, shrink_and_pad, map_kp_through_shrink,
    MEDIAPIPE_NOSTRIL_LEFT, MEDIAPIPE_NOSTRIL_RIGHT,
)
import cv2
import numpy as np

# we run on bhaskar so HOME is bhaskar's
HOME = Path.home()
FACE_PATH = HOME / "models/mediapipe/face_detector_full_range.tflite"
MESH_PATH = HOME / "models/mediapipe/face_landmarker.task"
OUT = HOME / "git/nostril-bench/runs/hier_v2"
IMG = HOME / "data/SFTL54/sftl/sftl/rgb/train/100_1_2_4_1047_25_3.png"


def label(img, txt, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)
    return img


def main():
    detector, landmarker, mp = make_landmarker(FACE_PATH, MESH_PATH)
    src = cv2.imread(str(IMG))
    H, W = src.shape[:2]
    gt_l, gt_r, _ = single_stage(landmarker, mp, src)
    print(f"original: {W}x{H}, GT nostril L={gt_l}, R={gt_r}")

    rows = []
    crop_size = 240
    for f in [1, 2, 4, 8]:
        if f == 1:
            shrunk = src.copy()
        else:
            shrunk, sw, sh = shrink_and_pad(src, f)
        gt_l_s = map_kp_through_shrink(gt_l, f)
        gt_r_s = map_kp_through_shrink(gt_r, f)
        ss_l, ss_r, _ = single_stage(landmarker, mp, shrunk)
        h_out = run_hier(detector, landmarker, mp, shrunk)

        # crop a window around the face
        if f == 1:
            x0 = max(0, int((gt_l[0] + gt_r[0]) / 2) - crop_size // 2)
            y0 = max(0, int((gt_l[1] + gt_r[1]) / 2) - crop_size // 2)
        else:
            x0 = 0; y0 = 0
        x1 = min(W, x0 + crop_size); y1 = min(H, y0 + crop_size)
        # ensure square
        # single-stage panel
        p1 = shrunk.copy()
        cv2.drawMarker(p1, (int(gt_l_s[0]), int(gt_l_s[1])),
                       (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(p1, (int(gt_r_s[0]), int(gt_r_s[1])),
                       (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        if ss_l is not None:
            cv2.circle(p1, (int(ss_l[0]), int(ss_l[1])), 6,
                       (0, 255, 0), -1)
            cv2.circle(p1, (int(ss_r[0]), int(ss_r[1])), 6,
                       (0, 255, 0), -1)
        # hier panel
        p2 = shrunk.copy()
        cv2.drawMarker(p2, (int(gt_l_s[0]), int(gt_l_s[1])),
                       (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(p2, (int(gt_r_s[0]), int(gt_r_s[1])),
                       (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        if h_out is not None and h_out[0] is not None:
            pred, bbox = h_out
            x0b, y0b, x1b, y1b = bbox
            cv2.rectangle(p2, (int(x0b), int(y0b)), (int(x1b), int(y1b)),
                          (0, 255, 255), 1)
            cv2.circle(p2, (int(pred[0][0]), int(pred[0][1])), 6,
                       (255, 0, 255), -1)
            cv2.circle(p2, (int(pred[1][0]), int(pred[1][1])), 6,
                       (255, 0, 255), -1)
        # crop both
        p1c = p1[y0:y1, x0:x1]; p2c = p2[y0:y1, x0:x1]
        # upscale shrunk panels for readability
        if f > 1:
            up = 256 // (crop_size // (crop_size if f == 1 else max(1, crop_size//f))) \
                 if False else 2
            new_h = p1c.shape[0] * 2
            new_w = p1c.shape[1] * 2
            p1c = cv2.resize(p1c, (new_w, new_h), cv2.INTER_NEAREST)
            p2c = cv2.resize(p2c, (new_w, new_h), cv2.INTER_NEAREST)
        ss_err = None
        if ss_l is not None:
            ss_err = (((ss_l[0]-gt_l_s[0])**2 + (ss_l[1]-gt_l_s[1])**2)**0.5 +
                      ((ss_r[0]-gt_r_s[0])**2 + (ss_r[1]-gt_r_s[1])**2)**0.5) / 2
        hi_err = None
        if h_out is not None and h_out[0] is not None:
            pred, _ = h_out
            hi_err = (((pred[0][0]-gt_l_s[0])**2 + (pred[0][1]-gt_l_s[1])**2)**0.5 +
                      ((pred[1][0]-gt_r_s[0])**2 + (pred[1][1]-gt_r_s[1])**2)**0.5) / 2
        label(p1c, f"single  {f}x  " + (f"err={ss_err:.1f}px" if ss_err is not None else "MISS"),
              (255, 200, 200))
        label(p2c, f"hier    {f}x  " + (f"err={hi_err:.1f}px" if hi_err is not None else "MISS"),
              (200, 200, 255))
        row = np.hstack([p1c, p2c])
        rows.append(row)

    # pad rows to max width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
    grid = np.vstack(padded)
    cv2.imwrite(str(OUT.parent.parent / "progressive_panel.png"), grid)
    print("wrote", OUT.parent.parent / "progressive_panel.png", grid.shape)


if __name__ == "__main__":
    main()
