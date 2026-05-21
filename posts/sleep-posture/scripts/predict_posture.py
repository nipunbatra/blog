"""Run DWPose body keypoints on an image, then classify sleep posture.

Usage:
    python predict_posture.py --image path/to/image.jpg --out outdir/

Annotates the image with the posture report + keypoints + body axis,
saves to outdir/<basename>_posture.png and outdir/<basename>_report.json.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from classify_posture import classify, PostureReport, NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP

ROOT = Path.home() / "git/nostril-bench"


def run_dwpose(image_bgr):
    """Run RTMPose (via rtmlib) and return COCO-WholeBody kpts truncated to body."""
    from rtmlib import Wholebody
    wb = Wholebody(to_openpose=False, mode="balanced",
                   backend="onnxruntime", device="cuda")
    t0 = time.perf_counter()
    keypoints, scores = wb(image_bgr)
    elapsed = time.perf_counter() - t0
    if keypoints is None or len(keypoints) == 0:
        return None, None, elapsed
    # Pick the most central person if multiple
    if len(keypoints) > 1:
        H, W = image_bgr.shape[:2]
        cx, cy = W / 2, H / 2
        dists = []
        for kp in keypoints:
            mh = (kp[L_HIP] + kp[R_HIP]) / 2
            ms = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2
            c = (mh + ms) / 2
            dists.append((c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        idx = int(np.argmin(dists))
    else:
        idx = 0
    kp = keypoints[idx][:17]   # body block of COCO-WholeBody
    sc = scores[idx][:17]
    return kp, sc, elapsed


def annotate(image, kp, sc, report: PostureReport, score_thresh=0.3):
    out = image.copy()
    H, W = out.shape[:2]
    # Skeleton
    skel = [(L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
            (L_HIP, R_HIP), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16), (NOSE, L_EYE),
            (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR)]
    for a, b in skel:
        if sc[a] > score_thresh and sc[b] > score_thresh:
            cv2.line(out, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     (200, 200, 200), 2)
    # Keypoints
    for i, (p, s) in enumerate(zip(kp, sc)):
        if s < score_thresh: continue
        color = (0, 255, 255) if i < 5 else (255, 128, 0)
        cv2.circle(out, tuple(p.astype(int)), 4, color, -1)
    # Body axis
    if sc[L_HIP] > score_thresh and sc[R_HIP] > score_thresh \
            and sc[L_SHOULDER] > score_thresh and sc[R_SHOULDER] > score_thresh:
        mh = (kp[L_HIP] + kp[R_HIP]) / 2
        ms = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2
        cv2.arrowedLine(out, tuple(mh.astype(int)), tuple(ms.astype(int)),
                        (0, 0, 255), 2, tipLength=0.15)
    # Posture text box
    txt_lines = [
        f"posture: {report.posture}",
        f"body theta: {report.theta_body:.0f} deg" if not np.isnan(report.theta_body) else "body theta: n/a",
        f"yaw:        {report.yaw:.0f} deg" if not np.isnan(report.yaw) else "yaw: n/a",
        f"pitch:      {report.pitch:.0f} deg" if not np.isnan(report.pitch) else "pitch: n/a",
        f"face_vis:   {report.face_visible:.2f}",
        f"confidence: {report.confidence:.2f}",
    ]
    box_h = 18 * len(txt_lines) + 12
    box_w = 240
    cv2.rectangle(out, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    for i, t in enumerate(txt_lines):
        cv2.putText(out, t, (14, 24 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(image_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"can't read {image_path}"); return
    kp, sc, elapsed = run_dwpose(img)
    if kp is None:
        print(f"no person detected in {image_path}"); return
    report = classify(kp, sc)
    print(f"[{Path(image_path).name}] {report.posture} "
          f"theta={report.theta_body:.0f} yaw={report.yaw:.0f} "
          f"pitch={report.pitch:.0f} (conf={report.confidence:.2f})")
    annotated = annotate(img, kp, sc, report)
    stem = Path(image_path).stem
    cv2.imwrite(str(out_dir / f"{stem}_posture.png"), annotated)
    with open(out_dir / f"{stem}_report.json", "w") as f:
        json.dump({
            "image": str(image_path), "posture": report.posture,
            "theta_body": float(report.theta_body),
            "yaw": float(report.yaw), "pitch": float(report.pitch),
            "confidence": float(report.confidence),
            "face_visible": float(report.face_visible),
            "notes": report.notes,
            "elapsed_ms": elapsed * 1000,
        }, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs/posture"))
    args = ap.parse_args()
    main(args.image, args.out)
