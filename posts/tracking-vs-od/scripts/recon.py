"""Reconnaissance: per-frame YOLO detection on the tennis clip.

Goal: confirm the presence/absence story before building the full post.
 - Do players (COCO 'person') leave and re-enter the frame?
 - Is the ball (COCO 'sports ball') detected at all, and does it flicker / vanish?

Run: python recon.py
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
POSTS = HERE.parent.parent                      # posts/
SRC = POSTS / "992695-hd_1920_1080_25fps.mp4"
MODEL = POSTS / "yolov8l.pt"

PERSON, BALL = 0, 32

def main():
    model = YOLO(str(MODEL))
    cap = cv2.VideoCapture(str(SRC))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"clip: {W}x{H}, {n} frames")

    persons, balls = [], []      # per-frame counts / best ball conf
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=0.20, classes=[PERSON, BALL],
                          imgsz=1280, device="mps", verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        np_persons = int((cls == PERSON).sum())
        ball_confs = conf[cls == BALL]
        persons.append(np_persons)
        balls.append(float(ball_confs.max()) if len(ball_confs) else 0.0)
        fi += 1
    cap.release()

    persons = np.array(persons)
    balls = np.array(balls)
    print(f"\nframes processed: {fi}")
    print(f"persons/frame: min={persons.min()} max={persons.max()} mean={persons.mean():.1f}")
    print(f"ball detected in {int((balls>0).sum())}/{fi} frames "
          f"({100*(balls>0).mean():.0f}%); mean conf when seen="
          f"{balls[balls>0].mean() if (balls>0).any() else 0:.2f}")

    # Person-count timeline (does the count change -> players leaving/entering?)
    print("\nperson-count timeline (every 5 frames):")
    print(" ".join(str(persons[i]) for i in range(0, fi, 5)))

    # Ball presence timeline:  X = detected, . = absent
    print("\nball presence (1 char/frame, X=seen .=absent):")
    print("".join("X" if b > 0 else "." for b in balls))

    # Longest absence gap for the ball
    gaps, cur = [], 0
    for b in balls:
        if b == 0:
            cur += 1
        else:
            if cur:
                gaps.append(cur)
            cur = 0
    if cur:
        gaps.append(cur)
    if gaps:
        print(f"\nball absence gaps (frames): {sorted(gaps, reverse=True)[:10]}")

if __name__ == "__main__":
    main()
