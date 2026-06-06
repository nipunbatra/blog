"""Shared config + helpers for the tracking-vs-detection post."""
from pathlib import Path
import cv2

HERE = Path(__file__).resolve().parent
POSTS = HERE.parent.parent                       # posts/
OUT = HERE.parent / "outputs"                    # posts/tracking-vs-od/outputs
OUT.mkdir(parents=True, exist_ok=True)

SRC = POSTS / "992695-hd_1920_1080_25fps.mp4"
MODEL = POSTS / "yolov8l.pt"

PERSON, BALL = 0, 32
NAMES = {PERSON: "player", BALL: "ball"}

# Detection / tracking knobs (shared so OD and tracking see the same detector)
IMGSZ = 1280
DEVICE = "mps"
# Hold the detector identical for both passes, so the ONLY difference between
# the detection video and the tracking video is the temporal association layer.
OD_CONF = 0.25
TRACK_CONF = 0.25
FPS = 25


def read_frames(path=SRC):
    """Yield (index, BGR frame) for every frame in the clip."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def clip_meta(path=SRC):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, n


class VideoWriter:
    """Thin mp4 writer (avc1) with lazy init from first frame size."""
    def __init__(self, path, fps=FPS):
        self.path, self.fps, self.w = Path(path), fps, None

    def write(self, frame):
        if self.w is None:
            h, w = frame.shape[:2]
            self.w = cv2.VideoWriter(str(self.path),
                                     cv2.VideoWriter_fourcc(*"avc1"), self.fps, (w, h))
        self.w.write(frame)

    def close(self):
        if self.w is not None:
            self.w.release()
