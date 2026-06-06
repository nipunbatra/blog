"""Extra tracking pass with BoT-SORT (GMC) -> botsort_tracks.json.

Same detector/thresholds as extract.py; only the tracker changes. Used for the
three-way Detection | ByteTrack | BoT-SORT comparison video.

  python extract_botsort.py
"""
import json
from pathlib import Path
from ultralytics import YOLO
import common as C
from extract import boxes_to_records

CFG = Path(__file__).resolve().parent / "configs" / "bot_30.yaml"


def main():
    w, h, n = C.clip_meta()
    meta = {"w": w, "h": h, "n": n, "fps": C.FPS, "tracker": "botsort (GMC, buffer=30)"}
    model = YOLO(str(C.MODEL))
    frames = []
    for r in model.track(str(C.SRC), conf=C.TRACK_CONF, classes=[C.PERSON, C.BALL],
                         imgsz=C.IMGSZ, device=C.DEVICE, tracker=str(CFG),
                         persist=True, stream=True, verbose=False):
        frames.append(boxes_to_records(r.boxes, with_id=True))
    (C.OUT / "botsort_tracks.json").write_text(json.dumps({"meta": meta, "frames": frames}))
    print(f"botsort: {len(frames)} frames")


if __name__ == "__main__":
    main()
