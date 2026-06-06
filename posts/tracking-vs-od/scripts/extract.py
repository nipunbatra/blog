"""Run two passes over the clip and cache results as JSON.

  Pass 1  detection  : model.predict per frame -> independent boxes, NO identity.
  Pass 2  tracking   : model.track (ByteTrack)  -> boxes WITH a persistent id.

Both use the same detector/weights so the only difference is the temporal
association layer that tracking adds on top.

  python extract.py
"""
import json
from ultralytics import YOLO
import common as C


def boxes_to_records(boxes, with_id=False):
    out = []
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    ids = (boxes.id.cpu().numpy().astype(int)
           if (with_id and boxes.id is not None) else [None] * len(cls))
    for k in range(len(cls)):
        rec = {"cls": int(cls[k]), "conf": round(float(conf[k]), 4),
               "xyxy": [round(float(v), 1) for v in xyxy[k]]}
        if with_id:
            rec["id"] = (int(ids[k]) if ids[k] is not None else None)
        out.append(rec)
    return out


def main():
    w, h, n = C.clip_meta()
    meta = {"w": w, "h": h, "n": n, "fps": C.FPS, "model": C.MODEL.name,
            "imgsz": C.IMGSZ, "od_conf": C.OD_CONF, "track_conf": C.TRACK_CONF}

    # ---- Pass 1: per-frame detection (no memory) ----
    model = YOLO(str(C.MODEL))
    det_frames = []
    for r in model.predict(str(C.SRC), conf=C.OD_CONF, classes=[C.PERSON, C.BALL],
                           imgsz=C.IMGSZ, device=C.DEVICE, stream=True, verbose=False):
        det_frames.append(boxes_to_records(r.boxes))
    (C.OUT / "detections.json").write_text(
        json.dumps({"meta": meta, "frames": det_frames}))
    print(f"detection: {len(det_frames)} frames")

    # ---- Pass 2: tracking (ByteTrack, default config) ----
    model = YOLO(str(C.MODEL))
    trk_frames = []
    for r in model.track(str(C.SRC), conf=C.TRACK_CONF, classes=[C.PERSON, C.BALL],
                         imgsz=C.IMGSZ, device=C.DEVICE, tracker="bytetrack.yaml",
                         persist=True, stream=True, verbose=False):
        trk_frames.append(boxes_to_records(r.boxes, with_id=True))
    (C.OUT / "tracks.json").write_text(
        json.dumps({"meta": meta, "frames": trk_frames}))
    print(f"tracking:  {len(trk_frames)} frames")


if __name__ == "__main__":
    main()
