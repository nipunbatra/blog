"""Render the detection-vs-tracking comparison from cached JSON.

  outputs/compare_od_vs_track.mp4   side-by-side video (detection | tracking)
  outputs/frame_strip.png           a montage of consecutive frames over a ball blink

No model needed here — reads detections.json / tracks.json + the source frames.
"""
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import cv2
import supervision as sv
import common as C

PANEL_W, PANEL_H = 960, 540          # each side
HEADER = 46

det = json.load(open(C.OUT / "detections.json"))["frames"]
trk = json.load(open(C.OUT / "tracks.json"))["frames"]
_bot_path = C.OUT / "botsort_tracks.json"
bot = json.load(open(_bot_path))["frames"] if _bot_path.exists() else None

# remap COCO ids -> 0/1 so a 2-colour palette indexes cleanly
REMAP = {C.PERSON: 0, C.BALL: 1}
PLAYER_COL, BALL_COL = sv.Color(44, 111, 187), sv.Color(196, 69, 54)   # blue, red
PALETTE = sv.ColorPalette([PLAYER_COL, BALL_COL])

box_class = sv.BoxAnnotator(color=PALETTE, color_lookup=sv.ColorLookup.CLASS, thickness=2)
lab_class = sv.LabelAnnotator(color=PALETTE, color_lookup=sv.ColorLookup.CLASS,
                              text_scale=0.5, text_thickness=1, text_padding=3)
box_track = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2)
lab_track = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK,
                              text_scale=0.5, text_thickness=1, text_padding=3)
trace = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK, trace_length=25, thickness=2)


def to_det(records, with_id=False):
    if not records:
        return sv.Detections.empty()
    xyxy = np.array([r["xyxy"] for r in records], float)
    cls = np.array([REMAP[r["cls"]] for r in records], int)
    conf = np.array([r["conf"] for r in records], float)
    tid = None
    if with_id:
        tid = np.array([r["id"] if r["id"] is not None else -1 for r in records], int)
    d = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls, tracker_id=tid)
    if with_id:
        d = d[d.tracker_id >= 0]                       # drop unconfirmed (no id)
    return d


def label_det(d):
    name = {0: "player", 1: "ball"}
    return [f"{name[c]} {p:.2f}" for c, p in zip(d.class_id, d.confidence)]


def label_trk(d):
    name = {0: "player", 1: "ball"}
    return [f"{name[c]} #{t}" for c, t in zip(d.class_id, d.tracker_id)]


def header(panel, text, sub):
    h, w = panel.shape[:2]
    bar = np.full((HEADER, w, 3), 28, np.uint8)
    cv2.putText(bar, text, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bar, sub, (12, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 200, 255), 1)
    return np.vstack([bar, panel])


def render_video():
    frames = list(C.read_frames())
    tmp = Path(tempfile.mkdtemp()) / "raw.mp4"
    vw = C.VideoWriter(tmp)
    for i, frame in frames:
        left = frame.copy()
        dd = to_det(det[i])
        left = box_class.annotate(left, dd)
        left = lab_class.annotate(left, dd, labels=label_det(dd))

        right = frame.copy()
        td = to_det(trk[i], with_id=True)
        right = trace.annotate(right, td)
        right = box_track.annotate(right, td)
        right = lab_track.annotate(right, td, labels=label_trk(td))

        left = cv2.resize(left, (PANEL_W, PANEL_H))
        right = cv2.resize(right, (PANEL_W, PANEL_H))
        left = header(left, "DETECTION", "per frame  -  no identity, no memory")
        right = header(right, "TRACKING (YOLO + ByteTrack)", "persistent id + motion trail")
        canvas = np.hstack([left, np.full((left.shape[0], 4, 3), 90, np.uint8), right])
        cv2.putText(canvas, f"frame {i:3d}", (canvas.shape[1] // 2 - 50, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vw.write(canvas)
    vw.close()

    out = C.OUT / "compare_od_vs_track.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-loglevel", "error",
                    str(out)], check=True)
    print("wrote", out)


def render_3up():
    """Detection | ByteTrack | BoT-SORT, side by side — tracking done two ways."""
    assert bot is not None, "run extract_botsort.py first"
    W3, H3 = 624, 351
    byte_trace = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK, trace_length=25, thickness=2)
    bot_trace = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK, trace_length=25, thickness=2)
    frames = list(C.read_frames())
    tmp = Path(tempfile.mkdtemp()) / "raw.mp4"
    vw = C.VideoWriter(tmp)
    for i, frame in frames:
        p1 = frame.copy()
        dd = to_det(det[i])
        p1 = box_class.annotate(p1, dd); p1 = lab_class.annotate(p1, dd, labels=label_det(dd))

        p2 = frame.copy()
        bt = to_det(trk[i], with_id=True)
        p2 = byte_trace.annotate(p2, bt); p2 = box_track.annotate(p2, bt)
        p2 = lab_track.annotate(p2, bt, labels=label_trk(bt))

        p3 = frame.copy()
        ot = to_det(bot[i], with_id=True)
        p3 = bot_trace.annotate(p3, ot); p3 = box_track.annotate(p3, ot)
        p3 = lab_track.annotate(p3, ot, labels=label_trk(ot))

        p1 = header(cv2.resize(p1, (W3, H3)), "DETECTION", "per frame - no memory")
        p2 = header(cv2.resize(p2, (W3, H3)), "TRACKING: ByteTrack", "IoU + motion only")
        p3 = header(cv2.resize(p3, (W3, H3)), "TRACKING: BoT-SORT", "+ camera-motion compensation")
        sep = np.full((p1.shape[0], 4, 3), 90, np.uint8)
        canvas = np.hstack([p1, sep, p2, sep, p3])
        cv2.putText(canvas, f"frame {i:3d}", (canvas.shape[1] // 2 - 48, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vw.write(canvas)
    vw.close()
    out = C.OUT / "compare_3up.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-loglevel", "error", str(out)], check=True)
    print("wrote", out)


def _annot(f, i):
    top, bot = f.copy(), f.copy()
    dd = to_det(det[i]); td = to_det(trk[i], with_id=True)
    top = box_class.annotate(top, dd); top = lab_class.annotate(top, dd, labels=label_det(dd))
    bot = trace.annotate(bot, td); bot = box_track.annotate(bot, td)
    bot = lab_track.annotate(bot, td, labels=label_trk(td))
    return top, bot


def render_strip(start=31, n=6, crop=(380, 525, 700, 805)):
    """Consecutive frames over a ball blink: detection (top) vs tracking (bottom)."""
    frames = {i: f for i, f in C.read_frames()}
    x0, y0, x1, y1 = crop
    cols = []
    for k in range(n):
        i = start + k
        top, bot = _annot(frames[i], i)
        top, bot = top[y0:y1, x0:x1], bot[y0:y1, x0:x1]
        tw = 300
        th = int(top.shape[0] * tw / top.shape[1])
        top = cv2.resize(top, (tw, th)); bot = cv2.resize(bot, (tw, th))
        cv2.putText(top, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        col = np.vstack([top, np.full((3, tw, 3), 255, np.uint8), bot])
        cols.append(col)
    grid = np.hstack([np.hstack([c, np.full((c.shape[0], 3, 3), 255, np.uint8)]) for c in cols])
    # row labels
    lab = np.full((grid.shape[0], 150, 3), 255, np.uint8)
    cv2.putText(lab, "DETECTION", (6, grid.shape[0] // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
    cv2.putText(lab, "TRACKING", (6, 3 * grid.shape[0] // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
    grid = np.hstack([lab, grid])
    cv2.imwrite(str(C.OUT / "frame_strip.png"), grid)
    print("wrote frame_strip.png")


def render_still(i, name):
    """A readable full-frame detection | tracking still."""
    frame = {j: f for j, f in C.read_frames()}[i]
    top, bot = _annot(frame, i)
    W = 760
    top = cv2.resize(top, (W, int(top.shape[0] * W / top.shape[1])))
    bot = cv2.resize(bot, (W, int(bot.shape[0] * W / bot.shape[1])))
    top = header(top, "DETECTION", "per frame  -  no identity, no memory")
    bot = header(bot, "TRACKING (YOLO + ByteTrack)", "persistent id + motion trail")
    canvas = np.hstack([top, np.full((top.shape[0], 4, 3), 90, np.uint8), bot])
    cv2.imwrite(str(C.OUT / name), canvas)
    print("wrote", name)


if __name__ == "__main__":
    render_video()
    render_strip()
    # readable full-frame stills (optional; not embedded in the post):
    # render_still(104, "still_f104.png")
