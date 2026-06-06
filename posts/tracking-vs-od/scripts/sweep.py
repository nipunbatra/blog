"""Can we fix the ball fragmentation? Sweep the tracker's memory knobs.

  1) track_buffer sweep (ByteTrack): how long to keep a lost track alive.
  2) ByteTrack vs BoT-SORT (camera-motion compensation, +optional ReID).

For each config: how many frames does the ball get a confirmed id, and how
many distinct ids does the single ball get (lower = less fragmentation)?

  outputs/sweep.json, outputs/sweep.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ultralytics import YOLO
import common as C

mpl.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12.5, "axes.titleweight": "bold"})

CFG = Path(__file__).resolve().parent / "configs"
CFG.mkdir(exist_ok=True)

BYTE = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {buffer}
match_thresh: 0.8
fuse_score: True
"""
BOT = """tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {buffer}
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: {reid}
model: auto
"""


def write_cfg(name, text):
    p = CFG / name
    p.write_text(text)
    return str(p)


def run(tracker_yaml):
    model = YOLO(str(C.MODEL))
    ball_ids, person_ids, ball_frames = [], set(), 0
    for r in model.track(str(C.SRC), conf=C.TRACK_CONF, classes=[C.PERSON, C.BALL],
                         imgsz=C.IMGSZ, device=C.DEVICE, tracker=tracker_yaml,
                         persist=True, stream=True, verbose=False):
        b = r.boxes
        if b.id is None:
            continue
        ids = b.id.cpu().numpy().astype(int)
        cls = b.cls.cpu().numpy().astype(int)
        bids = ids[cls == C.BALL]
        if len(bids):
            ball_frames += 1
            ball_ids.extend(bids.tolist())
        person_ids.update(ids[cls == C.PERSON].tolist())
    return {"ball_frames_with_id": ball_frames,
            "ball_unique_ids": len(set(ball_ids)),
            "player_unique_ids": len(person_ids)}


def main(replot_only=False):
    cache = C.OUT / "sweep.json"
    if replot_only and cache.exists():
        results = json.loads(cache.read_text())
        results["buffer_sweep"] = {int(k): v for k, v in results["buffer_sweep"].items()}
    else:
        results = {"buffer_sweep": {}, "trackers": {}}
        for buf in [15, 30, 60, 120]:
            cfg = write_cfg(f"byte_{buf}.yaml", BYTE.format(buffer=buf))
            results["buffer_sweep"][buf] = run(cfg)
            print(f"buffer={buf}: {results['buffer_sweep'][buf]}")
        trackers = {
            "ByteTrack": write_cfg("byte_30.yaml", BYTE.format(buffer=30)),
            "BoT-SORT": write_cfg("bot_30.yaml", BOT.format(buffer=30, reid="False")),
            "BoT-SORT+ReID": write_cfg("bot_reid.yaml", BOT.format(buffer=30, reid="True")),
        }
        for name, cfg in trackers.items():
            try:
                results["trackers"][name] = run(cfg)
            except Exception as e:
                results["trackers"][name] = {"error": str(e)[:200]}
            print(f"{name}: {results['trackers'][name]}")
        cache.write_text(json.dumps(results, indent=2))

    # ---------------- figure ----------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    bs = results["buffer_sweep"]
    xs = list(bs.keys())
    a1.plot(xs, [bs[b]["ball_unique_ids"] for b in xs], "o-", color="#c44536", label="ball ids (1 ball)")
    a1.plot(xs, [bs[b]["ball_frames_with_id"] for b in xs], "s-", color="#2c6fbb", label="ball frames w/ id")
    a1.set_xlabel("track_buffer (frames a lost track is kept alive)")
    a1.set_title("ByteTrack memory knob: no effect")
    a1.legend(frameon=False, fontsize=10, loc="lower right")
    a1.set_ylim(0, 56)
    a1.annotate("flat — the bottleneck is association,\nnot how long tracks are kept",
                xy=(62, 12.5), xytext=(30, 30), fontsize=9.5, color="#7a2a20",
                arrowprops=dict(arrowstyle="->", color="#7a2a20"))

    tr = {k: v for k, v in results["trackers"].items() if "error" not in v}
    names = list(tr.keys())
    x = np.arange(len(names))
    w = 0.35
    a2.bar(x - w / 2, [tr[n]["ball_unique_ids"] for n in names], w, color="#c44536", label="ball ids (1 ball)")
    a2.bar(x + w / 2, [tr[n]["player_unique_ids"] for n in names], w, color="#2c6fbb", label="player ids")
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=10)
    a2.set_title("Tracker family (buffer=30)")
    a2.legend(frameon=False, fontsize=10)
    for i, n in enumerate(names):
        a2.text(i - w / 2, tr[n]["ball_unique_ids"], str(tr[n]["ball_unique_ids"]), ha="center", va="bottom", fontsize=9)
        a2.text(i + w / 2, tr[n]["player_unique_ids"], str(tr[n]["player_unique_ids"]), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Fragmentation of one ball + the players, across tracker settings "
                 "(lower bars = more stable identity)", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(C.OUT / "sweep.png", bbox_inches="tight")
    print("wrote sweep.png")


if __name__ == "__main__":
    main()
