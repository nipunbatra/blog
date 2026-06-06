"""Compute metrics + identity figures from the cached detection/tracking JSON.

Outputs:
  outputs/metrics.json          all the headline numbers
  outputs/flicker_timeline.png  ball: per-frame detection vs tracker-id bands
  outputs/identity_timeline.png track-id vs frame for players & ball
"""
import json
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import common as C

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold",
})

det = json.load(open(C.OUT / "detections.json"))["frames"]
trk = json.load(open(C.OUT / "tracks.json"))["frames"]
N = len(det)


def class_present(frames, cls):
    return np.array([any(r["cls"] == cls for r in fr) for fr in frames])


def transitions(present):
    return int(np.sum(present[1:] != present[:-1]))


def gaps(present):
    g, cur = [], 0
    for p in present:
        if not p:
            cur += 1
        elif cur:
            g.append(cur); cur = 0
    if cur:
        g.append(cur)
    return g


# ---- detection-side (no memory) ----
od_ball = class_present(det, C.BALL)
od_person_counts = np.array([sum(r["cls"] == C.PERSON for r in fr) for fr in det])
ball_gaps = gaps(od_ball)

# ---- tracking-side (with memory) ----
ball_id_per_frame = [[r["id"] for r in fr if r["cls"] == C.BALL and r["id"] is not None]
                     for fr in trk]
ball_ids_all = [i for fr in ball_id_per_frame for i in fr]
person_id_per_frame = [[r["id"] for r in fr if r["cls"] == C.PERSON and r["id"] is not None]
                       for fr in trk]
person_ids_all = [i for fr in person_id_per_frame for i in fr]

# How many ball blinks did the tracker BRIDGE? After a 1+ frame gap, is the
# next id the same as before the gap (memory held) or new (identity lost)?
bridged = broke = 0
prev_seen_id = None
in_gap = False
for ids in ball_id_per_frame:
    if ids:
        if in_gap and prev_seen_id is not None:
            if ids[0] == prev_seen_id:
                bridged += 1
            else:
                broke += 1
        in_gap = False
        prev_seen_id = ids[0]
    else:
        if prev_seen_id is not None:
            in_gap = True

metrics = {
    "n_frames": N, "fps": C.FPS, "duration_s": round(N / C.FPS, 1),
    "detection": {
        "ball_detection_rate": round(float(od_ball.mean()), 3),
        "ball_frames_seen": int(od_ball.sum()),
        "ball_onoff_transitions": transitions(od_ball),
        "ball_absence_gaps_frames": sorted(ball_gaps, reverse=True),
        "longest_ball_gap_frames": max(ball_gaps) if ball_gaps else 0,
        "longest_ball_gap_seconds": round((max(ball_gaps) if ball_gaps else 0) / C.FPS, 2),
        "persons_per_frame_min": int(od_person_counts.min()),
        "persons_per_frame_max": int(od_person_counts.max()),
        "persons_per_frame_mean": round(float(od_person_counts.mean()), 2),
    },
    "tracking": {
        "ball_frames_with_id": int(sum(1 for ids in ball_id_per_frame if ids)),
        "ball_unique_ids": len(set(ball_ids_all)),
        "ball_ids": sorted(set(ball_ids_all)),
        "player_frames_with_id": int(sum(1 for ids in person_id_per_frame if ids)),
        "player_unique_ids": len(set(person_ids_all)),
        "player_ids": sorted(set(person_ids_all)),
        "ball_gaps_bridged_same_id": bridged,
        "ball_gaps_broke_new_id": broke,
        "longest_single_ball_track_frames": max(Counter(ball_ids_all).values()) if ball_ids_all else 0,
    },
}
(C.OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# ============================ Figure 1: flicker vs id bands ============================
fig, axes = plt.subplots(2, 1, figsize=(11, 3.6), sharex=True, layout="constrained")
n_det = int(od_ball.sum())
n_trk = int(sum(1 for ids in ball_id_per_frame if ids))

# top: detection presence barcode
ax = axes[0]
ax.imshow(od_ball[None, :], aspect="auto", cmap="Greens", vmin=0, vmax=1,
          extent=[0, N, 0, 1])
ax.set_yticks([])
ax.set_title(f"Detection (per-frame): ball found in {n_det}/{N} frames  "
             "(green = found, white = absent)", loc="left")

# bottom: tracker id band per frame (color = id)
ax = axes[1]
uniq = sorted(set(ball_ids_all))
cmap = plt.get_cmap("tab20")
idcolor = {i: cmap(k % 20) for k, i in enumerate(uniq)}
band = np.ones((1, N, 4))
for f, ids in enumerate(ball_id_per_frame):
    if ids:
        band[0, f] = idcolor[ids[0]]
ax.imshow(band, aspect="auto", extent=[0, N, 0, 1])
ax.set_yticks([])
ax.set_title(f"Tracking (ByteTrack): a confirmed ball id in only {n_trk}/{N} frames, "
             f"split across {len(uniq)} ids (each colour)", loc="left")
ax.set_xlabel("frame")
fig.suptitle("One tennis ball, same detector. Tracking discards detections it can't link "
             "in time — so the fast ball gets fewer boxes, not more.",
             fontsize=12, weight="bold")
fig.savefig(C.OUT / "flicker_timeline.png", bbox_inches="tight")
plt.close(fig)

# ============================ Figure 2: identity timeline ============================
fig, ax = plt.subplots(figsize=(11, 4.2))
# players (blue-ish) and ball (red-ish), id on y
for f, ids in enumerate(person_id_per_frame):
    for i in ids:
        ax.scatter(f, i, s=10, color="#2c6fbb", marker="s")
for f, ids in enumerate(ball_id_per_frame):
    for i in ids:
        ax.scatter(f, i, s=16, color="#c44536", marker="o")
ax.set_xlabel("frame")
ax.set_ylabel("track id")
ax.set_title("Track id vs time — long horizontal lines = stable identity, "
             "stubs = fragmentation")
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#2c6fbb",
           markersize=8, label=f"player ({metrics['tracking']['player_unique_ids']} ids)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44536",
           markersize=8, label=f"ball ({metrics['tracking']['ball_unique_ids']} ids)"),
], loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(C.OUT / "identity_timeline.png", bbox_inches="tight")
plt.close(fig)
print("\nwrote flicker_timeline.png, identity_timeline.png")
