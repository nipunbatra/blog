"""Step 5 — Aggregate INSID3 detections, dedupe tile overlaps, match to the 40
SentinelKilnDB ground-truth kilns in the AOI, and compute precision / recall /
F1 per config. Renders the headline detection map + the shot-sweep / ablation
plots. Runs locally (reads work/ rsynced back from Bhaskar).
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("/Users/nipun/git/blog/posts/kiln-insid3")
WORK = ROOT / "work"; OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"

geo = json.loads((WORK / "aoi_geo.json").read_text())
MPP = geo["mpp"]
DEDUP_M = 70.0
MATCH_M = 100.0
A_MIN_POST = 10000           # px (~53 m side): drop sub-kiln fragments, keeps recall
DEDUP_PX = DEDUP_M / MPP

gt = pd.read_csv(DATA / "aoi_kilns.csv")

def lonlat_to_gpx(lon, lat, z):
    n = 256 * 2 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)

gt_gx, gt_gy = zip(*[lonlat_to_gpx(lo, la, geo["z"]) for lo, la in zip(gt.lon, gt.lat)])
gt = gt.assign(gx=gt_gx, gy=gt_gy)


def dedupe(dets):
    """greedy merge of detections whose centroids are within DEDUP_PX."""
    if not dets:
        return []
    pts = np.array([[d["gx"], d["gy"]] for d in dets])
    areas = np.array([d["area"] for d in dets])
    order = np.argsort(-areas)
    used = np.zeros(len(dets), bool); merged = []
    for i in order:
        if used[i]:
            continue
        d2 = ((pts - pts[i]) ** 2).sum(1)
        grp = np.where((d2 <= DEDUP_PX ** 2) & (~used))[0]
        used[grp] = True
        gx, gy = pts[grp].mean(0)
        lon, lat = gpx_to_lonlat(gx, gy, geo["z"])
        merged.append(dict(gx=float(gx), gy=float(gy), lon=lon, lat=lat,
                           area=int(areas[grp].max())))
    return merged


def gpx_to_lonlat(x, y, z):
    n = 256 * 2 ** z
    return (x / n * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))))


def match(dets, gtg):
    """greedy nearest matching det<->gt within MATCH_M."""
    G = gtg[["gx", "gy"]].values
    D = np.array([[d["gx"], d["gy"]] for d in dets]) if dets else np.zeros((0, 2))
    gt_matched = np.zeros(len(G), bool)
    det_tp = np.zeros(len(D), bool)
    pairs = []
    for di in range(len(D)):
        dd = np.sqrt(((G - D[di]) ** 2).sum(1)) * MPP
        order = np.argsort(dd)
        for gi in order:
            if dd[gi] > MATCH_M:
                break
            if not gt_matched[gi]:
                gt_matched[gi] = True; det_tp[di] = True
                pairs.append((di, gi)); break
    tp = int(gt_matched.sum()); fn = int((~gt_matched).sum()); fp = int((~det_tp).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fn=fn, fp=fp, precision=round(prec, 3), recall=round(rec, 3),
                f1=round(f1, 3), n_det=len(D)), gt_matched, det_tp


CONFIGS = ["sam_k1", "sam_k2", "sam_k4", "sam_k8", "obb_k1"]
results = {}
per_config_dets = {}
for label in CONFIGS:
    p = WORK / "detections" / f"dets_{label}.json"
    if not p.exists():
        print("missing", p); continue
    raw = [d for d in json.loads(p.read_text()) if d["area"] >= A_MIN_POST]
    dd = dedupe(raw)
    m, gtm, dtp = match(dd, gt)
    results[label] = m
    per_config_dets[label] = (dd, gtm, dtp)
    print(f"{label:8s} raw={len(raw):4d} dedup={len(dd):3d}  TP={m['tp']:2d} FP={m['fp']:3d} "
          f"FN={m['fn']:2d}  P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

(OUT / "metrics.json").write_text(json.dumps(dict(
    n_gt=int(len(gt)), match_m=MATCH_M, dedup_m=DEDUP_M, a_min_post=A_MIN_POST,
    mpp=MPP, results=results), indent=2))

# persist headline deduped detections + matched flags for the FP spot-check
HEADc = "sam_k1"
dd_h, gtm_h, dtp_h = per_config_dets[HEADc]
for j, d in enumerate(dd_h):
    d["matched_gt"] = bool(dtp_h[j])
(WORK / "detections" / f"dedup_{HEADc}.json").write_text(json.dumps(dd_h, indent=2))
# per-GT matched flags (detection-level) so the qualitative grids match the metric
gt_out = gt.copy(); gt_out["found"] = gtm_h
gt_out.to_csv(WORK / "detections" / f"gt_matched_{HEADc}.csv", index=False)

# ---- shot-sweep + ablation figure ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ks = [1, 2, 4, 8]
sam = [results[f"sam_k{k}"] for k in ks]
ax = axes[0]
ax.plot(ks, [r["recall"] for r in sam], "o-", color="#0a84ff", label="recall", lw=2)
ax.plot(ks, [r["precision"] for r in sam], "s-", color="#ff9f0a", label="precision", lw=2)
ax.plot(ks, [r["f1"] for r in sam], "^-", color="#c44536", label="F1", lw=2)
ax.set_xticks(ks); ax.set_xlabel("shots (reference kilns)"); ax.set_ylim(0, 1.02)
ax.set_title("1-shot vs few-shot (SAM-refined reference)"); ax.legend(); ax.grid(alpha=.3)

ax = axes[1]
labels = ["OBB box\n(k=1)", "SAM mask\n(k=1)"]
o, s = results["obb_k1"], results["sam_k1"]
x = np.arange(2)
for j, key, c in [(0, "precision", "#ff9f0a"), (1, "recall", "#0a84ff"), (2, "f1", "#c44536")]:
    ax.bar(x + (j-1)*0.25, [o[key], s[key]], width=0.25, label=key, color=c)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)
ax.set_title("Does reference mask quality matter?"); ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(OUT / "02_shots_and_ablation.png", dpi=140, bbox_inches="tight")
print("saved 02_shots_and_ablation.png")

# ---- headline detection map (best SAM config) -----------------------------
HEAD = "sam_k1"
dd, gtm, dtp = per_config_dets[HEAD]
mos = Image.open(WORK / "aoi_z18.jpg")
scale = 1600 / mos.width
small = mos.resize((int(mos.width*scale), int(mos.height*scale)))
fig, ax = plt.subplots(figsize=(11, 11*small.height/small.width))
ax.imshow(small)
def to_small(gx, gy):
    return (gx - geo["X0"]) * scale, (gy - geo["Y0"]) * scale
for i, r in enumerate(gt.itertuples()):
    sx, sy = to_small(r.gx, r.gy)
    ax.scatter(sx, sy, s=90, facecolors="none",
               edgecolors=("#34c759" if gtm[i] else "#ff3b30"), linewidths=2)
for j, d in enumerate(dd):
    if not dtp[j]:
        sx, sy = to_small(d["gx"], d["gy"])
        ax.scatter(sx, sy, marker="x", s=55, c="#ffd60a", linewidths=2)
ax.set_axis_off()
m = results[HEAD]
ax.set_title(f"INSID3 {HEAD} over Bulandshahr AOI — "
             f"found {m['tp']}/{len(gt)} GT kilns (recall {m['recall']:.0%}), "
             f"{m['fp']} extra detections\n"
             f"green=found GT  red=missed GT  yellow ×=extra detection", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "03_detection_map.png", dpi=145, bbox_inches="tight")
print("saved 03_detection_map.png")
