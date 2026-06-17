"""Step 6 — qualitative grids that AGREE with the reported detection metric.

Uses step 5's detection-level matching (run 05 first):
  - WORKS:  GT kilns that were matched by a deduped detection.
  - MISSES: GT kilns NOT matched (the real false negatives).
  - FALSE ALARMS: deduped detections that matched no GT (candidate FPs).

Crops are taken straight from the ESRI z18 mosaic so each panel is centred on the
object, regardless of tile boundaries.
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

ROOT = Path("/Users/nipun/git/blog/posts/kiln-insid3")
WORK = ROOT / "work"; OUT = ROOT / "outputs"; DET = WORK / "detections"
Image.MAX_IMAGE_PIXELS = None
HEAD = "sam_k1"
geo = json.loads((WORK / "aoi_geo.json").read_text())
mos = Image.open(WORK / "aoi_z18.jpg").convert("RGB")

def gpx(lon, lat, z):
    n = 256 * 2 ** z
    return ((lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

gt = pd.read_csv(DET / f"gt_matched_{HEAD}.csv")
dd = json.loads((DET / f"dedup_{HEAD}.json").read_text())

def crop_at(gx, gy, half, ring):
    px, py = gx - geo["X0"], gy - geo["Y0"]
    c = mos.crop((int(px-half), int(py-half), int(px+half), int(py+half))).resize((260, 260))
    d = ImageDraw.Draw(c); d.ellipse([130-26, 130-26, 130+26, 130+26], outline=ring, width=3)
    return c

def grid(items, title, fname, ring):
    items = items[:8]; cols = 4; rows = (len(items)+cols-1)//cols or 1; TH = 260
    sheet = Image.new("RGB", (cols*TH, rows*TH+26), (12, 12, 12))
    d = ImageDraw.Draw(sheet); d.text((6, 6), title, fill="white")
    for i, c in enumerate(items):
        r, cc = divmod(i, cols); sheet.paste(c, (cc*TH, r*TH+26))
    sheet.save(OUT / fname); print("saved", fname, "n=", len(items))

works = [crop_at(*gpx(r.lon, r.lat, geo["z"]), 140, (52,199,89)) for r in gt[gt.found].itertuples()]
miss  = [crop_at(*gpx(r.lon, r.lat, geo["z"]), 140, (255,59,48)) for r in gt[~gt.found].itertuples()]
fps_d = [d for d in dd if not d["matched_gt"]]
fps_d = sorted(fps_d, key=lambda d: -d["area"])[::max(1, len(fps_d)//8)][:8]
fps   = [crop_at(d["gx"], d["gy"], max(140, int(math.sqrt(d["area"])*0.9)), (255,214,10)) for d in fps_d]

print(f"GT found={gt.found.sum()} missed={(~gt.found).sum()}  unmatched dets={sum(1 for d in dd if not d['matched_gt'])}")
grid(works, f"WORKS — GT kilns INSID3 ({HEAD}) detected (green)", "04_works.png", (52,199,89))
grid(miss,  f"MISSES — GT kilns INSID3 did not detect (red)",     "05_misses.png", (255,59,48))
grid(fps,   f"FALSE ALARMS — detections with no GT label (yellow)", "06_false_alarms.png", (255,214,10))
