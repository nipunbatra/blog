"""Step 7 — Spot-check the UNMATCHED detections of the headline config.

Precision against SentinelKilnDB is unfair: the AOI has far more kilns than the
40 tile-sampled labels. So we sample the detections that did NOT match a GT kiln,
crop the ESRI imagery around each, and build a montage to hand-classify as
{real kiln, kiln-like false alarm}. The verdict feeds an "effective precision".
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

ROOT = Path("/Users/nipun/git/blog/posts/kiln-insid3")
WORK = ROOT / "work"; OUT = ROOT / "outputs"
Image.MAX_IMAGE_PIXELS = None
geo = json.loads((WORK / "aoi_geo.json").read_text())
mos = Image.open(WORK / "aoi_z18.jpg").convert("RGB")
HEAD = "sam_k1"
dd = json.loads((WORK / "detections" / f"dedup_{HEAD}.json").read_text())
unm = [d for d in dd if not d["matched_gt"]]
print(f"{HEAD}: {len(dd)} dets, {len(unm)} unmatched (candidate FPs)")

# deterministic sample of up to 24 unmatched, sorted by area (bigger = more kiln-like)
unm = sorted(unm, key=lambda d: -d["area"])
step = max(1, len(unm) // 24)
sample = unm[::step][:24]

TH = 240; cols = 6; rows = (len(sample) + cols - 1) // cols
sheet = Image.new("RGB", (cols * TH, rows * TH + 24), (10, 10, 10))
d = ImageDraw.Draw(sheet); d.text((6, 6), f"{HEAD} unmatched detections (sorted by size) — classify each", fill="white")
for i, det in enumerate(sample):
    px = det["gx"] - geo["X0"]; py = det["gy"] - geo["Y0"]
    half = max(120, int(math.sqrt(det["area"]) * 0.9))
    crop = mos.crop((int(px - half), int(py - half), int(px + half), int(py + half))).resize((TH, TH))
    dd2 = ImageDraw.Draw(crop)
    dd2.ellipse([TH//2-24, TH//2-24, TH//2+24, TH//2+24], outline=(255, 214, 10), width=3)
    dd2.text((4, 4), f"#{i}", fill="yellow")
    r, c = divmod(i, cols); sheet.paste(crop, (c*TH, r*TH+24))
sheet.save(OUT / "08_spotcheck.png")
(WORK / "spotcheck_sample.json").write_text(json.dumps(sample, indent=2))
print(f"saved 08_spotcheck.png ({len(sample)} crops) — classify, then record verdicts")
