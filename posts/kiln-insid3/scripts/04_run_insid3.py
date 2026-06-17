"""Step 4 — Run INSID3 over every AOI tile for several configs and dump kiln
detections (connected components of the predicted mask -> geo centroids).

Configs (set in CONFIGS): 1/2/4/8-shot with SAM-refined masks, plus a 1-shot
OBB-rectangle config for the mask-quality ablation. The reference order is fixed
(REF_ORDER) so k-shot always uses the first k references.

Per tile we threshold the boolean mask, label connected components, drop blobs
outside a plausible kiln-area range, and record each component's centroid in
global pixel + lat/lon coordinates. Masks for every tile are also saved (packed)
so qualitative grids can be built later.

Run on Bhaskar GPU inside the INSID3 venv with TORCH_HOME set.
"""
import json, time, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scipy import ndimage

ROOT = Path.home() / "kiln-insid3"
WORK = ROOT / "work"; TILES = WORK / "tiles"; REF = WORK / "refpool"
OUT = WORK / "detections"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "INSID3"))
from models import build_insid3

geo = json.loads((WORK / "aoi_geo.json").read_text())
tiles = json.loads((WORK / "tiles_meta.json").read_text())
Z = geo["z"]; MPP = geo["mpp"]

# reference order (cleanest kilns first), chosen from the SAM comparison sheet
REF_ORDER = json.loads((WORK / "ref_order.json").read_text())   # e.g. [10,3,8,11,12,13,0,5]

CONFIGS = [
    dict(label="sam_k1", shots=1, mask="sam"),
    dict(label="sam_k2", shots=2, mask="sam"),
    dict(label="sam_k4", shots=4, mask="sam"),
    dict(label="sam_k8", shots=8, mask="sam"),
    dict(label="obb_k1", shots=1, mask="obb"),   # ablation: box vs SAM mask
]

# kiln area gate (px) at this resolution: ~25 m .. ~260 m equivalent box
A_MIN = int((25 / MPP) ** 2 * 0.4)
A_MAX = int((260 / MPP) ** 2)
print(f"mpp={MPP:.3f}  area gate [{A_MIN},{A_MAX}] px")

import math
def gpx_to_lonlat(x, y, z):
    n = 256 * 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def detections_from_mask(mask, tile):
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        area = len(xs)
        if area < A_MIN or area > A_MAX:
            continue
        cx = xs.mean() + tile["x0"]; cy = ys.mean() + tile["y0"]      # mosaic px
        gx = geo["X0"] + cx; gy = geo["Y0"] + cy                      # global px
        lon, lat = gpx_to_lonlat(gx, gy, Z)
        out.append(dict(tile=tile["name"], area=int(area),
                        gx=float(gx), gy=float(gy), lat=lat, lon=lon))
    return out


def run_config(cfg):
    refs = REF_ORDER[:cfg["shots"]]
    model = build_insid3(model_size="large", image_size=1024)
    masks_dir = OUT / cfg["label"]; masks_dir.mkdir(exist_ok=True)
    dets = []; t0 = time.time(); fg_tiles = 0
    for ti, tile in enumerate(tiles):
        for k in refs:
            model.set_reference(str(REF / f"ref_{k:02d}.jpg"),
                                str(REF / f"ref_{k:02d}_{cfg['mask']}.png"))
        model.set_target(str(TILES / tile["name"]))
        pred = model.segment().cpu().numpy() > 0.5
        if pred.any():
            fg_tiles += 1
            # save packed mask for qualitative figures (sparse -> small PNG)
            Image.fromarray((pred * 255).astype("uint8")).save(masks_dir / tile["name"].replace(".jpg", ".png"))
        dets.extend(detections_from_mask(pred, tile))
        if (ti + 1) % 25 == 0:
            print(f"  [{cfg['label']}] {ti+1}/{len(tiles)} tiles, {len(dets)} dets, {time.time()-t0:.0f}s")
    (OUT / f"dets_{cfg['label']}.json").write_text(json.dumps(dets, indent=2))
    print(f"[{cfg['label']}] DONE {len(dets)} raw dets over {len(tiles)} tiles "
          f"({fg_tiles} with fg) in {time.time()-t0:.0f}s")
    return len(dets)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for cfg in CONFIGS:
        if only and cfg["label"] != only:
            continue
        print(f"\n=== config {cfg} refs={REF_ORDER[:cfg['shots']]} ===")
        run_config(cfg)
