"""Step 3b — Build the one-/few-shot REFERENCE POOL.

Pick the most ISOLATED clean kilns OUTSIDE the AOI (so each chip shows a single
kiln), fetch a z=18 ESRI chip centred on each, and project the SentinelKilnDB
oriented bounding box onto the chip to get a precise binary reference mask.
Outputs chips, masks and overlay previews under work/refpool/.
"""
import io, json, math, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path.home() / "kiln-insid3"
DATA = ROOT / "data"
OUT = ROOT / "work" / "refpool"; OUT.mkdir(parents=True, exist_ok=True)
Z, TILE = 18, 1024
TILE_PX, TILE_M = 128, 1280.0
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")


def lonlat_to_gpx(lon, lat, z):
    n = 256 * 2 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def tilepx_to_lonlat(tile_lat, tile_lon, px, py):
    fx, fy = px / TILE_PX - 0.5, py / TILE_PX - 0.5
    dlat = -(fy * TILE_M) / 111111.0
    dlon = (fx * TILE_M) / (111111.0 * math.cos(math.radians(tile_lat)))
    return tile_lon + dlon, tile_lat + dlat


def fetch_tile(z, x, y, retries=4):
    for k in range(retries):
        try:
            req = urllib.request.Request(ESRI.format(z=z, x=x, y=y),
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            if k == retries - 1:
                return Image.new("RGB", (256, 256), (0, 0, 0))
            time.sleep(0.4 * (k + 1))


def fetch_chip(lon, lat):
    cx, cy = lonlat_to_gpx(lon, lat, Z)
    X0, Y0 = int(round(cx - TILE / 2)), int(round(cy - TILE / 2))
    tx0, ty0 = X0 // 256, Y0 // 256
    tx1, ty1 = (X0 + TILE - 1) // 256, (Y0 + TILE - 1) // 256
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_tile, Z, tx, ty): (tx, ty)
                for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)}
        for f in as_completed(futs):
            tx, ty = futs[f]
            canvas.paste(f.result(), ((tx - tx0) * 256, (ty - ty0) * 256))
    chip = canvas.crop((X0 - tx0 * 256, Y0 - ty0 * 256,
                        X0 - tx0 * 256 + TILE, Y0 - ty0 * 256 + TILE))
    return chip, X0, Y0


aoi = json.loads((DATA / "aoi.json").read_text())
gt = pd.read_csv(DATA / "bulandshahr_kilns_district.csv")

# nearest-neighbour distance (deg) to find ISOLATED kilns
lat = gt.lat.values; lon = gt.lon.values
nn = np.full(len(gt), np.inf)
for i in range(len(gt)):
    d = np.sqrt((lat - lat[i])**2 + ((lon - lon[i]) * math.cos(math.radians(28.4)))**2)
    d[i] = np.inf
    nn[i] = d.min()
gt = gt.assign(nn_deg=nn)

# candidates: OUTSIDE AOI (with margin), most isolated first, Zigzag (dominant class)
m = 0.02
outside = gt[(gt.lat < aoi["lat0"] - m) | (gt.lat > aoi["lat1"] + m) |
             (gt.lon < aoi["lon0"] - m) | (gt.lon > aoi["lon1"] + m)]
cand = outside[outside.cls == "Zigzag"].sort_values("nn_deg", ascending=False).head(14)

records = []
for rank, (_, r) in enumerate(cand.iterrows()):
    chip, X0, Y0 = fetch_chip(r.lon, r.lat)
    # project OBB vertices -> chip pixels
    pts = json.loads(r.pts)
    poly = []
    for k in range(4):
        plon, plat = tilepx_to_lonlat(r.tile_lat, r.tile_lon, pts[2*k], pts[2*k+1])
        gx, gy = lonlat_to_gpx(plon, plat, Z)
        poly.append((gx - X0, gy - Y0))
    mask = Image.new("L", (TILE, TILE), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    # overlay preview
    ov = chip.copy(); d = ImageDraw.Draw(ov, "RGBA")
    d.polygon(poly, outline=(255, 59, 48, 255), fill=(255, 59, 48, 70))
    chip.save(OUT / f"ref_{rank:02d}.jpg", quality=92)
    mask.save(OUT / f"ref_{rank:02d}_mask.png")
    ov.save(OUT / f"ref_{rank:02d}_overlay.jpg", quality=92)
    records.append(dict(rank=rank, lat=float(r.lat), lon=float(r.lon),
                        cls=r.cls, nn_deg=float(r.nn_deg),
                        mask_px=int(np.array(mask).sum() // 255)))
    print(f"ref_{rank:02d} ({r.lat:.4f},{r.lon:.4f}) nn={r.nn_deg*111:.2f}km mask_px={records[-1]['mask_px']}")
(OUT / "pool.json").write_text(json.dumps(records, indent=2))
print(f"\nDONE: {len(records)} reference candidates -> {OUT}")
