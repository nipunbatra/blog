"""Step 3 — Fetch ESRI World Imagery (z=18, ~0.5 m/px) over the AOI, build a
tile-aligned mosaic with an exact pixel<->lat/lon geotransform, cut overlapping
1024 px target tiles for INSID3, and build the single one-shot REFERENCE
(kiln chip + mask) from a clean kiln located OUTSIDE the AOI.

Runs on Bhaskar (good bandwidth). Writes everything under WORK/.
"""
import io, json, math, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path.home() / "kiln-insid3"
DATA = ROOT / "data"                 # rsynced from repo posts/kiln-insid3/data
WORK = ROOT / "work"; WORK.mkdir(exist_ok=True)
TILES = WORK / "tiles"; TILES.mkdir(exist_ok=True)
Z = 18
TILE = 1024          # INSID3 target tile size (px)
STRIDE = 768         # 25% overlap so border kilns survive
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")


def lonlat_to_gpx(lon, lat, z):
    n = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def gpx_to_lonlat(x, y, z):
    n = 256 * 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def fetch_tile(z, x, y, retries=4):
    url = ESRI.format(z=z, x=x, y=y)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            if k == retries - 1:
                return Image.new("RGB", (256, 256), (0, 0, 0))
            time.sleep(0.4 * (k + 1))


def build_mosaic(lon0, lat0, lon1, lat1, z, out_path):
    # global pixel bbox (note: lat0<lat1 but y grows southward)
    x_l, y_t = lonlat_to_gpx(lon0, lat1, z)
    x_r, y_b = lonlat_to_gpx(lon1, lat0, z)
    X0, Y0 = int(math.floor(x_l)), int(math.floor(y_t))
    X1, Y1 = int(math.ceil(x_r)), int(math.ceil(y_b))
    W, H = X1 - X0, Y1 - Y0
    tx0, ty0 = X0 // 256, Y0 // 256
    tx1, ty1 = (X1 - 1) // 256, (Y1 - 1) // 256
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    jobs = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
    print(f"  fetching {len(jobs)} z{z} tiles -> mosaic {W}x{H}px")

    def work(j):
        tx, ty = j
        return j, fetch_tile(z, tx, ty)
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for (tx, ty), im in ex.map(work, jobs):
            canvas.paste(im, ((tx - tx0) * 256, (ty - ty0) * 256))
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(jobs)}")
    # crop to exact AOI global-pixel bbox
    ox, oy = tx0 * 256, ty0 * 256
    mosaic = canvas.crop((X0 - ox, Y0 - oy, X1 - ox, Y1 - oy))
    mosaic.save(out_path, quality=90)
    geo = dict(z=z, X0=X0, Y0=Y0, W=W, H=H,
               lon0=lon0, lat0=lat0, lon1=lon1, lat1=lat1,
               mpp=156543.03392 / (2 ** z) * math.cos(math.radians((lat0 + lat1) / 2)))
    return mosaic, geo


def px_to_lonlat(px, py, geo):
    return gpx_to_lonlat(geo["X0"] + px, geo["Y0"] + py, geo["z"])


# ---- 1) AOI mosaic --------------------------------------------------------
aoi = json.loads((DATA / "aoi.json").read_text())
print("AOI:", aoi)
mosaic, geo = build_mosaic(aoi["lon0"], aoi["lat0"], aoi["lon1"], aoi["lat1"],
                           Z, WORK / "aoi_z18.jpg")
(WORK / "aoi_geo.json").write_text(json.dumps(geo, indent=2))
print(f"  mosaic {mosaic.size}, {geo['mpp']:.3f} m/px")

# ---- 2) cut overlapping 1024 target tiles, record each tile's origin ------
W, H = mosaic.size
tiles_meta = []
for ty in range(0, max(1, H - TILE + STRIDE), STRIDE):
    for tx in range(0, max(1, W - TILE + STRIDE), STRIDE):
        x0, y0 = min(tx, max(0, W - TILE)), min(ty, max(0, H - TILE))
        crop = mosaic.crop((x0, y0, x0 + TILE, y0 + TILE))
        name = f"tile_{y0:05d}_{x0:05d}.jpg"
        crop.save(TILES / name, quality=90)
        tiles_meta.append(dict(name=name, x0=x0, y0=y0))
seen = set()
tiles_meta = [t for t in tiles_meta if (t["x0"], t["y0"]) not in seen and not seen.add((t["x0"], t["y0"]))]
(WORK / "tiles_meta.json").write_text(json.dumps(tiles_meta, indent=2))
print(f"  cut {len(tiles_meta)} target tiles ({TILE}px, stride {STRIDE})")

# ---- 3) build the one-shot REFERENCE from a clean kiln OUTSIDE the AOI -----
gt = pd.read_csv(DATA / "bulandshahr_kilns_district.csv")
# candidates: kilns well outside the AOI bbox, prefer Zigzag (dominant class)
out = gt[(gt.lat < aoi["lat0"] - 0.02) | (gt.lat > aoi["lat1"] + 0.02) |
         (gt.lon < aoi["lon0"] - 0.02) | (gt.lon > aoi["lon1"] + 0.02)]
out = out[out.cls == "Zigzag"].reset_index(drop=True)

REF = WORK / "reference"; REF.mkdir(exist_ok=True)
# fetch a 1024px chip centred on each of a few candidates so we can hand-pick
cands = out.iloc[:: max(1, len(out) // 12)].head(12).reset_index(drop=True)
ref_records = []
for i, r in cands.iterrows():
    cx, cy = lonlat_to_gpx(r.lon, r.lat, Z)
    X0, Y0 = int(round(cx - TILE / 2)), int(round(cy - TILE / 2))
    # direct chip fetch via global-pixel crop
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
    chip.save(REF / f"cand_{i:02d}.jpg", quality=92)
    ref_records.append(dict(i=i, lat=float(r.lat), lon=float(r.lon), cls=r.cls,
                            X0=X0, Y0=Y0))
(REF / "candidates.json").write_text(json.dumps(ref_records, indent=2))
print(f"  fetched {len(ref_records)} reference candidate chips -> {REF}")
print("DONE step 3")
