"""Step 1 — Extract ground-truth brick-kiln locations in/around Bulandshahr (UP)
from SentinelKilnDB, then pick a kiln-dense AOI for the one-shot experiment.

We only read the light columns (image_name + dota_label) from the parquet files
via HF's parquet filesystem — the heavy `image` column is never downloaded.

Each SentinelKilnDB tile is 128 px of Sentinel-2 at 10 m/px = 1280 m on a side,
geocoded by its filename `{lat}_{lon}.png` (tile CENTRE). dota_label entries are
`x1 y1 x2 y2 x3 y3 x4 y4 ClassName difficult` in tile-pixel coordinates, so each
kiln's pixel centroid maps to a precise lat/lon.
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

REPO = "hf://datasets/SustainabilityLabIITGN/SentinelKilnDB"
ROOT = Path("/Users/nipun/git/blog/posts/kiln-insid3")
DATA = ROOT / "data"; DATA.mkdir(parents=True, exist_ok=True)

TILE_PX = 128            # SentinelKilnDB tile size
TILE_M = 1280.0          # 128 px * 10 m/px
# Broad Bulandshahr district bbox (refined to a tight AOI by density below)
BBOX = dict(lat0=28.00, lat1=28.80, lon0=77.50, lon1=78.50)


def parse_latlon(name):
    s = str(name).replace(".png", "")
    a, b = s.split("_")
    return float(a), float(b)


def kiln_centroid_latlon(tile_lat, tile_lon, pts):
    """pts = [x1,y1,...,x4,y4] in 128px tile space -> kiln centroid lat/lon."""
    cx = sum(pts[0::2]) / 4.0
    cy = sum(pts[1::2]) / 4.0
    fx = cx / TILE_PX - 0.5            # [-0.5, 0.5], +x = east
    fy = cy / TILE_PX - 0.5            # +y = south (image row)
    dx_m = fx * TILE_M
    dy_m = fy * TILE_M
    dlat = -dy_m / 111111.0
    dlon = dx_m / (111111.0 * math.cos(math.radians(tile_lat)))
    return tile_lat + dlat, tile_lon + dlon


def load_split(split):
    df = pd.read_parquet(f"{REPO}/{split}/{split}.parquet",
                         columns=["image_name", "dota_label"])
    ll = df["image_name"].map(parse_latlon)
    df["lat"] = ll.map(lambda t: t[0])
    df["lon"] = ll.map(lambda t: t[1])
    df["split"] = split
    m = (df.lat.between(BBOX["lat0"], BBOX["lat1"]) &
         df.lon.between(BBOX["lon0"], BBOX["lon1"]))
    return df[m].copy()


rows = []
for split in ["train", "val", "test"]:
    sub = load_split(split)
    print(f"[{split}] {len(sub)} tiles in Bulandshahr bbox")
    for _, r in sub.iterrows():
        labels = r["dota_label"]
        if labels is None:
            continue
        for entry in list(labels):
            parts = str(entry).split()
            if len(parts) < 9:
                continue
            pts = [float(x) for x in parts[:8]]
            cls = parts[8]
            klat, klon = kiln_centroid_latlon(r["lat"], r["lon"], pts)
            rows.append(dict(split=split, tile=r["image_name"], cls=cls,
                             tile_lat=r["lat"], tile_lon=r["lon"],
                             lat=klat, lon=klon, pts=json.dumps(pts)))

gt = pd.DataFrame(rows)
print(f"\nTotal GT kilns in broad bbox: {len(gt)}")
print(gt["cls"].value_counts().to_string())
gt.to_csv(DATA / "bulandshahr_kilns_all.csv", index=False)

# --- pick the densest ~AOI_DEG window -------------------------------------
AOI_DEG = 0.055   # ~6 km at this latitude
step = 0.005
best = None
lat_grid = np.arange(BBOX["lat0"], BBOX["lat1"] - AOI_DEG, step)
lon_grid = np.arange(BBOX["lon0"], BBOX["lon1"] - AOI_DEG, step)
for la in lat_grid:
    for lo in lon_grid:
        n = ((gt.lat.between(la, la + AOI_DEG)) &
             (gt.lon.between(lo, lo + AOI_DEG))).sum()
        if best is None or n > best["n"]:
            best = dict(n=int(n), lat0=float(la), lat1=float(la + AOI_DEG),
                        lon0=float(lo), lon1=float(lo + AOI_DEG))

# tighten AOI to the actual kiln extent inside the window (+small margin)
inside = gt[(gt.lat.between(best["lat0"], best["lat1"])) &
            (gt.lon.between(best["lon0"], best["lon1"]))]
margin = 0.004
aoi = dict(
    lat0=float(inside.lat.min() - margin), lat1=float(inside.lat.max() + margin),
    lon0=float(inside.lon.min() - margin), lon1=float(inside.lon.max() + margin),
    n_gt_kilns=int(len(inside)),
    cls_counts=inside["cls"].value_counts().to_dict(),
)
print("\nChosen AOI:", json.dumps(aoi, indent=2))
inside.to_csv(DATA / "aoi_kilns.csv", index=False)
(DATA / "aoi.json").write_text(json.dumps(aoi, indent=2))
print(f"AOI spans ~{(aoi['lon1']-aoi['lon0'])*111*math.cos(math.radians(28.4)):.1f} km "
      f"x {(aoi['lat1']-aoi['lat0'])*111:.1f} km")
