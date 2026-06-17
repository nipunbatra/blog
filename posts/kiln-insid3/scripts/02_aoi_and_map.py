"""Step 2 — Clip GT kilns to the true Bulandshahr district polygon, choose the
densest ~5-6 km AOI that lies fully inside the district, and render a locator /
density map. Outputs: data/aoi.json, data/aoi_kilns.csv, outputs/01_district_density.png
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/Users/nipun/git/blog/posts/kiln-insid3")
DATA = ROOT / "data"; OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)

dist = gpd.read_file(DATA / "bulandshahr.geojson").to_crs(4326)
poly = dist.geometry.union_all()
gt = pd.read_csv(DATA / "bulandshahr_kilns_all.csv")

# clip kilns to the actual district polygon
pts = gpd.GeoSeries([Point(lo, la) for lo, la in zip(gt.lon, gt.lat)], crs=4326)
gt = gt[pts.within(poly).values].reset_index(drop=True)
print(f"GT kilns inside Bulandshahr district: {len(gt)}")
print(gt["cls"].value_counts().to_string())
gt.to_csv(DATA / "bulandshahr_kilns_district.csv", index=False)

# densest AOI fully inside the district
AOI_DEG_LAT = 0.045          # ~5.0 km
AOI_DEG_LON = 0.051          # ~5.0 km at 28.4N
step = 0.004
best = None
minx, miny, maxx, maxy = poly.bounds
for la in np.arange(miny, maxy - AOI_DEG_LAT, step):
    for lo in np.arange(minx, maxx - AOI_DEG_LON, step):
        b = box(lo, la, lo + AOI_DEG_LON, la + AOI_DEG_LAT)
        if not poly.contains(b):
            continue
        n = ((gt.lat.between(la, la + AOI_DEG_LAT)) &
             (gt.lon.between(lo, lo + AOI_DEG_LON))).sum()
        if best is None or n > best["n"]:
            best = dict(n=int(n), lat0=float(la), lat1=float(la + AOI_DEG_LAT),
                        lon0=float(lo), lon1=float(lo + AOI_DEG_LON))

inside = gt[(gt.lat.between(best["lat0"], best["lat1"])) &
            (gt.lon.between(best["lon0"], best["lon1"]))].copy()
aoi = dict(lat0=best["lat0"], lat1=best["lat1"], lon0=best["lon0"], lon1=best["lon1"],
           n_gt_kilns=int(len(inside)), cls_counts=inside["cls"].value_counts().to_dict())
km_w = (aoi["lon1"]-aoi["lon0"])*111*math.cos(math.radians(28.4))
km_h = (aoi["lat1"]-aoi["lat0"])*111
aoi["km_w"], aoi["km_h"] = round(km_w, 2), round(km_h, 2)
inside.to_csv(DATA / "aoi_kilns.csv", index=False)
(DATA / "aoi.json").write_text(json.dumps(aoi, indent=2))
print("\nAOI:", json.dumps(aoi, indent=2))
print(f"AOI ~{km_w:.1f} km x {km_h:.1f} km")

# --- locator / density map -------------------------------------------------
states = gpd.read_file("/Users/nipun/git/blog/posts/shapefiles/India_State_Boundary.shp").to_crs(4326)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

# (a) India + district location
ax = axes[0]
states.boundary.plot(ax=ax, color="#bbb", linewidth=0.4)
dist.plot(ax=ax, color="#c44536", alpha=0.9)
ax.set_xlim(67, 90); ax.set_ylim(7, 37)
ax.set_title("Bulandshahr district, Uttar Pradesh", fontsize=12)
ax.set_xlabel("lon"); ax.set_ylabel("lat")
ax.annotate("Bulandshahr", xy=(77.9, 28.4), xytext=(82, 30),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="#c44536"))

# (b) district kiln density + AOI box
ax = axes[1]
dist.boundary.plot(ax=ax, color="#333", linewidth=1.2)
colors = {"Zigzag": "#0a84ff", "FCBK": "#ff9f0a", "CFCBK": "#ff3b30"}
for cls, c in colors.items():
    s = gt[gt.cls == cls]
    ax.scatter(s.lon, s.lat, s=7, c=c, label=f"{cls} ({len(s)})", alpha=0.7, edgecolors="none")
ax.add_patch(Rectangle((aoi["lon0"], aoi["lat0"]), aoi["lon1"]-aoi["lon0"], aoi["lat1"]-aoi["lat0"],
                       fill=False, edgecolor="#111", linewidth=2.2, zorder=5))
ax.annotate(f"AOI\n{aoi['n_gt_kilns']} GT kilns\n~{km_w:.1f}×{km_h:.1f} km",
            xy=(aoi["lon1"], aoi["lat1"]), xytext=(aoi["lon1"]+0.03, aoi["lat1"]+0.03),
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#111"))
ax.set_title(f"{len(gt)} SentinelKilnDB kilns in Bulandshahr", fontsize=12)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.set_xlabel("lon"); ax.set_ylabel("lat")
ax.set_aspect(1/math.cos(math.radians(28.4)))

plt.tight_layout()
plt.savefig(OUT / "01_district_density.png", dpi=140, bbox_inches="tight")
print("saved outputs/01_district_density.png")
