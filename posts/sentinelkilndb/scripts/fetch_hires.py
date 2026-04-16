"""For each labeled kiln tile, fetch ESRI World Imagery at z=17 (full Sentinel
extent, ~1.1 m/px) and z=19 (tight zoom on the actual kiln, ~0.3 m/px)."""
import math, urllib.request, io, json, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
OUT = ROOT / "hires"; OUT.mkdir(exist_ok=True)
manifest = json.loads((ROOT / "data" / "manifest.json").read_text())

def latlon_to_pixel(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * 256
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n * 256
    return x, y

def fetch_tile(z, x, y, retries=3):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:
            if k == retries - 1: raise
            time.sleep(0.5)

def fetch_extent(lat, lon, half_size_m, z):
    mpp = 156543.03 / (2 ** z) * math.cos(math.radians(lat))
    half_px = int(half_size_m / mpp)
    cx, cy = latlon_to_pixel(lat, lon, z)
    L, T = int(cx - half_px), int(cy - half_px)
    R, B = int(cx + half_px), int(cy + half_px)
    tx0, ty0 = L // 256, T // 256
    tx1, ty1 = R // 256 + 1, B // 256 + 1
    canvas = Image.new("RGB", ((tx1-tx0)*256, (ty1-ty0)*256))
    for tx in range(tx0, tx1):
        for ty in range(ty0, ty1):
            canvas.paste(fetch_tile(z, tx, ty), ((tx-tx0)*256, (ty-ty0)*256))
    canvas = canvas.crop((L - tx0*256, T - ty0*256, R - tx0*256, B - ty0*256))
    return canvas, mpp

def parse_coord_from_name(name):
    s = name.replace(".png", "")
    lat, lon = s.split("_")
    return float(lat), float(lon)

def kiln_offset_latlon(lat, lon, labels, patch_extent_m=1280):
    """Compute the lat/lon of the kiln using the GT box centroid in the 128-px patch."""
    if not labels:
        return lat, lon
    pts = labels[0]["pts"]
    cx_px = sum(pts[0::2]) / 4
    cy_px = sum(pts[1::2]) / 4
    # frac offset from centre, in [-0.5, 0.5]
    fx = (cx_px / 128.0) - 0.5
    fy = (cy_px / 128.0) - 0.5
    # world offsets in metres — note y is south
    dx_m = fx * patch_extent_m
    dy_m = fy * patch_extent_m
    # convert to lat/lon offsets (small angle approx)
    dlat = -dy_m / 111111.0   # north positive in latitude
    dlon = dx_m / (111111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon

CLASS_COLORS = {"CFCBK": "#ff3b30", "FCBK": "#0a84ff", "Zigzag": "#ffd60a"}

def annotate_kiln(img, label_text, color):
    """Draw a hollow circle + label at the centre of the image."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    cx, cy = W//2, H//2
    r = min(W, H) // 14
    # outer ring
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=4)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = ImageFont.load_default()
    tb = draw.textbbox((cx + r + 6, cy - 12), label_text, font=font)
    draw.rectangle([tb[0]-4, tb[1]-3, tb[2]+4, tb[3]+3], fill=(0,0,0,180))
    draw.text((cx + r + 6, cy - 12), label_text, fill="white", font=font)
    return img

# Pick representative items: 2 per class (so 6 total) plus 2 negatives
pick = []
for cls in ["CFCBK", "FCBK", "Zigzag"]:
    for entry in manifest[cls][:2]:
        pick.append({"cls": cls, **entry})
for entry in manifest["negative"][:2]:
    pick.append({"cls": "none", **entry})

print(f"fetching hi-res for {len(pick)} items ...")
records = []
for i, item in enumerate(pick):
    name = item["name"]; cls = item["cls"]
    lat, lon = parse_coord_from_name(name)
    klat, klon = kiln_offset_latlon(lat, lon, item.get("labels", []))
    print(f"  [{i:02d}] {cls:8s} {name}  patch=({lat:.4f},{lon:.4f})  kiln=({klat:.4f},{klon:.4f})")

    # 1) Wide view (z=17, 1280m extent — same as Sentinel patch)
    wide, mpp17 = fetch_extent(lat, lon, half_size_m=640, z=17)
    wide.save(OUT / f"{i:02d}_{cls}_wide_z17.jpg", quality=88)

    # 2) Tight view (z=19, 100m extent — centred on kiln)
    if cls != "none":
        tight, mpp19 = fetch_extent(klat, klon, half_size_m=80, z=19)
        # Annotate the centre
        tight_ann = annotate_kiln(tight, cls, CLASS_COLORS[cls])
        tight.save(OUT / f"{i:02d}_{cls}_tight_z19.jpg", quality=88)
        tight_ann.save(OUT / f"{i:02d}_{cls}_tight_z19_ann.jpg", quality=88)
    else:
        tight = mpp19 = None

    records.append({
        "idx": i, "cls": cls, "name": name,
        "patch_lat": lat, "patch_lon": lon,
        "kiln_lat": klat, "kiln_lon": klon,
        "wide_size": wide.size, "wide_mpp": round(mpp17, 2),
        "tight_size": tight.size if tight else None,
        "tight_mpp": round(mpp19, 2) if mpp19 else None,
    })

(OUT / "manifest.json").write_text(json.dumps(records, indent=2))
print("ALL DONE")
