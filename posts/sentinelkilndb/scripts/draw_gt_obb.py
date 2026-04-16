"""Overlay ground-truth OBB polygons (from dota_label) on the original 128-px
Sentinel tile AND on the corresponding ESRI z=18 hi-res crop.

Uses supervision's PolygonAnnotator where possible; falls back to plain PIL
for the 4-corner OBB case (supervision's annotator wants a single numpy polygon
per detection, which is exactly what we have).
"""
import io, json, math
from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import supervision as sv

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
HIRES = ROOT / "hires_extra"
OUT = ROOT / "results" / "gt_overlays"; OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((HIRES / "manifest.json").read_text())

# Re-stream dataset to pull the dota labels matching each tile
print("streaming test split to pick up GT labels ...")
ds = load_dataset("SustainabilityLabIITGN/SentinelKilnDB", split="test", streaming=True)
wanted = {f"{r['name']}": r for r in manifest}
by_name = {}
for row in ds:
    if row["image_name"] in wanted:
        by_name[row["image_name"]] = {
            "image_bytes": row["image"] if isinstance(row["image"], (bytes,)) else (row["image"].get("bytes") if isinstance(row["image"], dict) else None),
            "image_obj": row["image"] if isinstance(row["image"], Image.Image) else None,
            "dota": row["dota_label"],
        }
        if len(by_name) >= len(wanted): break
print(f"  got GT for {len(by_name)}/{len(wanted)} tiles")

CLASS_COLORS = {
    "CFCBK": sv.Color(r=255, g=59, b=48),
    "FCBK":  sv.Color(r=10,  g=132, b=255),
    "Zigzag":sv.Color(r=255, g=149, b=0),
}
CLASS_RGBA = {
    "CFCBK": (255, 59, 48),
    "FCBK":  (10, 132, 255),
    "Zigzag":(255, 149, 0),
}

def parse_dota_row(row):
    """Return list of dicts: {class, pts (8 floats in 128-px space)}."""
    out = []
    for item in row:
        if isinstance(item, dict):
            out.append({"class": item["class"], "pts": list(item["pts"])})
        elif isinstance(item, str):
            t = item.split()
            if len(t) >= 9:
                out.append({"class": t[8], "pts": [float(x) for x in t[:8]]})
    return out

def draw_obb_on_sentinel(img, boxes, scale=4):
    """img: 128×128 PIL; boxes: list of {class, pts}. Draw polygons scaled up for clarity."""
    tile = img.convert("RGB").resize((128*scale, 128*scale), Image.LANCZOS)
    draw = ImageDraw.Draw(tile, "RGBA")
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica-Bold.ttc", 14)
    except: font = ImageFont.load_default()
    for b in boxes:
        col = CLASS_RGBA.get(b["class"], (255, 0, 255))
        pts = [(b["pts"][i] * scale, b["pts"][i+1] * scale) for i in range(0, 8, 2)]
        pts.append(pts[0])
        draw.line(pts, fill=col + (255,), width=3)
        # label at centroid
        cx = sum(p[0] for p in pts[:4]) / 4
        cy = sum(p[1] for p in pts[:4]) / 4
        bb = draw.textbbox((cx+6, cy-10), b["class"], font=font)
        draw.rectangle([bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2], fill=(0,0,0,200))
        draw.text((cx+6, cy-10), b["class"], fill="white", font=font)
    return tile

def sentinel_pix_to_latlon(x_pix, y_pix, patch_lat, patch_lon, patch_extent_m=1280, tile_px=128):
    """Convert a Sentinel-tile pixel (128×128, patch_extent_m wide) to (lat, lon)."""
    dx_m = (x_pix - tile_px/2) * (patch_extent_m / tile_px)
    dy_m = (y_pix - tile_px/2) * (patch_extent_m / tile_px)
    # latitude decreases as y increases (y=0 top)
    dlat = -dy_m / 111111.0
    dlon = dx_m / (111111.0 * math.cos(math.radians(patch_lat)))
    return patch_lat + dlat, patch_lon + dlon

def latlon_to_esri_crop_pix(lat, lon, crop_center_lat, crop_center_lon, half_size_m, crop_size_px):
    """Convert a (lat, lon) to pixel coords inside the ESRI crop (square, centered on crop_center)."""
    dx_m = (lon - crop_center_lon) * 111111.0 * math.cos(math.radians(crop_center_lat))
    dy_m = -(lat - crop_center_lat) * 111111.0
    # px per metre
    ppm = (crop_size_px / 2.0) / half_size_m
    px = crop_size_px / 2.0 + dx_m * ppm
    py = crop_size_px / 2.0 + dy_m * ppm
    return px, py

def draw_obb_on_esri(img, boxes, patch_lat, patch_lon, crop_lat, crop_lon, half_size_m=200):
    """Map each GT OBB polygon to ESRI crop pixel space and draw."""
    tile = img.convert("RGB").copy()
    W, H = tile.size
    draw = ImageDraw.Draw(tile, "RGBA")
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica-Bold.ttc", 18)
    except: font = ImageFont.load_default()
    for b in boxes:
        col = CLASS_RGBA.get(b["class"], (255, 0, 255))
        obb_pts = []
        for i in range(0, 8, 2):
            xs, ys = b["pts"][i], b["pts"][i+1]
            lat, lon = sentinel_pix_to_latlon(xs, ys, patch_lat, patch_lon)
            px, py = latlon_to_esri_crop_pix(lat, lon, crop_lat, crop_lon, half_size_m, W)
            obb_pts.append((px, py))
        if not obb_pts: continue
        # semi-transparent fill
        draw.polygon(obb_pts + [obb_pts[0]], fill=col + (40,))
        # outline
        draw.line(obb_pts + [obb_pts[0]], fill=col + (255,), width=3)
        cx = sum(p[0] for p in obb_pts) / 4
        cy = sum(p[1] for p in obb_pts) / 4
        bb = draw.textbbox((cx+8, cy-12), b["class"], font=font)
        draw.rectangle([bb[0]-4, bb[1]-3, bb[2]+4, bb[3]+3], fill=(0,0,0,210))
        draw.text((cx+8, cy-12), b["class"], fill="white", font=font)
    return tile

def parse_coord(name):
    s = name.replace(".png", "")
    lat, lon = s.split("_")
    return float(lat), float(lon)

# Iterate and produce overlays
for rec in manifest:
    idx = rec["idx"]; cls = rec["cls"]; name = rec["name"]
    patch_lat, patch_lon = parse_coord(name)
    crop_lat = rec["kiln_lat"]; crop_lon = rec["kiln_lon"]

    data = by_name.get(name)
    if data is None:
        print(f"  [{idx}] no GT data found")
        continue
    boxes = parse_dota_row(data["dota"])

    # Sentinel tile
    sent_img = data["image_obj"]
    if sent_img is None and data["image_bytes"] is not None:
        sent_img = Image.open(io.BytesIO(data["image_bytes"])).convert("RGB")
    if sent_img is None:
        print(f"  [{idx}] no sentinel image")
        continue
    sent_overlay = draw_obb_on_sentinel(sent_img, boxes, scale=4)
    sent_overlay.save(OUT / f"{idx:02d}_{cls}_sentinel_obb.jpg", quality=90)

    # ESRI tile
    esri_path = HIRES / f"{idx:02d}_{cls}_{name.replace('.png','')}.jpg"
    if not esri_path.exists():
        print(f"  [{idx}] no ESRI tile at {esri_path}")
        continue
    esri_img = Image.open(esri_path).convert("RGB")
    esri_overlay = draw_obb_on_esri(esri_img, boxes, patch_lat, patch_lon, crop_lat, crop_lon, half_size_m=200)
    esri_overlay.save(OUT / f"{idx:02d}_{cls}_esri_obb.jpg", quality=90)
    print(f"  [{idx}] {cls} {name}: {len(boxes)} box(es) drawn on both scales")

print("DONE")
