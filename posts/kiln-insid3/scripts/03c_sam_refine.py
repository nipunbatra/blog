"""Step 3c — Turn each reference OBB into a tight segmentation MASK with SAM2.

SentinelKilnDB gives an oriented box; INSID3 wants a mask. We prompt SAM2.1
with the OBB's bounding box and keep the returned mask (restricted to the OBB
neighbourhood so SAM can't wander onto an adjacent field). We save the SAM mask,
an OBB-vs-SAM comparison strip, and IoU/area stats for ranking the pool.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from ultralytics import SAM

ROOT = Path.home() / "kiln-insid3"
REF = ROOT / "work" / "refpool"
sam = SAM("sam2.1_b.pt")

pool = json.loads((REF / "pool.json").read_text())
rows = []
for r in pool:
    k = r["rank"]
    chip_p = REF / f"ref_{k:02d}.jpg"
    obb = np.array(Image.open(REF / f"ref_{k:02d}_mask.png")) > 127
    ys, xs = np.where(obb)
    if len(xs) == 0:
        continue
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    # SAM2 with the OBB bbox as a box prompt
    res = sam(str(chip_p), bboxes=[[int(x0), int(y0), int(x1), int(y1)]], verbose=False)
    m = res[0].masks.data[0].cpu().numpy() > 0.5
    # restrict to a dilated OBB neighbourhood so SAM stays on the kiln
    pad = int(0.35 * max(x1 - x0, y1 - y0))
    gate = np.zeros_like(m)
    gate[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    m = m & gate
    inter = (m & obb).sum(); union = (m | obb).sum()
    iou = float(inter / union) if union else 0.0
    Image.fromarray((m * 255).astype("uint8")).save(REF / f"ref_{k:02d}_sam.png")

    # comparison strip: chip | OBB | SAM
    chip = Image.open(chip_p).convert("RGB")
    a = chip.copy(); da = ImageDraw.Draw(a, "RGBA")
    oy, ox = np.where(obb)
    da.rectangle([ox.min(), oy.min(), ox.max(), oy.max()], outline=(10, 132, 255, 255), width=4)
    b = chip.copy(); arr = np.array(b); arr[m] = (arr[m] * 0.45 + np.array([255, 59, 48]) * 0.55).astype("uint8")
    b = Image.fromarray(arr)
    strip = Image.new("RGB", (chip.width * 3 + 16, chip.height), (15, 15, 15))
    for j, im in enumerate([chip, a, b]):
        strip.paste(im, (j * (chip.width + 8), 0))
    strip.resize((strip.width // 2, strip.height // 2)).save(REF / f"ref_{k:02d}_compare.jpg", quality=90)

    rows.append(dict(rank=k, lat=r["lat"], lon=r["lon"], iou_obb=round(iou, 3),
                     sam_px=int(m.sum()), obb_px=int(obb.sum())))
    print(f"ref_{k:02d} iou_obb={iou:.3f} sam_px={int(m.sum())} obb_px={int(obb.sum())}")

(REF / "sam_pool.json").write_text(json.dumps(rows, indent=2))
print(f"\nDONE {len(rows)} SAM-refined references")
