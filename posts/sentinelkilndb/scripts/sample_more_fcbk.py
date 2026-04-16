"""Stream SentinelKilnDB test split, collect more FCBK tiles, pick
visually-isolated ones (single kiln, no overlapping structures).
"""
import io, json
from pathlib import Path
from datasets import load_dataset
from PIL import Image

OUT = Path("/Users/nipun/git/blog/posts/sentinelkilndb/data_extra")
OUT.mkdir(parents=True, exist_ok=True)

# Target: 8 FCBK tiles where labels show EXACTLY one kiln and it's a moderate size
WANT_PER_CLASS = {"FCBK": 8, "Zigzag": 2, "CFCBK": 2}
bucket = {k: [] for k in WANT_PER_CLASS}

print("streaming test split for cleaner examples...")
ds = load_dataset("SustainabilityLabIITGN/SentinelKilnDB", split="test", streaming=True)
for i, row in enumerate(ds):
    if all(len(v) >= WANT_PER_CLASS[k] for k, v in bucket.items()):
        break
    dota = row.get("dota_label", [])
    if not dota: continue
    # prefer tiles with exactly ONE kiln (cleaner case)
    if len(dota) != 1: continue
    cls = dota[0]["class"] if isinstance(dota[0], dict) else dota[0].split()[8]
    if cls not in bucket: continue
    if len(bucket[cls]) >= WANT_PER_CLASS[cls]: continue

    img_bytes = row["image"]
    if isinstance(img_bytes, dict): img_bytes = img_bytes["bytes"]
    if isinstance(img_bytes, Image.Image):
        img = img_bytes
    else:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    name = row["image_name"]
    img.save(OUT / f"{cls.lower()}_{name}")
    bucket[cls].append({"name": name, "cls": cls, "labels": dota})
    print(f"  +{cls}: {name} (total {len(bucket[cls])})", flush=True)

(OUT / "manifest.json").write_text(json.dumps(bucket, indent=2, default=str))
print("\nfinal:", {k: len(v) for k, v in bucket.items()})
