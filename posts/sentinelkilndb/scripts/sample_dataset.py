"""Stream SentinelKilnDB test split and collect a small balanced sample."""
import io, json
from pathlib import Path
from datasets import load_dataset
from PIL import Image

OUT = Path("/Users/nipun/git/blog/posts/sentinelkilndb/data")
WANT_PER_CLASS = 4  # 4 each of CFCBK, FCBK, Zigzag + 4 negatives = 16

print("[stream] test split ...")
ds = load_dataset("SustainabilityLabIITGN/SentinelKilnDB", split="test", streaming=True)

bucket = {"CFCBK": [], "FCBK": [], "Zigzag": [], "negative": []}
for i, row in enumerate(ds):
    if i % 500 == 0:
        print(f"  scanned {i}, have {[(k, len(v)) for k, v in bucket.items()]}", flush=True)

    img_bytes = row["image"]
    if isinstance(img_bytes, dict): img_bytes = img_bytes.get("bytes")
    if isinstance(img_bytes, Image.Image):
        img = img_bytes
    else:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    name = row["image_name"]
    dota = row.get("dota_label", [])

    # Decide: negative if no labels
    if not dota:
        if len(bucket["negative"]) < WANT_PER_CLASS:
            img.save(OUT / "negatives" / name)
            bucket["negative"].append({"name": name, "labels": []})
        # done collecting?
        if all(len(v) >= WANT_PER_CLASS for v in bucket.values()):
            break
        continue

    # Determine class via DOTA labels: each entry has a class
    # dota_label rows look like: x1 y1 x2 y2 x3 y3 x4 y4 class difficult
    classes_in_image = set()
    parsed_boxes = []
    for entry in dota:
        # Could be a dict or string; let's handle both
        if isinstance(entry, str):
            parts = entry.split()
            if len(parts) >= 9:
                cls = parts[8]
                pts = [float(x) for x in parts[:8]]
                parsed_boxes.append({"class": cls, "pts": pts})
                classes_in_image.add(cls)
        elif isinstance(entry, dict):
            cls = entry.get("class") or entry.get("category") or entry.get("name")
            classes_in_image.add(cls)
            parsed_boxes.append(entry)

    # Pick the dominant class (first one in the list)
    if not classes_in_image:
        continue
    cls = sorted(classes_in_image)[0]
    if cls in bucket and len(bucket[cls]) < WANT_PER_CLASS:
        img.save(OUT / cls.lower() / name)
        bucket[cls].append({"name": name, "labels": parsed_boxes,
                            "all_classes_in_image": list(classes_in_image)})
        print(f"  +{cls}: {name}", flush=True)

    if all(len(v) >= WANT_PER_CLASS for v in bucket.values()):
        print("[done] all buckets filled")
        break

# Save manifest
(OUT / "manifest.json").write_text(json.dumps(bucket, indent=2, default=str))
print(f"final counts: { {k: len(v) for k, v in bucket.items()} }")
