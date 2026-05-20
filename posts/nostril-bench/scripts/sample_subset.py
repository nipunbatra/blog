"""Pick 10 diverse-subject thermal images from SFTL54 test split + sample images
from gray/train. Writes image_list_subset.json."""
import json
import re
from pathlib import Path

ROOT = Path.home() / "data/SFTL54/sftl/sftl"

def files_of(modality, split):
    return sorted((ROOT / modality / split / "images").glob("*.png"))

# pick first-image-per-subject from gray/train and gray/test
def subject_of(p): return int(p.name.split("_")[0])

def pick_first_per_subject(files, n):
    seen, picked = set(), []
    for f in files:
        s = subject_of(f)
        if s in seen: continue
        seen.add(s); picked.append(f)
        if len(picked) >= n: break
    return picked

picked_gray = pick_first_per_subject(files_of("gray", "train"), 8)
picked_gray += pick_first_per_subject(files_of("gray", "test"), 3)
# unique by name
picked_gray = list({p.name: p for p in picked_gray}.values())[:10]

items = []
for p in picked_gray:
    # thermal
    items.append({
        "path": str(p),
        "jsonl": str(ROOT / "gray/train/labels.jsonl") if "train" in str(p)
                 else str(ROOT / "gray/test/labels.jsonl"),
        "modality": "gray",
    })
    # paired iron
    iron = ROOT / "iron" / ("train" if "train" in str(p) else "test") / "images" / p.name
    if iron.exists():
        items.append({
            "path": str(iron),
            "jsonl": str(ROOT / "iron/train/labels.jsonl") if "train" in str(iron)
                     else str(ROOT / "iron/test/labels.jsonl"),
            "modality": "iron",
        })
    # paired RGB (different filename suffix _3)
    rgb_name = re.sub(r"_1\.png$", "_3.png", p.name)
    rgb = ROOT / "rgb" / ("train" if "train" in str(p) else "test") / rgb_name
    if rgb.exists():
        items.append({"path": str(rgb), "jsonl": None, "modality": "rgb"})

out = Path("image_list_subset.json")
out.write_text(json.dumps(items, indent=2))
print(f"wrote {out} with {len(items)} items "
      f"({sum(1 for i in items if i['modality']=='gray')} gray,"
      f" {sum(1 for i in items if i['modality']=='iron')} iron,"
      f" {sum(1 for i in items if i['modality']=='rgb')} rgb)")
