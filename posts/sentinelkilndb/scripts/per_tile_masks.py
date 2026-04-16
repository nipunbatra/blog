"""Run Falcon with several queries per tile and save overlays + metadata.

For each tile, produces:
  results/per_tile/{idx}_{cls}/
    - 00_input.jpg
    - 01_gt.jpg
    - 02_falcon_brick_kiln.jpg
    - 03_falcon_oval_kiln.jpg
    - 04_falcon_rectangular_kiln.jpg
    - 05_falcon_circular_kiln.jpg
    - metadata.json (n_masks per query, gemma pred, reasoning)
"""
import sys, json, time, re, math
from pathlib import Path
sys.path.insert(0, "/Users/nipun/git/blog/posts/mlx_vlm_pr926")

from PIL import Image, ImageDraw, ImageFont
from mlx_vlm import load
from fp_tools import run_ground_expression
from viz import render_som
from agent import LocalVLMClient, run_baseline

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
HIRES = ROOT / "hires_extra"
OUT = ROOT / "results" / "per_tile"; OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((HIRES / "manifest.json").read_text())

QUERIES = [
    ("brick_kiln", "brick kiln"),
    ("oval_kiln", "oval brick kiln"),
    ("circular_kiln", "circular brick kiln"),
    ("rectangular_kiln", "rectangular brick kiln"),
]

CLASS_COLORS = {"CFCBK": "#ff3b30", "FCBK": "#0a84ff", "Zigzag": "#ffa500"}

# Load corrected Gemma prompt
PROMPT = (
    "High-resolution aerial image (~0.6 m/pixel), ~400 m across, South Asia.\n\n"
    "Brick-kiln classes (SHAPE is the primary cue):\n"
    "  - CFCBK: nearly CIRCULAR (aspect ≈ 1:1), central chimney.\n"
    "  - FCBK:  OVAL / elongated oval / stadium-track (aspect ≈ 1:1.5 to 1:2.5), single chimney.\n"
    "  - Zigzag: RECTANGULAR with sharp corners (aspect 1:2+), internal zigzag wall pattern, rows of bricks.\n"
    "  - none:  no kiln.\n\n"
    "Look at the centre. Answer as JSON: "
    "{\"present\": true/false, \"class\": \"CFCBK|FCBK|Zigzag|none\", "
    "\"confidence\": \"low|medium|high\", \"reason\": \"<one short sentence about the SHAPE you see>\"}"
)

def parse_class(answer):
    a = answer.lower()
    for c in ["cfcbk", "zigzag", "fcbk", "none"]:
        if c in a:
            return {"cfcbk":"CFCBK","fcbk":"FCBK","zigzag":"Zigzag","none":"none"}[c]
    return "??"

print("[load] Falcon Perception ...", flush=True)
fp_model, fp_processor = load("tiiuae/Falcon-Perception")
print("[load] Gemma 4 31B 4bit ...", flush=True)
vlm = LocalVLMClient("mlx-community/gemma-4-31b-it-4bit", max_tokens=768)

all_meta = []
for rec in manifest:
    idx = rec["idx"]; cls = rec["cls"]
    img_path = HIRES / f"{idx:02d}_{cls}_{rec['name'].replace('.png','')}.jpg"
    ann_path = HIRES / f"{idx:02d}_{cls}_{rec['name'].replace('.png','')}_ann.jpg"
    if not img_path.exists():
        continue

    out_dir = OUT / f"{idx:02d}_{cls}"
    out_dir.mkdir(exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    print(f"\n[tile {idx:02d} true={cls}]", flush=True)

    # 00 input, 01 GT annotation
    img.save(out_dir / "00_input.jpg", quality=88)
    if ann_path.exists():
        Image.open(ann_path).save(out_dir / "01_gt.jpg", quality=88)

    meta = {"idx": idx, "cls": cls, "falcon": {}}

    # Run each query
    for slug, expr in QUERIES:
        t = time.time()
        masks = run_ground_expression(fp_model, fp_processor, img, expr, max_new_tokens=512)
        som = render_som(img, masks)
        som.save(out_dir / f"02_falcon_{slug}.jpg", quality=85)
        meta["falcon"][slug] = {
            "expression": expr,
            "n_masks": len(masks),
            "elapsed_s": round(time.time()-t, 1),
            "areas": [round(m["area_fraction"], 4) for m in masks.values()],
        }
        print(f"  falcon '{expr}': {len(masks)} masks  {time.time()-t:.1f}s", flush=True)

    # Gemma classification
    t = time.time()
    gemma_ans = run_baseline(img, PROMPT, vlm)
    m = re.search(r'\{[^{}]+\}', gemma_ans, re.DOTALL)
    pred = "??"; conf = "?"; reason = ""
    if m:
        try:
            j = json.loads(m.group(0))
            pred = j.get("class", "??")
            conf = j.get("confidence", "?")
            reason = j.get("reason", "")
        except Exception:
            pass
    meta["gemma"] = {"pred": pred, "confidence": conf, "reason": reason,
                     "raw_answer": gemma_ans.strip()[:500],
                     "elapsed_s": round(time.time()-t, 1),
                     "correct": pred == cls}
    print(f"  gemma: pred={pred} conf={conf}  {'✓' if pred == cls else '✗'}  {time.time()-t:.1f}s", flush=True)

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    all_meta.append(meta)

(OUT / "all_metadata.json").write_text(json.dumps(all_meta, indent=2))
print("\nDONE")

# Print summary
correct = sum(1 for m in all_meta if m["gemma"]["correct"])
print(f"Gemma accuracy: {correct}/{len(all_meta)} = {correct/len(all_meta)*100:.0f}%")
from collections import defaultdict
bc = defaultdict(lambda: {"c": 0, "t": 0})
for m in all_meta:
    bc[m["cls"]]["t"] += 1
    if m["gemma"]["correct"]:
        bc[m["cls"]]["c"] += 1
for cls in ["CFCBK", "FCBK", "Zigzag"]:
    print(f"  {cls}: {bc[cls]['c']}/{bc[cls]['t']}")
