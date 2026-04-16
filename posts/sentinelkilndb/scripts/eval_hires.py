"""Run Falcon Perception + Gemma 4 on hi-res ESRI tiles."""
import sys, json, time, re
from pathlib import Path
sys.path.insert(0, "/Users/nipun/git/blog/posts/mlx_vlm_pr926")

from PIL import Image
from mlx_vlm import load
from fp_tools import run_ground_expression
from viz import render_som
from agent import LocalVLMClient, run_baseline

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
HIRES = ROOT / "hires"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

manifest = json.loads((HIRES / "manifest.json").read_text())
print(f"loaded {len(manifest)} items")

print("[load] Falcon Perception ...")
fp_model, fp_processor = load("tiiuae/Falcon-Perception")
print("[load] Gemma 4 31B 4bit ...")
vlm = LocalVLMClient("mlx-community/gemma-4-31b-it-4bit", max_tokens=512)

GEMMA_PROMPT = (
    "This is a high-resolution aerial image (ESRI World Imagery, ~0.6 m/pixel) "
    "of a small area in South Asia, 250 m across.\n\n"
    "Context: The dataset SentinelKilnDB distinguishes three brick-kiln types:\n"
    "  - CFCBK: continuous fixed-chimney bull's-trench kiln. Distinctive *circular* or oval shape, "
    "central chimney, often a ring of unfired bricks around the perimeter.\n"
    "  - FCBK: fixed-chimney bull's-trench kiln. Rectangular layout, single chimney visible, "
    "rows of stacked bricks alongside.\n"
    "  - Zigzag: zigzag brick kiln. Large rectangular footprint with internal zigzag wall pattern, "
    "rows of curing bricks around it.\n\n"
    "Look at the centre of the image and decide. Answer in this exact JSON:\n"
    "{\"present\": true/false, \"class\": \"CFCBK|FCBK|Zigzag|none\", "
    "\"confidence\": \"low|medium|high\", \"reason\": \"<one short sentence>\"}"
)

results = []
for rec in manifest:
    idx = rec["idx"]; cls = rec["cls"]
    # use the wide-z17 view (full 1280m, comparable to Sentinel patch) for evaluation
    if cls == "none":
        # No tight view for negatives — use wide z17
        img_path = HIRES / f"{idx:02d}_none_wide_z17.jpg"
    else:
        # Use the z18 tight (~250m around the kiln) — clearer subject
        img_path = HIRES / f"{idx:02d}_{cls}_tight_z18.jpg"

    if not img_path.exists():
        print(f"  [{idx:02d}] missing {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    record = {"idx": idx, "cls": cls, "image": img_path.name,
              "image_size": img.size, "falcon": {}, "gemma": None}

    # Falcon: try several queries
    for q in ["brick kiln", "circular kiln", "small building"]:
        t = time.time()
        masks = run_ground_expression(fp_model, fp_processor, img, q, max_new_tokens=512)
        record["falcon"][q] = {
            "n": len(masks),
            "elapsed_s": round(time.time()-t, 1),
        }
        if q == "brick kiln":
            som = render_som(img, masks)
            som.save(RES / f"{idx:02d}_falcon_brickkiln.jpg", quality=85)
        if q == "circular kiln":
            som = render_som(img, masks)
            som.save(RES / f"{idx:02d}_falcon_circular.jpg", quality=85)

    # Gemma classification
    t = time.time()
    ans = run_baseline(img, GEMMA_PROMPT, vlm)
    record["gemma"] = {"answer": ans.strip(), "elapsed_s": round(time.time()-t, 1)}

    # Parse Gemma JSON
    m = re.search(r'\{[^{}]+\}', ans, re.DOTALL)
    parsed = {"class": "??", "present": None, "confidence": "?", "reason": ""}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            pass
    record["gemma_parsed"] = parsed
    print(f"  [{idx:02d}] true={cls:8s} falcon_brickkiln={record['falcon']['brick kiln']['n']} "
          f"falcon_circular={record['falcon']['circular kiln']['n']} "
          f"gemma_class={parsed.get('class','??')} conf={parsed.get('confidence','?')}", flush=True)
    results.append(record)

(RES / "hires_results.json").write_text(json.dumps(results, indent=2))
print("ALL DONE")
