"""v4 — CORRECTED visual descriptors:
  CFCBK = circle
  FCBK  = oval / elongated oval (NOT rectangle)
  Zigzag = rectangle with internal zigzag
"""
import sys, json, time, re
from pathlib import Path
sys.path.insert(0, "/Users/nipun/git/blog/posts/mlx_vlm_pr926")

from PIL import Image
from mlx_vlm import load
from agent import LocalVLMClient, run_baseline, run_agent

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
HIRES = ROOT / "hires_extra"
RES = ROOT / "results"

manifest = json.loads((HIRES / "manifest.json").read_text())
print(f"loaded {len(manifest)} tiles")

PROMPT_A = (
    "High-resolution aerial image (~0.6 m/pixel), ~400 m across, South Asia.\n\n"
    "Brick-kiln classes (SHAPE is the primary cue):\n"
    "  - CFCBK: nearly CIRCULAR (aspect ≈ 1:1), central chimney.\n"
    "  - FCBK:  OVAL / elongated oval (aspect ≈ 1:1.5 to 1:2.5), single chimney.\n"
    "  - Zigzag: RECTANGULAR (sharp corners, aspect 1:2+), internal zigzag wall pattern, rows of bricks.\n"
    "  - none:  no kiln.\n\n"
    "Look at the centre. Answer as JSON: "
    "{\"present\": true/false, \"class\": \"CFCBK|FCBK|Zigzag|none\", "
    "\"confidence\": \"low|medium|high\", \"reason\": \"<one short sentence>\"}"
)

PROMPT_C = """Classify the brick kiln in this aerial image (~0.6 m/pixel, South Asia).

The three brick-kiln classes are distinguished by SHAPE, not size:

  CFCBK — Continuous Fixed-Chimney Bull's-Trench Kiln
    Shape: nearly CIRCULAR ring (aspect ratio ≈ 1:1, diameter 80-200 m)
    Cue: central chimney visible at the centre of a round ring

  FCBK  — Fixed-Chimney Bull's-Trench Kiln
    Shape: OVAL, elongated oval, or stadium-track (aspect ≈ 1:1.5 to 1:2.5)
    Cue: smooth curved ends, a single chimney at one end or on the long axis

  Zigzag — Zigzag brick kiln
    Shape: RECTANGULAR with SHARP CORNERS (aspect ≈ 1:1.5 to 1:3)
    Cue: internal parallel rows / zigzag wall pattern visible; rows of curing bricks around it.
    (Never oval — always angular.)

  none — no kiln visible.

Key decision: is the kiln's outline a CIRCLE, an OVAL, or an angular RECTANGLE?
Describe the shape in one sentence, then classify. Call answer with just the class name."""


def parse_class(answer):
    a = answer.lower()
    for c in ["cfcbk", "zigzag", "fcbk", "none"]:
        if c in a:
            return {"cfcbk":"CFCBK","fcbk":"FCBK","zigzag":"Zigzag","none":"none"}[c]
    return "??"


print("[load] Falcon Perception ...", flush=True)
fp_model, fp_processor = load("tiiuae/Falcon-Perception")
print("[load] Gemma 4 31B 4bit ...", flush=True)
vlm = LocalVLMClient("mlx-community/gemma-4-31b-it-4bit", max_tokens=1024)

results = []
for rec in manifest:
    idx = rec["idx"]; cls = rec["cls"]
    img_path = HIRES / f"{idx:02d}_{cls}_{rec['name'].replace('.png','')}.jpg"
    if not img_path.exists(): continue
    img = Image.open(img_path).convert("RGB")

    print(f"\n[tile {idx:02d} true={cls}]", flush=True)
    row = {"idx": idx, "cls": cls, "image": img_path.name}

    # A: independent probe with CORRECTED descriptions
    t = time.time()
    ans_a = run_baseline(img, PROMPT_A, vlm)
    m = re.search(r'\{[^{}]+\}', ans_a, re.DOTALL)
    pred_a = "??"
    if m:
        try: pred_a = json.loads(m.group(0)).get("class", "??")
        except: pass
    row["A_pred"] = pred_a
    row["A_answer"] = ans_a.strip()[:300]
    row["A_elapsed"] = round(time.time()-t, 1)
    print(f"  A: {pred_a:8s}  {'✓' if pred_a == cls else '✗'}  {row['A_elapsed']}s", flush=True)

    # C: agent loop with CORRECTED visual features
    t = time.time()
    result_c = run_agent(img, PROMPT_C, fp_model, fp_processor, vlm, max_steps=6, verbose=False)
    pred_c = parse_class(result_c.answer)
    row["C_pred"] = pred_c
    row["C_answer"] = result_c.answer.strip()[:300]
    row["C_vlm_calls"] = result_c.n_vlm_calls
    row["C_fp_calls"] = result_c.n_fp_calls
    row["C_elapsed"] = round(time.time()-t, 1)
    print(f"  C: {pred_c:8s}  {'✓' if pred_c == cls else '✗'}  vlm={result_c.n_vlm_calls} fp={result_c.n_fp_calls} {row['C_elapsed']}s", flush=True)
    if result_c.final_image is not None:
        result_c.final_image.save(RES / f"v4_{idx:02d}_final.jpg", quality=85)

    results.append(row)

(RES / "v4_results.json").write_text(json.dumps(results, indent=2, default=str))
correct_a = sum(1 for r in results if r["A_pred"] == r["cls"])
correct_c = sum(1 for r in results if r["C_pred"] == r["cls"])
print(f"\nA (independent): {correct_a}/{len(results)} = {correct_a/len(results)*100:.0f}%")
print(f"C (features):    {correct_c}/{len(results)} = {correct_c/len(results)*100:.0f}%")
for cls in ["CFCBK", "FCBK", "Zigzag"]:
    sub = [r for r in results if r["cls"] == cls]
    if not sub: continue
    ca = sum(1 for r in sub if r["A_pred"] == cls)
    cc = sum(1 for r in sub if r["C_pred"] == cls)
    print(f"  {cls}: A={ca}/{len(sub)}  C={cc}/{len(sub)}")
print("DONE")
