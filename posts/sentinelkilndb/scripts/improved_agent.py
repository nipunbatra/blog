"""Improved kiln classifier v3 — better visual features, no hard-coded rules.

Key changes vs v2:
  - Emphasise the SIZE of the kiln structure in the frame (Zigzag is large, FCBK smaller)
  - Emphasise the INTERNAL texture (zigzag has striated rows of bricks filling its interior)
  - Emphasise CHIMNEY (FCBK/CFCBK have a clear chimney, Zigzag often doesn't)
  - Do NOT prescribe rules — let Gemma use its vision
  - Keep multi-query grounding as an OPTION, not a requirement
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "/Users/nipun/git/blog/posts/mlx_vlm_pr926")

from PIL import Image
from mlx_vlm import load
from agent import LocalVLMClient, run_agent

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
RES = ROOT / "results"

KILN_QUERY_V3 = """Classify the brick kiln in this high-resolution overhead image (~0.6 m/pixel, South Asia).

Four possible labels: CFCBK, FCBK, Zigzag, none.

Distinguishing visual features (look for ALL of these, in order):

1. CFCBK — Continuous Fixed-Chimney Bull's-Trench Kiln
   - CIRCULAR or OVAL ring shape (aspect ratio ≈ 1:1 to 1:1.3)
   - Central chimney visible as a small dot at the centre of the ring
   - Typical diameter 80-200 metres
   - The ring encloses a cleared interior

2. FCBK — Fixed-Chimney Bull's-Trench Kiln
   - SMALL-to-medium RECTANGULAR kiln body (aspect 1:2 to 1:4, narrow)
   - SINGLE tall CHIMNEY at one end or in the middle
   - Surrounded by separate rectangular brick-curing yards / piles
   - Kiln body does NOT fill the whole frame — it's one structure among others

3. Zigzag — Zigzag brick kiln
   - LARGE rectangular footprint (aspect 1:1.5 to 1:3)
   - DOMINATES the frame (often > 30% of image area)
   - INTERNAL STRIATION: parallel rows of stacked curing bricks filling the interior
   - Often NO tall standalone chimney — draft is via the long walls
   - Surrounded by green fields or villages

4. none — no kiln visible

Steps:
  - First, describe what you see in one sentence.
  - Optionally ground an expression if it helps (e.g. "circular brick kiln" or "rectangular kiln").
  - Choose the label based on the visual features above.
  - Call answer with just the class name (CFCBK, FCBK, Zigzag, or none)."""


if __name__ == "__main__":
    print("[load] Falcon Perception ...")
    fp_model, fp_processor = load("tiiuae/Falcon-Perception")
    print("[load] Gemma 4 31B 4bit ...")
    vlm = LocalVLMClient("mlx-community/gemma-4-31b-it-4bit", max_tokens=1024)

    manifest = json.loads((ROOT / "hires" / "manifest.json").read_text())
    results = []
    for rec in manifest:
        idx = rec["idx"]; cls = rec["cls"]
        if cls == "none":
            img_path = ROOT / "hires" / f"{idx:02d}_none_wide_z17.jpg"
        else:
            img_path = ROOT / "hires" / f"{idx:02d}_{cls}_tight_z18.jpg"
        img = Image.open(img_path).convert("RGB")

        print(f"\n[idx={idx} true={cls}]")
        t = time.time()
        result = run_agent(img, KILN_QUERY_V3, fp_model, fp_processor, vlm,
                           max_steps=6, verbose=False)
        dt = round(time.time() - t, 1)

        if result.final_image is not None:
            result.final_image.save(RES / f"v3_{idx:02d}_final.jpg", quality=85)

        # Parse — check cfcbk before fcbk
        a = result.answer.lower()
        pred = "??"
        for cand in ["cfcbk", "zigzag", "fcbk", "none"]:
            if cand in a:
                pred = {"cfcbk":"CFCBK","fcbk":"FCBK","zigzag":"Zigzag","none":"none"}[cand]
                break

        rec_out = {"idx": idx, "true": cls, "pred": pred, "answer": result.answer.strip(),
                   "vlm_calls": result.n_vlm_calls, "fp_calls": result.n_fp_calls,
                   "elapsed_s": dt}
        results.append(rec_out)
        mark = "✓" if pred == cls else "✗"
        print(f"  → pred={pred:6s} (true={cls}) {mark}  vlm={result.n_vlm_calls} fp={result.n_fp_calls} {dt}s")
        print(f"    answer: {result.answer.strip()[:160]}")

    (RES / "v3_results.json").write_text(json.dumps(results, indent=2, default=str))
    correct = sum(1 for r in results if r["pred"] == r["true"])
    print(f"\nv3 accuracy: {correct}/{len(results)} = {correct/len(results)*100:.0f}%")
    for cls in ["CFCBK", "FCBK", "Zigzag", "none"]:
        sub = [r for r in results if r["true"] == cls]
        c = sum(1 for r in sub if r["pred"] == cls)
        print(f"  {cls}: {c}/{len(sub)}  preds={[r['pred'] for r in sub]}")
