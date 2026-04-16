"""Improved kiln classifier — keeps the PR's full SYSTEM_PROMPT (with tool schema)
and *appends* a brick-kiln-specific addendum at the user-prompt layer."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "/Users/nipun/git/blog/posts/mlx_vlm_pr926")

from PIL import Image
from mlx_vlm import load
from agent import LocalVLMClient, run_agent

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
RES = ROOT / "results"

KILN_QUERY = """Classify the brick kiln in this high-resolution overhead image (~0.6 m/pixel, South Asia).

There are exactly four labels: CFCBK, FCBK, Zigzag, none.
  - CFCBK: continuous fixed-chimney bull's-trench kiln. CIRCULAR or oval ring (aspect ratio < 1.4), central chimney.
  - FCBK:  fixed-chimney bull's-trench kiln. RECTANGULAR (aspect ratio 1.4-2.5), single chimney, brick-curing yards.
  - Zigzag: zigzag kiln. LARGE RECTANGLE (aspect ratio > 2.5 or visible internal zigzag pattern).
  - none:  no kiln.

Plan you SHOULD follow:
  1. Look at the image. Note the most plausible kiln structure(s).
  2. Ground at least ONE expression like "circular brick kiln" or "rectangular brick kiln" using ground_expression.
     If both shapes are plausible, ground both into separate slots ("circular_kiln", "rect_kiln").
  3. For each grounded mask of interest, call compute_relations on its single id to get its bbox geometry.
  4. Read the bbox aspect ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h) from compute_relations output.
     - aspect < 1.4 + circular ring  → CFCBK
     - aspect 1.4-2.5 + single chimney  → FCBK
     - aspect > 2.5 OR visible zigzag pattern  → Zigzag
  5. Combine the visual cue with the measured aspect ratio. Call answer with the label and supporting_mask_ids.

Use the measurement, not just visual gestalt. Aspect ratio is decisive."""

if __name__ == "__main__":
    print("[load] Falcon Perception ...")
    fp_model, fp_processor = load("tiiuae/Falcon-Perception")
    print("[load] Gemma 4 31B 4bit ...")
    vlm = LocalVLMClient("mlx-community/gemma-4-31b-it-4bit", max_tokens=2048)

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
        result = run_agent(img, KILN_QUERY, fp_model, fp_processor, vlm,
                           max_steps=10, verbose=False)
        dt = round(time.time() - t, 1)

        if result.final_image is not None:
            result.final_image.save(RES / f"improved_{idx:02d}_final.jpg", quality=85)

        # Parse predicted class from answer text
        pred = "??"
        ans_lower = result.answer.lower()
        # Search for class names; prefer the one that appears with a strong signal
        for cand in ["zigzag", "fcbk", "cfcbk", "none"]:
            if cand in ans_lower:
                pred = cand.upper() if cand != "none" else "none"
                # Map zigzag → "Zigzag" canonical
                pred = {"ZIGZAG": "Zigzag", "FCBK": "FCBK", "CFCBK": "CFCBK", "none": "none"}.get(pred, pred)
                break

        rec_out = {"idx": idx, "true": cls, "pred": pred, "answer": result.answer.strip(),
                   "vlm_calls": result.n_vlm_calls, "fp_calls": result.n_fp_calls,
                   "supporting_mask_ids": result.supporting_mask_ids,
                   "elapsed_s": dt}
        results.append(rec_out)
        print(f"  → pred={pred:6s} (true={cls})  vlm={result.n_vlm_calls} fp={result.n_fp_calls} {dt}s")
        print(f"     answer: {result.answer.strip()[:140]}")

    (RES / "improved_results.json").write_text(json.dumps(results, indent=2, default=str))
    correct = sum(1 for r in results if r["pred"] == r["true"])
    print(f"\nImproved agent accuracy: {correct}/{len(results)} = {correct/len(results)*100:.0f}%")
    for cls in ["CFCBK", "FCBK", "Zigzag", "none"]:
        sub = [r for r in results if r["true"] == cls]
        c = sum(1 for r in sub if r["pred"] == cls)
        print(f"  {cls}: {c}/{len(sub)}  preds={[r['pred'] for r in sub]}")
