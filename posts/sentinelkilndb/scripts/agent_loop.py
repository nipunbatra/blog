"""Demo: Gemma 4 in the actual agent loop, calling Falcon Perception as a tool,
on a high-res ESRI tile of a brick kiln."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "/Users/nipun/git/blog/posts/mlx_vlm_pr926")

from PIL import Image
from mlx_vlm import load
from agent import LocalVLMClient, run_agent

ROOT = Path("/Users/nipun/git/blog/posts/sentinelkilndb")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

# Load on the CFCBK z18 tight image — most visible kiln
TILE = ROOT / "hires" / "00_CFCBK_tight_z18.jpg"
QUERY = ("This is a high-resolution aerial image (~0.6 m/pixel) of a small area in India. "
         "I'm looking for a brick kiln. Specifically, the dataset SentinelKilnDB recognises three types: "
         "CFCBK (circular/oval, central chimney), FCBK (rectangular with a single chimney), "
         "and Zigzag (large rectangle, internal zigzag walls). "
         "Identify whether a brick kiln is visible, and if so, classify it.")

print("[load] Falcon Perception ...", flush=True)
fp_model, fp_processor = load("tiiuae/Falcon-Perception")
print("[load] Gemma 4 31B 4bit ...", flush=True)
vlm = LocalVLMClient("mlx-community/gemma-4-31b-it-4bit", max_tokens=2048)

img = Image.open(TILE).convert("RGB")
print(f"image: {img.size}", flush=True)

t = time.time()
result = run_agent(img, QUERY, fp_model, fp_processor, vlm, max_steps=8, verbose=True)
print(f"\nELAPSED {time.time()-t:.1f}s", flush=True)
print(f"answer: {result.answer}")

# Save artefacts
(RES / "agent_loop_answer.txt").write_text(result.answer)
(RES / "agent_loop_stats.json").write_text(json.dumps({
    "fp_calls": result.n_fp_calls,
    "vlm_calls": result.n_vlm_calls,
    "supporting_mask_ids": result.supporting_mask_ids,
    "elapsed_s": round(time.time()-t, 1),
}, indent=2))
if result.final_image is not None:
    result.final_image.save(RES / "agent_loop_final.jpg", quality=88)

trace_items = []
for step in result.trace:
    item = {"step": step.step, "think": step.think,
            "tool_name": step.tool_name, "tool_params": step.tool_params,
            "result_text": step.result_text[:1500]}
    if step.som_image is not None:
        p = RES / f"agent_loop_som_step_{step.step}.jpg"
        step.som_image.save(p, quality=85)
        item["som_image"] = p.name
    trace_items.append(item)
(RES / "agent_loop_trace.json").write_text(json.dumps(trace_items, indent=2))
print("ALL DONE")
