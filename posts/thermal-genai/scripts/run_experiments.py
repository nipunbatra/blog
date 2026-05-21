"""Run all Gemini-thermal-generation experiments and save outputs to disk.

Each output PNG is written under ../outputs/ and a top-level summary.json
records the prompts, model IDs, latencies, and Gemini-judge scores so the
notebook can render the whole story without re-calling the API.

Models used:
  - gemini-3.1-flash-image-preview ("Nano Banana 2") — image generation
  - gemini-3.5-flash                  — text captions, critiques, scoring
"""
import argparse, io, json, time
from pathlib import Path
from PIL import Image

from google import genai

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
client = genai.Client()
IMAGE_MODEL = "gemini-3.1-flash-image-preview"
TEXT_MODEL = "gemini-3.5-flash"


def gen_image(prompt_parts, save_as):
    """Generate one image; returns (saved_path or None, latency_s, text_response)."""
    t0 = time.perf_counter()
    resp = client.models.generate_content(model=IMAGE_MODEL, contents=prompt_parts)
    latency = time.perf_counter() - t0
    saved = None; text = []
    for p in resp.parts:
        if hasattr(p, "inline_data") and p.inline_data is not None:
            data = p.inline_data.data
            img = Image.open(io.BytesIO(data))
            img.save(OUT / save_as)
            saved = save_as
        elif hasattr(p, "text") and p.text:
            text.append(p.text)
    return saved, latency, "\n".join(text)


def gen_text(prompt, system=None):
    parts = [prompt] if not system else [system, prompt]
    t0 = time.perf_counter()
    resp = client.models.generate_content(model=TEXT_MODEL, contents=parts)
    return resp.text.strip(), time.perf_counter() - t0


def caption(image_path, system_prompt):
    img = Image.open(image_path)
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[system_prompt, img])
    return resp.text.strip(), time.perf_counter() - t0


def judge(real, generated, prompt):
    real_img = Image.open(real); gen_img = Image.open(generated)
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt,
                  "REFERENCE (real thermal):", real_img,
                  "CANDIDATE (Gemini-generated thermal):", gen_img])
    return resp.text.strip(), time.perf_counter() - t0


def main():
    summary = {"models": {"image": IMAGE_MODEL, "text": TEXT_MODEL}, "steps": []}

    rgb_path = OUT / "source_rgb.png"
    real_thermal_path = OUT / "source_thermal_real.png"
    thermeval_path = OUT / "source_thermeval.png"

    # ----- step 1: Gemini captions for the RGB and the real thermal -----
    print("[1] captioning RGB + real thermal")
    rgb_cap, t = caption(rgb_path,
        "Write a single-sentence caption describing this person/scene. "
        "Be specific about visible objects, posture, and clothing.")
    print(f"  RGB caption:    {rgb_cap[:120]}")
    summary["steps"].append({"name": "caption_rgb", "caption": rgb_cap, "latency_s": t})

    therm_cap, t = caption(real_thermal_path,
        "This is a long-wave-infrared thermal image. Write a single-sentence "
        "caption that describes ONLY what you see in this thermal image: which "
        "regions are warmer (brighter) vs cooler (darker), and which face/body "
        "features are visible from the temperature signal. Don't invent things "
        "you can't see.")
    print(f"  Thermal caption: {therm_cap[:120]}")
    summary["steps"].append({"name": "caption_thermal", "caption": therm_cap,
                             "latency_s": t})

    thermeval_cap, t = caption(thermeval_path,
        "This is a thermal infrared image. Describe what you see — number of "
        "people, posture, what's warm vs cold.")
    print(f"  ThermEval cap:  {thermeval_cap[:120]}")
    summary["steps"].append({"name": "caption_thermeval", "caption": thermeval_cap,
                             "latency_s": t})

    # ----- step 2: caption-only generation, 4 caption styles -----
    captions = {
        "exp1a_generic": "A thermal infrared image of a human face. Iron color palette.",
        "exp1b_rgb_derived": f"A thermal infrared image of: {rgb_cap}. Iron color palette.",
        "exp1c_thermal_physics": (
            "A thermal infrared image of a human face on iron color palette. "
            "Hot regions (bright yellow/white): periorbital area (around eyes), "
            "carotid region (sides of neck), exhaled air around the nostrils. "
            "Cool regions (dark purple/black): hair, eyeglasses (if any), the "
            "tip of the nose, ears, and any cloth fabric. The face skin should be "
            "warmer (yellow/orange) than the background room which should be cool "
            "(dark blue/black)."),
        "exp1d_gemini_thermcap": (
            f"A thermal infrared image of a human face. {therm_cap} Iron palette."),
    }
    for name, prompt in captions.items():
        print(f"[2] {name}")
        saved, lat, _ = gen_image([prompt], f"{name}.png")
        summary["steps"].append({"name": name, "prompt": prompt, "saved": saved,
                                 "latency_s": lat})

    # ----- step 3: RGB + thermal-physics caption -> thermal -----
    print("[3] RGB+thermal-physics caption -> thermal")
    rgb_img = Image.open(rgb_path)
    prompt3 = (
        "I am giving you a real RGB photograph of a person. Generate the "
        "corresponding long-wave infrared (LWIR) thermal image of the SAME "
        "subject in the SAME pose. Use the iron color palette: warm regions "
        "(skin) should be yellow/orange/red, cool regions should be dark "
        "purple/black. The eyes / periorbital area should be the hottest, "
        "the eyeglasses (if any) should be cold (dark), the hair cool, the "
        "background room cool. Match the silhouette of the input photo exactly.")
    saved, lat, _ = gen_image([prompt3, rgb_img], "exp3_rgb_plus_thermcap.png")
    summary["steps"].append({"name": "exp3_rgb_plus_thermcap",
                             "prompt": prompt3, "saved": saved,
                             "latency_s": lat})

    # ----- step 4: self-critique loop -----
    print("[4] self-critique loop (2 iterations)")
    current_gen = OUT / "exp3_rgb_plus_thermcap.png"
    for it in range(2):
        # Show real thermal + generated, ask Gemini to find differences
        crit_prompt = (
            "Compare these two long-wave-infrared thermal face images. The "
            "REFERENCE is a real thermal photograph. The CANDIDATE was generated "
            "by an AI image model from an RGB photograph. List 3-5 SPECIFIC "
            "physical errors in the candidate's thermal physics (e.g. 'the "
            "eyeglasses appear hot when in reality glass blocks LWIR and should "
            "be cold', 'the hair is too warm', 'the nostrils don't show exhale "
            "plume'). Be concrete about WHERE in the image and WHAT'S WRONG.")
        critique, t = judge(real_thermal_path, current_gen, crit_prompt)
        print(f"  iter {it} critique: {critique[:200]}")
        summary["steps"].append({"name": f"critique_iter{it}",
                                 "critique": critique, "latency_s": t})
        # Use the critique to regenerate
        refine_prompt = (
            "I will give you an RGB photograph and a list of thermal-physics "
            "corrections. Generate a new long-wave-infrared (LWIR) thermal image "
            "of the same subject, fixing all the listed issues. Iron color "
            "palette. Corrections: " + critique)
        saved, lat, _ = gen_image([refine_prompt, rgb_img],
                                  f"exp4_refined_iter{it+1}.png")
        summary["steps"].append({"name": f"exp4_refined_iter{it+1}",
                                 "prompt": refine_prompt, "saved": saved,
                                 "latency_s": lat})
        current_gen = OUT / f"exp4_refined_iter{it+1}.png"

    # ----- step 5: Gemini judges all generated images vs real thermal -----
    print("[5] Gemini-as-judge scores")
    scores = {}
    judge_prompt = (
        "You are evaluating an AI-generated long-wave infrared (LWIR) thermal "
        "face image against a real thermal photograph (REFERENCE). Score the "
        "candidate on a 1-10 scale for THERMAL PHYSICAL PLAUSIBILITY, where "
        "10 = correct relative temperatures (eyes hottest, glasses cold, hair "
        "cool, etc) and 1 = visually thermal-looking but completely wrong "
        "physics. Output exactly: 'SCORE: N/10 — <one sentence reason>'.")
    for name in ["exp1a_generic", "exp1b_rgb_derived", "exp1c_thermal_physics",
                 "exp1d_gemini_thermcap", "exp3_rgb_plus_thermcap",
                 "exp4_refined_iter1", "exp4_refined_iter2"]:
        path = OUT / f"{name}.png"
        if not path.exists():
            scores[name] = None; continue
        s, _ = judge(real_thermal_path, path, judge_prompt)
        scores[name] = s
        print(f"  {name}: {s[:100]}")
    summary["judge_scores"] = scores

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
