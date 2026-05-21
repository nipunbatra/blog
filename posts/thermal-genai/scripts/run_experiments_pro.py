"""Re-run the thermal-generation experiments with Gemini 3 Pro Image Preview
(Nano Banana Pro) to compare against the Nano Banana 2 baseline.

Identical setup to run_experiments.py except IMAGE_MODEL is changed.
Outputs land in ../outputs_pro/ to avoid clobbering the baseline.
"""
import io, json, time
from pathlib import Path
from PIL import Image
from google import genai

OUT = Path(__file__).resolve().parent.parent / "outputs_pro"
OUT.mkdir(parents=True, exist_ok=True)
BASE_OUT = Path(__file__).resolve().parent.parent / "outputs"

client = genai.Client()
IMAGE_MODEL = "gemini-3-pro-image-preview"   # Nano Banana Pro
TEXT_MODEL = "gemini-3.5-flash"


def gen_image(prompt_parts, save_as):
    t0 = time.perf_counter()
    resp = client.models.generate_content(model=IMAGE_MODEL, contents=prompt_parts)
    latency = time.perf_counter() - t0
    saved = None; text = []
    for p in resp.parts:
        if hasattr(p, "inline_data") and p.inline_data is not None:
            data = p.inline_data.data
            img = Image.open(io.BytesIO(data))
            img.save(OUT / save_as); saved = save_as
        elif hasattr(p, "text") and p.text:
            text.append(p.text)
    return saved, latency, "\n".join(text)


def caption(image_path, system_prompt):
    img = Image.open(image_path)
    resp = client.models.generate_content(
        model=TEXT_MODEL, contents=[system_prompt, img])
    return resp.text.strip()


def judge(real, generated, prompt):
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt,
                  "REFERENCE (real thermal):", Image.open(real),
                  "CANDIDATE (Gemini-generated thermal):", Image.open(generated)])
    return resp.text.strip()


def main():
    summary = {"image_model": IMAGE_MODEL, "text_model": TEXT_MODEL, "steps": []}
    rgb_path = BASE_OUT / "source_rgb.png"
    real_thermal_path = BASE_OUT / "source_thermal_real.png"

    # Re-use the captions from the baseline run (so prompts are identical):
    base_summary = json.load(open(BASE_OUT / "summary.json"))
    rgb_cap = next(s["caption"] for s in base_summary["steps"]
                   if s["name"] == "caption_rgb")
    therm_cap = next(s["caption"] for s in base_summary["steps"]
                     if s["name"] == "caption_thermal")

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

    print("[3] RGB + thermal-physics caption -> thermal")
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
                             "prompt": prompt3, "saved": saved, "latency_s": lat})

    # Self-critique loop, 2 iterations
    current_gen = OUT / "exp3_rgb_plus_thermcap.png"
    for it in range(2):
        crit_prompt = (
            "Compare these two long-wave-infrared thermal face images. The "
            "REFERENCE is a real thermal photograph. The CANDIDATE was generated "
            "by an AI image model from an RGB photograph. List 3-5 SPECIFIC "
            "physical errors in the candidate's thermal physics (e.g. 'the "
            "eyeglasses appear hot when in reality glass blocks LWIR and should "
            "be cold', 'the hair is too warm', 'the nostrils don't show exhale "
            "plume'). Be concrete about WHERE in the image and WHAT'S WRONG.")
        critique = judge(real_thermal_path, current_gen, crit_prompt)
        print(f"  iter {it}: {critique[:150]}")
        summary["steps"].append({"name": f"critique_iter{it}", "critique": critique})
        refine = (
            "I will give you an RGB photograph and a list of thermal-physics "
            "corrections. Generate a new long-wave-infrared (LWIR) thermal image "
            "of the same subject, fixing all the listed issues. Iron color "
            "palette. Corrections: " + critique)
        saved, lat, _ = gen_image([refine, rgb_img], f"exp4_refined_iter{it+1}.png")
        summary["steps"].append({"name": f"exp4_refined_iter{it+1}",
                                 "prompt": refine, "saved": saved, "latency_s": lat})
        current_gen = OUT / f"exp4_refined_iter{it+1}.png"

    # Judge each output
    judge_prompt = (
        "You are evaluating an AI-generated long-wave infrared (LWIR) thermal "
        "face image against a real thermal photograph (REFERENCE). Score the "
        "candidate on a 1-10 scale for THERMAL PHYSICAL PLAUSIBILITY, where "
        "10 = correct relative temperatures (eyes hottest, glasses cold, hair "
        "cool, etc) and 1 = visually thermal-looking but completely wrong "
        "physics. Output exactly: 'SCORE: N/10 — <one sentence reason>'.")
    scores = {}
    for name in ["exp1a_generic", "exp1b_rgb_derived", "exp1c_thermal_physics",
                 "exp1d_gemini_thermcap", "exp3_rgb_plus_thermcap",
                 "exp4_refined_iter1", "exp4_refined_iter2"]:
        path = OUT / f"{name}.png"
        if not path.exists():
            scores[name] = None; continue
        s = judge(real_thermal_path, path, judge_prompt)
        scores[name] = s
        print(f"  {name}: {s[:100]}")
    summary["judge_scores"] = scores

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
