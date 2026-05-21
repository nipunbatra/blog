"""Apply iron palette to ThermalGen grayscale output, then have Gemini-3.5
judge it against the real SF-TL54 thermal using the same prompt as the
Gemini-generated images."""
import io, json, re
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from google import genai

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TG_DIR = OUT / "thermalgen"
REAL_THERMAL = OUT / "source_thermal_real.png"

client = genai.Client()
TEXT_MODEL = "gemini-3.5-flash"
JUDGE_PROMPT = (
    "You are evaluating an AI-generated long-wave infrared (LWIR) thermal "
    "face image against a real thermal photograph (REFERENCE). Score the "
    "candidate on a 1-10 scale for THERMAL PHYSICAL PLAUSIBILITY, where "
    "10 = correct relative temperatures (eyes hottest, glasses cold, hair "
    "cool, etc) and 1 = visually thermal-looking but completely wrong "
    "physics. Output exactly: 'SCORE: N/10 — <one sentence reason>'.")

# OpenCV's COLORMAP_INFERNO is a close visual match to the FLIR "Iron" palette.
def to_iron(gray_path, out_path):
    g = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
    iron = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out_path), iron)


def judge(real, generated):
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[JUDGE_PROMPT,
                  "REFERENCE (real thermal):", Image.open(real),
                  "CANDIDATE (AI-generated thermal):", Image.open(generated)])
    return resp.text.strip()


def main():
    color_dir = OUT / "thermalgen_iron"
    color_dir.mkdir(exist_ok=True)
    scores = {}
    for p in sorted(TG_DIR.glob("*.png")):
        iron_path = color_dir / p.name
        to_iron(p, iron_path)
        s = judge(REAL_THERMAL, iron_path)
        scores[p.stem] = s
        print(f"  {p.stem}: {s[:120]}")
    with open(OUT / "thermalgen_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"wrote {OUT / 'thermalgen_scores.json'}")


if __name__ == "__main__":
    main()
