"""Use SAM 3 (Meta, text-prompt segmentation) to get clean anatomical masks
for each thermal image, then compute region-ratio metrics with those masks.

For each image (real + each generated):
  - Prompt SAM with text "eye", "hair", "nose", "skin" to get masks.
  - Compute mean luminance per region.
  - Compute ratios eye/orbital, hair/face, nose/cheek.

Why this matters vs the MediaPipe-mask version:
  MediaPipe FaceMesh is RGB-trained and unreliable on thermal images
  (the eye landmarks float to wrong positions; the hair mask is
  hand-defined by image coordinates). SAM 3 is text-promptable and
  thermal-agnostic — it'll find the actual eye / hair / nose regardless
  of pixel statistics.

Outputs:
  outputs/sam3_metrics.json — region luminances and ratios per image
  outputs/sam3_masks_grid.png — visual showing SAM masks on each image
"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SAM3_ROOT = Path(sam3.__file__).resolve().parent.parent
BPE = SAM3_ROOT / "assets/bpe_simple_vocab_16e6.txt.gz"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[init] device={device}")

torch.backends.cuda.matmul.allow_tf32 = True
if device == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

model = build_sam3_image_model(bpe_path=str(BPE)).to(device).eval()
processor = Sam3Processor(model, confidence_threshold=0.3)


def masks_for(image_pil, prompts):
    """For each text prompt, run SAM3 and return a binary mask (HxW)."""
    H = image_pil.height; W = image_pil.width
    state = processor.set_image(image_pil)
    out = {}
    for p in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=p)
        # state has masks under various keys depending on SAM3 version
        masks_field = state.get("masks", None) or state.get("seg_masks", None)
        if masks_field is None:
            out[p] = None; continue
        # masks_field is a tensor (N, 1, H, W) or list
        m = masks_field
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
        if m is None or len(m) == 0:
            out[p] = None; continue
        # Take union over all detected instances of the prompted concept
        m = (m > 0.5).astype(np.uint8).squeeze()
        if m.ndim == 3:  # (N, H, W)
            m = m.any(axis=0).astype(np.uint8)
        out[p] = m
    return out


def lum(img_rgb, mask):
    if mask is None or not mask.any(): return None
    y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if mask.shape != y.shape:
        mask = cv2.resize(mask, (y.shape[1], y.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return float(y[mask > 0].mean())


def process(name, image_path, prompts):
    print(f"[seg] {name}")
    img_pil = Image.open(image_path).convert("RGB")
    masks = masks_for(img_pil, prompts)
    img_rgb = np.array(img_pil)
    region_lums = {p: lum(img_rgb, masks[p]) for p in prompts}
    return masks, region_lums


def main():
    prompts = ["eye", "hair", "nose", "face", "mouth"]
    candidates = [
        ("real_sftl54", OUT / "source_thermal_real.png"),
        ("real_thermeval", OUT / "source_thermeval.png"),
        ("flash_exp4_iter1", OUT / "exp4_refined_iter1.png"),
        ("pro_exp4_iter1", ROOT / "outputs_pro/exp4_refined_iter1.png"),
        ("pro_exp3_RGBplus", ROOT / "outputs_pro/exp3_rgb_plus_thermcap.png"),
        ("thermalgen_L_iron", OUT / "thermalgen_iron/ThermalGen-L-2-concat_ds7_cfg1.0.png"),
        ("thermalgen_L_gray", OUT / "thermalgen/ThermalGen-L-2-concat_ds7_cfg1.0.png"),
    ]
    results = {}; mask_dir = OUT / "sam3_masks"
    mask_dir.mkdir(exist_ok=True)
    for name, p in candidates:
        if not p.exists(): print(f"missing {p}"); continue
        masks, lums = process(name, p, prompts)
        # Save visualisation of each mask
        img_rgb = np.array(Image.open(p).convert("RGB"))
        for q, m in masks.items():
            if m is None: continue
            overlay = img_rgb.copy()
            mr = cv2.resize(m, (overlay.shape[1], overlay.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            overlay[mr > 0] = (0.4 * np.array([0, 255, 0]) + 0.6 * overlay[mr > 0]).astype(np.uint8)
            Image.fromarray(overlay).save(mask_dir / f"{name}__{q}.png")
        results[name] = lums
        print(f"   lums: {lums}")

    # Compute ratios
    summary = {}
    for name, lums in results.items():
        eye = lums.get("eye"); hair = lums.get("hair")
        nose = lums.get("nose"); face = lums.get("face"); mouth = lums.get("mouth")
        ratios = {}
        if eye is not None and face is not None: ratios["eye_to_face"] = eye / face
        if hair is not None and face is not None: ratios["hair_to_face"] = hair / face
        if nose is not None and face is not None: ratios["nose_to_face"] = nose / face
        summary[name] = {"lums": lums, "ratios": ratios}

    # Real reference
    real_ratios = summary.get("real_sftl54", {}).get("ratios", {})
    print("\nREAL SF-TL54 reference ratios:", real_ratios)
    print()
    print(f"{'Method':<22} {'eye/face':>8} {'hair/face':>10} {'nose/face':>10} "
          f"{'eye dev':>8} {'hair dev':>9} {'nose dev':>9}")
    for name, info in summary.items():
        r = info["ratios"]
        def fmt(v, ndp=3):
            return f"{v:>.{ndp}f}" if v is not None else "n/a"
        def dev(k):
            cand = r.get(k); ref = real_ratios.get(k)
            if cand is None or ref is None: return None
            return abs(cand - ref) / max(ref, 1e-6)
        print(f"{name:<22} {fmt(r.get('eye_to_face')):>8} "
              f"{fmt(r.get('hair_to_face')):>10} "
              f"{fmt(r.get('nose_to_face')):>10} "
              f"{fmt(dev('eye_to_face'), 2):>8} "
              f"{fmt(dev('hair_to_face'), 2):>9} "
              f"{fmt(dev('nose_to_face'), 2):>9}")

    with open(OUT / "sam3_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nwrote {OUT / 'sam3_metrics.json'}")


if __name__ == "__main__":
    main()
