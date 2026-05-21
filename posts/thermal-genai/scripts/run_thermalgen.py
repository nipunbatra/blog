"""Run ThermalGen on the SF-TL54 RGB used in the Gemini experiments, save
the predicted thermal back to disk for the comparison.

Pretrained checkpoints from https://github.com/arplaboratory/ThermalGen
(NeurIPS 2025). Flow-based SiT model conditioned on RGB + dataset index.
"""
import argparse
import sys
import time
from pathlib import Path

# This script must be run from inside ~/git/ThermalGen on bhaskar
sys.path.insert(0, str(Path.home() / "git/ThermalGen"))

import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from thermalgen_demo import ThermalGenSIT  # noqa: E402

RGB_IMG = Path.home() / "git/ThermalGen/sf_tl54_subject100_rgb.png"
OUT = Path.home() / "git/ThermalGen/sftl54_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main(model_id, dataset_indices, cfg_scales):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] loading {model_id} on {device}")
    t0 = time.perf_counter()
    model = ThermalGenSIT.from_pretrained(model_id).to(device)
    model.eval()
    print(f"[init] loaded in {time.perf_counter()-t0:.1f}s")

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR,
                  antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    rgb_pil = Image.open(RGB_IMG).convert("RGB")
    rgb_tensor = eval_transform(rgb_pil).unsqueeze(0).to(device)

    short_id = model_id.split("/")[-1]
    for ds_idx in dataset_indices:
        for cfg in cfg_scales:
            model.use_cfg = cfg > 1.0
            model.model_config["cfg_scale"] = cfg
            with torch.no_grad():
                ds = torch.ones(1, dtype=torch.long, device=device) * ds_idx
                t0 = time.perf_counter()
                pred = model(rgb_tensor, ds)
                latency = time.perf_counter() - t0
            pred = (pred * 0.5 + 0.5).clamp(0, 1)
            out_path = OUT / f"{short_id}_ds{ds_idx}_cfg{cfg}.png"
            to_pil_image(pred[0].cpu()).save(out_path)
            print(f"  {out_path.name}  {latency:.1f}s")
    print(f"wrote {len(dataset_indices)*len(cfg_scales)} images to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xjh19972/ThermalGen-L-2-concat",
                    help="HF model id")
    ap.add_argument("--dataset-indices", default="7,4,5,8",
                    help="comma-sep dataset indices to try")
    ap.add_argument("--cfg", default="1.0,4.0",
                    help="comma-sep cfg scales")
    args = ap.parse_args()
    main(args.model,
         [int(x) for x in args.dataset_indices.split(",")],
         [float(x) for x in args.cfg.split(",")])
