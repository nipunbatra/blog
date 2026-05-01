"""Side-by-side comparison of Sapiens2 normal-head sizes on the same image.

Runs 0.4B / 0.8B / 1B (and 5B if present) on a target image, records cold +
warm forward time and peak resident memory, and emits a horizontally-stacked
visual + a small CSV.
"""

import os
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sapiens.dense.models import init_model

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent
OUT = POST_DIR / "outputs"
OUT.mkdir(exist_ok=True)

CONFIGS = {
    "0.4b": "/tmp/sapiens2/sapiens/dense/configs/normal/metasim_render_people/sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
    "0.8b": "/tmp/sapiens2/sapiens/dense/configs/normal/metasim_render_people/sapiens2_0.8b_normal_metasim_render_people-1024x768.py",
    "1b":   "/tmp/sapiens2/sapiens/dense/configs/normal/metasim_render_people/sapiens2_1b_normal_metasim_render_people-1024x768.py",
    "5b":   "/tmp/sapiens2/sapiens/dense/configs/normal/metasim_render_people/sapiens2_5b_normal_metasim_render_people-1024x768.py",
}
CKPTS = {
    s: os.path.expanduser(f"~/sapiens2_host/normal/sapiens2_{s}_normal.safetensors")
    for s in CONFIGS
}


def predict_normal(model, image):
    data = model.pipeline(dict(img=image))
    data = model.data_preprocessor(data)
    inputs, ds = data["inputs"], data["data_samples"]
    with torch.no_grad():
        n = model(inputs)
        n = n / torch.norm(n, dim=1, keepdim=True).clamp(1e-8)
    pl, pr, pt, pb = ds["meta"]["padding_size"]
    n = n[:, :, pt:inputs.shape[2] - pb, pl:inputs.shape[3] - pr]
    n = F.interpolate(n, size=(image.shape[0], image.shape[1]),
                      mode="bilinear", align_corners=False)
    return n.squeeze(0).float().cpu().numpy()


def encode_rgb(n):
    rgb = ((n.transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def panel(images, labels):
    h = max(im.shape[0] for im in images)
    rs = [cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h))
          for im in images]
    strip = np.concatenate(rs, axis=1)
    cap = np.full((50, strip.shape[1], 3), 255, np.uint8)
    cw = strip.shape[1] // len(images)
    for i, lab in enumerate(labels):
        cv2.putText(cap, lab, (i * cw + 14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 40), 2)
    return np.concatenate([strip, cap], axis=0)


def main(image_path: str = None, out_stem: str = "size_compare"):
    if image_path is None:
        image_path = str(POST_DIR / "inputs/desk_worker.jpg")
    image = cv2.imread(image_path)
    print(f"[image] {image_path}  {image.shape[1]}x{image.shape[0]}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = []
    rgbs = [image]
    labels = ["original"]

    for size, cfg in CONFIGS.items():
        ckpt = CKPTS[size]
        if not os.path.exists(ckpt):
            print(f"[skip] {size}: {ckpt} not present")
            continue
        ckpt_gb = os.path.getsize(ckpt) / 1e9
        print(f"\n[load] {size} ({ckpt_gb:.1f} GB) on {device} …")
        t0 = time.perf_counter()
        model = init_model(cfg, ckpt, device=device)
        t_load = time.perf_counter() - t0
        # Cold forward
        t1 = time.perf_counter()
        n = predict_normal(model, image)
        t_cold = time.perf_counter() - t1
        # Warm forward (averaged over 3)
        ts = []
        for _ in range(3):
            tx = time.perf_counter()
            _ = predict_normal(model, image)
            ts.append(time.perf_counter() - tx)
        t_warm = sum(ts) / len(ts)
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"[time] {size}: load {t_load:5.1f} s | cold {t_cold:.2f} s | "
              f"warm {t_warm:.2f} s | rss {rss_gb:5.1f} GB")
        results.append({
            "size": size, "ckpt_gb": round(ckpt_gb, 2),
            "load_s": round(t_load, 2), "cold_s": round(t_cold, 2),
            "warm_s": round(t_warm, 2), "rss_gb": round(rss_gb, 2),
        })
        rgbs.append(encode_rgb(n))
        labels.append(f"{size}  ({t_warm*1000:.0f} ms warm)")
        del model
        import gc; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    strip = panel(rgbs, labels)
    cv2.imwrite(str(OUT / f"{out_stem}.jpg"), strip)
    with open(OUT / f"{out_stem}.csv", "w") as f:
        f.write("size,ckpt_gb,load_s,cold_s,warm_s,rss_gb\n")
        for r in results:
            f.write(f"{r['size']},{r['ckpt_gb']},{r['load_s']},"
                    f"{r['cold_s']},{r['warm_s']},{r['rss_gb']}\n")
    print(f"\n[done] {OUT / f'{out_stem}.jpg'}")
    print(f"[done] {OUT / f'{out_stem}.csv'}")


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else None
    stem = sys.argv[2] if len(sys.argv) > 2 else "size_compare"
    main(image, stem)
