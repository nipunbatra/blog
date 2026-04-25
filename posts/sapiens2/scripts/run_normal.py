"""Run Sapiens2 surface-normal estimation on a single image.

Tries Apple-Silicon MPS first, falls back to CPU. Saves both the
RGB-encoded normal map and a per-channel breakdown.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent

# The sapiens.dense.tools.vis script lives inside the cloned repo.
# We import it via init_model directly.
from sapiens.dense.models import init_model  # noqa: E402


def encode_normal_rgb(normal_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) unit-vector normals in [-1, 1] -> uint8 RGB."""
    n = normal_chw.transpose(1, 2, 0)
    rgb = ((n + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return rgb  # H x W x 3, RGB


def main(image_path: str, output_dir: str,
         config: str = None, checkpoint: str = None,
         device: str = None):
    if config is None:
        config = ("/tmp/sapiens2/sapiens/dense/configs/normal/"
                  "metasim_render_people/"
                  "sapiens2_0.4b_normal_metasim_render_people-1024x768.py")
    if checkpoint is None:
        checkpoint = os.path.expanduser(
            "~/sapiens2_host/normal/sapiens2_0.4b_normal.safetensors")

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[init] device={device}")
    print(f"[init] config={config}")
    print(f"[init] checkpoint={checkpoint}")
    t = time.perf_counter()
    model = init_model(config, checkpoint, device=device)
    print(f"[init] done in {time.perf_counter() - t:.1f} s")

    image = cv2.imread(image_path)        # BGR
    if image is None:
        raise SystemExit(f"could not read {image_path}")
    H0, W0 = image.shape[:2]
    print(f"[image] {image_path}  {W0}x{H0}")

    data = model.pipeline(dict(img=image))
    data = model.data_preprocessor(data)
    inputs, data_samples = data["inputs"], data["data_samples"]

    t = time.perf_counter()
    with torch.no_grad():
        normal = model(inputs)                            # 1 x 3 x H x W
        normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(1e-8)
    print(f"[infer] forward+normalise: {time.perf_counter() - t:.2f} s")

    pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"]["padding_size"]
    normal = normal[:, :,
                    pad_top:inputs.shape[2] - pad_bottom,
                    pad_left:inputs.shape[3] - pad_right]
    normal = F.interpolate(normal, size=(H0, W0),
                           mode="bilinear", align_corners=False)
    normal_np = normal.squeeze(0).float().cpu().numpy()   # 3 x H x W

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    # 1) standard normal-as-RGB
    cv2.imwrite(str(out / f"{stem}_normal.jpg"),
                cv2.cvtColor(encode_normal_rgb(normal_np), cv2.COLOR_RGB2BGR))

    # 2) per-channel grayscale (X, Y, Z)
    for i, name in enumerate("xyz"):
        ch = ((normal_np[i] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{stem}_normal_{name}.jpg"), ch)

    # 3) raw .npy for downstream use
    np.save(out / f"{stem}_normal.npy", normal_np)

    print(f"[done] outputs in {out}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(POST_DIR / "inputs/person1.jpg")
    outdir = sys.argv[2] if len(sys.argv) > 2 else str(POST_DIR / "outputs")
    main(img, outdir)
