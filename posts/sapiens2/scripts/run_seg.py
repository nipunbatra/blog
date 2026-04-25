"""Run Sapiens2 body-part segmentation on a single image. Produces a
side-by-side input/segmentation visualization plus a raw class-id .npy."""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sapiens.dense.models import init_model
from sapiens.dense.visualizers import SegVisualizer

THIS_DIR = Path(__file__).resolve().parent
POST_DIR = THIS_DIR.parent


def main(image_path: str, output_dir: str,
         config: str = None, checkpoint: str = None,
         device: str = None):
    if config is None:
        config = ("/tmp/sapiens2/sapiens/dense/configs/seg/"
                  "shutterstock_goliath/"
                  "sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py")
    if checkpoint is None:
        checkpoint = os.path.expanduser(
            "~/sapiens2_host/seg/sapiens2_0.4b_seg.safetensors")
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[init] device={device}")
    t = time.perf_counter()
    model = init_model(config, checkpoint, device=device)
    print(f"[init] done in {time.perf_counter() - t:.1f} s")

    image = cv2.imread(image_path)
    print(f"[image] {image_path}  {image.shape[1]}x{image.shape[0]}")

    data = model.pipeline(dict(img=image))
    data = model.data_preprocessor(data)
    inputs = data["inputs"]

    t = time.perf_counter()
    with torch.no_grad():
        seg_logits = model(inputs)
    print(f"[infer] forward: {time.perf_counter() - t:.2f} s")

    seg_logits = F.interpolate(seg_logits, size=image.shape[:2], mode="bilinear")
    pred_labels = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()

    vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
    seg_vis = vis._visualize_segmentation(image, pred_labels)
    side_by_side = np.concatenate([image, seg_vis], axis=1)

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(str(out / f"{stem}_seg.jpg"), side_by_side)
    np.save(out / f"{stem}_seg.npy", pred_labels.astype(np.int16))

    counts = np.bincount(pred_labels.flatten(), minlength=30)
    print("[classes] non-zero pixels per class:")
    for cid, n in enumerate(counts):
        if n > 0:
            print(f"  class {cid:2d}: {n:>10,d} px ({100*n/pred_labels.size:5.2f}%)")
    print(f"[done] {out / f'{stem}_seg.jpg'}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(POST_DIR / "inputs/person1.jpg")
    outdir = sys.argv[2] if len(sys.argv) > 2 else str(POST_DIR / "outputs")
    main(img, outdir)
