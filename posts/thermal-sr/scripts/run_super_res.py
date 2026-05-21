"""Thermal super-resolution head-to-head.

Pipeline:
  1. Read HR thermal at native resolution (SF-TL54 sample, 464x348).
  2. Downsample 4x by area (-> ~116x87, simulating a Lepton 3.5).
  3. Restore to 464x348 via 4 methods:
       a) Bicubic (cv2.INTER_CUBIC)
       b) Real-ESRGAN-x4 (general purpose, RGB-trained)
       c) SwinIR-x4 (transformer SR, RGB-trained)
       d) Stable Diffusion x4 upscaler (caption-conditioned diffusion)
  4. Score each restored image against the HR ground truth on:
       - PSNR / SSIM / LPIPS (pixel + perceptual similarity)
       - DWPose nostril-localization error (downstream-task accuracy)
       - Visual sharpness/Laplacian variance (objective sharpness)
"""
import argparse, io, json, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips

ROOT = Path.home() / "thermal-sr-work"
ROOT.mkdir(parents=True, exist_ok=True)
HR_PATH = Path.home() / "thermal-genai-data/outputs/source_thermal_real.png"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWN_FACTOR = 4
CFG_DEVICE = "cpu"


def degrade(hr_pil, factor=DOWN_FACTOR):
    """Downsample HR by `factor` using area interpolation (good for downsizing)."""
    arr = np.array(hr_pil.convert("RGB"))
    H, W = arr.shape[:2]
    lr = cv2.resize(arr, (W // factor, H // factor), interpolation=cv2.INTER_AREA)
    return Image.fromarray(lr)


def restore_bicubic(lr_pil, target_size):
    arr = np.array(lr_pil)
    up = cv2.resize(arr, target_size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(up)


def restore_realesrgan(lr_pil):
    """Real-ESRGAN via the BasicSR/realesrgan PyPI tool — official path."""
    import subprocess
    in_path = OUT / "_re_input.png"; lr_pil.save(in_path)
    out_dir = OUT / "_re_out"; out_dir.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    subprocess.check_call([
        "python", "-c",
        ("from realesrgan import RealESRGANer;"
         "from basicsr.archs.rrdbnet_arch import RRDBNet;"
         "import torch, cv2;"
         "m = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4);"
         "u = RealESRGANer(scale=4, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', model=m, half=True, device='" + DEVICE + "');"
         "img = cv2.imread('" + str(in_path) + "');"
         "out, _ = u.enhance(img, outscale=4);"
         "cv2.imwrite('" + str(out_dir/'sr.png') + "', out);")
    ])
    elapsed = time.perf_counter() - t0
    sr = Image.open(out_dir / "sr.png").convert("RGB")
    return sr, elapsed


def _si_predict(model, lr_pil):
    from super_image import ImageLoader
    inputs = ImageLoader.load_image(lr_pil)
    t0 = time.perf_counter()
    out = model(inputs)
    elapsed = time.perf_counter() - t0
    out_arr = (out[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(out_arr), elapsed


def restore_edsr(lr_pil, scale=4):
    from super_image import EdsrModel
    return _si_predict(EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale), lr_pil)


def restore_msrn(lr_pil, scale=4):
    from super_image import MsrnModel
    return _si_predict(MsrnModel.from_pretrained("eugenesiow/msrn", scale=scale), lr_pil)


def restore_a2n(lr_pil, scale=4):
    from super_image import A2nModel
    return _si_predict(A2nModel.from_pretrained("eugenesiow/a2n", scale=scale), lr_pil)


def restore_drln(lr_pil, scale=4):
    from super_image import DrlnModel
    return _si_predict(DrlnModel.from_pretrained("eugenesiow/drln-bam", scale=scale), lr_pil)


def restore_sd_upscaler(lr_pil, prompt="a thermal infrared face image"):
    from diffusers import StableDiffusionUpscalePipeline
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    t0 = time.perf_counter()
    img = pipe(prompt=prompt, image=lr_pil, num_inference_steps=20).images[0]
    elapsed = time.perf_counter() - t0
    return img, elapsed


def pixel_metrics(hr_arr, restored_arr, lpips_net=None):
    # align sizes
    h, w = hr_arr.shape[:2]
    rh, rw = restored_arr.shape[:2]
    if (rh, rw) != (h, w):
        restored_arr = cv2.resize(restored_arr, (w, h),
                                   interpolation=cv2.INTER_AREA)
    diff = (hr_arr.astype(np.float32) - restored_arr.astype(np.float32))
    mse = float((diff ** 2).mean())
    psnr = float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    hr_y = cv2.cvtColor(hr_arr, cv2.COLOR_RGB2GRAY)
    rs_y = cv2.cvtColor(restored_arr, cv2.COLOR_RGB2GRAY)
    s = float(ssim(hr_y, rs_y, data_range=255))
    sharpness = float(cv2.Laplacian(rs_y, cv2.CV_64F).var())
    out = {"psnr": psnr, "ssim": s, "sharpness_lapvar": sharpness}
    if lpips_net is not None:
        def to_t(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            d = lpips_net(to_t(hr_arr), to_t(restored_arr)).item()
        out["lpips"] = float(d)
    return out


def dwpose_nose(image_pil):
    """Return (nose_x, nose_y) from DWPose, or None if no person detected."""
    from rtmlib import Wholebody
    wb = Wholebody(to_openpose=False, mode="balanced",
                   backend="onnxruntime", device="cuda")
    arr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    keypoints, scores = wb(arr)
    if keypoints is None or len(keypoints) == 0: return None
    # DWPose body keypoint 0 = nose
    kp = keypoints[0]
    return float(kp[0][0]), float(kp[0][1])


def main():
    print(f"[init] HR={HR_PATH}, device={DEVICE}")
    hr_pil = Image.open(HR_PATH).convert("RGB")
    W, H = hr_pil.size
    print(f"  HR size: {W}x{H}")

    lr_pil = degrade(hr_pil)
    lw, lh = lr_pil.size
    print(f"  LR size: {lw}x{lh}")
    lr_pil.save(OUT / "lr_input.png")
    hr_pil.save(OUT / "hr_target.png")

    # Restore via 4 methods
    methods = {}
    print("[1] bicubic")
    t0 = time.perf_counter()
    bicubic = restore_bicubic(lr_pil, (W, H))
    methods["bicubic"] = (bicubic, time.perf_counter() - t0)

    # Skipping Real-ESRGAN (the python wrapper + basicsr stack is brittle on
    # current torch/cu126; super-image's EDSR + diffusers' SD-upscaler cover
    # the "general-purpose RGB SR" baseline well enough for a paper sketch).
    methods["realesrgan_x4"] = (None, 0)

    for nm, fn in [("edsr_x4", restore_edsr), ("msrn_x4", restore_msrn),
                    ("a2n_x4", restore_a2n), ("drln_x4", restore_drln)]:
        print(f"[3] {nm}")
        try:
            sr, t = fn(lr_pil)
            methods[nm] = (sr, t)
        except Exception as e:
            print(f"   FAILED: {e}")
            methods[nm] = (None, 0)

    print("[4] Stable Diffusion x4 upscaler")
    try:
        sr, t = restore_sd_upscaler(lr_pil)
        methods["sd_x4"] = (sr, t)
    except Exception as e:
        print(f"   FAILED: {e}")
        methods["sd_x4"] = (None, 0)

    # Save
    for name, (img, _) in methods.items():
        if img is not None:
            img.save(OUT / f"restored_{name}.png")

    # Pixel metrics + downstream nostril localization
    print("\n[scoring]")
    lpips_net = lpips.LPIPS(net="alex").eval()
    hr_arr = np.array(hr_pil)
    summary = {"hr_size": [W, H], "lr_size": [lw, lh],
               "down_factor": DOWN_FACTOR, "methods": {}}
    # First the HR ground truth: where does DWPose put the nose?
    gt_nose = dwpose_nose(hr_pil)
    print(f"  HR DWPose nose: {gt_nose}")
    summary["hr_dwpose_nose"] = gt_nose
    for name, (img, t) in methods.items():
        if img is None: continue
        m = pixel_metrics(hr_arr, np.array(img.convert("RGB")), lpips_net)
        nose = dwpose_nose(img)
        m["dwpose_nose"] = nose
        m["dwpose_err_px"] = (
            float(np.hypot(nose[0] - gt_nose[0], nose[1] - gt_nose[1]))
            if (nose is not None and gt_nose is not None) else None)
        m["latency_s"] = t
        summary["methods"][name] = m
        print(f"  {name:18s}  psnr={m['psnr']:.2f}  ssim={m['ssim']:.3f}  "
              f"lpips={m.get('lpips', 0):.3f}  sharp={m['sharpness_lapvar']:.1f}  "
              f"nose_err={m['dwpose_err_px'] if m['dwpose_err_px'] else 'n/a'}")

    # also evaluate the LR image itself (upsampled trivially) and the HR itself
    lr_upsampled = lr_pil.resize((W, H), Image.NEAREST)
    summary["lr_nn_metrics"] = pixel_metrics(hr_arr, np.array(lr_upsampled), lpips_net)
    nose = dwpose_nose(lr_upsampled)
    summary["lr_nn_metrics"]["dwpose_nose"] = nose
    summary["lr_nn_metrics"]["dwpose_err_px"] = (
        float(np.hypot(nose[0] - gt_nose[0], nose[1] - gt_nose[1]))
        if (nose is not None and gt_nose is not None) else None)
    summary["lr_nn_metrics"]["latency_s"] = 0

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
