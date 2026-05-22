"""Compare all three DRLN states on the same test set: zero-shot, v1 finetune
(L1 only, 60 ep), v2 finetune (L1 + 0.05*LPIPS, EMA, 200 ep). Plus per-
dataset breakdown + downstream DWPose nostril error."""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import lpips

from super_image import DrlnModel

ROOT = Path.home() / "data/thermal-sr"
WORK = Path.home() / "thermal-sr-work"
OUT = WORK / "test_v2"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pair(item, scale=4, hr_size=192):
    hr = cv2.imread(str(ROOT / item["image"]))
    if hr.shape[:2] != (hr_size, hr_size):
        hr = cv2.resize(hr, (hr_size, hr_size))
    lr = cv2.resize(hr, (hr_size // scale, hr_size // scale),
                     interpolation=cv2.INTER_AREA)
    hr_t = torch.from_numpy(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    lr_t = torch.from_numpy(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    return lr_t, hr_t, hr


def dwpose_nose(img_bgr):
    from rtmlib import Wholebody
    if not hasattr(dwpose_nose, "wb"):
        dwpose_nose.wb = Wholebody(to_openpose=False, mode="balanced",
                                    backend="onnxruntime", device="cuda")
    keypoints, scores = dwpose_nose.wb(img_bgr)
    if keypoints is None or len(keypoints) == 0: return None
    return float(keypoints[0][0][0]), float(keypoints[0][0][1])


def psnr(a, b):
    mse = float(F.mse_loss(a, b).item())
    return 10 * np.log10(1.0 / max(mse, 1e-9))


def eval_model(model, items, lpips_net):
    res = {"sftl54": [], "thermeval": [], "all": []}
    for it in items:
        lr_t, hr_t, hr_bgr = load_pair(it)
        with torch.no_grad():
            sr_t = model(lr_t.unsqueeze(0).to(DEVICE)).clamp(0, 1)
        sr_bgr = cv2.cvtColor((sr_t[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
        p = psnr(sr_t[0].cpu(), hr_t)
        s = float(ssim(cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2GRAY), data_range=255))
        d = float(lpips_net((sr_t.to(DEVICE) * 2 - 1),
                            (hr_t.unsqueeze(0).to(DEVICE) * 2 - 1)).item())
        gt = dwpose_nose(hr_bgr); pr = dwpose_nose(sr_bgr)
        ne = float(np.hypot(pr[0] - gt[0], pr[1] - gt[1])) if (gt and pr) else None
        rec = {"psnr": p, "ssim": s, "lpips": d, "nose_err": ne, "dataset": it["dataset"]}
        res["all"].append(rec); res[it["dataset"]].append(rec)
    return res


def agg(res):
    out = {}
    for k, items in res.items():
        if not items: continue
        nes = [r["nose_err"] for r in items if r["nose_err"] is not None]
        out[k] = {
            "n": len(items),
            "psnr_mean": float(np.mean([r["psnr"] for r in items])),
            "ssim_mean": float(np.mean([r["ssim"] for r in items])),
            "lpips_mean": float(np.mean([r["lpips"] for r in items])),
            "nose_err_mean": float(np.mean(nes)) if nes else None,
            "nose_err_median": float(np.median(nes)) if nes else None,
        }
    return out


def main(n_test=120):
    manifest = json.load(open(ROOT / "manifest.json"))
    test_items = manifest["test"]
    sftl = [r for r in test_items if r["dataset"] == "sftl54"][:n_test // 2]
    te = [r for r in test_items if r["dataset"] == "thermeval"][:n_test // 2]
    items = sftl + te
    print(f"[init] {len(items)} test items")

    lpips_net = lpips.LPIPS(net="alex").to(DEVICE).eval()

    print("[a] zero-shot")
    m = DrlnModel.from_pretrained("eugenesiow/drln-bam", scale=4).to(DEVICE).eval()
    zs = eval_model(m, items, lpips_net)

    print("[b] v1 finetune (L1 only)")
    m.load_state_dict(torch.load(WORK / "ft" / "drln_best.pt"))
    m.eval()
    v1 = eval_model(m, items, lpips_net)

    print("[c] v2 finetune (L1 + LPIPS + EMA)")
    m.load_state_dict(torch.load(WORK / "ft_v2" / "drln_best.pt"))
    m.eval()
    v2 = eval_model(m, items, lpips_net)

    summary = {"zero_shot": agg(zs), "ft_v1": agg(v1), "ft_v2": agg(v2)}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Method':<14} {'Split':<10} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} "
          f"{'NoseMed':>9} {'NoseMean':>9}")
    for k, name in [("zero_shot", "zero-shot"), ("ft_v1", "v1 L1-only"),
                     ("ft_v2", "v2 L1+LPIPS")]:
        for split in ["all", "sftl54", "thermeval"]:
            if split not in summary[k]: continue
            s = summary[k][split]
            print(f"{name:<14} {split:<10} {s['psnr_mean']:>7.2f} "
                  f"{s['ssim_mean']:>7.3f} {s['lpips_mean']:>7.3f} "
                  f"{s['nose_err_median']:>9.2f} {s['nose_err_mean']:>9.2f}")
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    main(args.n)
