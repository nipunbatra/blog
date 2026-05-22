"""Evaluate zero-shot vs finetuned DRLN on the multi-dataset thermal SR test
set. Reports per-dataset PSNR / SSIM / LPIPS + downstream DWPose nostril
localisation error.
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import lpips

from super_image import DrlnModel

ROOT = Path.home() / "data/thermal-sr"
SAVE = Path.home() / "thermal-sr-work" / "ft"
OUT = Path.home() / "thermal-sr-work" / "test"
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
    return lr_t, hr_t, hr, lr


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


def eval_one(model, items, lpips_net):
    res = {"sftl54": [], "thermeval": [], "all": []}
    for it in items:
        lr_t, hr_t, hr_bgr, lr_bgr = load_pair(it)
        with torch.no_grad():
            sr_t = model(lr_t.unsqueeze(0).to(DEVICE)).clamp(0, 1)
        sr_bgr = cv2.cvtColor((sr_t[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
        p = psnr(sr_t[0].cpu(), hr_t)
        sgray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY)
        hgray = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2GRAY)
        s = float(ssim(sgray, hgray, data_range=255))
        d = float(lpips_net((sr_t.to(DEVICE) * 2 - 1),
                            (hr_t.unsqueeze(0).to(DEVICE) * 2 - 1)).item())
        # downstream nostril
        gt_nose = dwpose_nose(hr_bgr)
        pr_nose = dwpose_nose(sr_bgr)
        if gt_nose is not None and pr_nose is not None:
            nose_err = float(np.hypot(pr_nose[0] - gt_nose[0],
                                       pr_nose[1] - gt_nose[1]))
        else:
            nose_err = None
        rec = {"psnr": p, "ssim": s, "lpips": d, "nose_err": nose_err,
               "dataset": it["dataset"]}
        res["all"].append(rec)
        res[it["dataset"]].append(rec)
    return res


def aggregate(res):
    out = {}
    for k, items in res.items():
        if not items: continue
        out[k] = {
            "n": len(items),
            "psnr_mean": float(np.mean([r["psnr"] for r in items])),
            "ssim_mean": float(np.mean([r["ssim"] for r in items])),
            "lpips_mean": float(np.mean([r["lpips"] for r in items])),
            "nose_err_mean": float(np.mean([r["nose_err"] for r in items
                                              if r["nose_err"] is not None])),
            "nose_err_median": float(np.median([r["nose_err"] for r in items
                                                  if r["nose_err"] is not None])),
        }
    return out


def main(n_test=120, ckpt=None):
    manifest = json.load(open(ROOT / "manifest.json"))
    test_items = manifest["test"]
    # Subsample for speed, keeping per-dataset balance
    sftl = [r for r in test_items if r["dataset"] == "sftl54"][:n_test // 2]
    te = [r for r in test_items if r["dataset"] == "thermeval"][:n_test // 2]
    items = sftl + te
    print(f"[init] {len(items)} test items ({len(sftl)} SFTL54, {len(te)} ThermEval)")

    lpips_net = lpips.LPIPS(net="alex").to(DEVICE).eval()

    print("[a] zero-shot DRLN")
    model = DrlnModel.from_pretrained("eugenesiow/drln-bam", scale=4).to(DEVICE).eval()
    zs = eval_one(model, items, lpips_net)

    print("[b] finetuned DRLN")
    model.load_state_dict(torch.load(ckpt or SAVE / "drln_best.pt"))
    model.eval()
    ft = eval_one(model, items, lpips_net)

    summary = {"zero_shot": aggregate(zs), "finetuned": aggregate(ft)}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Method':<14} {'Split':<10} {'N':>4} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} "
          f"{'NoseErr(med)':>12} {'NoseErr(mean)':>13}")
    for method, agg in [("zero-shot", summary["zero_shot"]),
                          ("finetuned", summary["finetuned"])]:
        for split in ["all", "sftl54", "thermeval"]:
            if split not in agg: continue
            s = agg[split]
            print(f"{method:<14} {split:<10} {s['n']:>4} {s['psnr_mean']:>7.2f} "
                  f"{s['ssim_mean']:>7.3f} {s['lpips_mean']:>7.3f} "
                  f"{s['nose_err_median']:>12.2f} {s['nose_err_mean']:>13.2f}")
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    main(args.n, args.ckpt)
