"""Finetune DRLN (super-image package) on multi-dataset thermal SR.

Starts from RGB-pretrained eugenesiow/drln-bam (x4).
LR=1e-5, AdamW, cosine schedule, 80 epochs, patch-based training:
  - For each HR 192x192 crop, randomly sample a 128x128 sub-crop.
  - Generate LR = downsample(HR, factor=4) -> 32x32.
  - Loss: L1 between SR and HR.
Best val PSNR checkpoint saved.

Reports per-dataset eval on val every 5 epochs (PSNR / SSIM / LPIPS).
"""
import argparse, json, random, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips

from super_image import DrlnModel, ImageLoader

ROOT = Path.home() / "data/thermal-sr"
SAVE = Path.home() / "thermal-sr-work" / "ft"
SAVE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SRSet(Dataset):
    def __init__(self, manifest, root=ROOT, patch_hr=128, scale=4, train=True):
        self.items = manifest
        self.root = root
        self.patch_hr = patch_hr
        self.scale = scale
        self.train = train

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        hr = cv2.imread(str(self.root / r["image"]))
        H, W = hr.shape[:2]
        if self.train:
            x = random.randint(0, max(0, W - self.patch_hr))
            y = random.randint(0, max(0, H - self.patch_hr))
            hr = hr[y:y + self.patch_hr, x:x + self.patch_hr]
        # match patch_hr exactly
        if hr.shape[:2] != (self.patch_hr, self.patch_hr):
            hr = cv2.resize(hr, (self.patch_hr, self.patch_hr))
        lr = cv2.resize(hr, (self.patch_hr // self.scale,
                              self.patch_hr // self.scale),
                         interpolation=cv2.INTER_AREA)
        # BGR -> RGB, [0,1] float
        hr_t = torch.from_numpy(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t, r["dataset"]


def psnr(a, b):
    mse = float(F.mse_loss(a, b).item())
    return 10 * np.log10(1.0 / max(mse, 1e-9))


def evaluate(model, dl, lpips_net=None):
    model.eval()
    scores = {"all": [], "sftl54": [], "thermeval": []}
    lps = {"all": [], "sftl54": [], "thermeval": []}
    sss = {"all": [], "sftl54": [], "thermeval": []}
    with torch.no_grad():
        for lr, hr, ds in dl:
            lr = lr.to(DEVICE); hr = hr.to(DEVICE)
            sr = model(lr).clamp(0, 1)
            for j in range(lr.size(0)):
                p = psnr(sr[j].cpu(), hr[j].cpu())
                scores["all"].append(p); scores[ds[j]].append(p)
                a = sr[j].cpu().permute(1, 2, 0).numpy()
                b = hr[j].cpu().permute(1, 2, 0).numpy()
                s = float(ssim(cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                                data_range=255))
                sss["all"].append(s); sss[ds[j]].append(s)
                if lpips_net is not None:
                    d = float(lpips_net((sr[j:j+1] * 2 - 1), (hr[j:j+1] * 2 - 1)).item())
                    lps["all"].append(d); lps[ds[j]].append(d)
    out = {}
    for k in ["all", "sftl54", "thermeval"]:
        if scores[k]:
            out[k] = {
                "psnr_mean": float(np.mean(scores[k])),
                "psnr_std": float(np.std(scores[k])),
                "ssim_mean": float(np.mean(sss[k])),
                "lpips_mean": float(np.mean(lps[k])) if lps[k] else None,
                "n": len(scores[k]),
            }
    return out


def main(epochs=60, lr=1e-5, batch_size=16):
    manifest = json.load(open(ROOT / "manifest.json"))
    train_ds = SRSet(manifest["train"], train=True)
    val_ds = SRSet(manifest["val"], train=False, patch_hr=192)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=2)

    model = DrlnModel.from_pretrained("eugenesiow/drln-bam", scale=4).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    lpips_net = lpips.LPIPS(net="alex").to(DEVICE).eval()
    print(f"[init] DRLN {sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(train_ds)} train / {len(val_ds)} val, device={DEVICE}")

    # zero-shot baseline eval
    print("[eval 0] zero-shot...")
    base = evaluate(model, val_dl, lpips_net)
    for k, v in base.items():
        print(f"   {k}: psnr={v['psnr_mean']:.2f} ssim={v['ssim_mean']:.3f}")
    history = [{"epoch": -1, "metrics": base, "lr": lr}]

    best_psnr = base.get("all", {}).get("psnr_mean", 0)
    torch.save(model.state_dict(), SAVE / "drln_zero_shot.pt")

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        t0 = time.perf_counter()
        for lr_b, hr_b, _ in train_dl:
            lr_b = lr_b.to(DEVICE); hr_b = hr_b.to(DEVICE)
            sr = model(lr_b)
            loss = F.l1_loss(sr, hr_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * lr_b.size(0); n += lr_b.size(0)
        train_loss = tot / max(n, 1)
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            metrics = evaluate(model, val_dl, lpips_net)
            print(f"ep {ep:3d}  loss={train_loss:.4f}  "
                  f"psnr={metrics['all']['psnr_mean']:.2f}  "
                  f"ssim={metrics['all']['ssim_mean']:.3f}  "
                  f"lpips={metrics['all'].get('lpips_mean',0):.3f}  "
                  f"sftl={metrics.get('sftl54',{}).get('psnr_mean',0):.2f}  "
                  f"te={metrics.get('thermeval',{}).get('psnr_mean',0):.2f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            history.append({"epoch": ep, "metrics": metrics,
                             "lr": opt.param_groups[0]["lr"]})
            if metrics["all"]["psnr_mean"] > best_psnr:
                best_psnr = metrics["all"]["psnr_mean"]
                torch.save(model.state_dict(), SAVE / "drln_best.pt")

    with open(SAVE / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nbest val PSNR: {best_psnr:.2f}")
    print(f"saved best to {SAVE / 'drln_best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    main(args.epochs, args.lr, args.batch_size)
