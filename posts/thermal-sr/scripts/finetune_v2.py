"""Stronger thermal SR finetune.

Improvements over v1:
  - LPIPS perceptual loss term (alpha=0.05) — preserves edges, fixes
    downstream-task regression.
  - 192x192 HR patches (was 128) — larger receptive field.
  - EMA model copy — smoother test performance.
  - 200 epochs (was 60) — let it actually converge.
  - Warmup (5 ep) + cosine schedule, peak LR 2e-5 — gentler ramp.
  - Random horizontal flip augmentation.
  - Multi-backbone: DRLN, HAN, EDSR — test which transfers best.
"""
import argparse, copy, json, random, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
import lpips

from super_image import DrlnModel, EdsrModel, HanModel, MsrnModel, A2nModel

ROOT = Path.home() / "data/thermal-sr"
SAVE = Path.home() / "thermal-sr-work" / "ft_v2"
SAVE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SRSet(Dataset):
    def __init__(self, manifest, root=ROOT, patch_hr=192, scale=4, train=True):
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
        if self.train and H >= self.patch_hr and W >= self.patch_hr:
            x = random.randint(0, W - self.patch_hr)
            y = random.randint(0, H - self.patch_hr)
            hr = hr[y:y + self.patch_hr, x:x + self.patch_hr]
            # random horizontal flip
            if random.random() < 0.5:
                hr = hr[:, ::-1].copy()
            # random rotation in {0, 90, 180, 270}
            k = random.randint(0, 3)
            if k > 0:
                hr = np.rot90(hr, k).copy()
        else:
            if hr.shape[:2] != (self.patch_hr, self.patch_hr):
                hr = cv2.resize(hr, (self.patch_hr, self.patch_hr))
        lr = cv2.resize(hr, (self.patch_hr // self.scale,
                              self.patch_hr // self.scale),
                         interpolation=cv2.INTER_AREA)
        hr_t = torch.from_numpy(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        lr_t = torch.from_numpy(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t, r["dataset"]


def psnr(a, b):
    mse = float(F.mse_loss(a, b).item())
    return 10 * np.log10(1.0 / max(mse, 1e-9))


def evaluate(model, dl, lpips_net):
    model.eval()
    out = {"all": {"psnr": [], "ssim": [], "lpips": []},
           "sftl54": {"psnr": [], "ssim": [], "lpips": []},
           "thermeval": {"psnr": [], "ssim": [], "lpips": []}}
    with torch.no_grad():
        for lr, hr, ds in dl:
            lr = lr.to(DEVICE); hr = hr.to(DEVICE)
            sr = model(lr).clamp(0, 1)
            for j in range(lr.size(0)):
                p = psnr(sr[j].cpu(), hr[j].cpu())
                a = sr[j].cpu().permute(1, 2, 0).numpy()
                b = hr[j].cpu().permute(1, 2, 0).numpy()
                s = float(ssim(cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                                data_range=255))
                d = float(lpips_net((sr[j:j+1] * 2 - 1), (hr[j:j+1] * 2 - 1)).item())
                for k in ("all", ds[j]):
                    out[k]["psnr"].append(p); out[k]["ssim"].append(s); out[k]["lpips"].append(d)
    return {k: {"psnr_mean": float(np.mean(v["psnr"])) if v["psnr"] else 0,
                "ssim_mean": float(np.mean(v["ssim"])) if v["ssim"] else 0,
                "lpips_mean": float(np.mean(v["lpips"])) if v["lpips"] else 0,
                "n": len(v["psnr"])} for k, v in out.items()}


def get_backbone(name):
    if name == "drln": return DrlnModel.from_pretrained("eugenesiow/drln-bam", scale=4)
    if name == "han": return HanModel.from_pretrained("eugenesiow/han", scale=4)
    if name == "edsr": return EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=4)
    if name == "msrn": return MsrnModel.from_pretrained("eugenesiow/msrn", scale=4)
    if name == "a2n":  return A2nModel.from_pretrained("eugenesiow/a2n", scale=4)
    raise ValueError(name)


@torch.no_grad()
def ema_update(ema_model, model, decay=0.999):
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


def main(epochs, lr, batch_size, backbone, alpha_lpips, warmup_eps):
    manifest = json.load(open(ROOT / "manifest.json"))
    train_ds = SRSet(manifest["train"], train=True, patch_hr=192)
    val_ds = SRSet(manifest["val"], train=False, patch_hr=192)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=2)

    model = get_backbone(backbone).to(DEVICE)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def lr_lambda(ep):
        if ep < warmup_eps: return (ep + 1) / warmup_eps
        progress = (ep - warmup_eps) / max(1, epochs - warmup_eps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    lpips_net = lpips.LPIPS(net="alex").to(DEVICE).eval()
    for p in lpips_net.parameters(): p.requires_grad_(False)

    print(f"[init] {backbone} {sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(train_ds)} train / {len(val_ds)} val, alpha_lpips={alpha_lpips}, device={DEVICE}")

    # zero-shot baseline
    print("[eval 0] zero-shot baseline...")
    base = evaluate(model, val_dl, lpips_net)
    for k, v in base.items():
        print(f"   {k}: psnr={v['psnr_mean']:.2f} ssim={v['ssim_mean']:.3f} lpips={v['lpips_mean']:.3f}")
    history = [{"epoch": -1, "metrics": base, "lr": 0}]
    best_psnr = base["all"]["psnr_mean"]
    torch.save(model.state_dict(), SAVE / f"{backbone}_zs.pt")

    for ep in range(epochs):
        model.train()
        tot_l1, tot_lpips, n = 0.0, 0.0, 0
        t0 = time.perf_counter()
        for lr_b, hr_b, _ in train_dl:
            lr_b = lr_b.to(DEVICE); hr_b = hr_b.to(DEVICE)
            sr = model(lr_b)
            sr_c = sr.clamp(0, 1)
            l1 = F.l1_loss(sr, hr_b)
            lp = lpips_net((sr_c * 2 - 1), (hr_b * 2 - 1)).mean()
            loss = l1 + alpha_lpips * lp
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(ema_model, model, decay=0.999)
            tot_l1 += l1.item() * lr_b.size(0)
            tot_lpips += lp.item() * lr_b.size(0)
            n += lr_b.size(0)
        sched.step()
        train_l1 = tot_l1 / max(n, 1); train_lp = tot_lpips / max(n, 1)
        if ep % 5 == 0 or ep == epochs - 1:
            m_ema = evaluate(ema_model, val_dl, lpips_net)
            elapsed = time.perf_counter() - t0
            print(f"ep {ep:3d}  l1={train_l1:.4f}  lpips={train_lp:.4f}  "
                  f"EMA: psnr={m_ema['all']['psnr_mean']:.2f}  "
                  f"ssim={m_ema['all']['ssim_mean']:.3f}  "
                  f"lpips={m_ema['all']['lpips_mean']:.3f}  "
                  f"sftl={m_ema['sftl54']['psnr_mean']:.2f}  "
                  f"te={m_ema['thermeval']['psnr_mean']:.2f}  "
                  f"({elapsed:.0f}s, lr={opt.param_groups[0]['lr']:.1e})")
            history.append({"epoch": ep, "metrics": m_ema,
                             "lr": opt.param_groups[0]["lr"],
                             "train_l1": train_l1, "train_lpips": train_lp})
            if m_ema["all"]["psnr_mean"] > best_psnr:
                best_psnr = m_ema["all"]["psnr_mean"]
                torch.save(ema_model.state_dict(), SAVE / f"{backbone}_best.pt")

    with open(SAVE / f"{backbone}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nbest val PSNR (EMA): {best_psnr:.2f}")
    print(f"saved best to {SAVE / f'{backbone}_best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--backbone", default="drln",
                    choices=["drln", "han", "edsr", "msrn", "a2n"])
    ap.add_argument("--alpha-lpips", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    main(args.epochs, args.lr, args.batch_size, args.backbone,
         args.alpha_lpips, args.warmup)
