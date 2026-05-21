"""Finetune a tiny 1-keypoint heatmap head on frozen Sapiens2-0.4b backbone,
using ThermEval-D crops (one Nose centroid per person crop).

Architecture and loss are the same as `train_head.py`, but with 1-channel
heatmap output (just the single Nose centroid GT per crop).
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.modules["mmpretrain"] = None
from sapiens.pose.models import init_model

ROOT = Path.home() / "data/nostril-thermeval"
SAPIENS_REPO = Path.home() / "git/sapiens2"
SAPIENS_CKPT = (Path.home() / "models/sapiens2_pose_0.4b"
                / "sapiens2_0.4b_pose.safetensors")
SAPIENS_CFG = (SAPIENS_REPO / "sapiens/pose/configs/keypoints308/"
               "shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_"
               "shutterstock_goliath_3po-1024x768.py")
IMG_SIZE = 256; HEATMAP_SIZE = 64; SIGMA = 2.0; DEVICE = "cuda:0"


class NostrilSet(Dataset):
    def __init__(self, split):
        self.dir = ROOT / split
        with open(self.dir / "labels.jsonl") as f:
            self.items = [json.loads(l) for l in f]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        img = cv2.imread(str(self.dir / "images" / r["image"]),
                         cv2.IMREAD_GRAYSCALE)
        img = cv2.merge([img, img, img]).astype(np.float32) / 255.0
        mean = np.array([123.675, 116.28, 103.53]) / 255.0
        std = np.array([58.395, 57.12, 57.375]) / 255.0
        img = (img - mean) / std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # heatmap (1 channel)
        cx = r["nostril"][0] * HEATMAP_SIZE / IMG_SIZE
        cy = r["nostril"][1] * HEATMAP_SIZE / IMG_SIZE
        ys, xs = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE]
        hm = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * SIGMA ** 2))
        hm = hm[None].astype(np.float32)  # (1, H, W)
        return img, torch.from_numpy(hm), torch.tensor(r["nostril"], dtype=torch.float32)


class TinyHead(nn.Module):
    def __init__(self, n_kpts=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, n_kpts, 1),
        )

    def forward(self, feats): return self.net(feats)


def build_backbone():
    m = init_model(str(SAPIENS_CFG), str(SAPIENS_CKPT), device=DEVICE)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def decode_xy(heatmaps):
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    idx = flat.argmax(dim=-1)
    y = (idx // W).float() * IMG_SIZE / H
    x = (idx % W).float() * IMG_SIZE / W
    return torch.stack([x, y], dim=-1)


def pck(preds, gts, radius=10):
    err = (preds.squeeze(1) - gts).norm(dim=-1)
    return (err <= radius).float().mean().item()


def main(epochs=120, lr=3e-3, batch_size=8, save_dir=None):
    train_ds = NostrilSet("train"); val_ds = NostrilSet("val")
    test_ds = NostrilSet("test")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

    backbone = build_backbone()
    head = TinyHead(n_kpts=1).to(DEVICE)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[head] {n_params:,} trainable params")
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    best_val_err = float("inf")
    save_dir = Path(save_dir) if save_dir else (Path.home() / "git/nostril-bench/runs/finetune_thermeval")
    save_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        head.train(); tot, tot_n = 0.0, 0
        for img, hm, _ in train_dl:
            img, hm = img.to(DEVICE), hm.to(DEVICE)
            with torch.no_grad():
                feats = backbone.backbone(img)[0]
            pred = head(feats)
            loss = F.mse_loss(pred, hm)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * img.size(0); tot_n += img.size(0)
        train_loss = tot / tot_n
        head.eval()
        with torch.no_grad():
            val_preds, val_gts = [], []
            for img, hm, gt in val_dl:
                img = img.to(DEVICE)
                feats = backbone.backbone(img)[0]
                pred = head(feats)
                val_preds.append(decode_xy(pred).cpu())
                val_gts.append(gt)
            val_preds = torch.cat(val_preds); val_gts = torch.cat(val_gts)
            err = (val_preds.squeeze(1) - val_gts).norm(dim=-1).mean().item()
            p10 = pck(val_preds, val_gts, 10)
        sched.step()
        history.append({"epoch": ep, "train_loss": train_loss,
                        "val_mean_err": err, "val_pck10": p10})
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  loss={train_loss:.4f}  val_err={err:.1f}px  pck@10={p10:.2f}")
        if err < best_val_err:
            best_val_err = err
            torch.save(head.state_dict(), save_dir / "head_best.pt")

    head.load_state_dict(torch.load(save_dir / "head_best.pt"))
    head.eval()
    with torch.no_grad():
        tp, tg = [], []
        for img, hm, gt in test_dl:
            feats = backbone.backbone(img.to(DEVICE))[0]
            tp.append(decode_xy(head(feats)).cpu()); tg.append(gt)
        tp = torch.cat(tp); tg = torch.cat(tg)
        err = (tp.squeeze(1) - tg).norm(dim=-1)
        test_mean = err.mean().item(); test_med = err.median().item()
        for r in [3, 5, 10, 20]:
            print(f"test PCK@{r}px = {pck(tp, tg, r):.2f}")
        print(f"test mean: {test_mean:.1f} px / median: {test_med:.1f} px")
    summary = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "head_params": n_params,
        "best_val_err": best_val_err,
        "test_mean_err": test_mean, "test_median_err": test_med,
        "test_pck5": pck(tp, tg, 5), "test_pck10": pck(tp, tg, 10),
        "test_pck20": pck(tp, tg, 20),
        "history": history,
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", save_dir / "summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    main(args.epochs, args.lr, args.batch_size, args.save_dir)
