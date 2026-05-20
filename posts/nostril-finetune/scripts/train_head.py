"""Finetune a tiny 2-keypoint heatmap head on a frozen Sapiens2-0.4b backbone.

Architecture:
    Sapiens2-0.4b backbone (frozen, ~400M params, RGB-pretrained on humans)
        -> ViT features (1, 1024, 16, 16) for a 256x256 input
    Head (trainable, ~50k params)
        -> Conv 1024->256, BN, GELU
        -> Conv 256->64, BN, GELU
        -> 4x upsample (bilinear) to 64x64
        -> Conv 64->2 (left_nostril_heatmap, right_nostril_heatmap)

Loss: MSE between predicted heatmap and a 2D Gaussian centred at the GT
nostril, sigma=2 px in the 64x64 heatmap (= 8 px in input space).

We feed a 3-channel grayscale (replicated) thermal image into Sapiens2 — the
backbone has never seen thermal, but that's exactly the point we're testing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.modules["mmpretrain"] = None
from sapiens.pose.models import init_model

ROOT = Path.home() / "data/nostril-few-shot"
SAPIENS_REPO = Path.home() / "git/sapiens2"
SAPIENS_CKPT = (Path.home() / "models/sapiens2_pose_0.4b"
                / "sapiens2_0.4b_pose.safetensors")
SAPIENS_CFG = (SAPIENS_REPO / "sapiens/pose/configs/keypoints308/"
               "shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_"
               "shutterstock_goliath_3po-1024x768.py")

IMG_SIZE = 256
HEATMAP_SIZE = 64           # 4x downsample relative to input
SIGMA = 2.0                  # gaussian std in heatmap pixels
DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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
        # replicate to 3 channels
        img = cv2.merge([img, img, img]).astype(np.float32) / 255.0
        # Sapiens2 normalisation: subtract ImageNet mean / std then ×255 (the
        # data_preprocessor in Sapiens2 expects pixels in 0..255 with
        # ImageNet mean/std applied). We bypass it for this lightweight loop.
        mean = np.array([123.675, 116.28, 103.53]) / 255.0
        std = np.array([58.395, 57.12, 57.375]) / 255.0
        img = (img - mean) / std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # heatmap
        hm = np.zeros((2, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        for k, kp in enumerate([r["nostril_left"], r["nostril_right"]]):
            cx = kp[0] * HEATMAP_SIZE / IMG_SIZE
            cy = kp[1] * HEATMAP_SIZE / IMG_SIZE
            ys, xs = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE]
            hm[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2)
                           / (2 * SIGMA ** 2))
        return img, torch.from_numpy(hm), torch.tensor(
            [r["nostril_left"], r["nostril_right"]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, feats): return self.net(feats)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def build_backbone():
    print("[init] loading Sapiens2-0.4b backbone (frozen)…")
    m = init_model(str(SAPIENS_CFG), str(SAPIENS_CKPT), device=DEVICE)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def decode_heatmaps_to_xy(heatmaps):
    """heatmaps: (B, 2, H, W). Returns (B, 2, 2) xy in input-image coords."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    idx = flat.argmax(dim=-1)
    y = (idx // W).float() * IMG_SIZE / H
    x = (idx % W).float() * IMG_SIZE / W
    return torch.stack([x, y], dim=-1)  # (B, K, 2)


def pck(preds, gts, radius=10):
    # preds, gts: (N, 2, 2) in input-pixel coords
    err = (preds - gts).norm(dim=-1)  # (N, 2)
    return (err <= radius).float().mean().item()


def main(epochs=120, lr=3e-3, batch_size=8, save_dir=None):
    train_ds = NostrilSet("train"); val_ds = NostrilSet("val")
    test_ds = NostrilSet("test")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

    backbone = build_backbone()
    head = TinyHead().to(DEVICE)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[head] {n_params:,} trainable params")

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    best_val_err = float("inf")
    save_dir = Path(save_dir) if save_dir else (
        Path.home() / "git/nostril-bench/runs/finetune")
    save_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        # train
        head.train(); tot, tot_n = 0.0, 0
        for img, hm, _ in train_dl:
            img, hm = img.to(DEVICE), hm.to(DEVICE)
            with torch.no_grad():
                feats = backbone.backbone(img)[0]   # (B, 1024, 16, 16)
            pred = head(feats)                       # (B, 2, 64, 64)
            loss = F.mse_loss(pred, hm)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * img.size(0); tot_n += img.size(0)
        train_loss = tot / tot_n
        # val + pck
        head.eval()
        with torch.no_grad():
            val_preds, val_gts = [], []
            for img, hm, gt in val_dl:
                img = img.to(DEVICE)
                feats = backbone.backbone(img)[0]
                pred = head(feats)
                val_preds.append(decode_heatmaps_to_xy(pred).cpu())
                val_gts.append(gt)
            val_preds = torch.cat(val_preds); val_gts = torch.cat(val_gts)
            err = (val_preds - val_gts).norm(dim=-1).mean().item()
            pck10 = pck(val_preds, val_gts, 10)
            pck5 = pck(val_preds, val_gts, 5)
        sched.step()
        history.append({"epoch": ep, "train_loss": train_loss,
                        "val_mean_err": err,
                        "val_pck5": pck5, "val_pck10": pck10})
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  train_loss={train_loss:.4f}  "
                  f"val_err={err:.1f}px  pck@5={pck5:.2f} pck@10={pck10:.2f}")
        if err < best_val_err:
            best_val_err = err
            torch.save(head.state_dict(), save_dir / "head_best.pt")
    # test
    head.load_state_dict(torch.load(save_dir / "head_best.pt"))
    head.eval()
    with torch.no_grad():
        test_preds, test_gts = [], []
        for img, hm, gt in test_dl:
            img = img.to(DEVICE)
            feats = backbone.backbone(img)[0]
            pred = head(feats)
            test_preds.append(decode_heatmaps_to_xy(pred).cpu())
            test_gts.append(gt)
        test_preds = torch.cat(test_preds); test_gts = torch.cat(test_gts)
        test_err = (test_preds - test_gts).norm(dim=-1).mean().item()
        test_err_median = (test_preds - test_gts).norm(dim=-1).median().item()
        for r in [3, 5, 10, 20]:
            print(f"test PCK@{r}px = {pck(test_preds, test_gts, r):.2f}")
        print(f"test mean error = {test_err:.1f} px")
        print(f"test median error = {test_err_median:.1f} px")

    summary = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "head_params": n_params,
        "best_val_err": best_val_err,
        "test_mean_err": test_err,
        "test_median_err": test_err_median,
        "test_pck5": pck(test_preds, test_gts, 5),
        "test_pck10": pck(test_preds, test_gts, 10),
        "test_pck20": pck(test_preds, test_gts, 20),
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
