"""Generate per-image SR comparisons across datasets and build a panel.

For ~3 SF-TL54 and ~3 ThermEval-D test crops, produce:
  HR target | LR input (NN-upscaled for display) | zero-shot | v1 | v2

Save individual panels + a stacked grid for the blog post.
"""
import json
from pathlib import Path
import cv2, numpy as np, torch
from super_image import DrlnModel

ROOT = Path.home() / "data/thermal-sr"
WORK = Path.home() / "thermal-sr-work"
OUT = WORK / "panels"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HR_SIZE = 192
SCALE = 4


def load_pair(item):
    hr = cv2.imread(str(ROOT / item["image"]))
    if hr.shape[:2] != (HR_SIZE, HR_SIZE):
        hr = cv2.resize(hr, (HR_SIZE, HR_SIZE))
    lr = cv2.resize(hr, (HR_SIZE // SCALE, HR_SIZE // SCALE),
                     interpolation=cv2.INTER_AREA)
    return hr, lr


def restore(model, lr_bgr):
    lr_t = torch.from_numpy(cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        sr_t = model(lr_t.unsqueeze(0).to(DEVICE)).clamp(0, 1)
    return cv2.cvtColor((sr_t[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                         cv2.COLOR_RGB2BGR)


def label(img, txt, color=(255, 255, 255)):
    img = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)
    return img


def main():
    manifest = json.load(open(ROOT / "manifest.json"))
    test = manifest["test"]
    sftl = [r for r in test if r["dataset"] == "sftl54"]
    te = [r for r in test if r["dataset"] == "thermeval"]
    # pick diverse-looking items
    picks = sftl[:3] + te[:3]

    print("[init] loading 3 model states")
    m = DrlnModel.from_pretrained("eugenesiow/drln-bam", scale=4).to(DEVICE)
    zs_state = {k: v.clone() for k, v in m.state_dict().items()}
    v1_state = torch.load(WORK / "ft" / "drln_best.pt", weights_only=True)
    v2_state = torch.load(WORK / "ft_v2" / "drln_best.pt", weights_only=True)

    rows = []
    for item in picks:
        hr, lr = load_pair(item)
        lr_disp = cv2.resize(lr, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_NEAREST)
        m.load_state_dict(zs_state); m.eval()
        zs_sr = restore(m, lr)
        m.load_state_dict(v1_state); m.eval()
        v1_sr = restore(m, lr)
        m.load_state_dict(v2_state); m.eval()
        v2_sr = restore(m, lr)

        # scale each up 2x for visibility in the panel
        UP = 2
        hr_d = cv2.resize(hr, (HR_SIZE * UP, HR_SIZE * UP), interpolation=cv2.INTER_CUBIC)
        lr_d = cv2.resize(lr_disp, (HR_SIZE * UP, HR_SIZE * UP), interpolation=cv2.INTER_NEAREST)
        zs_d = cv2.resize(zs_sr, (HR_SIZE * UP, HR_SIZE * UP), interpolation=cv2.INTER_CUBIC)
        v1_d = cv2.resize(v1_sr, (HR_SIZE * UP, HR_SIZE * UP), interpolation=cv2.INTER_CUBIC)
        v2_d = cv2.resize(v2_sr, (HR_SIZE * UP, HR_SIZE * UP), interpolation=cv2.INTER_CUBIC)

        # Save individually first
        ds = item["dataset"]; iid = item["id"]
        cv2.imwrite(str(OUT / f"{ds}_{iid}_hr.png"), hr)
        cv2.imwrite(str(OUT / f"{ds}_{iid}_lr.png"), lr)
        cv2.imwrite(str(OUT / f"{ds}_{iid}_zs.png"), zs_sr)
        cv2.imwrite(str(OUT / f"{ds}_{iid}_v1.png"), v1_sr)
        cv2.imwrite(str(OUT / f"{ds}_{iid}_v2.png"), v2_sr)

        # Build a row: dataset/id | HR | LR | zs | v1 | v2
        ds_label = "SF-TL54" if ds == "sftl54" else "ThermEval-D"
        labeled = np.hstack([
            label(hr_d, f"{ds_label}: HR target"),
            label(lr_d, "LR input (4x down)"),
            label(zs_d, "zero-shot DRLN"),
            label(v1_d, "v1 L1-only"),
            label(v2_d, "v2 L1+LPIPS+EMA"),
        ])
        rows.append(labeled)

    grid = np.vstack(rows)
    cv2.imwrite(str(OUT / "panel_all.png"), grid)
    print(f"wrote {OUT / 'panel_all.png'}  shape={grid.shape}")

    # Also build per-dataset panels
    sftl_rows = rows[:3]; te_rows = rows[3:]
    cv2.imwrite(str(OUT / "panel_sftl54.png"), np.vstack(sftl_rows))
    cv2.imwrite(str(OUT / "panel_thermeval.png"), np.vstack(te_rows))
    print("wrote per-dataset panels")


if __name__ == "__main__":
    main()
