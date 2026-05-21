"""Generate 6 qualitative predictions from the ThermEval finetune model."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2, numpy as np, torch
from train_head_thermeval import (NostrilSet, TinyHead, build_backbone,
                                   decode_xy, DEVICE)

OUT = Path.home() / "git/nostril-bench/runs/finetune_thermeval"
test = NostrilSet("test")
backbone = build_backbone()
head = TinyHead(n_kpts=1).to(DEVICE)
head.load_state_dict(torch.load(OUT / "head_best.pt"))
head.eval()
rows = []
for i in range(6):
    img_t, hm, gt = test[i]
    with torch.no_grad():
        feats = backbone.backbone(img_t.unsqueeze(0).to(DEVICE))[0]
        pred_hm = head(feats)
        pred_xy = decode_xy(pred_hm)[0, 0].cpu().numpy()
    item = test.items[i]
    raw = cv2.imread(str(test.dir / "images" / item["image"]), cv2.IMREAD_GRAYSCALE)
    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    out = raw.copy()
    gt_np = gt.numpy()
    cv2.drawMarker(out, (int(gt_np[0]), int(gt_np[1])), (0, 0, 255),
                   cv2.MARKER_CROSS, 16, 2)
    cv2.circle(out, (int(pred_xy[0]), int(pred_xy[1])), 5, (0, 255, 0), -1)
    err = float(np.linalg.norm(pred_xy - gt_np))
    cv2.rectangle(out, (0, 0), (256, 22), (0, 0, 0), -1)
    cv2.putText(out, f"err={err:.1f}px  src=img{item.get('source_iid')}",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    rows.append(out)
grid = np.vstack([np.hstack(rows[:3]), np.hstack(rows[3:])])
cv2.imwrite(str(OUT / "qualitative_test.png"), grid)
print("wrote", OUT / "qualitative_test.png", grid.shape)
