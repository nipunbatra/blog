"""Make a 4-row panel (4 subjects) x 3-column (Sapiens2, DWPose, MediaPipe) on
thermal-gray for the bake-off post. Reads cached grid PNGs and crops out each
panel."""
import cv2
import numpy as np
from pathlib import Path

OUT = Path("/Users/nipun/git/blog/posts/nostril-bench/outputs")
SUBSET = OUT / "subset_0.4b_v5"

SUBJECTS = [
    "10_1_2_1_1020_36_1",
    "100_1_2_1_1134_36_1",
    "11_1_2_1_1174_100_1",
    "12_1_2_1_1249_26_1",
]


def grid_panels(img):
    """Take the 2x2 grid PNG and return 4 cropped panels in order:
    Sapiens, ViTPose, DWPose, MediaPipe."""
    h, w = img.shape[:2]
    return [img[0:h//2, 0:w//2], img[0:h//2, w//2:],
            img[h//2:, 0:w//2], img[h//2:, w//2:]]


rows = []
for s in SUBJECTS:
    p = SUBSET / "gray" / f"{s}_grid.png"
    img = cv2.imread(str(p))
    if img is None:
        print("missing", p); continue
    panels = grid_panels(img)
    # we want sapiens, dwpose, mediapipe (indices 0, 2, 3)
    row = np.hstack([panels[0], panels[2], panels[3]])
    rows.append(row)

panel = np.vstack(rows)
cv2.imwrite(str(OUT / "multi_subject_panel.png"), panel)
print(f"wrote {OUT / 'multi_subject_panel.png'}  {panel.shape}")
