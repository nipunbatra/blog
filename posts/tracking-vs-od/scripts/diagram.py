"""Conceptual schematic: per-frame detection vs tracking.

Renders outputs/how_tracking_works.svg (+ .png). Six consecutive frames; the
object is missed in frame 4. Top row = independent detection (no links, a miss
is a hole). Bottom row = tracking (detect -> predict -> associate; the motion
model coasts through the gap and identity persists).

  python diagram.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import common as C

GRAY_F, GRAY_E = "#f4f4f4", "#cfcfcf"
RED, BLUE, ORANGE, INK = "#c44536", "#2c6fbb", "#e08a2e", "#222222"
MUTE = "#8a8a8a"

N = 6
CARD_W, CARD_H = 1.42, 1.46
X0, STEP = 1.95, 1.70
CARDX = [X0 + i * STEP for i in range(N)]
ROW_A_Y, ROW_B_Y = 4.25, 1.15

MISS = 3                                   # frame index where detection fails
# ball position inside a card, as (x-fraction, y-fraction); bounce arc
LX = [0.26, 0.40, 0.54, 0.64, 0.74, 0.82]
LY = [0.30, 0.60, 0.82, 0.82, 0.58, 0.32]  # LY[MISS] = predicted apex


def center(i, row_y):
    return (CARDX[i] + LX[i] * CARD_W, row_y + LY[i] * CARD_H)


def draw_card(ax, i, row_y, label=None):
    ax.add_patch(FancyBboxPatch((CARDX[i], row_y), CARD_W, CARD_H,
                 boxstyle="round,pad=0.0,rounding_size=0.08",
                 fc=GRAY_F, ec=GRAY_E, lw=1.2, zorder=1))
    if label:
        ax.text(CARDX[i] + CARD_W / 2, row_y - 0.22, label, ha="center", va="top",
                fontsize=10, color=MUTE)


def draw_ball(ax, cx, cy, color, tag=None, box=True, dashed=False, faded=False):
    a = 0.45 if faded else 1.0
    if box:
        ax.add_patch(Rectangle((cx - 0.17, cy - 0.17), 0.34, 0.34, fill=False,
                     ec=color, lw=2.0, ls="--" if dashed else "-", alpha=a, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.085, fc=color, ec="white", lw=0.8, alpha=a, zorder=5))
    if tag:
        ax.text(cx + 0.22, cy + 0.20, tag, fontsize=9.5, color=color, weight="bold", zorder=6)


def arrow(ax, p, q, color, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                 color=color, lw=2.0, ls="--" if dashed else "-",
                 shrinkA=10, shrinkB=10, zorder=3))


fig, ax = plt.subplots(figsize=(12, 6.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

ax.text(0.1, 6.72, "Detection runs each frame from scratch. Tracking links frames into identities.",
        fontsize=15, weight="bold", color=INK)

# ---------------- Row A: detection ----------------
ax.text(0.1, ROW_A_Y + CARD_H / 2 + 0.15, "OBJECT\nDETECTION", fontsize=11.5, weight="bold",
        color=INK, va="center")
ax.text(0.1, ROW_A_Y + CARD_H / 2 - 0.42, "each frame\nindependent", fontsize=9, color=MUTE, va="center")
for i in range(N):
    draw_card(ax, i, ROW_A_Y, label=f"frame {i + 1}")
    if i == MISS:
        cx, cy = center(i, ROW_A_Y)
        ax.text(cx, cy, "?", ha="center", va="center", fontsize=20, color="#c0392b", weight="bold")
        ax.text(CARDX[i] + CARD_W / 2, ROW_A_Y + 0.16, "no detection", ha="center",
                fontsize=8, color="#c0392b")
    else:
        cx, cy = center(i, ROW_A_Y)
        draw_ball(ax, cx, cy, ORANGE, tag="?")
ax.text(11.95, ROW_A_Y + CARD_H + 0.02, "no arrows  -  no identity, no memory; a miss is just a hole",
        ha="right", fontsize=9.5, color=MUTE, style="italic")

# ---------------- Row B: tracking ----------------
ax.text(0.1, ROW_B_Y + CARD_H / 2 + 0.15, "TRACKING", fontsize=11.5, weight="bold", color=INK, va="center")
ax.text(0.1, ROW_B_Y + CARD_H / 2 - 0.42, "detect -> predict\n-> associate", fontsize=9, color=MUTE, va="center")
cB = [center(i, ROW_B_Y) for i in range(N)]
for i in range(N):
    draw_card(ax, i, ROW_B_Y, label=f"frame {i + 1}")
for i in range(N):
    cx, cy = cB[i]
    if i == MISS:
        draw_ball(ax, cx, cy, RED, tag="predicted", box=True, dashed=True, faded=True)
        ax.text(CARDX[i] + CARD_W / 2, ROW_B_Y + 0.16, "motion model coasts",
                ha="center", fontsize=8, color=RED)
    else:
        draw_ball(ax, cx, cy, RED, tag="#1", box=True)
# association arrows across cards (dashed through the predicted/gap segment)
for i in range(N - 1):
    arrow(ax, cB[i], cB[i + 1], BLUE, dashed=(i == MISS - 1 or i == MISS))
ax.text(11.95, ROW_B_Y - 0.55, "arrows = association; same #1 re-acquired after the gap",
        ha="right", fontsize=9.5, color=MUTE, style="italic")

# small "match (IoU)" callout at re-acquire
mx, my = cB[MISS + 1]
ax.annotate("re-match\n(IoU / appearance)", xy=(mx - 0.12, my + 0.12), xytext=(mx - 0.1, my + 1.05),
            fontsize=8.5, color=BLUE, ha="center",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))

# ---------------- 3-step strip ----------------
steps = [("1  Detect", "boxes in the current frame", ORANGE),
         ("2  Predict", "motion model -> where next", RED),
         ("3  Associate", "match new boxes to tracks", BLUE)]
sx = 1.95
for title, sub, col in steps:
    ax.add_patch(Rectangle((sx, 0.12), 0.12, 0.42, fc=col, ec="none"))
    ax.text(sx + 0.22, 0.46, title, fontsize=10, weight="bold", color=INK, va="center")
    ax.text(sx + 0.22, 0.22, sub, fontsize=8.5, color=MUTE, va="center")
    sx += 3.25

fig.tight_layout()
fig.savefig(C.OUT / "how_tracking_works.svg", bbox_inches="tight")
fig.savefig(C.OUT / "how_tracking_works.png", dpi=150, bbox_inches="tight")
print("wrote how_tracking_works.svg / .png")
