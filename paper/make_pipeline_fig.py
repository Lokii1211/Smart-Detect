"""Render the per-frame decision-path diagram as a print figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/private/tmp/claude-501/-Users-lokii-Downloads-Smart-Detect-main/"
           "4e0d6171-becd-4766-a354-d53068b6a2ba/scratchpad/figs")

INK, INK2, FAINT = "#16181d", "#3c414c", "#6a7080"
ACC, ACCBG = "#1f3a68", "#e8edf6"
PANEL, RULE = "#f2f3f7", "#d3d6de"
SURF = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 300,
})

STAGES = [
    ("1", "Frame acquisition", "capture thread; drop-oldest queue of depth 1 —\n"
     "display rate decoupled from analysis cost", False),
    ("2", "Person detection — YOLOv8", "resolution-aware input sizing:\n"
     "416 px webcam · 640 px HD · 960 px 4K", False),
    ("3", "Multi-object tracking — ByteTrack", "persistent track ids; arbitration runs once per\n"
     "TRACK, not per frame — what makes CPU-only viable", False),
    ("4", "Full-frame face analysis — ArcFace", "one detector pass per cycle; faces bound to person\n"
     "boxes by centroid in the upper 60% of the box", False),
    ("5", "Face quality gate", "< 48 px height or < 0.60 det. score: drawn, but NOT\n"
     "admitted as identity evidence", True),
    ("6", "ID-switch guard", "cached identifier contradicted by a gated face on 2\n"
     "consecutive cycles: cache dropped, re-identify", True),
    ("7", "Identity arbitration", "face match then colour (face-vetoed), then body re-ID\n"
     "(face-vetoed), then pose-gated enrolment", True),
    ("8", "Evidence gate", "sighting row + photo written ONLY on same-cycle face\n"
     "confirmation; inherited labels display but do not record", True),
    ("9", "Persistence and audit", "sighting, evidence crop, watchlist check, retention\n"
     "refresh, append-only biometric-access audit entry", False),
]

fig, ax = plt.subplots(figsize=(7.0, 6.15))
ax.set_xlim(0, 10); ax.set_ylim(0, len(STAGES) * 1.13 + 0.5)
ax.axis("off")

H = 0.92
for i, (num, name, desc, anchor) in enumerate(STAGES):
    y = (len(STAGES) - 1 - i) * 1.13 + 0.35
    face = ACCBG if anchor else PANEL
    edge = ACC if anchor else RULE
    ax.add_patch(FancyBboxPatch((0.62, y), 8.9, H,
                                boxstyle="round,pad=0.008,rounding_size=0.03",
                                facecolor=face, edgecolor="none", zorder=2))
    # left accent rule
    ax.add_patch(plt.Rectangle((0.62, y), 0.055, H, color=edge, zorder=3))
    ax.text(0.30, y + H / 2, num, ha="center", va="center", fontsize=10,
            color=ACC if anchor else FAINT, family="monospace", fontweight="bold")
    ax.text(0.86, y + H - 0.245, name, ha="left", va="center", fontsize=8.6,
            color=INK, fontweight="bold")
    ax.text(0.86, y + 0.30, desc, ha="left", va="center", fontsize=7.0,
            color=INK2, linespacing=1.42)
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch((5.07, y - 0.02), (5.07, y - 0.19),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=FAINT, lw=0.9, zorder=4))

# legend
ax.add_patch(plt.Rectangle((0.62, 0.02), 0.055, 0.2, color=ACC))
ax.text(0.86, 0.12, "stages where identity may be created or invalidated —"
                    " all others may only propose or annotate",
        ha="left", va="center", fontsize=7.0, color=INK2)

fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight")
print("wrote", OUT / "fig_pipeline.png")
