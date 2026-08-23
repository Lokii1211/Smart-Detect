"""
Figures for the SmartDetect paper, computed from the real ablation output in
eval/results/_full_corpus/. No value is typed in by hand: every number is read
from metrics.json / assignments.json.

Palette follows the validated reference instance (dataviz skill):
  categorical slots 1-3  #2a78d6 / #eb6834 / #1baf7a   (all-pairs validated)
  status                 good #0ca30c  critical #d03b3b
  sequential blue ramp   #cde2fb -> #0d366b
Print target, light surface.
"""
from __future__ import annotations
import json, collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path("/Users/lokii/Downloads/Smart-Detect-main")
RES = ROOT / "eval" / "results" / "_full_corpus"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CFGS = ["A_pre_hardening", "B_face_anchor", "C_plus_guard", "D_full"]
SHORT = {"A_pre_hardening": "A\npre-hardening",
         "B_face_anchor": "B\nface anchor",
         "C_plus_guard": "C\n+ ID-switch guard",
         "D_full": "D\nfull pipeline"}

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRIT = "#0ca30c", "#d03b3b"
INK, INK2, MUTED = "#16181d", "#3c414c", "#8b91a0"
SURF = "#ffffff"
GRID = "#e6e8ee"
SEQ = LinearSegmentedColormap.from_list(
    "seqblue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

M = {c: json.loads((RES / c / "metrics.json").read_text()) for c in CFGS}
A = {c: json.loads((RES / c / "assignments.json").read_text()) for c in CFGS}


def tidy(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


# ── Fig 3: bucket partition ───────────────────────────────────────────────
def fig_buckets():
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    labels = [SHORT[c] for c in CFGS]
    ys = np.arange(len(CFGS))[::-1]
    tot = M[CFGS[0]]["buckets"]["n_total"]
    for i, c in enumerate(CFGS):
        b = M[c]["buckets"]
        vals = [b["n_correct"], b["n_contaminated"], b["n_unassigned"]]
        cols = [GOOD, CRIT, "#c8ccd6"]
        left = 0
        for v, col in zip(vals, cols):
            if v <= 0:
                left += v
                continue
            ax.barh(ys[i], v, left=left, height=0.6, color=col,
                    edgecolor=SURF, linewidth=1.6)
            pct = 100 * v / tot
            if pct > 6:
                ax.text(left + v / 2, ys[i], f"{pct:.1f}%", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            left += v
        if b["n_contaminated"] <= 5 and b["n_contaminated"] > 0:
            ax.annotate(f"{b['n_contaminated']} frame",
                        xy=(b["n_correct"], ys[i]), xytext=(b["n_correct"] + 90, ys[i] - 0.42),
                        fontsize=7, color=CRIT,
                        arrowprops=dict(arrowstyle="-", color=CRIT, lw=0.7))
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, tot)
    ax.set_xlabel(f"ground-truth person-frames (n = {tot})")
    ax.xaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)
    ax.spines["left"].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=k) for k in [GOOD, CRIT, "#c8ccd6"]]
    ax.legend(h, ["CORRECT", "CONTAMINATED", "UNASSIGNED"], ncol=3, frameon=False,
              fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(OUT / "fig_buckets.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 4: precision / coverage / purity trajectory ───────────────────────
def fig_trajectory():
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(CFGS))
    prec = [100 * M[c]["identity_precision"]["value"] for c in CFGS]
    cov = [100 * M[c]["identity_coverage"]["value"] for c in CFGS]
    pur = [100 * M[c]["identity_purity"]["value"] for c in CFGS]
    plo = [100 * M[c]["identity_precision"]["ci"]["lo"] for c in CFGS]
    phi = [100 * M[c]["identity_precision"]["ci"]["hi"] for c in CFGS]

    ax.fill_between(x, plo, phi, color=S1, alpha=0.14, lw=0)
    ax.plot(x, prec, color=S1, lw=2, marker="o", ms=6, zorder=3,
            markeredgecolor=SURF, markeredgewidth=1.4)
    ax.plot(x, cov, color=S2, lw=2, marker="s", ms=5.5, zorder=3,
            markeredgecolor=SURF, markeredgewidth=1.4)
    ax.plot(x, pur, color=S3, lw=2, marker="^", ms=6, zorder=3,
            markeredgecolor=SURF, markeredgewidth=1.4)

    for xi, v in zip(x, prec):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.5, color=INK, fontweight="bold")
    ax.annotate("identity precision", (x[-1], prec[-1]), textcoords="offset points",
                xytext=(-6, -16), ha="right", fontsize=8, color=S1, fontweight="bold")
    ax.annotate("coverage", (x[0], cov[0]), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=S2, fontweight="bold")
    ax.annotate("identity purity", (x[1], pur[1]), textcoords="offset points",
                xytext=(8, -14), fontsize=8, color=S3, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CFGS], fontsize=8)
    ax.set_ylim(-6, 108); ax.set_ylabel("percent")
    ax.set_yticks([0, 25, 50, 75, 100])
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_trajectory.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 2: identity collapse matrix, A vs D ───────────────────────────────
def fig_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    for ax, cfg, ttl in zip(axes, ["A_pre_hardening", "D_full"],
                            ["(a)  A — pre-hardening", "(b)  D — full pipeline"]):
        recs = A[cfg]
        codes, people = [], []
        cnt = collections.Counter()
        for r in recs:
            if r["code"] == "Detecting...":
                continue
            cnt[(r["code"], r["gt_person"])] += 1
        codes = sorted({c for c, _ in cnt})
        people = sorted({p for _, p in cnt})
        Mx = np.zeros((len(codes), len(people)))
        for (c, p), v in cnt.items():
            Mx[codes.index(c), people.index(p)] = v
        # Row-normalise: each cell is the share of THAT identifier's frames
        # belonging to that person. A pure identifier is one full-value cell
        # in its row; a collapsed one smears across the row.
        rowsum = Mx.sum(axis=1, keepdims=True)
        Mn = np.divide(Mx, np.where(rowsum == 0, 1, rowsum))
        ax.imshow(Mn, cmap=SEQ, aspect="auto", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(range(len(people)))
        ax.set_xticklabels(people, rotation=90, fontsize=5.6)
        # With ~60 identifiers the labels collide into an unreadable stack;
        # thin them so every tick still lands on a real row.
        step = 1 if len(codes) <= 20 else (2 if len(codes) <= 40 else 3)
        pos = list(range(0, len(codes), step))
        ax.set_yticks(pos)
        ax.set_yticklabels([codes[i].replace("SDT-", "") for i in pos], fontsize=5.4)
        ax.set_yticks(range(len(codes)), minor=True)
        ax.tick_params(axis="y", which="minor", length=1, color=GRID)
        ax.set_xlabel("ground-truth person", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("identifier minted (SDT-)", fontsize=8)
        ax.set_title(ttl, fontsize=9, loc="left", pad=6)
        ax.tick_params(length=2, width=0.6)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
        npure = sum(1 for i in range(len(codes)) if (Mx[i] > 0).sum() == 1)
        ax.text(0.99, -0.30, f"{len(codes)} identifiers · {npure} pure "
                             f"({100*npure/max(len(codes),1):.0f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=INK, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label("share of that identifier's person-frames (%)", fontsize=7)
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_edgecolor(GRID)
    fig.savefig(OUT / "fig_collapse.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 5: evidence precision vs yield ────────────────────────────────────
def fig_evidence():
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    x = np.arange(len(CFGS))
    w = 0.38
    ep = [100 * M[c]["evidence_precision"]["value"] for c in CFGS]
    ey = [100 * M[c]["evidence_precision"]["evidence_yield"] for c in CFGS]
    b1 = ax.bar(x - w/2, ep, w, color=S1, edgecolor=SURF, linewidth=1.4, zorder=3)
    b2 = ax.bar(x + w/2, ey, w, color="#c8ccd6", edgecolor=SURF, linewidth=1.4, zorder=3)
    for xi, v in zip(x, ep):
        ax.text(xi - w/2, v + 2, f"{v:.1f}", ha="center", fontsize=7.2,
                color=INK, fontweight="bold")
    for xi, v in zip(x, ey):
        ax.text(xi + w/2, v + 2, f"{v:.1f}", ha="center", fontsize=7.2, color=INK2)
    nw = [M[c]["evidence_precision"]["n_wrong"] for c in CFGS]
    for xi, v, n in zip(x, ep, nw):
        if n == 0:
            ax.text(xi - w/2, v/2, "0 wrong", ha="center", va="center", rotation=90,
                    fontsize=6.8, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CFGS], fontsize=7.5)
    ax.set_ylim(0, 118); ax.set_ylabel("percent"); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend([b1, b2], ["evidence precision", "evidence yield"], frameon=False,
              fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_evidence.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 6: duplicates vs contamination (the split/merge trade) ────────────
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    contam = [M[c]["buckets"]["n_contaminated"] for c in CFGS]
    dup = [M[c]["duplicate_identities"]["value"] for c in CFGS]
    ncode = [M[c]["identity_purity"]["n_codes"] for c in CFGS]
    ax.plot(contam, dup, color=MUTED, lw=1, ls="--", zorder=1)
    # C and D coincide exactly on both axes (the evidence gate changes neither
    # contamination nor duplicates) — draw once, label both.
    drawn = {}
    for c, ct, dp, nc in zip(CFGS, contam, dup, ncode):
        key = (round(ct, 6), round(dp, 6))
        drawn.setdefault(key, []).append((c.split("_")[0], nc))
    for (ct, dp), members in drawn.items():
        names = ", ".join(m[0] for m in members)
        nc = members[-1][1]
        col = CRIT if names.startswith("A") else (S1 if "D" in names else INK2)
        ax.scatter(ct, dp, s=42 + nc * 3.0, color=col, zorder=3,
                   edgecolor=SURF, linewidth=1.4)
        off = (10, -14) if "D" in names else (10, 5)
        ax.annotate(f"{names}   ({nc} IDs)", (ct, dp), textcoords="offset points",
                    xytext=off, fontsize=7.5, color=INK, fontweight="bold")
    ax.set_xlabel("contaminated person-frames  (merging error)")
    ax.set_ylabel("duplicates per person\n(splitting error)")
    xmax = max(contam) if contam else 1
    ymin, ymax = min(dup), max(dup)
    ax.set_xlim(-0.06 * xmax, 1.18 * xmax)
    pad = max(0.25, 0.45 * (ymax - ymin))
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.annotate("preferred: splitting errors are\nvisible and repairable",
                xy=(0.14 * xmax, ymax + 0.80 * pad), fontsize=7.2, color=S1,
                fontweight="bold", ha="left", va="top")
    tidy(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 7: latency ────────────────────────────────────────────────────────
def fig_latency():
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    x = np.arange(len(CFGS))
    mean = [M[c]["runtime"]["mean_frame_latency_ms"] for c in CFGS]
    med = [M[c]["runtime"]["median_frame_latency_ms"] for c in CFGS]
    ax.bar(x, mean, 0.55, color="#c8ccd6", edgecolor=SURF, linewidth=1.4,
           zorder=3, label="mean")
    ax.scatter(x, med, s=34, color=S1, zorder=4, marker="D",
               edgecolor=SURF, linewidth=1.2, label="median")
    for xi, v in zip(x, mean):
        ax.text(xi, v + 1.8, f"{v:.1f}", ha="center", fontsize=7.4, color=INK,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c.split("_")[0] for c in CFGS], fontsize=8)
    ax.set_ylabel("ms / frame"); ax.set_ylim(0, 1.30 * max(mean))
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.png", bbox_inches="tight")
    plt.close(fig)


for fn in (fig_buckets, fig_trajectory, fig_collapse, fig_evidence,
           fig_tradeoff, fig_latency):
    fn(); print("ok", fn.__name__)
print("figures in", OUT)
