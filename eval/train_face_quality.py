"""
eval/train_face_quality.py
───────────────────────────
Train and validate a face-quality predictor against the fixed 48 px / 0.60 gate.

VALIDATION IS BY IDENTITY, NEVER BY FRAME.
Consecutive frames of one person are near-duplicates; splitting on frames puts
the same face in train and test and inflates every number. GroupKFold on the
identity is the only honest split here.

Usage:
    python eval/train_face_quality.py --data eval/data/face_quality.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The shipped gate.
FIXED_MIN_H = 48.0
FIXED_MIN_DET = 0.60


def load(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"], d["y"], d["groups"], [str(s) for s in d["feature_names"]])


def make_model():
    """Tree model — used for the analysis, because permutation importance on
    trees is the interpretable artefact this task asks for."""
    return HistGradientBoostingRegressor(
        max_depth=4, max_iter=200, learning_rate=0.07,
        min_samples_leaf=25, l2_regularization=1.0, random_state=0,
    )


def make_mlp():
    """
    Deployable model. The tree model costs ~22 ms per SINGLE-ROW call — a fixed
    thread-pool overhead, not compute (256 rows also cost ~23 ms). The gate is
    inherently one-face-at-a-time, so that overhead cannot be amortised and the
    tree fails the < 1 ms budget outright. A 2-layer MLP exported to a raw
    numpy forward pass runs in ~0.006 ms.
    """
    return Pipeline([("sc", StandardScaler()),
                     ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=800,
                                          early_stopping=True, random_state=0))])


def cross_val_model(factory, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), "identity leaked across folds"
        oof[te] = factory().fit(X[tr], y[tr]).predict(X[te])
    return oof


def export_mlp(pipe, names, X=None, y=None):
    """
    Raw weights for a dependency-light forward pass, PLUS the trained input and
    output ranges.

    An MLP extrapolates without bound. Fed a feature far outside training range
    — a blur variance of 50 000 where training saw hundreds — it returns an
    arbitrary number, and a gate comparing that to a threshold would admit
    anything. The ranges let inference clip inputs and clamp the output to what
    the model has actually seen.
    """
    sc, mlp = pipe.named_steps["sc"], pipe.named_steps["mlp"]
    art = {
        "kind": "mlp",
        "feature_names": names,
        "mean": sc.mean_.astype(np.float32),
        "scale": sc.scale_.astype(np.float32),
        "W": [w.astype(np.float32) for w in mlp.coefs_],
        "b": [b.astype(np.float32) for b in mlp.intercepts_],
    }
    if X is not None:
        art["feat_lo"] = X.min(axis=0).astype(np.float32)
        art["feat_hi"] = X.max(axis=0).astype(np.float32)
    if y is not None:
        art["y_lo"] = float(y.min())
        art["y_hi"] = float(y.max())
    return art


def cross_val(X, y, groups, n_splits=5):
    """Out-of-fold predictions with identities disjoint across folds."""
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y)
    per_fold = []
    for tr, te in gkf.split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), "identity leaked across folds"
        m = make_model().fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        per_fold.append({
            "n_train_ids": len(set(groups[tr])), "n_test_ids": len(set(groups[te])),
            "n_test_rows": int(len(te)),
            "pearson": float(pearsonr(y[te], oof[te])[0]),
            "spearman": float(spearmanr(y[te], oof[te])[0]),
            "mae": float(np.mean(np.abs(y[te] - oof[te]))),
        })
    return oof, per_fold


def operating_curve(score, y, safe, n=200):
    """Sweep a threshold on `score`; report coverage and precision-among-admitted."""
    out = []
    for t in np.quantile(score, np.linspace(0, 1, n)):
        adm = score >= t
        if adm.sum() == 0:
            continue
        out.append({"threshold": float(t),
                    "coverage": float(adm.mean()),
                    "precision": float(safe[adm].mean()),
                    "n_admitted": int(adm.sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/data/face_quality.npz")
    ap.add_argument("--model-out", default="models/face_quality.pkl")
    ap.add_argument("--report-out", default="eval/results/face_quality_report.json")
    ap.add_argument("--safe-margin", type=float, default=0.0,
                    help="y above this counts as a reliable face")
    args = ap.parse_args()

    X, y, groups, names = load(args.data)
    ids = sorted(set(groups.tolist()))
    print(f"rows {len(y)} | identities {len(ids)} | features {len(names)}")
    print(f"margin y: min={y.min():.3f} p05={np.quantile(y,.05):.3f} "
          f"median={np.median(y):.3f} max={y.max():.3f}")

    safe = y > args.safe_margin
    print(f"\nfaces with y > {args.safe_margin}: {safe.sum()}/{len(y)} ({100*safe.mean():.1f}%)")
    if safe.all():
        print("  !! EVERY face is discriminative — there are no negatives in this corpus.")

    # ── the fixed gate's operating point ────────────────────────────────────
    h = X[:, names.index("face_h_px")]
    ds = X[:, names.index("det_score")]
    fixed = (h >= FIXED_MIN_H) & (ds >= FIXED_MIN_DET)
    print(f"\nFIXED GATE ({FIXED_MIN_H:.0f}px / {FIXED_MIN_DET})")
    print(f"  coverage  : {100*fixed.mean():.1f}%  ({fixed.sum()}/{len(y)})")
    print(f"  precision : {100*safe[fixed].mean():.1f}%" if fixed.any() else "  precision : n/a")
    if (~fixed).any():
        print(f"  REFUSED   : {int((~fixed).sum())} faces, of which "
              f"{int(safe[~fixed].sum())} were actually discriminative "
              f"({100*safe[~fixed].mean():.1f}%)")
        print(f"              their margin: min={y[~fixed].min():.3f} "
              f"median={np.median(y[~fixed]):.3f}")

    # ── learned predictor, identity-disjoint folds ──────────────────────────
    oof, folds = cross_val(X, y, groups)
    print("\nHELD-OUT-IDENTITY CROSS-VALIDATION")
    for i, f in enumerate(folds):
        print(f"  fold {i}: {f['n_test_ids']:2d} ids, {f['n_test_rows']:5d} rows | "
              f"pearson {f['pearson']:+.3f} spearman {f['spearman']:+.3f} mae {f['mae']:.3f}")
    pr = float(pearsonr(y, oof)[0]); sp = float(spearmanr(y, oof)[0])
    print(f"  POOLED tree : pearson {pr:+.3f}  spearman {sp:+.3f}  "
          f"mae {np.mean(np.abs(y-oof)):.3f}")
    oof_mlp = cross_val_model(make_mlp, X, y, groups)
    pr_m = float(pearsonr(y, oof_mlp)[0]); sp_m = float(spearmanr(y, oof_mlp)[0])
    print(f"  POOLED mlp  : pearson {pr_m:+.3f}  spearman {sp_m:+.3f}  "
          f"mae {np.mean(np.abs(y-oof_mlp)):.3f}")

    # ── operating curves ────────────────────────────────────────────────────
    learned = operating_curve(oof, y, safe)
    height_only = operating_curve(h, y, safe)

    fixed_prec = float(safe[fixed].mean()) if fixed.any() else 1.0
    match = [p for p in learned if p["precision"] >= fixed_prec]
    best = max(match, key=lambda p: p["coverage"]) if match else None
    print(f"\nLEARNED GATE AT THE FIXED GATE'S PRECISION ({100*fixed_prec:.1f}%)")
    if best:
        print(f"  coverage {100*best['coverage']:.1f}%  vs fixed {100*fixed.mean():.1f}%"
              f"   -> {100*(best['coverage']-fixed.mean()):+.1f} pp")
    else:
        print("  no learned threshold reaches the fixed gate's precision")

    # ── feature importance (permutation, on held-out identities) ────────────
    from sklearn.inspection import permutation_importance
    gkf = GroupKFold(n_splits=5)
    tr, te = next(iter(gkf.split(X, y, groups)))
    m = make_model().fit(X[tr], y[tr])
    imp = permutation_importance(m, X[te], y[te], n_repeats=15, random_state=0,
                                 scoring="r2")
    order = np.argsort(-imp.importances_mean)
    print("\nPERMUTATION IMPORTANCE (held-out identities, r2 drop)")
    for i in order:
        bar = "#" * max(0, int(60 * imp.importances_mean[i] / max(imp.importances_mean.max(), 1e-9)))
        print(f"  {names[i]:22s} {imp.importances_mean[i]:+.4f} ± {imp.importances_std[i]:.4f} {bar}")
    top2 = [names[i] for i in order[:2]]
    share = float(imp.importances_mean[order[:2]].sum() /
                  max(imp.importances_mean[imp.importances_mean > 0].sum(), 1e-9))
    print(f"  top-2 = {top2}  ({100*share:.0f}% of positive importance)")

    # ── persist + timing ────────────────────────────────────────────────────
    import joblib
    final = make_mlp().fit(X, y)
    art = export_mlp(final, names, X, y)
    art.update({"trained_rows": int(len(y)), "trained_identities": len(ids),
                "cv_pearson": pr_m, "cv_spearman": sp_m})
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, args.model_out)

    # Time the DEPLOYED form (raw numpy forward), not sklearn's predict().
    from recognition.face_quality import _forward
    v = X[0].astype(np.float32)
    _forward(art, v)
    t0 = time.perf_counter()
    for _ in range(20000):
        _forward(art, v)
    per = (time.perf_counter() - t0) / 20000 * 1000
    print(f"\nmodel -> {args.model_out}  ({Path(args.model_out).stat().st_size/1024:.0f} KB)")
    print(f"inference: {per:.3f} ms/call  [{'PASS' if per < 1.0 else 'FAIL'} < 1 ms]")

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps({
        "n_rows": int(len(y)), "n_identities": len(ids),
        "margin": {"min": float(y.min()), "median": float(np.median(y)),
                   "max": float(y.max()), "frac_safe": float(safe.mean())},
        "fixed_gate": {"coverage": float(fixed.mean()), "precision": fixed_prec,
                       "n_refused": int((~fixed).sum()),
                       "n_refused_but_discriminative": int(safe[~fixed].sum())},
        "cv": {"tree_pearson": pr, "tree_spearman": sp,
               "mlp_pearson": pr_m, "mlp_spearman": sp_m, "folds": folds},
        "learned_at_fixed_precision": best,
        "importance": {names[i]: float(imp.importances_mean[i]) for i in order},
        "inference_ms": per,
        "curve_learned": learned, "curve_height_only": height_only,
    }, indent=2))
    print(f"report -> {args.report_out}")


if __name__ == "__main__":
    main()
