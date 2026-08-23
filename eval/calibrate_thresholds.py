"""
eval/calibrate_thresholds.py
─────────────────────────────
Fit the score calibration, report the FMR/FNMR curve, and answer the question
the hand-tuned threshold cannot: what error rate does 0.56 actually buy?

Held-out IDENTITIES, never held-out frames.

Usage:
    python eval/calibrate_thresholds.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import GroupKFold

from recognition.score_calibration import (fit, fmr_at, fnmr_at, probability,
                                           resolution_limit, threshold_for_fmr)

# Thresholds currently hand-set in config/identity_config.py.
CURRENT = {
    "face_match_threshold": 0.56,
    "template_blend_threshold": 0.55,
    "gallery_add_threshold": 0.60,
    "face_veto_threshold": 0.45,
    "evidence_face_sim_threshold": 0.45,
    "id_switch_similarity_threshold": 0.35,
    "duplicate_suggestion_threshold": 0.50,
}
FMR_TARGETS = [0.01, 0.001, 0.0001]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/data/face_quality.npz")
    ap.add_argument("--out", default="models/score_calibration.pkl")
    ap.add_argument("--report", default="eval/results/score_calibration.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    gen_all, imp_all, groups = d["genuine"], d["impostor"], d["groups"]
    ids = sorted(set(groups.tolist()))
    print(f"rows {len(gen_all)} | identities {len(ids)}")
    print("scores are 1:N — max cosine over a 24-identity gallery, which is "
          "what\nfind_person_by_embedding actually computes.\n")

    print("SCORE DISTRIBUTIONS")
    for name, v in (("genuine", gen_all), ("impostor", imp_all)):
        print(f"  {name:8s} n={len(v):5d} min={v.min():.3f} p01={np.quantile(v,.01):.3f} "
              f"median={np.median(v):.3f} p99={np.quantile(v,.99):.3f} max={v.max():.3f}")
    gap = gen_all.min() - imp_all.max()
    print(f"  separation: max impostor {imp_all.max():.3f} vs min genuine "
          f"{gen_all.min():.3f}  -> gap {gap:+.3f}")
    if gap > 0:
        print("  the distributions do NOT overlap on this corpus")

    # ── held-out-identity evaluation ────────────────────────────────────────
    print("\nHELD-OUT-IDENTITY FOLDS (fit on train ids, evaluate on test ids)")
    gkf = GroupKFold(n_splits=5)
    oof_fmr56, oof_fnmr56 = [], []
    for k, (tr, te) in enumerate(gkf.split(gen_all, gen_all, groups)):
        assert not (set(groups[tr]) & set(groups[te])), "identity leaked"
        cal = fit(gen_all[tr], imp_all[tr])
        f = fmr_at(0.56, imp_all[te]); n = fnmr_at(0.56, gen_all[te])
        oof_fmr56.append(f); oof_fnmr56.append(n)
        print(f"  fold {k}: {len(set(groups[te])):2d} test ids | "
              f"FMR@0.56 {100*f:6.3f}%  FNMR@0.56 {100*n:6.2f}%")
    print(f"  mean : FMR {100*np.mean(oof_fmr56):.3f}%  "
          f"FNMR {100*np.mean(oof_fnmr56):.2f}%")

    # ── full-corpus calibration ─────────────────────────────────────────────
    cal = fit(gen_all, imp_all)
    lim = resolution_limit(cal["n_impostor"])
    print(f"\nFMR resolution limit with {cal['n_impostor']} impostor scores: "
          f"{100*lim:.4f}%  (1/n)")

    print("\nOPERATING POINT OF EACH CURRENT THRESHOLD")
    print(f"  {'parameter':32s} {'cosine':>7} {'FMR':>10} {'FNMR':>8} {'P(genuine)':>11}")
    rows = {}
    for name, t in sorted(CURRENT.items(), key=lambda kv: -kv[1]):
        f, n = fmr_at(t, imp_all), fnmr_at(t, gen_all)
        p = probability(t, cal)
        fs = "0 (unmeasured)" if f == 0 else f"{100*f:.3f}%"
        print(f"  {name:32s} {t:7.2f} {fs:>10} {100*n:7.2f}% {p:11.3f}")
        rows[name] = {"cosine": t, "fmr": f, "fnmr": n, "p_genuine": p}

    print("\nTHRESHOLDS FOR TARGET FMR")
    print(f"  {'target FMR':>11} {'cosine':>8} {'FNMR':>8} {'coverage cost vs 0.56':>24}")
    base_fnmr = fnmr_at(0.56, gen_all)
    targets = {}
    for tgt in FMR_TARGETS:
        t = threshold_for_fmr(tgt, imp_all)
        if t is None:
            print(f"  {100*tgt:10.4g}% {'n/a':>8} {'n/a':>8} "
                  f"{'below resolution (1/n = %.4g%%)' % (100*lim):>24}")
            targets[str(tgt)] = None
            continue
        n = fnmr_at(t, gen_all)
        delta = n - base_fnmr
        print(f"  {100*tgt:10.4g}% {t:8.3f} {100*n:7.2f}% "
              f"{100*delta:+23.2f} pp")
        targets[str(tgt)] = {"cosine": t, "fnmr": n, "coverage_delta_pp": 100*delta}

    # ── was 0.56 near-optimal or lucky? ─────────────────────────────────────
    print("\nWAS 0.56 NEAR-OPTIMAL?")
    zero_fmr_min = float(imp_all.max())
    print(f"  lowest threshold with zero observed false matches : {zero_fmr_min:.3f}")
    print(f"  0.56 sits {0.56 - zero_fmr_min:+.3f} above it")
    recoverable = int(((gen_all >= zero_fmr_min) & (gen_all < 0.56)).sum())
    print(f"  genuine comparisons rejected by 0.56 but accepted at "
          f"{zero_fmr_min:.3f}: {recoverable} ({100*recoverable/len(gen_all):.2f}%)")
    print(f"  FNMR at 0.56          : {100*fnmr_at(0.56, gen_all):.2f}%")
    print(f"  FNMR at {zero_fmr_min:.3f}         : {100*fnmr_at(zero_fmr_min, gen_all):.2f}%")

    import joblib
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, args.out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "n_genuine": cal["n_genuine"], "n_impostor": cal["n_impostor"],
        "n_identities": len(ids),
        "separation_gap": float(gap),
        "max_impostor": float(imp_all.max()), "min_genuine": float(gen_all.min()),
        "fmr_resolution_limit": lim,
        "current_thresholds": rows,
        "fmr_targets": targets,
        "held_out": {"fmr_at_0.56": float(np.mean(oof_fmr56)),
                     "fnmr_at_0.56": float(np.mean(oof_fnmr56))},
        "zero_fmr_min_threshold": zero_fmr_min,
        "recoverable_genuine_at_zero_fmr": recoverable,
    }, indent=2))
    print(f"\ncalibration -> {args.out}")
    print(f"report      -> {args.report}")


if __name__ == "__main__":
    main()
