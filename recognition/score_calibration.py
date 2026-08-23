"""
recognition/score_calibration.py
─────────────────────────────────
Calibrate ArcFace cosine similarity to P(genuine), and derive match thresholds
from target error rates instead of hand-tuning.

WHY
───
`face_match_threshold = 0.56` is a number that was moved until specific observed
merges stopped: 0.35 cross-matched strangers, 0.50 still merged two pairs, 0.56
did not. That is evidence, but it says nothing about the error rate the value
buys. A threshold should be answerable in the units an operator cares about:
"how often will this system attach the wrong person's identity?"

WHAT IS CALIBRATED
──────────────────
A monotone map cosine -> P(genuine), fitted with isotonic regression. Isotonic
rather than Platt because the true relationship need not be a sigmoid, and
imposing one biases exactly the tail that sets the threshold.

FMR IS 1:N HERE, NOT 1:1
────────────────────────
SmartDetect matches by scanning the whole gallery and taking the maximum
(`find_person_by_embedding`). The operationally relevant impostor score is
therefore the MAXIMUM over all non-self identities, not a random impostor pair.
Every rate below is that open-set identification rate. A 1:1 verification FMR
computed from random pairs would be far smaller and would not describe this
system.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PATH = "models/score_calibration.pkl"
_CACHE: Dict[str, Optional[Dict]] = {}


# ─── Error rates ─────────────────────────────────────────────────────────────

def fmr_at(threshold: float, impostor: Sequence[float]) -> float:
    """False-match rate: impostor comparisons that would be accepted."""
    imp = np.asarray(impostor, dtype=np.float64)
    return float((imp >= threshold).mean()) if imp.size else float("nan")


def fnmr_at(threshold: float, genuine: Sequence[float]) -> float:
    """False-non-match rate: genuine comparisons that would be rejected."""
    gen = np.asarray(genuine, dtype=np.float64)
    return float((gen < threshold).mean()) if gen.size else float("nan")


def threshold_for_fmr(target: float, impostor: Sequence[float]) -> Optional[float]:
    """
    Lowest cosine whose FMR is at or below `target`.

    Returns None when the target is below the corpus's resolution: with n
    impostor scores the smallest non-zero measurable rate is 1/n, and any
    threshold beyond the observed maximum has an empirically zero — that is,
    unmeasured — FMR. Reporting a number there would be extrapolation dressed
    as measurement.
    """
    imp = np.sort(np.asarray(impostor, dtype=np.float64))[::-1]
    n = imp.size
    if n == 0:
        return None
    if target < 1.0 / n:
        return None                     # below what n samples can resolve
    k = int(np.floor(target * n))
    if k <= 0:
        return None
    return float(imp[k - 1])


def resolution_limit(n_impostor: int) -> float:
    """Smallest FMR distinguishable from zero with this many impostor scores."""
    return 1.0 / n_impostor if n_impostor else float("nan")


# ─── Calibration ─────────────────────────────────────────────────────────────

def fit(genuine: Sequence[float], impostor: Sequence[float]) -> Dict:
    """
    Isotonic cosine -> P(genuine), plus the raw score distributions the error
    rates are read from.
    """
    from sklearn.isotonic import IsotonicRegression

    gen = np.asarray(genuine, dtype=np.float64)
    imp = np.asarray(impostor, dtype=np.float64)
    x = np.concatenate([gen, imp])
    y = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                             out_of_bounds="clip").fit(x, y)
    grid = np.linspace(float(x.min()), float(x.max()), 512)
    return {
        "grid": grid.astype(np.float32),
        "prob": iso.predict(grid).astype(np.float32),
        "genuine": gen.astype(np.float32),
        "impostor": imp.astype(np.float32),
        "n_genuine": int(gen.size),
        "n_impostor": int(imp.size),
    }


def probability(cosine: float, cal: Dict) -> float:
    """P(genuine | cosine) from the fitted map."""
    return float(np.interp(cosine, cal["grid"], cal["prob"]))


def load(path: str = DEFAULT_PATH) -> Optional[Dict]:
    if path in _CACHE:
        return _CACHE[path]
    cal = None
    try:
        p = Path(path)
        if p.is_file():
            import joblib
            cal = joblib.load(p)
        else:
            logger.debug("score_calibration: %s not present", path)
    except Exception as exc:                        # noqa: BLE001
        logger.error("score_calibration: cannot load %s: %s", path, exc)
    _CACHE[path] = cal
    return cal


# ─── Config integration ──────────────────────────────────────────────────────

def resolve_threshold(cfg, raw_attr: str = "face_match_threshold",
                      fmr_attr: str = "face_match_target_fmr") -> float:
    """
    The threshold actually in force.

    A target FMR, when set and satisfiable from the calibration, wins over the
    raw cosine. Otherwise the raw value stands — an absent or under-resolved
    calibration must never silently loosen the gate.
    """
    raw = float(getattr(cfg, raw_attr))
    target = getattr(cfg, fmr_attr, None)
    if not target:
        return raw
    cal = load(getattr(cfg, "score_calibration_path", DEFAULT_PATH))
    if cal is None:
        logger.warning("score_calibration: target FMR %.4g requested but no "
                       "calibration is available — keeping raw %.3f", target, raw)
        return raw
    t = threshold_for_fmr(float(target), cal["impostor"])
    if t is None:
        logger.warning("score_calibration: FMR %.4g is below the resolution of "
                       "%d impostor scores (limit %.4g) — keeping raw %.3f",
                       target, cal["n_impostor"],
                       resolution_limit(cal["n_impostor"]), raw)
        return raw
    return t
