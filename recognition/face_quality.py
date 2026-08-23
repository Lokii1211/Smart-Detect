"""
recognition/face_quality.py
────────────────────────────
Learned face-quality gate — predicts the DISCRIMINATIVE MARGIN of a face
embedding (genuine similarity minus best-impostor similarity) from cheap
geometric and photometric features, and admits a face only above a threshold.

Replaces the fixed 48 px / 0.60 det_score pair, which are two hand-set
constants standing in for a quantity nobody measured.

OFF BY DEFAULT. `enable_learned_quality_gate` must be set explicitly; with it
off, nothing in this module is imported on the hot path and the fixed gate runs
byte-identically to before.

INFERENCE COST
The model is a 2-layer MLP exported as raw weights and evaluated with a plain
numpy forward pass — measured ~0.006 ms/call. The tree model that produced the
feature-importance analysis is NOT deployed: sklearn's single-row `predict()`
carries a fixed ~22 ms thread-pool overhead that cannot be amortised, because
the gate is inherently one face at a time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/face_quality.pkl"

# Must match eval/build_quality_dataset.py FEATURE_NAMES exactly. The loader
# verifies this against the artefact and refuses a mismatch rather than
# silently scoring a permuted vector.
FEATURE_NAMES = [
    "face_h_px", "face_w_px", "det_score",
    "yaw", "pitch", "roll", "frontality",
    "blur_var_laplacian", "brightness", "contrast",
    "occlusion_frac", "face_to_person_ratio",
]

_CACHE: Dict[str, Optional[Dict]] = {}


def _forward(art: Dict, v: np.ndarray) -> float:
    """
    Standardise, then ReLU-MLP forward. ~0.006 ms for a 12-32-16-1 net.

    Inputs are CLIPPED to the trained feature range and the output CLAMPED to
    the trained margin range. An MLP extrapolates without bound: an unusual
    crop (measured: blur variance 50 000 where training saw hundreds) otherwise
    produces an arbitrary margin, and a gate comparing that to a threshold
    would admit anything. Clipping makes out-of-distribution input degrade to
    the nearest seen case instead of to nonsense.
    """
    lo, hi = art.get("feat_lo"), art.get("feat_hi")
    if lo is not None and hi is not None:
        v = np.clip(v, lo, hi)
    h = (v - art["mean"]) / art["scale"]
    W, b = art["W"], art["b"]
    for i in range(len(W) - 1):
        h = np.maximum(0.0, h @ W[i] + b[i])
    out = float(h @ W[-1] + b[-1])
    ylo, yhi = art.get("y_lo"), art.get("y_hi")
    if ylo is not None and yhi is not None:
        out = min(max(out, ylo), yhi)
    return out


def load_model(path: str = DEFAULT_MODEL_PATH) -> Optional[Dict]:
    """Cached load. Returns None (and logs once) when unavailable — a missing
    model must fall back to the fixed gate, never crash the pipeline."""
    if path in _CACHE:
        return _CACHE[path]
    art = None
    try:
        p = Path(path)
        if p.is_file():
            import joblib
            art = joblib.load(p)
            got = list(art.get("feature_names", []))
            if got != FEATURE_NAMES:
                logger.error("face_quality: feature order mismatch in %s "
                             "(got %s) — refusing to use it", path, got)
                art = None
        else:
            logger.warning("face_quality: %s not found — falling back to the "
                           "fixed gate", path)
    except Exception as exc:                       # noqa: BLE001
        logger.error("face_quality: could not load %s: %s", path, exc)
        art = None
    _CACHE[path] = art
    return art


def build_features(face_box: Sequence[int], det_score: float, pose,
                   crop: Optional[np.ndarray] = None,
                   person_boxes: Optional[List[Sequence[int]]] = None,
                   owner_box: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Assemble the 12-feature vector in FEATURE_NAMES order.

    Every input is already computed by the caller for other reasons: the box
    and det_score come from the detector, `pose` from face_pose.estimate_pose
    over landmarks InsightFace already returned, and the crop is a view.
    """
    x, y, w, h = [float(v) for v in face_box]

    blur = brightness = contrast = 0.0
    if crop is not None and crop.size:
        try:
            import cv2
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
            brightness = float(g.mean())
            contrast = float(g.std())
        except Exception:                          # noqa: BLE001
            pass

    occl = 0.0
    if person_boxes:
        area = max(1.0, w * h)
        cov = 0.0
        for pb in person_boxes:
            if owner_box is not None and list(pb) == list(owner_box):
                continue
            px, py, pw, ph = pb
            ix = max(0.0, min(x + w, px + pw) - max(x, px))
            iy = max(0.0, min(y + h, py + ph) - max(y, py))
            cov += ix * iy
        occl = min(1.0, cov / area)

    ratio = 0.0
    if owner_box is not None and owner_box[3]:
        ratio = h / float(owner_box[3])

    return np.array([
        h, w, float(det_score),
        float(pose.yaw) if pose else 0.0,
        float(pose.pitch) if pose else 0.5,
        float(pose.roll) if pose else 0.0,
        float(pose.frontality) if pose else 0.0,
        blur, brightness, contrast, occl, ratio,
    ], dtype=np.float32)


def predict_margin(features: np.ndarray, path: str = DEFAULT_MODEL_PATH) -> Optional[float]:
    """Predicted genuine-minus-impostor margin, or None if no model is loaded."""
    art = load_model(path)
    if art is None:
        return None
    try:
        return _forward(art, np.asarray(features, dtype=np.float32))
    except Exception as exc:                       # noqa: BLE001
        logger.debug("face_quality: prediction failed: %s", exc)
        return None


def gate_passes(face_box, det_score, pose, cfg, crop=None,
                person_boxes=None, owner_box=None) -> bool:
    """
    The gate decision.

    With `enable_learned_quality_gate` off — the default — this is exactly the
    original two-constant test, evaluated in the same order, so behaviour is
    unchanged. With it on, a face is admitted when its predicted margin clears
    `learned_quality_min_margin`.

    Falls back to the fixed test whenever the model is missing or prediction
    fails: an unavailable artefact must degrade to shipped behaviour, not to
    an open gate.
    """
    h = float(face_box[3])
    fixed = (h >= cfg.face_quality_min_height_px
             and float(det_score) >= cfg.face_quality_min_det_score)

    if not getattr(cfg, "enable_learned_quality_gate", False):
        return fixed

    feats = build_features(face_box, det_score, pose, crop, person_boxes, owner_box)
    margin = predict_margin(feats, getattr(cfg, "face_quality_model_path",
                                           DEFAULT_MODEL_PATH))
    if margin is None:
        return fixed
    return margin >= float(getattr(cfg, "learned_quality_min_margin", 0.35))
