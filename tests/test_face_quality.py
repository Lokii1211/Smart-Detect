"""
tests/test_face_quality.py
───────────────────────────
The learned face-quality gate.

The single most important property is that with `enable_learned_quality_gate`
off — the shipped default — `gate_passes()` is exactly the original
two-constant test. A quality gate that quietly changed behaviour when nobody
enabled it would alter every identity decision in the system.
"""
from __future__ import annotations

import numpy as np
import pytest

from config.identity_config import IdentityConfig
from recognition.face_pose import estimate_pose
from recognition.face_quality import (FEATURE_NAMES, build_features, _forward,
                                      gate_passes, load_model, predict_margin)

POSE = estimate_pose([[0, 0], [40, 0], [20, 15], [8, 35], [32, 35]])


def crop(h=60, w=40, noisy=False):
    """A face-like crop: smooth with mild texture. Uniform random noise gives a
    Laplacian variance ~50 000 where real face crops sit in the hundreds, which
    is far outside anything the model was trained on."""
    rng = np.random.default_rng(0)
    h, w = max(1, int(h)), max(1, int(w))
    if noisy:
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    base = np.linspace(60, 190, h * w).reshape(h, w).astype(np.float32)
    base += rng.normal(0, 4, (h, w))
    return np.clip(np.stack([base] * 3, -1), 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 1. OFF is byte-identical to the fixed gate
# ═══════════════════════════════════════════════════════════════════════════

class TestDisabledIsUnchanged:

    def test_flag_defaults_off(self):
        assert IdentityConfig().enable_learned_quality_gate is False

    @pytest.mark.parametrize("h", [10, 30, 47, 47.9, 48, 48.1, 60, 120, 300])
    @pytest.mark.parametrize("ds", [0.0, 0.35, 0.599, 0.6, 0.601, 0.9, 1.0])
    def test_matches_the_original_expression(self, h, ds):
        cfg = IdentityConfig()
        want = (h >= cfg.face_quality_min_height_px
                and ds >= cfg.face_quality_min_det_score)
        assert gate_passes([0, 0, 40, h], ds, POSE, cfg, crop=crop(h)) is want

    def test_boundaries_are_inclusive_exactly_as_before(self):
        cfg = IdentityConfig()
        assert gate_passes([0, 0, 40, 48], 0.60, POSE, cfg) is True
        assert gate_passes([0, 0, 40, 47.9], 0.60, POSE, cfg) is False
        assert gate_passes([0, 0, 40, 48], 0.599, POSE, cfg) is False

    def test_no_model_is_loaded_when_disabled(self, monkeypatch):
        """Proves the model is not merely unused but never touched."""
        import recognition.face_quality as fq
        monkeypatch.setattr(fq, "load_model",
                            lambda *a, **k: pytest.fail("model loaded while disabled"))
        gate_passes([0, 0, 40, 60], 0.9, POSE, IdentityConfig(), crop=crop())

    def test_pose_and_crop_are_irrelevant_when_disabled(self):
        cfg = IdentityConfig()
        a = gate_passes([0, 0, 40, 60], 0.9, None, cfg)
        b = gate_passes([0, 0, 40, 60], 0.9, POSE, cfg, crop=crop(),
                        person_boxes=[[0, 0, 500, 500]], owner_box=[0, 0, 100, 200])
        assert a is b is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature vector contract
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatures:

    def test_length_and_order(self):
        v = build_features([0, 0, 40, 60], 0.9, POSE, crop=crop())
        assert len(v) == len(FEATURE_NAMES) == 12
        assert v[FEATURE_NAMES.index("face_h_px")] == 60
        assert v[FEATURE_NAMES.index("face_w_px")] == 40
        assert v[FEATURE_NAMES.index("det_score")] == pytest.approx(0.9)

    def test_matches_the_training_extractor_order(self):
        """A permuted vector would score silently and wrongly."""
        from eval.build_quality_dataset import FEATURE_NAMES as TRAIN_NAMES
        assert FEATURE_NAMES == list(TRAIN_NAMES)

    def test_missing_pose_uses_neutral_values(self):
        v = build_features([0, 0, 40, 60], 0.9, None)
        assert v[FEATURE_NAMES.index("yaw")] == 0.0
        assert v[FEATURE_NAMES.index("pitch")] == 0.5

    def test_occlusion_zero_without_other_people(self):
        v = build_features([10, 10, 40, 60], 0.9, POSE,
                           person_boxes=[[0, 0, 100, 200]], owner_box=[0, 0, 100, 200])
        assert v[FEATURE_NAMES.index("occlusion_frac")] == 0.0

    def test_occlusion_detects_an_overlapping_person(self):
        v = build_features([10, 10, 40, 60], 0.9, POSE,
                           person_boxes=[[0, 0, 100, 200], [0, 0, 200, 200]],
                           owner_box=[0, 0, 100, 200])
        assert v[FEATURE_NAMES.index("occlusion_frac")] == pytest.approx(1.0)

    def test_face_to_person_ratio(self):
        v = build_features([0, 0, 40, 50], 0.9, POSE, owner_box=[0, 0, 100, 200])
        assert v[FEATURE_NAMES.index("face_to_person_ratio")] == pytest.approx(0.25)

    def test_no_crop_gives_zero_photometrics(self):
        v = build_features([0, 0, 40, 60], 0.9, POSE, crop=None)
        for k in ("blur_var_laplacian", "brightness", "contrast"):
            assert v[FEATURE_NAMES.index(k)] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Enabled behaviour, and safe degradation
# ═══════════════════════════════════════════════════════════════════════════

class TestEnabled:

    def test_missing_model_falls_back_to_the_fixed_gate(self):
        """An absent artefact must degrade to shipped behaviour, never to an
        open gate."""
        cfg = IdentityConfig(enable_learned_quality_gate=True,
                             face_quality_model_path="models/does_not_exist.pkl")
        assert gate_passes([0, 0, 40, 60], 0.9, POSE, cfg, crop=crop()) is True
        assert gate_passes([0, 0, 40, 20], 0.9, POSE, cfg, crop=crop(20)) is False

    def test_feature_mismatch_is_refused(self, tmp_path, monkeypatch):
        import joblib
        import recognition.face_quality as fq
        bad = tmp_path / "bad.pkl"
        joblib.dump({"kind": "mlp", "feature_names": ["wrong", "order"],
                     "mean": np.zeros(2, np.float32), "scale": np.ones(2, np.float32),
                     "W": [np.zeros((2, 1), np.float32)], "b": [np.zeros(1, np.float32)]},
                    bad)
        monkeypatch.setattr(fq, "_CACHE", {})
        assert fq.load_model(str(bad)) is None, "a permuted feature list must be refused"

    @pytest.mark.skipif(not __import__("pathlib").Path("models/face_quality.pkl").is_file(),
                        reason="trained model not present")
    def test_trained_model_predicts_a_plausible_margin(self):
        v = build_features([0, 0, 80, 120], 0.95, POSE, crop=crop(120, 80))
        m = predict_margin(v)
        assert m is not None
        art = load_model()
        assert art["y_lo"] <= m <= art["y_hi"], (
            f"{m} outside the trained margin range "
            f"[{art['y_lo']:.3f}, {art['y_hi']:.3f}]")

    @pytest.mark.skipif(not __import__("pathlib").Path("models/face_quality.pkl").is_file(),
                        reason="trained model not present")
    def test_inference_is_under_one_millisecond(self):
        import time
        art = load_model()
        v = build_features([0, 0, 80, 120], 0.95, POSE, crop=crop(120, 80))
        _forward(art, v)
        t0 = time.perf_counter()
        for _ in range(5000):
            _forward(art, v)
        per_ms = (time.perf_counter() - t0) / 5000 * 1000
        assert per_ms < 1.0, f"{per_ms:.3f} ms/call exceeds the 1 ms budget"

    @pytest.mark.skipif(not __import__("pathlib").Path("models/face_quality.pkl").is_file(),
                        reason="trained model not present")
    def test_a_larger_face_scores_above_a_smaller_one(self):
        """Face size is the strongest single predictor of margin in the training
        data (Spearman +0.71; mean margin rises monotonically across every
        height decile, 0.41 -> 0.71). Only height/width vary here: det_score is
        held equal because it is NEGATIVELY correlated with margin on this
        corpus, and inputs are clipped to the trained range, so comparing a
        20 px face would really be comparing the 52 px training floor."""
        small = build_features([0, 0, 44, 60], 0.90, POSE, crop=crop(60, 44))
        large = build_features([0, 0, 150, 210], 0.90, POSE, crop=crop(210, 150))
        assert predict_margin(large) > predict_margin(small)

    @pytest.mark.skipif(not __import__("pathlib").Path("models/face_quality.pkl").is_file(),
                        reason="trained model not present")
    def test_out_of_distribution_input_cannot_produce_an_absurd_margin(self):
        """REGRESSION. An MLP extrapolates without bound; a noise crop gave a
        margin of 4.5 against a trained range of [0.18, 0.86], which would have
        made the gate admit anything. Inputs are clipped and the output clamped."""
        art = load_model()
        wild = build_features([0, 0, 4000, 9000], 99.0, POSE, crop=crop(200, 200, noisy=True))
        m = predict_margin(wild)
        assert art["y_lo"] <= m <= art["y_hi"], f"{m} escaped the trained range"

    @pytest.mark.skipif(not __import__("pathlib").Path("models/face_quality.pkl").is_file(),
                        reason="trained model not present")
    def test_clipping_bounds_every_feature(self):
        art = load_model()
        assert "feat_lo" in art and "feat_hi" in art
        assert len(art["feat_lo"]) == len(FEATURE_NAMES)
