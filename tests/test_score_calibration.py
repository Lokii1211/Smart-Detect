"""
tests/test_score_calibration.py
────────────────────────────────
Score calibration and FMR-derived thresholds.

The property that matters most for safety: an unavailable or under-resolved
calibration must fall back to the raw cosine. A target FMR that cannot be
honoured must never quietly loosen the gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from config.identity_config import IdentityConfig
from recognition.score_calibration import (fit, fmr_at, fnmr_at, probability,
                                           resolution_limit, resolve_threshold,
                                           threshold_for_fmr)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    genuine = np.clip(rng.normal(0.78, 0.10, 2000), 0, 1)
    impostor = np.clip(rng.normal(0.15, 0.06, 2000), 0, 1)
    return genuine, impostor


class TestErrorRates:

    def test_fmr_counts_accepted_impostors(self):
        imp = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert fmr_at(0.35, imp) == pytest.approx(0.4)
        assert fmr_at(0.0, imp) == pytest.approx(1.0)
        assert fmr_at(0.99, imp) == pytest.approx(0.0)

    def test_fnmr_counts_rejected_genuines(self):
        gen = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
        assert fnmr_at(0.55, gen) == pytest.approx(0.4)
        assert fnmr_at(0.0, gen) == pytest.approx(0.0)

    def test_raising_the_threshold_trades_fmr_for_fnmr(self, scores):
        gen, imp = scores
        lo, hi = 0.35, 0.65
        assert fmr_at(hi, imp) <= fmr_at(lo, imp)
        assert fnmr_at(hi, gen) >= fnmr_at(lo, gen)

    def test_empty_is_nan_not_zero(self):
        assert np.isnan(fmr_at(0.5, []))
        assert np.isnan(fnmr_at(0.5, []))


class TestThresholdForFmr:

    def test_recovers_the_requested_rate(self, scores):
        _gen, imp = scores
        for target in (0.10, 0.01):
            t = threshold_for_fmr(target, imp)
            assert t is not None
            assert fmr_at(t, imp) <= target + 1e-9

    def test_stricter_target_gives_higher_threshold(self, scores):
        _gen, imp = scores
        assert threshold_for_fmr(0.001, imp) > threshold_for_fmr(0.10, imp)

    def test_below_resolution_returns_none(self):
        """With n impostor scores the smallest measurable rate is 1/n.
        Returning a number below that would be extrapolation, not measurement."""
        imp = np.linspace(0.0, 0.4, 100)
        assert resolution_limit(100) == pytest.approx(0.01)
        assert threshold_for_fmr(0.001, imp) is None
        assert threshold_for_fmr(0.05, imp) is not None

    def test_empty_impostors(self):
        assert threshold_for_fmr(0.01, []) is None


class TestIsotonicFit:

    def test_probability_is_monotone(self, scores):
        gen, imp = scores
        cal = fit(gen, imp)
        p = [probability(x, cal) for x in np.linspace(0.0, 1.0, 60)]
        assert all(b >= a - 1e-6 for a, b in zip(p, p[1:])), "must be non-decreasing"

    def test_probability_is_bounded(self, scores):
        gen, imp = scores
        cal = fit(gen, imp)
        for x in np.linspace(-0.5, 1.5, 50):
            assert 0.0 <= probability(x, cal) <= 1.0

    def test_low_scores_are_impostors_high_scores_genuine(self, scores):
        gen, imp = scores
        cal = fit(gen, imp)
        assert probability(0.05, cal) < 0.10
        assert probability(0.95, cal) > 0.90

    def test_records_sample_counts(self, scores):
        gen, imp = scores
        cal = fit(gen, imp)
        assert cal["n_genuine"] == len(gen) and cal["n_impostor"] == len(imp)


class TestResolveThreshold:

    def test_default_uses_the_raw_cosine(self):
        assert resolve_threshold(IdentityConfig()) == pytest.approx(0.56)

    def test_no_calibration_keeps_the_raw_value(self):
        """Safety: a missing artefact must not loosen the gate."""
        cfg = IdentityConfig(face_match_target_fmr=0.01,
                             score_calibration_path="models/absent.pkl")
        assert resolve_threshold(cfg) == pytest.approx(0.56)

    @pytest.mark.skipif(not __import__("pathlib").Path(
        "models/score_calibration.pkl").is_file(), reason="calibration not built")
    def test_target_fmr_overrides_the_raw_value(self):
        cfg = IdentityConfig(face_match_target_fmr=0.01)
        assert resolve_threshold(cfg) != pytest.approx(0.56)

    @pytest.mark.skipif(not __import__("pathlib").Path(
        "models/score_calibration.pkl").is_file(), reason="calibration not built")
    def test_stricter_target_is_a_higher_threshold(self):
        a = resolve_threshold(IdentityConfig(face_match_target_fmr=0.01))
        b = resolve_threshold(IdentityConfig(face_match_target_fmr=0.001))
        assert b > a

    @pytest.mark.skipif(not __import__("pathlib").Path(
        "models/score_calibration.pkl").is_file(), reason="calibration not built")
    def test_unresolvable_target_falls_back_rather_than_guessing(self):
        """0.01% needs ~10 000 impostor scores; the corpus has 2833."""
        cfg = IdentityConfig(face_match_target_fmr=0.0001)
        assert resolve_threshold(cfg) == pytest.approx(0.56)
