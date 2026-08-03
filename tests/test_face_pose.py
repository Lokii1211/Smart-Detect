"""
tests/test_face_pose.py
────────────────────────
Head-pose estimation and the registration pose gate.

The gate reduces duplicate identities by refusing to MINT a code from an
extreme viewpoint. These tests pin three things:
  * the geometry is right (calibrated against a real head-pose sweep),
  * the gate only ever blocks REGISTRATION, never matching,
  * it fails open — a missing pose, or a track that has waited too long,
    must never permanently prevent enrolment.

Synthetic landmarks, no models. See tests/conftest.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import add_person, make_embedding

from config.identity_config import IdentityConfig
from recognition.face_pose import (HeadPose, estimate_pose,
                                   pose_ok_for_registration)
from recognition.smart_identifier import SmartIdentifier


def kps(yaw_off=0.0, pitch_frac=0.5, roll_deg=0.0, iod=40.0, cx=100.0, cy=100.0):
    """
    Build 5-point landmarks with a known pose.

    yaw_off    : nose offset from eye-centre, in units of inter-ocular distance
    pitch_frac : nose position between eye line (0) and mouth line (1)
    """
    half = iod / 2.0
    r = np.radians(roll_deg)
    le = np.array([cx - half * np.cos(r), cy - half * np.sin(r)])
    re = np.array([cx + half * np.cos(r), cy + half * np.sin(r)])
    span = iod * 1.1                       # eye-line to mouth-line
    mouth_cy = cy + span
    nose = np.array([cx + yaw_off * iod, cy + pitch_frac * span])
    lm = np.array([cx - iod * 0.35, mouth_cy])
    rm = np.array([cx + iod * 0.35, mouth_cy])
    return np.stack([le, re, nose, lm, rm]).astype(np.float32)


class TestPoseEstimation:

    def test_frontal_face_is_neutral(self):
        p = estimate_pose(kps(yaw_off=0.0, pitch_frac=0.5))
        assert p.yaw == pytest.approx(0.0, abs=1e-3)
        assert p.pitch == pytest.approx(0.5, abs=1e-3)
        assert p.roll == pytest.approx(0.0, abs=1e-3)

    def test_yaw_sign_follows_turn_direction(self):
        assert estimate_pose(kps(yaw_off=+0.4)).yaw > 0
        assert estimate_pose(kps(yaw_off=-0.4)).yaw < 0

    def test_pitch_above_half_is_head_down(self):
        """Calibrated on real footage: bowed heads measured pitch ~1.0,
        raised chins ~0.3, frontal ~0.5."""
        assert estimate_pose(kps(pitch_frac=1.0)).pitch > 0.5    # bowed
        assert estimate_pose(kps(pitch_frac=0.3)).pitch < 0.5    # chin up

    def test_roll_is_degrees(self):
        assert estimate_pose(kps(roll_deg=30)).roll == pytest.approx(30, abs=1.0)

    def test_pose_is_scale_invariant(self):
        """Ratios, not pixels — a face twice the size has the same pose."""
        near = estimate_pose(kps(yaw_off=0.3, pitch_frac=0.7, iod=80))
        far = estimate_pose(kps(yaw_off=0.3, pitch_frac=0.7, iod=20))
        assert near.yaw == pytest.approx(far.yaw, abs=1e-3)
        assert near.pitch == pytest.approx(far.pitch, abs=1e-3)

    def test_frontality_peaks_at_frontal(self):
        assert estimate_pose(kps()).frontality > 0.95
        assert estimate_pose(kps(yaw_off=0.6)).frontality < 0.6
        assert estimate_pose(kps(pitch_frac=1.0)).frontality < 0.6

    def test_degenerate_landmarks_return_none(self):
        assert estimate_pose(None) is None
        assert estimate_pose(np.zeros((2, 2), dtype=np.float32)) is None
        coincident = np.zeros((5, 2), dtype=np.float32)      # all points equal
        assert estimate_pose(coincident) is None

    def test_never_raises_on_garbage(self):
        for bad in ("nonsense", 42, [[1]], np.array([np.nan] * 10)):
            estimate_pose(bad)          # must not raise


class TestRegistrationGate:

    def test_frontal_passes(self):
        ok, _ = pose_ok_for_registration(estimate_pose(kps()), IdentityConfig())
        assert ok is True

    def test_profile_rejected(self):
        ok, why = pose_ok_for_registration(
            estimate_pose(kps(yaw_off=0.6)), IdentityConfig())
        assert ok is False and "yaw" in why

    def test_head_down_rejected(self):
        """The exact failure that produced a duplicate identity in real data:
        a code minted from a bowed-head frame."""
        ok, why = pose_ok_for_registration(
            estimate_pose(kps(pitch_frac=1.0)), IdentityConfig())
        assert ok is False and "pitch" in why

    def test_chin_up_rejected(self):
        ok, why = pose_ok_for_registration(
            estimate_pose(kps(pitch_frac=0.2)), IdentityConfig())
        assert ok is False and "pitch" in why

    def test_extreme_roll_rejected(self):
        ok, why = pose_ok_for_registration(
            estimate_pose(kps(roll_deg=60)), IdentityConfig())
        assert ok is False and "roll" in why

    def test_missing_pose_fails_open(self):
        """A detector that stops returning landmarks must not silently stop
        all enrolment."""
        ok, why = pose_ok_for_registration(None, IdentityConfig())
        assert ok is True and why == "no_pose"

    def test_disabled_gate_passes_everything(self):
        cfg = IdentityConfig(enable_pose_gate_at_registration=False)
        ok, _ = pose_ok_for_registration(estimate_pose(kps(pitch_frac=1.0)), cfg)
        assert ok is True

    def test_thresholds_are_config_driven(self):
        pose = estimate_pose(kps(yaw_off=0.5))
        assert pose_ok_for_registration(pose, IdentityConfig())[0] is False
        loose = IdentityConfig(registration_max_abs_yaw=0.9)
        assert pose_ok_for_registration(pose, loose)[0] is True


class TestGateAppliesToRegistrationOnly:
    """The gate must never cost recall for someone already enrolled."""

    def test_extreme_pose_blocks_registration(self, db, blank_frame, full_bbox):
        from database.models import Person
        r = SmartIdentifier(IdentityConfig()).identify(
            blank_frame, full_bbox, db, allow_new=True,
            face_embedding=make_embedding(1),
            face_pose=estimate_pose(kps(pitch_frac=1.0)),
            extract_face_if_missing=False)
        assert r["unique_code"] == "Detecting..."
        assert r["method"] == "pending_pose"
        assert db.query(Person).count() == 0, "no code may be minted"

    def test_extreme_pose_still_MATCHES_an_enrolled_person(self, db, blank_frame, full_bbox):
        """A profile view of someone already enrolled must still match —
        blocking this would lose recall, which the gate must never do."""
        stored = make_embedding(2)
        add_person(db, "SDT-0001", face_emb=stored)
        r = SmartIdentifier(IdentityConfig()).identify(
            blank_frame, full_bbox, db, allow_new=True,
            face_embedding=stored,                       # same face
            face_pose=estimate_pose(kps(yaw_off=0.9)),   # extreme profile
            extract_face_if_missing=False)
        assert r["unique_code"] == "SDT-0001"
        assert r["method"] == "face"

    def test_good_pose_registers_normally(self, db, blank_frame, full_bbox):
        from database.models import Person
        r = SmartIdentifier(IdentityConfig()).identify(
            blank_frame, full_bbox, db, allow_new=True,
            face_embedding=make_embedding(3),
            face_pose=estimate_pose(kps()),
            extract_face_if_missing=False)
        assert r["unique_code"].startswith("SDT-")
        assert db.query(Person).count() == 1


class TestStarvationValve:
    """
    Deferring assumes a better frame arrives on the same track. For a person
    who is never frontal that frame never comes, so the valve must eventually
    let them enrol from the best available view.
    """

    @pytest.fixture
    def ls(self):
        from cameras.live_stream import LiveStream
        s = LiveStream(source=0, camera_id="CAM-T")
        s._identity_config = IdentityConfig(pose_gate_max_deferrals=3)
        return s

    def test_pose_passed_through_below_limit(self, ls):
        pose = estimate_pose(kps(pitch_frac=1.0))
        for _ in range(3):
            assert ls._pose_for_registration(1, pose) is pose
            ls._note_pose_deferral(1, "pending_pose")

    def test_valve_opens_at_limit(self, ls):
        pose = estimate_pose(kps(pitch_frac=1.0))
        for _ in range(3):
            ls._note_pose_deferral(1, "pending_pose")
        assert ls._pose_for_registration(1, pose) is None, (
            "after the deferral limit the gate must fail open")

    def test_success_resets_the_counter(self, ls):
        pose = estimate_pose(kps(pitch_frac=1.0))
        for _ in range(2):
            ls._note_pose_deferral(1, "pending_pose")
        ls._note_pose_deferral(1, "face")           # a real decision
        assert ls._track_pose_defer.get(1, 0) == 0
        assert ls._pose_for_registration(1, pose) is pose

    def test_counters_are_per_track(self, ls):
        pose = estimate_pose(kps(pitch_frac=1.0))
        for _ in range(3):
            ls._note_pose_deferral(1, "pending_pose")
        assert ls._pose_for_registration(1, pose) is None
        assert ls._pose_for_registration(2, pose) is pose, "track 2 unaffected"

    def test_valve_disabled_by_zero(self):
        from cameras.live_stream import LiveStream
        s = LiveStream(source=0, camera_id="CAM-T")
        s._identity_config = IdentityConfig(pose_gate_max_deferrals=0)
        pose = estimate_pose(kps(pitch_frac=1.0))
        for _ in range(50):
            s._note_pose_deferral(1, "pending_pose")
        assert s._pose_for_registration(1, pose) is pose, "0 disables the valve"


class TestMergeConfidenceBands:
    """Bands are advisory. Nothing auto-merges at any confidence."""

    def test_bands_partition_by_similarity(self):
        from backend.main import _merge_confidence_band
        cfg = IdentityConfig()
        assert _merge_confidence_band(0.95, cfg)[0] == "high"
        assert _merge_confidence_band(0.80, cfg)[0] == "high"
        assert _merge_confidence_band(0.79, cfg)[0] == "medium"
        assert _merge_confidence_band(0.65, cfg)[0] == "medium"
        assert _merge_confidence_band(0.64, cfg)[0] == "low"
        assert _merge_confidence_band(0.50, cfg)[0] == "low"

    def test_every_band_carries_guidance(self):
        from backend.main import _merge_confidence_band
        cfg = IdentityConfig()
        for sim in (0.95, 0.70, 0.52):
            band, guidance = _merge_confidence_band(sim, cfg)
            assert guidance and len(guidance) > 20

    def test_low_band_warns_about_lookalikes(self):
        from backend.main import _merge_confidence_band
        _, g = _merge_confidence_band(0.51, IdentityConfig())
        assert "ook-alike" in g or "not merge" in g.lower()

    def test_bands_are_config_driven(self):
        from backend.main import _merge_confidence_band
        strict = IdentityConfig(merge_band_high=0.95, merge_band_medium=0.90)
        assert _merge_confidence_band(0.92, strict)[0] == "medium"
        assert _merge_confidence_band(0.85, strict)[0] == "low"
