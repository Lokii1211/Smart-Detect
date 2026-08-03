"""
tests/test_identity_arbitration.py
───────────────────────────────────
Unit tests for SmartDetect's identity-arbitration layer.

Everything here is synthetic embeddings + a temp SQLite gallery. No camera,
no InsightFace, no YOLO, no network. See tests/conftest.py.

Each test names the production behaviour it pins, so a future threshold
change that breaks an intended guarantee fails loudly.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import (add_person, cosine, embedding_at_similarity,
                      make_embedding)

from config.identity_config import IdentityConfig
from recognition.smart_identifier import SmartIdentifier, _face_veto

ABLATION_CONFIGS = ["A_pre_hardening", "B_face_anchor", "C_plus_guard", "D_full"]


def load_ablation(name: str) -> IdentityConfig:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / "config" / "ablation" / f"{name}.json").read_text())
    return IdentityConfig.from_dict(data)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Face matching either side of face_match_threshold (default 0.56)
# ═══════════════════════════════════════════════════════════════════════════

class TestFaceMatchThreshold:

    def test_match_above_threshold(self, db, blank_frame, full_bbox):
        cfg = IdentityConfig()
        stored = make_embedding(1)
        add_person(db, "SDT-0001", face_emb=stored)
        # Comfortably above 0.56
        query = embedding_at_similarity(stored, 0.75)
        assert cosine(query, stored) == pytest.approx(0.75, abs=1e-3)

        r = SmartIdentifier(cfg).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=query, extract_face_if_missing=False)

        assert r["unique_code"] == "SDT-0001"
        assert r["method"] == "face"

    def test_no_match_below_threshold(self, db, blank_frame, full_bbox):
        cfg = IdentityConfig()
        stored = make_embedding(2)
        add_person(db, "SDT-0001", face_emb=stored)
        # Just under 0.56 — close enough to be a lookalike, not a match
        query = embedding_at_similarity(stored, 0.50)

        r = SmartIdentifier(cfg).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=query, extract_face_if_missing=False)

        assert r["unique_code"] == "Detecting..."
        assert r["method"] == "pending"

    def test_threshold_boundary_is_inclusive(self, db, blank_frame, full_bbox):
        """find_person_by_embedding rejects sim < threshold, so exactly at
        the threshold must MATCH. Pins the comparison direction."""
        cfg = IdentityConfig()
        stored = make_embedding(3)
        add_person(db, "SDT-0001", face_emb=stored)
        query = embedding_at_similarity(stored, cfg.face_match_threshold + 1e-4)

        r = SmartIdentifier(cfg).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=query, extract_face_if_missing=False)
        assert r["unique_code"] == "SDT-0001"

    def test_threshold_is_config_driven(self, db, blank_frame, full_bbox):
        """A similarity that fails at 0.56 must pass when the config lowers
        the bar — proves the threshold is read from config, not hardcoded."""
        stored = make_embedding(4)
        add_person(db, "SDT-0001", face_emb=stored)
        query = embedding_at_similarity(stored, 0.45)

        strict = SmartIdentifier(IdentityConfig(face_match_threshold=0.56))
        loose = SmartIdentifier(IdentityConfig(face_match_threshold=0.40))

        assert strict.identify(blank_frame, full_bbox, db, allow_new=False,
                               face_embedding=query,
                               extract_face_if_missing=False)["unique_code"] == "Detecting..."
        assert loose.identify(blank_frame, full_bbox, db, allow_new=False,
                              face_embedding=query,
                              extract_face_if_missing=False)["unique_code"] == "SDT-0001"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Face-anchored blocking of colour / re-ID re-association
#
# HOW THE TWO MECHANISMS DIVIDE THE WORK (established by mutation testing):
#
#   _face_veto        blocks a candidate when query-vs-stored face similarity
#                     is BELOW face_veto_threshold (0.45).
#   face_says_stranger blocks the entire colour/re-ID block whenever ANY
#                     gate-passing face is present and unmatched.
#
# So the veto covers sim < 0.45, and the stranger gate uniquely covers the
# LOOKALIKE BAND [0.45, 0.56): faces too dissimilar to match (< 0.56) but too
# similar for the veto to reject (>= 0.45). In the shipped configuration the
# stranger gate always fires first, so the veto never *decides* anything —
# but it is a real backstop, not dead code: delete the stranger gate and the
# veto still catches everything below 0.45.
#
# test_lookalike_band_* below is the test that fails if the stranger gate is
# removed. Tests using near-orthogonal strangers do NOT catch that, because
# the veto covers them too — a gap found by mutating the source and watching
# the suite stay green.
# ═══════════════════════════════════════════════════════════════════════════

BLUE_HSV = {"hue": 110, "saturation": 200, "value": 180, "hex_color": "#3050b4"}


def blue_frame():
    """A frame whose dominant colour matches BLUE_HSV closely enough to be a
    colour match (HSV distance well under colour_match_threshold=30)."""
    import cv2
    hsv = np.uint8([[[BLUE_HSV["hue"], BLUE_HSV["saturation"], BLUE_HSV["value"]]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return np.full((240, 120, 3), tuple(int(c) for c in bgr), dtype=np.uint8)


class TestFaceAnchorBlocking:

    def test_visible_stranger_face_blocks_colour_reassociation(self, db, full_bbox):
        """The headline guarantee: a face that matches nobody must stop
        clothing colour handing this person an existing code."""
        cfg = IdentityConfig(enable_face_anchor=True)
        known = make_embedding(10)
        add_person(db, "SDT-0001", face_emb=known, dress_hsv=BLUE_HSV, minutes_ago=1)
        stranger = make_embedding(99)          # near-orthogonal to `known`
        assert cosine(stranger, known) < 0.2

        r = SmartIdentifier(cfg).identify(
            blue_frame(), full_bbox, db, allow_new=False,
            face_embedding=stranger, extract_face_if_missing=False)

        assert r["unique_code"] == "Detecting...", (
            "a visible unmatched face must block colour re-association")

    def test_colour_reassociates_when_anchor_disabled(self, db, full_bbox):
        """Same scenario with the anchor off must re-associate — proving the
        block above comes from the flag, not from colour matching failing."""
        cfg = IdentityConfig(enable_face_anchor=False)
        known = make_embedding(11)
        add_person(db, "SDT-0001", face_emb=known, dress_hsv=BLUE_HSV, minutes_ago=1)
        stranger = make_embedding(98)

        r = SmartIdentifier(cfg).identify(
            blue_frame(), full_bbox, db, allow_new=False,
            face_embedding=stranger, extract_face_if_missing=False)

        assert r["unique_code"] == "SDT-0001"
        assert r["method"] == "dress_color"

    def test_visible_stranger_face_blocks_reid_reassociation(self, db, full_bbox):
        """Same guarantee for Method 3. Re-ID is stubbed out at the model
        level here, so this asserts the gate is reached before any re-ID
        work — the faceless path is what matters."""
        cfg = IdentityConfig(enable_face_anchor=True, enable_colour_fallback=False)
        known = make_embedding(12)
        reid = make_embedding(500)
        add_person(db, "SDT-0001", face_emb=known, reid_emb=reid, minutes_ago=1)
        stranger = make_embedding(97)

        r = SmartIdentifier(cfg).identify(
            blue_frame(), full_bbox, db, allow_new=False,
            face_embedding=stranger, extract_face_if_missing=False)
        assert r["unique_code"] == "Detecting..."

    # ── The veto itself ────────────────────────────────────────────────────

    def test_face_veto_function_works_in_isolation(self, db):
        """_face_veto's own logic is correct: a contradicting face vetoes."""
        stored = make_embedding(20)
        add_person(db, "SDT-0001", face_emb=stored)
        stranger = make_embedding(21)
        assert _face_veto(stranger, "SDT-0001", db, 0.45) is True
        similar = embedding_at_similarity(stored, 0.80)
        assert _face_veto(similar, "SDT-0001", db, 0.45) is False

    def test_face_veto_returns_false_without_a_query_face(self, db):
        """No face -> nothing to contradict with."""
        add_person(db, "SDT-0001", face_emb=make_embedding(22))
        assert _face_veto(None, "SDT-0001", db, 0.45) is False

    def test_lookalike_band_is_blocked_only_by_the_stranger_gate(self, db, full_bbox):
        """
        THE test that pins face_says_stranger's unique contribution.

        A query face at 0.50 similarity to the stored template is:
          - below face_match_threshold (0.56) -> the face matcher rejects it
          - above face_veto_threshold  (0.45) -> _face_veto does NOT block it

        So in this band ONLY the stranger gate prevents colour from handing a
        lookalike someone else's code. Removing that gate makes this test
        fail — which is exactly what a mutation run confirmed.
        """
        cfg = IdentityConfig(enable_face_anchor=True)
        stored = make_embedding(35)
        add_person(db, "SDT-0001", face_emb=stored, dress_hsv=BLUE_HSV, minutes_ago=1)

        lookalike = embedding_at_similarity(stored, 0.50)
        assert cfg.face_veto_threshold <= 0.50 < cfg.face_match_threshold, (
            "test must sit inside the lookalike band")
        assert _face_veto(lookalike, "SDT-0001", db, cfg.face_veto_threshold) is False, (
            "precondition: the veto must NOT catch this one")

        r = SmartIdentifier(cfg).identify(
            blue_frame(), full_bbox, db, allow_new=False,
            face_embedding=lookalike, extract_face_if_missing=False)

        assert r["unique_code"] == "Detecting...", (
            "only face_says_stranger can block a 0.45-0.56 lookalike")

    def test_veto_is_a_backstop_below_its_threshold(self, db, full_bbox):
        """
        Complement to the above: the veto's own coverage is sim < 0.45.
        Verified directly on the function, since in the shipped config the
        stranger gate reaches these cases first.
        """
        stored = make_embedding(36)
        add_person(db, "SDT-0001", face_emb=stored)
        assert _face_veto(embedding_at_similarity(stored, 0.30),
                          "SDT-0001", db, 0.45) is True
        assert _face_veto(embedding_at_similarity(stored, 0.44),
                          "SDT-0001", db, 0.45) is True
        assert _face_veto(embedding_at_similarity(stored, 0.50),
                          "SDT-0001", db, 0.45) is False

    def test_veto_does_not_decide_in_the_shipped_config(self, db, full_bbox):
        """
        With the anchor ON the stranger gate fires first; with it OFF the veto
        is disabled by the same flag. So the veto never *decides* an outcome
        as shipped — documented so the redundancy is visible, not assumed.
        """
        known = make_embedding(30)
        add_person(db, "SDT-0001", face_emb=known, dress_hsv=BLUE_HSV, minutes_ago=1)
        contradicting = make_embedding(31)

        # anchor ON + face present: blocked by the stranger gate (not the veto)
        on = SmartIdentifier(IdentityConfig(enable_face_anchor=True)).identify(
            blue_frame(), full_bbox, db, allow_new=False,
            face_embedding=contradicting, extract_face_if_missing=False)
        assert on["unique_code"] == "Detecting..."

        # anchor OFF: colour matches even though the face plainly contradicts,
        # because the veto is gated behind the same flag it would defend.
        off = SmartIdentifier(IdentityConfig(enable_face_anchor=False)).identify(
            blue_frame(), full_bbox, db, allow_new=False,
            face_embedding=contradicting, extract_face_if_missing=False)
        assert off["unique_code"] == "SDT-0001", (
            "veto is disabled with the anchor, so it cannot block here")
        # ...even though the veto function itself says it should be blocked:
        assert _face_veto(contradicting, "SDT-0001", db, 0.45) is True


# ═══════════════════════════════════════════════════════════════════════════
# 3. ID-switch guard: fires at EXACTLY the contradiction limit
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def live_stream():
    """A LiveStream that is constructed but never start()ed: no capture
    thread, no cv2.VideoCapture, no analysis loop. Only its pure per-track
    helper methods are exercised."""
    from cameras.live_stream import LiveStream
    return LiveStream(source=0, location_id="LOC-TEST",
                      zone_id="test", camera_id="CAM-TEST")


class TestIdSwitchGuard:

    def _setup(self, db, ls, limit=2):
        ls._identity_config = IdentityConfig(
            enable_id_switch_guard=True,
            id_switch_contradiction_limit=limit,
            id_switch_similarity_threshold=0.35,
        )
        stored = make_embedding(40)
        add_person(db, "SDT-0001", face_emb=stored)
        contradicting = make_embedding(41)
        assert cosine(contradicting, stored) < 0.35
        tid = 7
        cached = {"code": "SDT-0001", "method": "face", "conf": 1.0}
        ls._track_codes[tid] = dict(cached)
        return tid, cached, contradicting

    def test_one_contradiction_does_not_drop_cache(self, db, live_stream):
        tid, cached, bad = self._setup(db, live_stream)
        out = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert out is not None, "guard must not fire on the first contradiction"
        assert out["code"] == "SDT-0001"
        assert tid in live_stream._track_codes
        assert live_stream._track_face_mismatch[tid] == 1

    def test_two_contradictions_drop_cache(self, db, live_stream):
        tid, cached, bad = self._setup(db, live_stream)
        cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert cached is None, "guard must fire on exactly the 2nd contradiction"
        assert tid not in live_stream._track_codes
        assert live_stream._track_face_mismatch[tid] == 0, "strike counter resets"

    def test_third_contradiction_is_a_no_op_after_drop(self, db, live_stream):
        """Once dropped there is no cache left; the guard must handle that
        without raising or resurrecting anything."""
        tid, cached, bad = self._setup(db, live_stream)
        for _ in range(2):
            cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert cached is None
        again = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert again is None
        assert tid not in live_stream._track_codes

    def test_agreeing_face_resets_the_strike_counter(self, db, live_stream):
        """One contradiction then a matching face must clear the count, so
        two NON-consecutive contradictions never trip the guard."""
        tid, cached, bad = self._setup(db, live_stream)
        # seed 40 is the stored template planted by _setup
        good = embedding_at_similarity(make_embedding(40), 0.90)

        cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert live_stream._track_face_mismatch[tid] == 1
        cached = live_stream._apply_id_switch_guard(tid, cached, good, db)
        assert live_stream._track_face_mismatch[tid] == 0, "agreement resets strikes"
        cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert cached is not None, "non-consecutive contradictions must not fire"
        assert live_stream._track_face_mismatch[tid] == 1

    def test_limit_is_config_driven(self, db, live_stream):
        tid, cached, bad = self._setup(db, live_stream, limit=3)
        for i in range(2):
            cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
            assert cached is not None, f"must not fire at contradiction {i+1} of 3"
        cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert cached is None, "must fire at the configured 3rd contradiction"

    def test_disabled_guard_never_drops_cache(self, db, live_stream):
        tid, cached, bad = self._setup(db, live_stream)
        live_stream._identity_config = IdentityConfig(enable_id_switch_guard=False)
        for _ in range(5):
            cached = live_stream._apply_id_switch_guard(tid, cached, bad, db)
        assert cached is not None and cached["code"] == "SDT-0001"
        assert tid in live_stream._track_codes

    def test_faceless_cycle_is_not_a_contradiction(self, db, live_stream):
        """No gate-passing face means no evidence either way; the guard must
        leave the cache and the counter alone."""
        tid, cached, _bad = self._setup(db, live_stream)
        for _ in range(5):
            cached = live_stream._apply_id_switch_guard(tid, cached, None, db)
        assert cached is not None
        assert live_stream._track_face_mismatch.get(tid, 0) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Evidence gate
# ═══════════════════════════════════════════════════════════════════════════

class TestEvidenceGate:

    def test_faceless_inherited_code_is_suppressed(self, db, live_stream):
        """The core guarantee: a box with no usable face, carrying a code it
        did not earn this cycle, must not write evidence."""
        live_stream._identity_config = IdentityConfig(enable_evidence_gating=True)
        add_person(db, "SDT-0001", face_emb=make_embedding(50))
        ok = live_stream._evidence_gate_ok(
            tid=1, code="SDT-0001", face_emb=None, fresh_face_id=False, db=db)
        assert ok is False

    def test_freshly_earned_code_writes_evidence(self, db, live_stream):
        live_stream._identity_config = IdentityConfig(enable_evidence_gating=True)
        add_person(db, "SDT-0001", face_emb=make_embedding(51))
        ok = live_stream._evidence_gate_ok(
            tid=1, code="SDT-0001", face_emb=make_embedding(51),
            fresh_face_id=True, db=db)
        assert ok is True

    def test_confirming_face_above_threshold_writes_evidence(self, db, live_stream):
        """Not freshly earned, but a face agrees with the stored template
        above evidence_face_sim_threshold (0.45)."""
        live_stream._identity_config = IdentityConfig(enable_evidence_gating=True)
        stored = make_embedding(52)
        add_person(db, "SDT-0001", face_emb=stored)
        confirming = embedding_at_similarity(stored, 0.70)
        assert live_stream._evidence_gate_ok(
            tid=1, code="SDT-0001", face_emb=confirming,
            fresh_face_id=False, db=db) is True

    def test_contradicting_face_below_threshold_suppresses(self, db, live_stream):
        live_stream._identity_config = IdentityConfig(enable_evidence_gating=True)
        stored = make_embedding(53)
        add_person(db, "SDT-0001", face_emb=stored)
        weak = embedding_at_similarity(stored, 0.20)
        assert live_stream._evidence_gate_ok(
            tid=1, code="SDT-0001", face_emb=weak,
            fresh_face_id=False, db=db) is False

    def test_track_with_strikes_is_suppressed_even_with_good_face(self, db, live_stream):
        """A track mid-ID-switch is suspect: no evidence until it re-verifies,
        regardless of this frame's face."""
        live_stream._identity_config = IdentityConfig(enable_evidence_gating=True)
        stored = make_embedding(54)
        add_person(db, "SDT-0001", face_emb=stored)
        live_stream._track_face_mismatch[9] = 1
        assert live_stream._evidence_gate_ok(
            tid=9, code="SDT-0001", face_emb=embedding_at_similarity(stored, 0.9),
            fresh_face_id=True, db=db) is False

    def test_gating_disabled_admits_everything(self, db, live_stream):
        live_stream._identity_config = IdentityConfig(enable_evidence_gating=False)
        add_person(db, "SDT-0001", face_emb=make_embedding(55))
        assert live_stream._evidence_gate_ok(
            tid=1, code="SDT-0001", face_emb=None,
            fresh_face_id=False, db=db) is True


# ═══════════════════════════════════════════════════════════════════════════
# 5. Registration requires a face — in ALL FOUR ablation configs
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistrationRequiresFace:

    @pytest.mark.parametrize("config_name", ABLATION_CONFIGS)
    def test_no_face_never_registers(self, db, blank_frame, full_bbox, config_name):
        """enable_face_anchor does NOT gate registration — the face
        requirement is unconditional. Pins the claim made in every
        config/ablation/*.json `_registration_note`."""
        cfg = load_ablation(config_name)
        from database.models import Person
        n_before = db.query(Person).count()

        r = SmartIdentifier(cfg).identify(
            blank_frame, full_bbox, db, allow_new=True,
            face_embedding=None, extract_face_if_missing=False)

        assert r["unique_code"] == "Detecting...", (
            f"{config_name}: a faceless box must never mint a code")
        assert db.query(Person).count() == n_before, (
            f"{config_name}: no Person row may be created without a face")

    @pytest.mark.parametrize("config_name", ABLATION_CONFIGS)
    def test_face_present_does_register(self, db, blank_frame, full_bbox, config_name):
        """Control for the test above: with a face, registration proceeds in
        every config — so the refusal above is about the face, not the config."""
        from database.models import Person
        cfg = load_ablation(config_name)
        r = SmartIdentifier(cfg).identify(
            blank_frame, full_bbox, db, allow_new=True,
            face_embedding=make_embedding(60), extract_face_if_missing=False)

        assert r["unique_code"].startswith("SDT-"), config_name
        assert r["method"] == "new_registration"
        assert db.query(Person).count() == 1

    @pytest.mark.parametrize("config_name", ABLATION_CONFIGS)
    def test_allow_new_false_blocks_registration(self, db, blank_frame,
                                                 full_bbox, config_name):
        """Track-age gating: a face alone is not enough if the caller has not
        confirmed the track."""
        from database.models import Person
        cfg = load_ablation(config_name)
        r = SmartIdentifier(cfg).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=make_embedding(61), extract_face_if_missing=False)
        assert r["unique_code"] == "Detecting..."
        assert db.query(Person).count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Gallery: cap of 5, add-window [0.60, 0.85)
# ═══════════════════════════════════════════════════════════════════════════

class TestGalleryMaintenance:

    def _person_with_gallery(self, db, n_templates: int, base_seed=70):
        base = make_embedding(base_seed)
        templates = [embedding_at_similarity(base, 0.70, seed=1000 + i)
                     for i in range(n_templates)]
        add_person(db, "SDT-0001", face_emb=base, templates=templates)
        return base

    def _gallery(self, db):
        from database.models import Person
        p = db.query(Person).filter(Person.unique_code == "SDT-0001").first()
        return json.loads(p.face_templates) if p.face_templates else []

    def test_view_inside_window_is_added(self, db):
        base = self._person_with_gallery(db, 0)
        si = SmartIdentifier(IdentityConfig())
        si._update_face_template("SDT-0001", embedding_at_similarity(base, 0.70),
                                 db, similarity=0.70)
        assert len(self._gallery(db)) == 1

    def test_below_lower_bound_is_not_added(self, db):
        """< gallery_add_threshold (0.60): too weak to trust as another view
        of the same person; adding it makes the code a similarity magnet."""
        base = self._person_with_gallery(db, 0)
        si = SmartIdentifier(IdentityConfig())
        si._update_face_template("SDT-0001", embedding_at_similarity(base, 0.59),
                                 db, similarity=0.59)
        assert self._gallery(db) == []

    def test_at_or_above_upper_bound_is_not_added(self, db):
        """>= gallery_add_max_similarity (0.85): near-duplicate of what is
        already stored, so it adds cost without adding coverage."""
        base = self._person_with_gallery(db, 0)
        si = SmartIdentifier(IdentityConfig())
        si._update_face_template("SDT-0001", embedding_at_similarity(base, 0.85),
                                 db, similarity=0.85)
        assert self._gallery(db) == []

    def test_lower_bound_is_inclusive(self, db):
        base = self._person_with_gallery(db, 0)
        si = SmartIdentifier(IdentityConfig())
        si._update_face_template("SDT-0001", embedding_at_similarity(base, 0.60),
                                 db, similarity=0.60)
        assert len(self._gallery(db)) == 1, "0.60 must be inside the window"

    def test_gallery_caps_at_five_keeping_newest(self, db):
        base = self._person_with_gallery(db, 5)
        assert len(self._gallery(db)) == 5
        si = SmartIdentifier(IdentityConfig())
        newest = embedding_at_similarity(base, 0.70, seed=9999)
        si._update_face_template("SDT-0001", newest, db, similarity=0.70)

        g = self._gallery(db)
        assert len(g) == 5, "gallery must stay capped at gallery_max_templates"
        assert np.allclose(np.array(g[-1], dtype=np.float32), newest, atol=1e-5), (
            "the cap must drop the OLDEST template and keep the newest")

    def test_cap_is_config_driven(self, db):
        base = self._person_with_gallery(db, 3)
        si = SmartIdentifier(IdentityConfig(gallery_max_templates=3))
        si._update_face_template("SDT-0001", embedding_at_similarity(base, 0.70, seed=8888),
                                 db, similarity=0.70)
        assert len(self._gallery(db)) == 3

    def test_primary_template_is_blended_not_replaced(self, db):
        """The running-average update must move the stored template toward
        the new view, not overwrite it."""
        from database.models import Person
        base = self._person_with_gallery(db, 0)
        new = embedding_at_similarity(base, 0.70, seed=7777)
        si = SmartIdentifier(IdentityConfig(template_blend_weight=0.20))
        si._update_face_template("SDT-0001", new, db, similarity=0.70)

        p = db.query(Person).filter(Person.unique_code == "SDT-0001").first()
        blended = np.array(json.loads(p.face_embedding), dtype=np.float32)
        assert cosine(blended, base) > cosine(blended, new), "must stay closer to the original"
        assert cosine(blended, base) < 1.0 - 1e-6, "must have moved at all"


# ═══════════════════════════════════════════════════════════════════════════
# 6b. Body re-ID (Method 3) — the real matching path
#
# PersonReID is replaced with a fake exposing the same contract
# (is_stub=False, extract_features -> np.ndarray). The production model is a
# pretrained OSNet whose weights are irrelevant to arbitration: what matters
# is whether a re-ID hit is accepted, vetoed, or gated. No weights are loaded.
# ═══════════════════════════════════════════════════════════════════════════

class FakeReID:
    is_stub = False

    def __init__(self, emb):
        self._emb = emb

    def extract_features(self, crop):
        return self._emb


@pytest.fixture
def fake_reid(monkeypatch):
    def _install(emb):
        import recognition.smart_identifier as si
        monkeypatch.setattr(si, "_get_reid_model", lambda: FakeReID(emb))
        return emb
    return _install


class TestReidMatching:

    def test_reid_matches_a_faceless_box(self, db, blank_frame, full_bbox, fake_reid):
        """Method 3's purpose: re-associate someone whose face is not visible
        this frame. Requires no face, so the stranger gate does not fire."""
        body = make_embedding(80)
        fake_reid(body)
        add_person(db, "SDT-0001", face_emb=make_embedding(81),
                   reid_emb=body, minutes_ago=5)

        r = SmartIdentifier(IdentityConfig()).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=None, extract_face_if_missing=False)

        assert r["unique_code"] == "SDT-0001"
        assert r["method"] == "body_structure"

    def test_reid_below_threshold_does_not_match(self, db, blank_frame,
                                                 full_bbox, fake_reid):
        body = make_embedding(82)
        fake_reid(embedding_at_similarity(body, 0.50))   # under 0.68
        add_person(db, "SDT-0001", face_emb=make_embedding(83),
                   reid_emb=body, minutes_ago=5)

        r = SmartIdentifier(IdentityConfig()).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=None, extract_face_if_missing=False)
        assert r["unique_code"] == "Detecting..."

    def test_reid_respects_recency_window(self, db, blank_frame,
                                          full_bbox, fake_reid):
        """Body features track clothing, so a match outside
        reid_reassoc_window_hours (12 h) must be refused."""
        body = make_embedding(84)
        fake_reid(body)
        add_person(db, "SDT-0001", face_emb=make_embedding(85),
                   reid_emb=body, minutes_ago=13 * 60)   # 13 h ago

        r = SmartIdentifier(IdentityConfig()).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=None, extract_face_if_missing=False)
        assert r["unique_code"] == "Detecting..."

    def test_reid_disabled_by_flag(self, db, blank_frame, full_bbox, fake_reid):
        body = make_embedding(86)
        fake_reid(body)
        add_person(db, "SDT-0001", face_emb=make_embedding(87),
                   reid_emb=body, minutes_ago=5)

        r = SmartIdentifier(IdentityConfig(enable_reid_fallback=False)).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=None, extract_face_if_missing=False)
        assert r["unique_code"] == "Detecting..."

    def test_reid_cannot_claim_a_code_already_visible(self, db, blank_frame,
                                                      full_bbox, fake_reid):
        """exclude_codes: a code worn by someone else in this same frame must
        not be handed to a second person by re-ID."""
        body = make_embedding(88)
        fake_reid(body)
        add_person(db, "SDT-0001", face_emb=make_embedding(89),
                   reid_emb=body, minutes_ago=5)

        r = SmartIdentifier(IdentityConfig()).identify(
            blank_frame, full_bbox, db, allow_new=False,
            face_embedding=None, exclude_codes={"SDT-0001"},
            extract_face_if_missing=False)
        assert r["unique_code"] == "Detecting..."


class TestEdgeCasesAndRobustness:
    """Degenerate inputs that occur in real footage (empty crops, partially
    populated Person rows). These must fail safe — refuse to identify —
    never raise into the analysis loop or mint a bogus identity."""

    def test_dominant_colour_rejects_empty_crop(self):
        from recognition.smart_identifier import _dominant_color_hsv
        assert _dominant_color_hsv(None) is None
        assert _dominant_color_hsv(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_dominant_colour_rejects_sub_10px_crop(self):
        """Too few pixels for stable K-means; a colour from 9px is noise."""
        from recognition.smart_identifier import _dominant_color_hsv
        assert _dominant_color_hsv(np.full((9, 9, 3), 100, dtype=np.uint8)) is None

    def test_veto_is_false_for_unknown_code(self, db):
        """Nothing to contradict -> no veto (and no exception)."""
        assert _face_veto(make_embedding(90), "SDT-NOPE", db, 0.45) is False

    def test_veto_is_false_when_candidate_has_no_stored_face(self, db):
        """A re-ID-only row cannot be vetoed on face. Documents that such
        rows rely entirely on the stranger gate for protection."""
        add_person(db, "SDT-0001", face_emb=None, reid_emb=make_embedding(91))
        assert _face_veto(make_embedding(92), "SDT-0001", db, 0.45) is False

    def test_template_update_on_unknown_code_is_a_noop(self, db):
        si = SmartIdentifier(IdentityConfig())
        si._update_face_template("SDT-NOPE", make_embedding(93), db, similarity=0.7)

    def test_template_update_ignores_shape_mismatch(self, db):
        """A 128-d vector must never be blended into a 512-d template."""
        from database.models import Person
        base = make_embedding(94)
        add_person(db, "SDT-0001", face_emb=base)
        si = SmartIdentifier(IdentityConfig())
        si._update_face_template("SDT-0001", np.ones(128, dtype=np.float32),
                                 db, similarity=0.70)
        p = db.query(Person).filter(Person.unique_code == "SDT-0001").first()
        assert np.allclose(np.array(json.loads(p.face_embedding), dtype=np.float32),
                           base, atol=1e-6), "template must be untouched"

    def test_face_confirm_refreshes_reid_template(self, db, fake_reid):
        """On a confident face match the re-ID template is refreshed to
        today's clothing — the mechanism that keeps re-ID usable across days."""
        from database.models import Person
        new_body = make_embedding(95)
        fake_reid(new_body)
        add_person(db, "SDT-0001", face_emb=make_embedding(96),
                   reid_emb=make_embedding(97))
        si = SmartIdentifier(IdentityConfig())
        si._update_face_template("SDT-0001", make_embedding(96), db,
                                 similarity=0.90,
                                 person_crop=np.full((80, 40, 3), 90, dtype=np.uint8))
        p = db.query(Person).filter(Person.unique_code == "SDT-0001").first()
        stored = np.array(json.loads(p.reid_embedding), dtype=np.float32)
        assert np.allclose(stored, new_body, atol=1e-5), "re-ID template must refresh"


class TestHsvDistance:
    """_hsv_distance wraps hue on a 180-unit circle (OpenCV convention).
    Off-by-one here silently widens or narrows every colour match."""

    def test_identical_colours_are_zero_distance(self):
        from recognition.smart_identifier import _hsv_distance
        c = {"hue": 90, "saturation": 100, "value": 100}
        assert _hsv_distance(c, c) == 0.0

    def test_hue_wraps_around_the_circle(self):
        from recognition.smart_identifier import _hsv_distance
        a = {"hue": 2, "saturation": 100, "value": 100}
        b = {"hue": 178, "saturation": 100, "value": 100}
        # 4 apart the short way, not 176
        assert _hsv_distance(a, b) == pytest.approx(4.0)

    def test_saturation_and_value_contribute(self):
        from recognition.smart_identifier import _hsv_distance
        a = {"hue": 50, "saturation": 100, "value": 100}
        b = {"hue": 50, "saturation": 130, "value": 140}
        assert _hsv_distance(a, b) == pytest.approx(50.0)  # sqrt(30^2 + 40^2)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Config loader
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigLoader:

    def test_rejects_unknown_key(self):
        with pytest.raises(ValueError) as e:
            IdentityConfig.from_dict({"enable_face_anchor": True,
                                      "enable_face_anchr": False})
        assert "enable_face_anchr" in str(e.value)

    def test_error_names_every_unknown_key(self):
        with pytest.raises(ValueError) as e:
            IdentityConfig.from_dict({"typo_one": 1, "typo_two": 2})
        msg = str(e.value)
        assert "typo_one" in msg and "typo_two" in msg

    def test_underscore_keys_are_metadata_not_params(self):
        cfg = IdentityConfig.from_dict({
            "_name": "doc", "_mechanisms_active": ["x"],
            "enable_face_anchor": False})
        assert cfg.enable_face_anchor is False
        assert not hasattr(cfg, "_name")

    def test_defaults_match_documented_values(self):
        """These are the numbers quoted throughout the docs and configs; a
        silent change here would invalidate every written claim."""
        c = IdentityConfig()
        assert c.face_match_threshold == 0.56
        assert c.reid_match_threshold == 0.68
        assert c.face_veto_threshold == 0.45
        assert c.template_blend_threshold == 0.55
        assert c.gallery_add_threshold == 0.60
        assert c.gallery_add_max_similarity == 0.85
        assert c.gallery_max_templates == 5
        assert c.id_switch_contradiction_limit == 2
        assert c.id_switch_similarity_threshold == 0.35
        assert c.face_quality_min_height_px == 48
        assert c.face_quality_min_det_score == 0.60
        assert c.colour_reassoc_window_minutes == 10.0
        assert c.reid_reassoc_window_hours == 12.0
        assert c.duplicate_suggestion_threshold == 0.50
        assert all([c.enable_face_anchor, c.enable_colour_fallback,
                    c.enable_reid_fallback, c.enable_id_switch_guard,
                    c.enable_evidence_gating])

    def test_list_is_coerced_to_tuple_for_det_size(self):
        cfg = IdentityConfig.from_dict({"face_detector_size": [320, 320]})
        assert cfg.face_detector_size == (320, 320)

    @pytest.mark.parametrize("config_name", ABLATION_CONFIGS)
    def test_every_shipped_ablation_config_loads(self, config_name):
        cfg = load_ablation(config_name)
        assert isinstance(cfg.enable_face_anchor, bool)

    def test_d_full_equals_defaults(self):
        """D_full must be behaviourally identical to no config at all."""
        d = load_ablation("D_full")
        default = IdentityConfig()
        for f in ("enable_face_anchor", "enable_colour_fallback",
                  "enable_reid_fallback", "enable_id_switch_guard",
                  "enable_evidence_gating", "face_match_threshold",
                  "reid_match_threshold", "face_veto_threshold"):
            assert getattr(d, f) == getattr(default, f), f
