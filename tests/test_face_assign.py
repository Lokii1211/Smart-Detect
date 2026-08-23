"""
tests/test_face_assign.py
──────────────────────────
Optimal face-to-person assignment.

ChokePoint contains almost no dense scenes — on 901 labelled frames exactly one
has two person boxes overlapping at IoU > 0.3, and none has that plus two
ground-truth people. So the failure this module fixes cannot be exercised on the
available corpus, and these tests construct the geometry directly instead.

That is not a weaker test. Assignment is pure geometry: a hand-built frame has a
known-correct answer, whereas a real frame only has a plausible one.
"""
from __future__ import annotations

import numpy as np
import pytest

from config.identity_config import IdentityConfig
from recognition.face_assign import (SCALE_HI, SCALE_LO, assign,
                                     build_cost_matrix, match_faces_to_boxes,
                                     pair_cost)


def person(x, y, w=100, h=300):
    return [x, y, w, h]


def face_for(p, scale=0.16, dx=0.0):
    """A well-placed face near the top of person box `p`."""
    px, py, pw, ph = p
    fh = ph * scale
    fw = fh * 0.8
    return [px + pw / 2 - fw / 2 + dx, py + ph * 0.06, fw, fh]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Cost terms
# ═══════════════════════════════════════════════════════════════════════════

class TestPairCost:

    def test_perfect_fit_is_near_zero(self):
        p = person(100, 100)
        assert pair_cost(face_for(p), p) < 0.05

    def test_face_outside_the_box_is_expensive(self):
        p = person(100, 100)
        far = [900, 900, 40, 50]
        assert pair_cost(far, p) >= 1.0

    def test_containment_dominates(self):
        """Half outside costs more than any single soft term can offset."""
        p = person(100, 100)
        good = face_for(p)
        half_out = [p[0] - 40, p[1] + 10, 80, 48]
        assert pair_cost(half_out, p) > pair_cost(good, p) + 0.3

    def test_face_at_the_feet_is_penalised(self):
        """Vertical plausibility: heads are not at the bottom of a person."""
        p = person(100, 100)
        top = face_for(p)
        bottom = [p[0] + 30, p[1] + p[3] * 0.85, 40, 48]
        assert pair_cost(bottom, p) > pair_cost(top, p)

    def test_anatomically_impossible_scale_is_penalised(self):
        p = person(100, 100, 100, 300)
        ok = face_for(p, scale=0.16)
        huge = face_for(p, scale=0.90)       # face 90% of body height
        tiny = face_for(p, scale=0.01)
        assert pair_cost(huge, p) > pair_cost(ok, p)
        assert pair_cost(tiny, p) > pair_cost(ok, p)

    @pytest.mark.parametrize("ratio", [SCALE_LO + 0.01, 0.15, SCALE_HI - 0.01])
    def test_inside_the_anatomical_band_is_unpenalised(self, ratio):
        p = person(100, 100)
        assert pair_cost(face_for(p, scale=ratio), p) < 0.06

    def test_depth_prefers_the_nearer_box(self):
        """Overlapping boxes: the one whose bottom edge is lower is in front."""
        near = person(100, 150, 100, 300)     # x 100-200, bottom 450
        far = person(140, 100, 100, 300)      # x 140-240, bottom 400, behind
        # Fully inside BOTH (overlap is x 140-200), so containment ties at 1.0
        # and depth is what actually decides.
        shared = [155, 170, 40, 48]
        boxes = [near, far]
        assert (pair_cost(shared, near, boxes, 600)
                < pair_cost(shared, far, boxes, 600))

    def test_depth_is_inert_without_overlap(self):
        a = person(0, 100)
        b = person(500, 100)
        f = face_for(a)
        assert pair_cost(f, a, [a, b], 600) == pytest.approx(pair_cost(f, a))


# ═══════════════════════════════════════════════════════════════════════════
# 2. The failure this exists to fix
# ═══════════════════════════════════════════════════════════════════════════

class TestOverlappingBoxes:

    def _scene(self):
        """
        Two overlapping people. A is nearer (lower bottom edge) and its face
        lies fully inside BOTH boxes; B has its own unambiguous face.

            A: x 100-260, y 180-480, bottom 480   <- nearer
            B: x 200-360, y 130-430, bottom 430   <- behind

        fA centroid (230, 240) sits inside both boxes AND inside both upper-60%
        bands, so the greedy rule genuinely has the chance to take it for B.
        Containment ties at 1.0 for both; depth is what makes A correct.
        """
        A = person(100, 180, 160, 300)
        B = person(200, 130, 160, 300)
        fA = [211, 216, 38, 48]               # centroid (230, 240) — in both
        fB = [291, 166, 38, 48]               # centroid (310, 190) — B only
        return A, B, fA, fB

    def test_greedy_can_take_the_wrong_face(self):
        """Pins the motivation: visiting B first lets it claim A's face."""
        A, B, fA, fB = self._scene()
        cfg = IdentityConfig()                # optimal assignment OFF
        got = match_faces_to_boxes([fA, fB], [B, A], cfg, frame_h=600)
        # boxes are [B, A]; fA is face 0. B is visited first and claims it.
        assert got.get(0) == 0, "B claimed A's face — the error optimal fixes"
        # A is now starved rather than double-claiming: the face-consumed fix
        # stops two boxes sharing one face, but it cannot decide WHICH box was
        # right. Only optimal assignment does that.
        assert got.get(1) is None, "A gets nothing — greedy still mis-attributes"
        assert len(set(got.values())) == len(got), "but no face is double-claimed"

    def test_optimal_assignment_gets_it_right(self):
        A, B, fA, fB = self._scene()
        cfg = IdentityConfig(enable_optimal_face_assignment=True)
        got = match_faces_to_boxes([fA, fB], [B, A], cfg, frame_h=600)
        assert got.get(1) == 0, "A (box 1) must get A's face (face 0)"
        assert got.get(0) == 1, "B (box 0) must get B's own face (face 1)"

    def test_assignment_is_one_to_one(self):
        A, B, fA, fB = self._scene()
        got = assign([fA, fB], [A, B], frame_h=600)
        assert len(set(got.values())) == len(got), "each face used at most once"
        assert len(set(got.keys())) == len(got), "each box gets at most one face"

    def test_one_face_two_boxes_goes_to_the_better_explanation(self):
        A, B, fA, _ = self._scene()
        got = assign([fA], [A, B], frame_h=600)
        assert got == {0: 0}, "the single face goes to A (box 0), the nearer box"


# ═══════════════════════════════════════════════════════════════════════════
# 3. The cost ceiling
# ═══════════════════════════════════════════════════════════════════════════

class TestCostCeiling:

    def test_an_orphan_face_is_left_unassigned(self):
        """Refusing to guess is the safe failure."""
        p = person(100, 100)
        orphan = [900, 900, 40, 50]
        assert assign([orphan], [p]) == {}

    def test_a_high_ceiling_would_have_accepted_it(self):
        p = person(100, 100)
        orphan = [900, 900, 40, 50]
        assert assign([orphan], [p], max_cost=99.0) == {0: 0}

    def test_good_faces_survive_the_ceiling(self):
        p = person(100, 100)
        assert assign([face_for(p)], [p]) == {0: 0}

    def test_surplus_faces_do_not_displace_correct_ones(self):
        p = person(100, 100)
        good, orphan = face_for(p), [900, 900, 40, 50]
        got = assign([orphan, good], [p])
        assert got == {0: 1}, "the only box must take the real face (index 1)"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Shape, degenerate input, and the default path
# ═══════════════════════════════════════════════════════════════════════════

class TestMatrixAndEdges:

    def test_matrix_shape(self):
        C = build_cost_matrix([[0, 0, 10, 12]] * 3, [person(0, 0)] * 5)
        assert C.shape == (3, 5)
        assert np.isfinite(C).all()

    @pytest.mark.parametrize("faces,boxes", [([], []), ([[0, 0, 1, 1]], []),
                                             ([], [person(0, 0)])])
    def test_empty_inputs(self, faces, boxes):
        assert assign(faces, boxes) == {}

    def test_zero_height_box_does_not_divide_by_zero(self):
        got = assign([[0, 0, 10, 12]], [[0, 0, 10, 0]])
        assert isinstance(got, dict)

    def test_more_faces_than_boxes(self):
        p = person(100, 100)
        got = assign([face_for(p), face_for(p, dx=5)], [p])
        assert len(got) <= 1

    def test_more_boxes_than_faces(self):
        a, b = person(0, 100), person(400, 100)
        got = assign([face_for(a)], [a, b])
        assert got == {0: 0}, "face 0 -> box 0"


class TestDefaultOff:

    def test_flag_defaults_off(self):
        assert IdentityConfig().enable_optimal_face_assignment is False

    def test_disabled_reproduces_the_upper_60_percent_rule(self):
        cfg = IdentityConfig()
        p = person(100, 100, 100, 300)
        inside_top = [140, 130, 30, 36]                  # 10% down — accepted
        below_60 = [140, 100 + 200, 30, 36]              # 67% down — rejected
        assert match_faces_to_boxes([inside_top], [p], cfg) == {0: 0}
        assert match_faces_to_boxes([below_60], [p], cfg) == {}

    def test_disabled_no_longer_lets_two_boxes_claim_one_face(self):
        """REGRESSION. The shipped rule had no "already taken" guard, so one
        face could be claimed by several boxes at once and identify two people
        simultaneously — with nothing downstream catching it, because the face
        is real and gate-passing. A face is now consumed once."""
        cfg = IdentityConfig()
        a, b = person(0, 100), person(0, 100)            # identical boxes
        f = face_for(a)
        got = match_faces_to_boxes([f], [a, b], cfg)
        assert got == {0: 0}, "the second box must not reuse a consumed face"
        assert len(set(got.values())) == len(got)

    def test_disabled_second_box_falls_through_to_its_own_face(self):
        """The fix must not starve the second box when it has a face of its
        own — it continues the scan rather than stopping."""
        cfg = IdentityConfig()
        a, b = person(0, 100), person(0, 100)
        fa, fb = face_for(a), face_for(a, dx=6)
        assert match_faces_to_boxes([fa, fb], [a, b], cfg) == {0: 0, 1: 1}

    def test_optimal_enforces_one_to_one(self):
        cfg = IdentityConfig(enable_optimal_face_assignment=True)
        a, b = person(0, 100), person(0, 100)
        f = face_for(a)
        got = match_faces_to_boxes([f], [a, b], cfg)
        assert len(got) == 1 and len(set(got.values())) == 1
