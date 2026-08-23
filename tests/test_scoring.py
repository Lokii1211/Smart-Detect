"""
tests/test_scoring.py
──────────────────────
Unit tests for eval/scoring.py — the module every reported number comes from.

It had 0% coverage while producing the results table of a paper. These tests
are hand-computed: each expectation is derived on paper from the definitions in
eval/METRICS.md and written as a literal, so a change in scoring.py that alters
a metric fails here rather than silently changing a published figure.

Wilson values are cross-checked against published 95% intervals, not against
the implementation.
"""
from __future__ import annotations

import math

import pytest

from eval.scoring import (CONTAMINATED, CORRECT, NO_ASSIGN, UNASSIGNED,
                          Assignment, buckets, classify, compute_all,
                          cross_camera_reassociation, duplicate_identities,
                          evidence_precision, fmt_ci, fragmentation,
                          id_switches, identity_coverage, identity_precision,
                          identity_purity, wilson)


def rec(gt, code, *, order, cam="cam1", tid=None, method="face", ev=False,
        frame=None):
    return Assignment(
        frame_id=frame or f"f{order}", camera_id=cam, seq_id=cam,
        gt_person=gt, code=code, tracker_id=tid, method=method,
        order=order, evidence_written=ev)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Wilson score interval — against published values
# ═══════════════════════════════════════════════════════════════════════════

class TestWilsonPublishedValues:
    """95% Wilson intervals for n=10 are standard textbook values."""

    @pytest.mark.parametrize("k,n,lo,hi", [
        (0,  10, 0.0000, 0.2775),
        (1,  10, 0.0179, 0.4042),
        (5,  10, 0.2366, 0.7634),
        (10, 10, 0.7225, 1.0000),
    ])
    def test_matches_published_interval(self, k, n, lo, hi):
        ci = wilson(k, n)
        assert ci["value"] == pytest.approx(k / n)
        assert ci["lo"] == pytest.approx(lo, abs=5e-4)
        assert ci["hi"] == pytest.approx(hi, abs=5e-4)

    @pytest.mark.parametrize("k,n,lo,hi", [
        (0, 1, 0.0000, 0.7935),
        (1, 1, 0.2065, 1.0000),
    ])
    def test_n_equals_one(self, k, n, lo, hi):
        """n=1 is where Wald degenerates; Wilson must stay informative."""
        ci = wilson(k, n)
        assert ci["lo"] == pytest.approx(lo, abs=5e-4)
        assert ci["hi"] == pytest.approx(hi, abs=5e-4)


class TestWilsonBoundaries:

    def test_p_equals_one_has_nonzero_width(self):
        """The reason Wilson was chosen over Wald: at p=1 Wald gives a
        zero-width interval, which is nonsense at n=30."""
        ci = wilson(30, 30)
        assert ci["value"] == 1.0
        assert ci["hi"] == 1.0
        assert ci["lo"] < 1.0
        assert ci["hi"] - ci["lo"] > 0.05

    def test_p_equals_one_hi_is_exactly_one(self):
        """Documented float-rounding clamp: 30/30 produced hi=0.9999999999999999."""
        for n in (1, 5, 30, 100, 1630):
            assert wilson(n, n)["hi"] == 1.0, n

    def test_p_equals_zero_lo_is_exactly_zero(self):
        for n in (1, 5, 30, 100):
            assert wilson(0, n)["lo"] == 0.0, n

    def test_interval_always_contains_the_point_estimate(self):
        for n in (1, 2, 7, 30, 100, 1630):
            for k in range(n + 1):
                ci = wilson(k, n)
                assert ci["lo"] <= ci["value"] <= ci["hi"], (k, n)
                assert 0.0 <= ci["lo"] and ci["hi"] <= 1.0, (k, n)

    def test_n_zero_is_undefined_not_zero(self):
        """A proportion with no denominator is undefined. Reporting 0.0 would
        make an unmeasured metric look like a failed one."""
        ci = wilson(0, 0)
        assert ci == {"value": None, "lo": None, "hi": None,
                      "half_width": None, "n": 0}

    def test_interval_narrows_as_n_grows(self):
        widths = [wilson(int(0.5 * n), n)["hi"] - wilson(int(0.5 * n), n)["lo"]
                  for n in (10, 100, 1000, 10000)]
        assert widths == sorted(widths, reverse=True)

    def test_half_width_near_one_point_four_pp_at_n_1000(self):
        """MIN_FRAMES=1000 is justified in validate_dataset.py by this claim."""
        ci = wilson(950, 1000)
        assert 100 * ci["half_width"] == pytest.approx(1.35, abs=0.15)


class TestFmtCi:

    def test_none_renders_as_na(self):
        assert fmt_ci(None) == "n/a"
        assert fmt_ci(wilson(0, 0)) == "n/a"

    def test_percent_form(self):
        assert fmt_ci(wilson(5, 10)) == "50.0% [23.7-76.3]"


# ═══════════════════════════════════════════════════════════════════════════
# 2. The bucket partition — hand-built corpus
# ═══════════════════════════════════════════════════════════════════════════
#
#   code X: A,A,A,B   -> maj(X) = A   (X spans TWO people)
#   code Y: A         -> maj(Y) = A   (person A holds TWO codes)
#   person C: never assigned (one detector miss, one gate refusal)
#
#   n_total       = 7
#   CORRECT       = 4  (3x X/A, 1x Y/A)
#   CONTAMINATED  = 1  (X/B)
#   UNASSIGNED    = 2
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def corpus():
    return [
        rec("A", "SDT-X", order=0, tid=1, ev=True),
        rec("A", "SDT-X", order=1, tid=1, ev=True),
        rec("A", "SDT-X", order=2, tid=1, ev=False),
        rec("B", "SDT-X", order=3, tid=2, ev=True),      # contaminated + evidenced
        rec("A", "SDT-Y", order=4, tid=3, ev=True),      # A's duplicate identity
        rec("C", UNASSIGNED, order=5, tid=4, method="no_detection"),
        rec("C", UNASSIGNED, order=6, tid=None, method="pending"),
    ]


class TestBuckets:

    def test_counts(self, corpus):
        b = buckets(corpus)
        assert b["n_total"] == 7
        assert b["n_correct"] == 4
        assert b["n_contaminated"] == 1
        assert b["n_unassigned"] == 2

    def test_fractions_sum_to_exactly_one(self, corpus):
        b = buckets(corpus)
        s = b["frac_correct"] + b["frac_contaminated"] + b["frac_unassigned"]
        assert s == pytest.approx(1.0, abs=1e-12)
        assert b["frac_correct"] == pytest.approx(4 / 7)
        assert b["frac_contaminated"] == pytest.approx(1 / 7)
        assert b["frac_unassigned"] == pytest.approx(2 / 7)

    def test_unassigned_split_by_cause(self, corpus):
        """A detector miss and a quality-gate refusal are different problems."""
        b = buckets(corpus)
        assert b["n_unassigned_no_detection"] == 1
        assert b["n_unassigned_detecting"] == 1

    def test_majority_decides_meaning_of_a_code(self, corpus):
        """X was assigned to A three times and B once, so X *means* A and the
        B record is the error — not the other way round."""
        cls = classify(corpus)
        assert cls["3:f3:B"] == CONTAMINATED
        assert cls["0:f0:A"] == CORRECT
        assert cls["5:f5:C"] == NO_ASSIGN

    def test_empty_input(self):
        b = buckets([])
        assert b["n_total"] == 0
        assert b["frac_correct"] is None

    def test_tie_breaks_to_first_encountered(self):
        """Documented behaviour: Counter.most_common is insertion-stable."""
        recs = [rec("A", "SDT-Z", order=0), rec("B", "SDT-Z", order=1)]
        b = buckets(recs)
        assert b["n_correct"] == 1 and b["n_contaminated"] == 1


class TestPartitionAssertionFires:

    def test_broken_classifier_aborts_rather_than_reporting(self, corpus, monkeypatch):
        """The invariant must be enforced, not merely documented: a partition
        that does not sum has to raise instead of emitting a plausible table."""
        import eval.scoring as sc
        good = sc.classify(corpus)
        broken = dict(good)
        broken.pop("3:f3:B")                     # lose one record
        monkeypatch.setattr(sc, "classify", lambda r: broken)
        with pytest.raises(AssertionError, match="not exhaustive"):
            sc.buckets(corpus)

    def test_compute_all_reasserts_at_top_level(self, corpus, monkeypatch):
        import eval.scoring as sc
        bad = dict(sc.buckets(corpus))
        bad["n_correct"] = 99
        monkeypatch.setattr(sc, "buckets", lambda r: bad)
        with pytest.raises(AssertionError, match="partition broken"):
            sc.compute_all(corpus, {"n_frames": 1})


# ═══════════════════════════════════════════════════════════════════════════
# 3. Headline metrics — hand-computed
# ═══════════════════════════════════════════════════════════════════════════

class TestHeadlineMetrics:

    def test_identity_precision_is_4_of_5(self, corpus):
        m = identity_precision(corpus)
        assert m["n_correct"] == 4 and m["n_assigned"] == 5
        assert m["value"] == pytest.approx(0.8)

    def test_coverage_is_5_of_7(self, corpus):
        m = identity_coverage(corpus)
        assert m["n_assigned"] == 5 and m["n_total"] == 7
        assert m["value"] == pytest.approx(5 / 7)

    def test_precision_and_coverage_have_different_denominators(self, corpus):
        """The whole argument of METRICS.md §1-2: they are not two views of
        one number and must never be blended."""
        assert identity_precision(corpus)["value"] != identity_coverage(corpus)["value"]

    def test_evidence_precision_scores_only_written_rows(self, corpus):
        """4 rows written (3 correct, 1 wrong) out of 7 person-frames."""
        m = evidence_precision(corpus)
        assert m["n_written"] == 4
        assert m["n_correct"] == 3
        assert m["n_wrong"] == 1
        assert m["value"] == pytest.approx(0.75)
        assert m["evidence_yield"] == pytest.approx(4 / 7)

    def test_evidence_precision_differs_from_identity_precision(self, corpus):
        """The only metric that can separate ablation configs C and D."""
        assert evidence_precision(corpus)["value"] == pytest.approx(0.75)
        assert identity_precision(corpus)["value"] == pytest.approx(0.80)

    def test_gate_writing_nothing_is_undefined_not_perfect(self):
        recs = [rec("A", "SDT-X", order=0), rec("B", "SDT-Y", order=1)]
        m = evidence_precision(recs)
        assert m["n_written"] == 0
        assert m["value"] is None, "0/0 must be undefined, never 100%"


class TestStructuralMetrics:

    def test_purity_one_of_two_codes_is_pure(self, corpus):
        """X spans {A,B} -> impure; Y spans {A} -> pure."""
        m = identity_purity(corpus)
        assert m["n_codes"] == 2
        assert m["value"] == pytest.approx(0.5)
        assert [c["code"] for c in m["impure_codes"]] == ["SDT-X"]
        assert m["impure_codes"][0]["gt_ids"] == ["A", "B"]

    def test_fragmentation_is_mean_codes_per_person(self, corpus):
        """A holds {X,Y}=2, B holds {X}=1, C excluded (never assigned)."""
        m = fragmentation(corpus)
        assert m["n_people"] == 2
        assert m["value"] == pytest.approx(1.5)
        assert m["per_person"] == {"A": 2, "B": 1}

    def test_duplicates_count_only_majority_owned_codes(self, corpus):
        """B's single stray X frame is contamination, not a duplicate identity:
        B owns no code, so B contributes nothing here. This separation is the
        reason duplicate_identities exists alongside fragmentation."""
        m = duplicate_identities(corpus)
        assert m["codes_owned"] == {"A": ["SDT-X", "SDT-Y"]}
        assert m["per_person"] == {"A": 1}
        assert m["total_duplicates"] == 1
        assert m["n_people"] == 1
        assert m["value"] == pytest.approx(1.0)

    def test_fragmentation_and_duplicates_disagree_by_design(self, corpus):
        """1.5 vs 1.0 on the same records — reading either alone misleads."""
        assert fragmentation(corpus)["value"] == pytest.approx(1.5)
        assert duplicate_identities(corpus)["value"] == pytest.approx(1.0)

    def test_pure_system_has_zero_duplicates(self):
        recs = [rec("A", "SDT-1", order=0), rec("A", "SDT-1", order=1),
                rec("B", "SDT-2", order=2)]
        assert duplicate_identities(recs)["value"] == pytest.approx(0.0)
        assert identity_purity(recs)["value"] == pytest.approx(1.0)
        assert fragmentation(recs)["value"] == pytest.approx(1.0)

    def test_one_code_per_frame_degenerate_case(self):
        """Perfect precision by construction, terrible fragmentation — the
        failure mode METRICS.md warns fragmentation must catch."""
        recs = [rec("A", f"SDT-{i}", order=i) for i in range(5)]
        assert identity_precision(recs)["value"] == pytest.approx(1.0)
        assert fragmentation(recs)["value"] == pytest.approx(5.0)
        assert duplicate_identities(recs)["value"] == pytest.approx(4.0)


class TestIdSwitches:

    def test_counts_code_changes_within_a_track(self):
        recs = [rec("A", "SDT-X", order=0, tid=7), rec("A", "SDT-X", order=1, tid=7),
                rec("A", "SDT-Y", order=2, tid=7), rec("A", "SDT-Y", order=3, tid=7)]
        assert id_switches(recs)["value"] == 1

    def test_transitions_through_unassigned_are_not_switches(self):
        """Coverage, not a switch."""
        recs = [rec("A", "SDT-X", order=0, tid=8),
                rec("A", UNASSIGNED, order=1, tid=8),
                rec("A", "SDT-X", order=2, tid=8)]
        assert id_switches(recs)["value"] == 0

    def test_untracked_records_excluded(self):
        recs = [rec("A", "SDT-X", order=0, tid=None),
                rec("A", "SDT-Y", order=1, tid=None)]
        assert id_switches(recs)["value"] == 0
        assert id_switches(recs)["n_tracks"] == 0

    def test_ordering_uses_order_field_not_list_position(self):
        recs = [rec("A", "SDT-Y", order=5, tid=9), rec("A", "SDT-X", order=1, tid=9)]
        assert id_switches(recs)["value"] == 1


class TestCrossCamera:

    def test_undefined_when_nobody_crosses(self, corpus):
        """n/a, never 0 — a single-camera corpus cannot fail this test."""
        m = cross_camera_reassociation(corpus)
        assert m["value"] is None
        assert m["n_pairs"] == 0

    def test_hit_and_miss(self):
        recs = [
            rec("P", "SDT-1", order=0, cam="c1"), rec("P", "SDT-1", order=1, cam="c1"),
            rec("P", "SDT-1", order=2, cam="c2"),          # kept the code -> hit
            rec("Q", "SDT-2", order=3, cam="c1"),
            rec("Q", "SDT-3", order=4, cam="c2"),          # changed code -> miss
        ]
        m = cross_camera_reassociation(recs)
        assert m["n_pairs"] == 2
        assert m["value"] == pytest.approx(0.5)

    def test_three_cameras_give_two_pairs(self):
        recs = [rec("P", "SDT-1", order=i, cam=c)
                for i, c in enumerate(["c1", "c2", "c3"])]
        assert cross_camera_reassociation(recs)["n_pairs"] == 2


class TestComputeAll:

    def test_assembles_every_metric(self, corpus):
        out = compute_all(corpus, {"n_frames": 7, "mean_frame_latency_ms": 1.0})
        for key in ("buckets", "identity_precision", "identity_coverage",
                    "evidence_precision", "identity_purity", "fragmentation",
                    "duplicate_identities", "id_switches",
                    "cross_camera_reassoc", "runtime", "n_records"):
            assert key in out, key
        assert out["n_records"] == 7
        assert out["identity_precision"]["value"] == pytest.approx(0.8)
