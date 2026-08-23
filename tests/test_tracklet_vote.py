"""
tests/test_tracklet_vote.py
────────────────────────────
Tracklet-level identity voting.

Two properties matter most:
  * commitment is DEFERRED — before commit the track is "Detecting...", and
    that coverage cost must be real, not quietly bypassed;
  * commitment is RETROACTIVE — frames emitted while undecided are relabelled
    with the code the track ultimately earned.
"""
from __future__ import annotations

import numpy as np
import pytest

from config.identity_config import IdentityConfig
from recognition.tracklet_vote import (UNASSIGNED, Observation, TrackletVoter,
                                       aggregate_mean, aggregate_vote,
                                       quality_score)


def unit(seed, dim=512):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def cfg(**kw):
    base = dict(enable_tracklet_voting=True, tracklet_commit_k=3,
                tracklet_buffer_size=10, tracklet_vote_strategy="mean")
    base.update(kw)
    return IdentityConfig(**base)


class Rec:
    """Stand-in for an Assignment record."""
    def __init__(self):
        self.code, self.method = UNASSIGNED, "pending"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Aggregation
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregateMean:

    def test_returns_a_unit_vector(self):
        obs = [Observation(unit(i), 1.0) for i in range(4)]
        v = aggregate_mean(obs)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)

    def test_identical_embeddings_average_to_themselves(self):
        e = unit(1)
        v = aggregate_mean([Observation(e, 1.0), Observation(e, 3.0)])
        assert float(v @ e) == pytest.approx(1.0, abs=1e-5)

    def test_quality_weighting_pulls_toward_the_better_frame(self):
        good, bad = unit(1), unit(2)
        v = aggregate_mean([Observation(good, 9.0), Observation(bad, 0.1)])
        assert float(v @ good) > float(v @ bad)

    def test_empty(self):
        assert aggregate_mean([]) is None

    def test_zero_quality_does_not_divide_by_zero(self):
        v = aggregate_mean([Observation(unit(1), 0.0), Observation(unit(2), 0.0)])
        assert v is not None and np.isfinite(v).all()


class TestAggregateVote:

    def test_majority_code_wins(self):
        obs = [Observation(unit(i), 1.0) for i in range(3)]
        codes = ["SDT-A", "SDT-A", "SDT-B"]
        code, share = aggregate_vote(obs, lambda e: codes.pop(0))
        assert code == "SDT-A"
        assert share == pytest.approx(2 / 3)

    def test_quality_outweighs_count(self):
        """One confident frame beats two poor ones — the point of weighting."""
        obs = [Observation(unit(0), 10.0), Observation(unit(1), 0.5),
               Observation(unit(2), 0.5)]
        codes = ["SDT-GOOD", "SDT-BAD", "SDT-BAD"]
        code, _ = aggregate_vote(obs, lambda e: codes.pop(0))
        assert code == "SDT-GOOD"

    def test_all_none_returns_none(self):
        """Everyone says stranger — the caller must enrol, not match."""
        obs = [Observation(unit(i), 1.0) for i in range(3)]
        code, _ = aggregate_vote(obs, lambda e: None)
        assert code is None

    def test_a_raising_matcher_counts_as_no_match(self):
        obs = [Observation(unit(i), 1.0) for i in range(2)]
        def boom(e):
            raise RuntimeError("db down")
        code, _ = aggregate_vote(obs, boom)
        assert code is None

    def test_bimodal_buffer_picks_the_majority_not_a_chimera(self):
        """The case MEAN handles badly: a mid-track ID switch."""
        obs = [Observation(unit(i), 1.0) for i in range(5)]
        codes = ["SDT-A", "SDT-A", "SDT-A", "SDT-B", "SDT-B"]
        code, share = aggregate_vote(obs, lambda e: codes.pop(0))
        assert code == "SDT-A" and share == pytest.approx(0.6)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Deferred commitment
# ═══════════════════════════════════════════════════════════════════════════

class TestCommitPolicy:

    def _resolve(self, code="SDT-0001", method="face"):
        return lambda e: (code, method)

    def test_not_ready_before_k(self):
        v = TrackletVoter(cfg(tracklet_commit_k=3))
        for i in range(2):
            v.observe(1, unit(i), 1.0, Rec())
            assert not v.ready(1)
        v.observe(1, unit(9), 1.0, Rec())
        assert v.ready(1)

    def test_commit_returns_the_resolved_code(self):
        v = TrackletVoter(cfg())
        for i in range(3):
            v.observe(1, unit(i), 1.0, Rec())
        code, method = v.commit(1, self._resolve())
        assert code == "SDT-0001"
        assert method.startswith("tracklet")

    def test_committing_twice_is_idempotent(self):
        v = TrackletVoter(cfg())
        for i in range(3):
            v.observe(1, unit(i), 1.0, Rec())
        v.commit(1, self._resolve("SDT-0001"))
        code, _ = v.commit(1, self._resolve("SDT-9999"))
        assert code == "SDT-0001", "a committed track must not be re-decided"

    def test_track_end_commits_below_k(self):
        """A short track still gets its one decision, from what it has."""
        v = TrackletVoter(cfg(tracklet_commit_k=5))
        v.observe(1, unit(1), 1.0, Rec())
        assert not v.ready(1)
        code, _ = v.end_track(1, self._resolve())
        assert code == "SDT-0001"

    def test_track_with_no_face_never_commits(self):
        """THE COVERAGE COST. No gate-passing face, no identity — ever."""
        v = TrackletVoter(cfg())
        r = Rec()
        v.observe(1, None, 0.0, r)
        assert not v.has_faces(1)
        code, _ = v.end_track(1, self._resolve())
        assert code is None
        assert r.code == UNASSIGNED, "must stay unassigned, not be guessed at"

    def test_buffer_is_capped(self):
        v = TrackletVoter(cfg(tracklet_buffer_size=10))
        for i in range(25):
            v.observe(1, unit(i), 1.0, Rec())
        assert len(v.buffered(1)) == 10

    def test_flush_commits_every_open_track(self):
        v = TrackletVoter(cfg())
        for tid in (1, 2, 3):
            v.observe(tid, unit(tid), 1.0, Rec())
        assert v.flush(self._resolve()) == 3
        assert v.stats()["uncommitted"] == 0

    def test_stats_counts_faceless_tracks(self):
        v = TrackletVoter(cfg())
        v.observe(1, unit(1), 1.0, Rec())
        v.observe(2, None, 0.0, Rec())
        v.flush(self._resolve())
        assert v.stats()["never_had_a_face"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Retroactive labelling
# ═══════════════════════════════════════════════════════════════════════════

class TestRetroactiveLabelling:

    def test_buffered_frames_are_relabelled_on_commit(self):
        v = TrackletVoter(cfg(tracklet_commit_k=3))
        recs = [Rec() for _ in range(3)]
        for i, r in enumerate(recs):
            v.observe(1, unit(i), 1.0, r)
            assert r.code == UNASSIGNED, "undecided frames read as Detecting..."
        v.commit(1, lambda e: ("SDT-0007", "face"))
        assert all(r.code == "SDT-0007" for r in recs), (
            "every frame held while undecided must be relabelled")

    def test_faceless_frames_on_a_committing_track_are_still_relabelled(self):
        """A frame with no face still belongs to the track and is scored under
        the track's code — it simply contributes no evidence."""
        v = TrackletVoter(cfg(tracklet_commit_k=2))
        faceless, withface = Rec(), Rec()
        v.observe(1, None, 0.0, faceless)
        v.observe(1, unit(1), 1.0, withface)
        v.observe(1, unit(2), 1.0, Rec())
        v.commit(1, lambda e: ("SDT-0003", "face"))
        assert faceless.code == "SDT-0003"
        assert withface.code == "SDT-0003"

    def test_frames_after_commit_are_not_held(self):
        v = TrackletVoter(cfg(tracklet_commit_k=1))
        v.observe(1, unit(0), 1.0, Rec())
        v.commit(1, lambda e: ("SDT-0001", "face"))
        later = Rec()
        v.observe(1, unit(5), 1.0, later)
        assert v.committed_code(1) == "SDT-0001"
        assert later.code == UNASSIGNED, "post-commit records are the caller's job"

    def test_no_relabelling_when_nothing_commits(self):
        v = TrackletVoter(cfg())
        r = Rec()
        v.observe(1, None, 0.0, r)
        v.flush(lambda e: ("SDT-0001", "face"))
        assert r.code == UNASSIGNED


# ═══════════════════════════════════════════════════════════════════════════
# 4. Quality scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestQualityScore:

    def test_fallback_is_det_score_times_normalised_height(self):
        c = IdentityConfig()          # learned gate off
        assert quality_score([0, 0, 40, 64], 0.8, cfg=c) == pytest.approx(0.8 * 0.5)

    def test_normalised_height_saturates(self):
        c = IdentityConfig()
        assert quality_score([0, 0, 40, 400], 1.0, cfg=c) == pytest.approx(1.0)

    def test_bigger_and_more_confident_scores_higher(self):
        c = IdentityConfig()
        assert (quality_score([0, 0, 60, 120], 0.95, cfg=c)
                > quality_score([0, 0, 30, 50], 0.65, cfg=c))

    def test_no_cfg_uses_the_fallback(self):
        assert quality_score([0, 0, 40, 64], 0.5) == pytest.approx(0.25)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Disabled by default
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultOff:

    def test_flag_defaults_off(self):
        assert IdentityConfig().enable_tracklet_voting is False

    def test_defaults_are_the_documented_values(self):
        c = IdentityConfig()
        assert c.tracklet_commit_k == 3
        assert c.tracklet_buffer_size == 10
        assert c.tracklet_vote_strategy == "mean"
