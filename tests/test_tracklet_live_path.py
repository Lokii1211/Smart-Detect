"""
tests/test_tracklet_live_path.py
─────────────────────────────────
Tracklet voting in the LIVE per-frame loop (cameras/live_stream.py).

The offline runner (eval/run_eval.py) has its own construction of the
pipeline; these tests drive LiveStream._analyze_frame() DIRECTLY — the real
production method — with stubbed detection / tracking / face recognition, so
no camera, no threads and no model weights are involved (conftest stub
mode). This is the same harness pattern used to catch the double-claim bug:
if a future change to the identity path only works in eval/run_eval.py's
copy, these tests catch it.

Properties under test:
  * commitment is DEFERRED — "Detecting..." until k gate-passing faces;
  * the k-th face triggers the commit on its own frame;
  * the buffer is capped at tracklet_buffer_size;
  * a track ending below k commits at track end;
  * the evidence gate is unchanged per frame — faceless frames after commit
    keep the label but write no evidence;
  * retroactive DB backfill writes the buffered frames' sighting rows under
    the committed code (on-screen history stays as displayed);
  * the ID-switch guard invalidating a committed code restarts the buffer;
  * enable_tracklet_voting defaults OFF (no voter, cached path unchanged).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from cameras.live_stream import LiveStream
from config.identity_config import IdentityConfig
from conftest import make_embedding
from database.models import Person, Sighting
from recognition.tracklet_vote import TrackletVoter

DIM = 512


def unit_emb(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


class FakeFace:
    """One InsightFace-style detection: a gate-passing face inside the stub
    person box (centroid in the upper 60% head region)."""
    def __init__(self, emb, box=(40.0, 40.0, 80.0, 200.0), det_score=0.9):
        self.bbox = np.array(box, dtype=np.float32)   # x1, y1, x2, y2
        self.kps = None
        self.det_score = det_score
        self.gender = None
        self.embedding = emb


class StubFaceRec:
    """Returns the configured face list on every extract_embedding() call."""
    def __init__(self, faces):
        self._faces = faces
        self._last_faces = []

    def extract_embedding(self, frame):
        self._last_faces = list(self._faces)


class StubDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, frame):
        return [{"bbox": list(b), "label": "person", "confidence": 0.9}
                for b in self._boxes]


class StubTracker:
    """Fixed (tracker_id, bbox) list per frame — bypasses real ByteTrack."""
    def __init__(self, tracked):
        self._tracked = list(tracked)

    def update_with_detections(self, sv_dets):
        out = sv.Detections(
            xyxy=np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]]
                           for _t, b in self._tracked], dtype=np.float32),
            confidence=np.array([0.9] * len(self._tracked), dtype=np.float32),
            class_id=np.zeros(len(self._tracked), dtype=int),
        )
        out.tracker_id = np.array([t for t, _b in self._tracked], dtype=int)
        return out


class LiveHarness:
    """A real LiveStream with stubbed components, ready to be driven frame
    by frame through the real _analyze_frame()."""

    FW, FH = 640, 400
    BOX = [10, 10, 100, 300]            # person box, head region y <= 190
    FACE_BOX = (40.0, 40.0, 80.0, 200.0)  # centroid (60,140) inside the head

    def __init__(self, cfg, faces_per_frame, tracked_per_frame,
                 boxes_per_frame=None):
        self.stream = LiveStream(source=0, location_id="LOC-T", zone_id="z",
                                 camera_id="CAM-T")
        self.stream._identity_config = cfg
        self.stream._voter = (TrackletVoter(cfg) if cfg.enable_tracklet_voting
                              else None)
        self.stream._detector = StubDetector(
            boxes_per_frame if boxes_per_frame is not None else [self.BOX])
        self.stream._tracker = StubTracker(tracked_per_frame)
        self.stream._face_rec = StubFaceRec(faces_per_frame)
        self.stream._face_scan_size = (self.FW, self.FH)  # identity scaling
        self.stream._identifier = self.stream._identifier  # real SmartIdentifier

    def analyze(self, n_frames=1):
        for _ in range(n_frames):
            self.stream._analyze_frame(
                np.zeros((self.FH, self.FW, 3), dtype=np.uint8))
        return self.stream._latest_results

    def persons(self):
        return self.stream._latest_results["persons"]


@pytest.fixture(autouse=True)
def _isolated_snapshots(tmp_path, monkeypatch):
    """Registration writes snapshots/ relative to cwd — redirect to tmp so
    no real biometric photo is ever touched (test_governance does the same)."""
    monkeypatch.chdir(tmp_path)


def cfg(**kw):
    # min_track_age_for_registration=1: the mechanism tests exercise the
    # voter boundaries (k=1, k=2) which can commit before a track is 3
    # analysis cycles old — the production default of 3 would (correctly)
    # refuse to register those, mirroring the offline resolve's allow_new.
    base = dict(enable_tracklet_voting=True, tracklet_commit_k=3,
                tracklet_buffer_size=10, tracklet_vote_strategy="mean",
                min_track_age_for_registration=1)
    base.update(kw)
    return IdentityConfig(**base)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Deferred commitment in the live loop
# ═══════════════════════════════════════════════════════════════════════════

class TestLiveCommitTiming:

    def test_shows_detecting_until_k_faces_then_commits(self, db):
        emb = unit_emb(7)
        h = LiveHarness(
            cfg(tracklet_commit_k=3),
            faces_per_frame=[FakeFace(emb)],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        h.analyze(1)
        assert h.persons()[0]["code"] == "Detecting..."
        assert h.persons()[0]["method"] == "pending"
        h.analyze(1)
        assert h.persons()[0]["code"] == "Detecting...", \
            "2 faces < k=3 must still be undecided"
        h.analyze(1)
        code, method = h.persons()[0]["code"], h.persons()[0]["method"]
        assert code != "Detecting...", "3rd gate-passing face must commit"
        assert method.startswith("tracklet_mean"), method
        # The committed code is cached and sticks on the next frame.
        h.analyze(1)
        assert h.persons()[0]["code"] == code

    def test_k_boundary_commits_on_the_kth_face(self, db):
        h = LiveHarness(
            cfg(tracklet_commit_k=2),
            faces_per_frame=[FakeFace(unit_emb(8))],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        h.analyze(1)
        assert h.persons()[0]["code"] == "Detecting..."
        h.analyze(1)
        assert h.persons()[0]["code"] != "Detecting...", \
            "k=2 must commit on the 2nd face, not later"

    def test_post_commit_frames_use_the_cached_code(self, db):
        h = LiveHarness(
            cfg(tracklet_commit_k=1),
            faces_per_frame=[FakeFace(unit_emb(9))],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        h.analyze(2)
        code = h.persons()[0]["code"]
        assert code != "Detecting..."
        assert h.persons()[0]["method"].startswith("tracklet_mean")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Buffer cap
# ═══════════════════════════════════════════════════════════════════════════

class TestLiveBufferCap:

    def test_buffer_caps_at_tracklet_buffer_size(self, db):
        # k > cap: the track can never reach k, so it must keep only the
        # newest `tracklet_buffer_size` embeddings and commit at track end.
        h = LiveHarness(
            cfg(tracklet_commit_k=15, tracklet_buffer_size=10),
            faces_per_frame=[FakeFace(unit_emb(10))],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        for _ in range(12):
            h.analyze(1)
        assert len(h.stream._voter.buffered(1)) == 10, \
            "buffer must cap at tracklet_buffer_size even before commit"
        # Track ends (no person next frame) -> commit from the capped buffer.
        h.stream._tracker = StubTracker([])
        h.stream._detector = StubDetector([])
        h.analyze(1)
        assert h.stream._voter.track_ids() == [], "voter state must be freed"
        assert db.query(Person).count() >= 1, \
            "track-end commit must have produced an identity"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Evidence gate interaction + retroactive DB correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestLiveEvidenceAndBackfill:

    def test_faceless_frame_after_commit_keeps_label_but_writes_no_evidence(self, db):
        h = LiveHarness(
            cfg(tracklet_commit_k=1),
            faces_per_frame=[FakeFace(unit_emb(11))],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        h.analyze(1)                       # commit
        code = h.persons()[0]["code"]
        before = db.query(Sighting).count()
        # Face disappears: the box keeps its cached code but the evidence
        # gate must refuse to write a sighting under it.
        h.stream._face_rec = StubFaceRec([])
        h.analyze(1)
        assert h.persons()[0]["code"] == code, \
            "cached code stays on screen without a face"
        assert db.query(Sighting).count() == before, \
            "faceless frame must not write evidence"

    def test_backfill_writes_buffered_frames_under_the_committed_code(self, db):
        h = LiveHarness(
            cfg(tracklet_commit_k=3),
            faces_per_frame=[FakeFace(unit_emb(12))],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        h.analyze(3)                       # commit on frame 3
        person = db.query(Person).first()
        assert person is not None
        rows = (db.query(Sighting)
                .filter(Sighting.person_id == person.id).all())
        # The three buffered frames agree with the committed code at the
        # evidence threshold, so their sighting rows are backfilled (the 30 s
        # write-rate limiter collapses the 3 back-to-back frames to 1 row —
        # exactly what production would have written had the code been known).
        assert len(rows) == 1, \
            f"expected 1 backfilled sighting row, got {len(rows)}"
        assert rows[0].unique_code == person.unique_code
        # The on-screen history is NOT rewritten: pre-commit frames were
        # shown as Detecting... and nothing retro-appears in the live feed.
        assert all(p["code"] == person.unique_code
                   for p in h.stream._latest_results["persons"])

    def test_no_evidence_rows_are_written_for_frames_that_never_commit(self, db):
        h = LiveHarness(
            cfg(tracklet_commit_k=5),
            faces_per_frame=[FakeFace(unit_emb(13))],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        h.analyze(2)                       # below k
        assert db.query(Sighting).count() == 0
        assert db.query(Person).count() == 0, \
            "no identity may be minted below k before track end"


# ═══════════════════════════════════════════════════════════════════════════
# 4. ID-switch guard invalidating a committed code restarts the buffer
# ═══════════════════════════════════════════════════════════════════════════

class TestLiveIdSwitchReset:

    def test_guard_drop_resets_the_voter_and_rebuffers(self, db):
        emb_a, emb_b = unit_emb(21), unit_emb(22)   # near-orthogonal
        h = LiveHarness(
            cfg(tracklet_commit_k=1),
            faces_per_frame=[FakeFace(emb_a)],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        h.analyze(1)
        code_a = h.persons()[0]["code"]
        assert code_a != "Detecting..."
        # First contradiction: strike 1, cache retained.
        h.stream._face_rec = StubFaceRec([FakeFace(emb_b)])
        h.analyze(1)
        assert h.persons()[0]["code"] == code_a
        # Second contradiction: guard drops the cache. With voting on, the
        # stale decision must be reset and the track re-buffer/re-commit.
        h.analyze(1)
        code_b = h.persons()[0]["code"]
        assert code_b != "Detecting..." and code_b != code_a, \
            "guard-dropped track must re-identify, not keep the stale code"
        assert h.stream._voter.committed_code(1) == code_b


# ═══════════════════════════════════════════════════════════════════════════
# 5. Flag default
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultOff:

    def test_voting_off_means_no_voter_and_cached_path_unchanged(self, db):
        c = IdentityConfig()
        assert c.enable_tracklet_voting is False
        h = LiveHarness(
            cfg(enable_tracklet_voting=False),
            faces_per_frame=[FakeFace(unit_emb(30))],
            tracked_per_frame=[(1, LiveHarness.BOX)])
        assert h.stream._voter is None
        h.analyze(3)
        # Without voting the identity commits on the FIRST gate-passing face.
        assert h.persons()[0]["code"] != "Detecting..."
        assert not h.persons()[0]["method"].startswith("tracklet_mean"), \
            "voting off must use the existing cached/identify path"
