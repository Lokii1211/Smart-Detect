"""
recognition/tracklet_vote.py
─────────────────────────────
Tracklet-level identity voting.

THE PROBLEM
───────────
Identity commits on the FIRST frame that clears the quality gate. If that frame
is a poor view — turned head, motion blur, unlucky lighting — the embedding it
mints is a bad anchor, and because arbitration is then cached per track, the
resulting duplicate persists for the whole track. The pose gate mitigates the
worst of this at enrolment; it does not help when the first gate-passing face is
merely mediocre rather than extreme.

THE APPROACH
────────────
Buffer up to N gate-passing embeddings per track with a quality score, and defer
the identity decision until either k faces have accumulated or the track ends.
Then decide once, from all of them.

Two aggregation strategies, both implemented so they can be compared:

  MEAN  quality-weighted mean embedding, L2-renormalised, matched once.
        Averaging suppresses per-frame noise, but a genuinely bimodal buffer
        (an ID-switch mid-track) averages two people into a chimera.

  VOTE  match each embedding independently, then take a quality-weighted vote
        over the resulting codes. Robust to a bimodal buffer — the minority
        view loses — at k times the matching cost.

THE COST, STATED PLAINLY
────────────────────────
Before commit the track shows "Detecting...". A track that never accumulates k
faces and ends early commits at track-end from whatever it has; a track with NO
gate-passing face never commits at all. Both cost coverage on short tracks. That
cost is real and is measured, not hidden.

THE EVIDENCE GATE STILL APPLIES
───────────────────────────────
Committing an identity from aggregated evidence does NOT license writing records
for frames that had no face confirmation. On commit, each buffered frame is
re-tested individually against the committed code; frames without a
gate-passing, agreeing face are labelled but write no evidence.
"""
from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

UNASSIGNED = "Detecting..."


def quality_score(face_box, det_score: float, pose=None, cfg=None,
                  crop=None, person_boxes=None, owner_box=None) -> float:
    """
    Per-face quality weight.

    Uses the learned margin predictor when a model is available (see
    recognition/face_quality.py), otherwise falls back to
    det_score x normalised height — the same two signals the fixed gate uses,
    combined rather than thresholded.
    """
    h = float(face_box[3])
    fallback = float(det_score) * min(1.0, h / 128.0)
    if cfg is None or not getattr(cfg, "enable_learned_quality_gate", False):
        return fallback
    try:
        from recognition.face_quality import build_features, predict_margin
        m = predict_margin(build_features(face_box, det_score, pose, crop,
                                          person_boxes, owner_box),
                           getattr(cfg, "face_quality_model_path",
                                   "models/face_quality.pkl"))
        return fallback if m is None else max(0.0, float(m))
    except Exception:                              # noqa: BLE001
        return fallback


@dataclass
class Observation:
    """One buffered gate-passing face on a track."""
    emb: np.ndarray
    quality: float
    record: Any = None            # scoring record to patch retroactively
    order: int = 0


@dataclass
class _Track:
    buf: Deque[Observation] = field(default_factory=lambda: deque(maxlen=10))
    pending: List[Any] = field(default_factory=list)   # records awaiting a code
    committed: Optional[str] = None
    method: str = "pending"


# ─── Aggregation strategies ──────────────────────────────────────────────────

def aggregate_mean(obs: List[Observation]) -> Optional[np.ndarray]:
    """Quality-weighted mean embedding, L2-renormalised."""
    if not obs:
        return None
    W = np.asarray([max(o.quality, 1e-6) for o in obs], dtype=np.float32)
    M = np.stack([o.emb for o in obs]).astype(np.float32)
    v = (W[:, None] * M).sum(axis=0) / W.sum()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def aggregate_vote(obs: List[Observation],
                   match_fn: Callable[[np.ndarray], Optional[str]]
                   ) -> Tuple[Optional[str], float]:
    """
    Match each embedding, then take a quality-weighted vote over the codes.

    Returns (winning_code_or_None, winning_weight_share). A None result means
    the majority of weight said "nobody" — the track is a stranger and should
    be enrolled rather than matched.
    """
    if not obs:
        return None, 0.0
    tally: Counter = Counter()
    total = 0.0
    for o in obs:
        w = max(o.quality, 1e-6)
        total += w
        try:
            code = match_fn(o.emb)
        except Exception:                          # noqa: BLE001
            code = None
        tally[code] += w
    if not tally:
        return None, 0.0
    code, w = tally.most_common(1)[0]
    return code, (w / total if total > 0 else 0.0)


# ─── The voter ───────────────────────────────────────────────────────────────

class TrackletVoter:
    """
    Per-track buffering and deferred commitment.

    Lifecycle per track:
        observe(...)         -> buffer a gate-passing face, hold the record
        ready(tid)           -> k faces accumulated?
        commit(tid, ...)     -> decide once, patch held records
        end_track(tid, ...)  -> commit from whatever is buffered
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.k = int(getattr(cfg, "tracklet_commit_k", 3))
        self.cap = int(getattr(cfg, "tracklet_buffer_size", 10))
        self.strategy = str(getattr(cfg, "tracklet_vote_strategy", "mean"))
        self._t: Dict[int, _Track] = {}

    def _track(self, tid: int) -> _Track:
        t = self._t.get(tid)
        if t is None:
            t = _Track(buf=deque(maxlen=self.cap))
            self._t[tid] = t
        return t

    def committed_code(self, tid: int) -> Optional[str]:
        t = self._t.get(tid)
        return t.committed if t else None

    def observe(self, tid: int, emb: Optional[np.ndarray], quality: float,
                record: Any = None, order: int = 0) -> None:
        """Buffer a gate-passing face and/or hold a record for later labelling."""
        t = self._track(tid)
        if record is not None and t.committed is None:
            t.pending.append(record)
        if emb is not None:
            t.buf.append(Observation(np.asarray(emb, dtype=np.float32),
                                     float(quality), record, order))

    def ready(self, tid: int) -> bool:
        t = self._t.get(tid)
        return bool(t and t.committed is None and len(t.buf) >= self.k)

    def has_faces(self, tid: int) -> bool:
        t = self._t.get(tid)
        return bool(t and t.buf)

    def buffered(self, tid: int) -> List[Observation]:
        t = self._t.get(tid)
        return list(t.buf) if t else []

    def commit(self, tid: int,
               resolve_mean: Callable[[np.ndarray], Tuple[str, str]],
               match_fn: Optional[Callable[[np.ndarray], Optional[str]]] = None,
               ) -> Tuple[Optional[str], str]:
        """
        Decide this track's identity from its buffer and patch held records.

        resolve_mean(embedding) -> (code, method)
            Full arbitration for one embedding — matching, and enrolment when
            nothing matches. This is the production identify() path; the voter
            never mints an identity itself.

        match_fn(embedding) -> code or None
            Match-only, no enrolment. Required for the VOTE strategy.
        """
        t = self._t.get(tid)
        if t is None or t.committed is not None or not t.buf:
            return (t.committed if t else None), (t.method if t else "pending")

        obs = list(t.buf)
        code: Optional[str] = None
        method = "tracklet_mean"

        if self.strategy == "vote" and match_fn is not None:
            winner, share = aggregate_vote(obs, match_fn)
            if winner is not None:
                code, method = winner, "tracklet_vote"
            else:
                # Majority said "nobody" — enrol from the single best view
                # rather than from an average that no frame actually looked
                # like. Enrolment stays inside production arbitration.
                best = max(obs, key=lambda o: o.quality)
                code, method = resolve_mean(best.emb)
                method = "tracklet_vote_new"
        else:
            v = aggregate_mean(obs)
            if v is not None:
                code, method = resolve_mean(v)
                method = "tracklet_mean" if method != "new_registration" \
                    else "tracklet_mean_new"

        if code and code != UNASSIGNED:
            t.committed, t.method = code, method
            for r in t.pending:
                # Retroactive labelling: frames displayed as "Detecting..."
                # are scored under the code the track ultimately earned.
                try:
                    r.code = code
                    r.method = method
                except Exception:                  # noqa: BLE001
                    pass
            t.pending.clear()
        return t.committed, t.method

    def end_track(self, tid: int, resolve_mean, match_fn=None):
        """Track ended — commit from whatever accumulated, however little."""
        t = self._t.get(tid)
        if t is None or t.committed is not None:
            return None, "pending"
        if not t.buf:
            t.pending.clear()      # never had a face; nothing to commit to
            return None, "pending"
        return self.commit(tid, resolve_mean, match_fn)

    def flush(self, resolve_mean, match_fn=None) -> int:
        """Commit every open track. Returns how many committed."""
        n = 0
        for tid in list(self._t):
            code, _m = self.end_track(tid, resolve_mean, match_fn)
            n += 1 if code else 0
        return n

    def stats(self) -> Dict[str, int]:
        return {
            "tracks": len(self._t),
            "committed": sum(1 for t in self._t.values() if t.committed),
            "uncommitted": sum(1 for t in self._t.values() if not t.committed),
            "never_had_a_face": sum(1 for t in self._t.values()
                                    if not t.buf and not t.committed),
        }

    def track_ids(self) -> List[int]:
        """Tids currently tracked by this voter (used by the live path when
        the camera stops and every open track must be decided)."""
        return list(self._t.keys())

    def reset(self, tid: int) -> None:
        """
        Drop one track's decision AND its buffer.

        Used by the live path when the ID-switch guard invalidates a
        committed code: the decision was wrong, so the track must re-buffer
        from scratch instead of trusting the stale vote.
        """
        self._t.pop(tid, None)

    def forget(self, tid: int) -> None:
        """Release all voter state for a finished track (post-commit cleanup
        in the live path). A later ByteTrack id reuse starts a fresh buffer."""
        self._t.pop(tid, None)
