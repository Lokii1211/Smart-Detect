"""
config/identity_config.py
──────────────────────────
Single source of truth for every identity-arbitration parameter and feature
flag used by recognition/smart_identifier.py, recognition/face_recognizer.py,
cameras/live_stream.py, and the duplicate-suggestions endpoint in
backend/main.py.

Every default below is IDENTICAL to the value that was previously hardcoded
in those files. Importing this module and doing nothing else changes no
behavior — SMARTDETECT_IDENTITY_CONFIG must be set to a JSON file for
anything to differ from the shipped defaults.

Ablation configs: see config/ablation/*.json.

Usage:
    from config.identity_config import get_identity_config
    cfg = get_identity_config()
    if cfg.enable_face_anchor: ...
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IdentityConfig:
    # ── Method 1: Face match (recognition/smart_identifier.py) ─────────────
    face_match_threshold: float = 0.56
    # ArcFace cosine sim required to match an existing person by face.
    # History: 0.35 cross-matched strangers; 0.50 still cross-matched two
    # pairs of different people on a 4-video eval set.

    # ── Method 2: Dress colour fallback ─────────────────────────────────────
    colour_match_threshold: float = 30.0
    # HSV Euclidean-distance ceiling for a dress-colour match (lower = stricter).
    colour_reassoc_window_minutes: float = 10.0
    # Colour may only re-associate someone seen this recently.

    # ── Method 3: Body re-ID fallback ───────────────────────────────────────
    reid_match_threshold: float = 0.68
    # OSNet cosine sim — 0.60 cross-matched different people in similar clothing.
    reid_reassoc_window_hours: float = 12.0
    # Body features track clothing — trust them across hours, not days.
    # (converted to minutes at the point of use: hours * 60 == 720.0,
    # identical to the previously-hardcoded REID_RECENT_MINUTES)

    # NOTE: "Method 4 — multi-feature fusion" was documented in the original
    # design and never implemented; its `multi_feature_threshold` field was
    # inert and has been REMOVED (2026-08-02). It is not planned: a weighted
    # combination of colour + body re-ID would let non-face evidence assert an
    # identity, which is precisely what the face-anchoring work removed. If
    # multi-signal fusion returns it must be face-gated by design, not a
    # score blend. See docs/CROSS_CAMERA_IDENTITY.md §4 for the direction
    # actually being pursued (tracklet-level voting).

    # ── Face veto: protects colour/re-ID from wrong-person hijack ──────────
    face_veto_threshold: float = 0.45
    # If the query face's similarity to a colour/re-ID candidate's stored
    # face is below this, the candidate match is rejected outright.
    # Gated by enable_face_anchor (see below).

    # ── Template / gallery maintenance ──────────────────────────────────────
    template_blend_threshold: float = 0.55
    # Only face matches at or above this similarity get blended into the
    # stored template; below it the match still counts as an ID, but the
    # template is left untouched (a weak match would drag it toward a
    # lookalike).
    template_blend_weight: float = 0.20
    # Weight of the NEW embedding in the running-average template blend.
    gallery_add_threshold: float = 0.60
    # Minimum similarity for a matched view to be added as a new gallery
    # template — below this it isn't confident enough to trust as "this
    # person, different angle".
    gallery_add_max_similarity: float = 0.85
    # Maximum similarity for a matched view to be worth adding — above this
    # it's near-identical to what's already stored, so it's skipped.
    gallery_max_templates: int = 5

    # ── ByteTrack ID-switch guard (cameras/live_stream.py) ──────────────────
    id_switch_similarity_threshold: float = 0.35
    # A gate-passing face scoring below this against a track's cached code's
    # stored face counts as one "contradiction".
    id_switch_contradiction_limit: int = 2
    # Consecutive contradictions before the cached code is dropped and the
    # track is forced to re-identify from scratch. Gated by
    # enable_id_switch_guard.

    # ── Face quality gate for identity decisions ────────────────────────────
    face_quality_min_height_px: int = 48
    # Faces shorter than this (in the original frame, after scale-back) are
    # drawn but never allowed to decide identity.
    face_quality_min_det_score: float = 0.60
    # InsightFace per-face detection-confidence floor for identity use —
    # downstream of face_detector_min_score below.

    # ── Evidence gating (cameras/live_stream.py) ────────────────────────────
    evidence_face_sim_threshold: float = 0.45
    # Minimum face similarity to a code's stored face required before a
    # sighting row / photo is logged as evidence under that code. Gated by
    # enable_evidence_gating.

    # ── Registration pose gate (recognition/face_pose.py) ───────────────────
    # A code minted from a profile or bowed-head frame embeds a viewpoint that
    # matches poorly against later frontal views of the SAME person, producing
    # a duplicate identity. Observed: one woman registered twice (frontal +
    # head-bowed) from a single continuous recording.
    #
    # Applies ONLY to minting, never to matching — refusing to match a profile
    # would lose recall; refusing to ENROL from one merely waits for a better
    # frame, and tracks last many frames.
    #
    # Thresholds are landmark RATIOS, not degrees (see face_pose.py), chosen
    # from a measured 138-sample distribution on a head-pose sweep:
    #     yaw    p05 -0.416  median -0.007  p95 0.479
    #     pitch  p05  0.398  median  0.495  p95 0.744
    # so the bounds below sit just inside the observed 5th/95th percentiles.
    enable_pose_gate_at_registration: bool = True
    registration_max_abs_yaw:      float = 0.35   # |yaw| above this = turned away
    registration_min_pitch:        float = 0.35   # below = chin raised
    registration_max_pitch:        float = 0.70   # above = head bowed
    registration_max_abs_roll_deg: float = 35.0   # head tilted sideways
    # SAFETY VALVE. The gate defers enrolment expecting a better frame to
    # arrive on the same track. On sparsely-sampled input (or a person who is
    # simply never frontal) that frame may never come, and the gate would
    # starve registration entirely — measured: 6.7pp coverage loss with no
    # duplicate reduction on a 10s-interval dataset. After this many
    # consecutive pose deferrals on one track, enrol from the best frame
    # available rather than losing the person. 0 disables the valve.
    pose_gate_max_deferrals: int = 5

    # ── Registration gate ────────────────────────────────────────────────────
    min_track_age_for_registration: int = 3
    # Analysis cycles a track must survive before a brand-new SDT code may
    # be minted from it — avoids one ghost row per flicker.

    # ── Face detector (InsightFace, recognition/face_recognizer.py) ────────
    face_detector_size: Tuple[int, int] = (640, 640)
    # InsightFace det_size. 320 halved detection recall on sub-50px faces
    # (measured on face-demographics-walking.mp4); 640 is InsightFace's own
    # library default.
    face_detector_min_score: float = 0.35
    # InsightFace's internal det_thresh — the pre-filter a face must clear
    # to be returned by the detector at all, upstream of
    # face_quality_min_det_score above.

    # ── Head-zoom second detection pass — recall booster, not arbitration ──
    zoom_min_person_height_px: int = 140
    # Person boxes shorter than this cannot hold a gate-passing face — the
    # zoom pass skips them.
    zoom_max_crops_per_frame: int = 4
    # Cap on how many faceless person boxes get a zoom re-scan per frame
    # (CPU budget) — nearest/tallest first.
    zoom_target_width_px: float = 384.0
    # Head crops are upscaled toward roughly this pixel width before re-scan.
    zoom_max_factor: float = 4.0
    # Upper bound on the digital zoom applied.

    # ── Score calibration (recognition/score_calibration.py) ───────────────
    # Thresholds may be given as a raw cosine OR as a target false-match rate.
    # A target FMR, when set and resolvable from the calibration, overrides the
    # raw cosine; otherwise the raw value stands, so a missing or
    # under-resolved calibration can never silently loosen the gate.
    #
    # FMR here is 1:N (max over the gallery), because that is what
    # find_person_by_embedding computes. It therefore grows with gallery size:
    # a threshold calibrated on 25 identities is not automatically safe at
    # 1000. Re-calibrate when the population changes materially.
    face_match_target_fmr: Optional[float] = None
    score_calibration_path: str = "models/score_calibration.pkl"

    # ── Optimal face-to-person assignment (recognition/face_assign.py) ─────
    # The shipped rule is greedy and local: each person box takes the first
    # face whose centroid falls in its upper 60%. Where two boxes overlap, one
    # face can satisfy both and whichever box is visited first wins. The face
    # is real and gate-passing, so the face veto, ID-switch guard and evidence
    # gate all see legitimate evidence and none of them objects.
    #
    # The alternative scores every (face, box) pair on containment, vertical
    # position, anatomical scale and a depth proxy, then solves the frame as a
    # one-to-one linear assignment.
    enable_optimal_face_assignment: bool = False
    face_assign_max_cost: float = 0.60
    # Assignments above this are REJECTED and the face is left unattached.
    # Asymmetric on purpose: an unattached face costs one frame of coverage; a
    # misattached one identifies the wrong human and nothing downstream catches
    # it.

    # ── Tracklet-level identity voting (recognition/tracklet_vote.py) ──────
    # Identity otherwise commits on the FIRST gate-passing frame, so one poor
    # enrolment view produces a duplicate that lasts the track's lifetime.
    # Voting buffers gate-passing faces and decides once, from all of them.
    #
    # The cost is coverage: before commit the track shows "Detecting...", so
    # short tracks that never reach k faces are labelled late or not at all.
    enable_tracklet_voting: bool = False
    tracklet_vote_strategy: str = "mean"     # "mean" | "vote"
    # mean : quality-weighted mean embedding, matched once. Cheap; a bimodal
    #        buffer (mid-track ID switch) averages two people into a chimera.
    # vote : match each embedding, weighted vote over codes. Robust to a
    #        bimodal buffer, at k times the matching cost.
    tracklet_buffer_size: int = 10
    tracklet_commit_k: int = 3
    # Commit when k gate-passing faces have accumulated OR the track ends.

    # ── Learned face-quality gate (recognition/face_quality.py) ────────────
    # face_quality_min_height_px / _min_det_score above are two hand-set
    # constants standing in for embedding reliability. The learned gate
    # predicts that quantity directly — the discriminative margin, genuine
    # similarity minus best-impostor similarity — and admits above a threshold.
    #
    # OFF by default. When off, gate_passes() runs the identical two-constant
    # test and nothing in face_quality.py is touched.
    enable_learned_quality_gate: bool = False
    face_quality_model_path: str = "models/face_quality.pkl"
    learned_quality_min_margin: float = 0.35
    # Predicted margin required to admit a face. 0.0 is the point at which the
    # embedding is as close to a stranger as to its owner; 0.35 keeps headroom.

    # ── Sliced (tiled) inference — DETECTION ONLY ───────────────────────────
    # Downscaling to the detector input is what makes CPU operation affordable
    # and what destroys distant pedestrians. Tiling re-runs detection at native
    # resolution inside overlapping tiles and merges the boxes back.
    #
    # These are detection parameters, not arbitration parameters. They live
    # here because this dataclass is the single mechanism the ablation configs
    # and the offline runner already load; nothing downstream of detection
    # reads them, and no identity decision depends on them.
    enable_tiled_detection: bool = False
    # OFF by default: tiling multiplies detection cost and the gain is
    # confined to sources where subjects are small relative to the frame.
    tile_size: int = 640
    tile_overlap: float = 0.2
    # Overlap must exceed the widest subject that may straddle a seam, or that
    # subject is seen only in fragments by both tiles. 0.2 of 640 = 128 px.
    tiled_min_source_resolution: int = 1280
    # Longer-edge floor, below which tiling is skipped as pure cost — a 640 px
    # webcam frame is already at native scale for the detector.
    tile_merge_iou: float = 0.50
    tile_merge_containment: float = 0.70
    # Merge uses IoU OR intersection-over-smaller. IoU alone cannot absorb a
    # tile-edge fragment into the whole person (a sliver under half the parent
    # box scores IoU < 0.5), which would leave one person with two boxes.

    # ── Duplicate-suggestion endpoint (backend/main.py) ─────────────────────
    duplicate_suggestion_threshold: float = 0.50
    # Advisory confidence bands shown to the operator. These change PRESENTATION
    # ONLY — nothing auto-merges at any band. See _merge_confidence_band().
    merge_band_high:   float = 0.80   # >= : very likely the same person
    merge_band_medium: float = 0.65   # >= : probable; below this = look-alike risk
    # Minimum cross-gallery face similarity for two SDT codes to be flagged
    # as a likely-duplicate pair on the People page.

    # ── Feature flags — all default True (current shipped behavior) ────────
    enable_face_anchor: bool = True
    # True: (a) a gate-passing face that matches nobody blocks the colour/
    # re-ID fallback methods from running at all for that box, and (b)
    # face_veto_threshold is enforced against colour/re-ID candidates.
    # False: colour/re-ID run and match unconditionally — the pre-hardening
    # behavior most prone to merging two different people.
    enable_colour_fallback: bool = True
    # Master switch for the dress-colour matching method (Method 2).
    enable_reid_fallback: bool = True
    # Master switch for the body re-ID matching method (Method 3).
    enable_id_switch_guard: bool = True
    # Master switch for the ByteTrack ID-switch contradiction guard.
    enable_evidence_gating: bool = True
    # Master switch for face-confirmed-only sighting/photo evidence.

    # ── Construction / (de)serialization ────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityConfig":
        """
        Build a config where JSON keys override defaults. Keys starting with
        "_" are treated as comments/metadata (e.g. "_description") and
        ignored. Any other unknown key raises — typo protection for ablation
        JSON, since a research tool should fail loudly on a misspelled param
        rather than silently keep the default.
        """
        params = {k: v for k, v in data.items() if not k.startswith("_")}
        known = {f.name for f in fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"Unknown identity config key(s): {sorted(unknown)}")
        cfg = cls()
        for key, value in params.items():
            if key == "face_detector_size" and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
        return cfg

    def as_loggable_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["face_detector_size"] = list(self.face_detector_size)
        return d


# ── Process-wide singleton ──────────────────────────────────────────────────

_active_config: Optional[IdentityConfig] = None


def load_identity_config() -> IdentityConfig:
    """
    Load the identity config once per process from SMARTDETECT_IDENTITY_CONFIG
    (a path to a JSON file overriding a subset of fields) if set, otherwise
    return pure defaults — identical to the values previously hardcoded
    across recognition/smart_identifier.py, recognition/face_recognizer.py
    and cameras/live_stream.py.
    """
    global _active_config
    path = os.getenv("SMARTDETECT_IDENTITY_CONFIG")
    if not path:
        _active_config = IdentityConfig()
        logger.info("identity_config.loaded source=defaults")
        return _active_config

    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        _active_config = IdentityConfig.from_dict(data)
        logger.info("identity_config.loaded source=%s overrides=%s", path, sorted(data))
    except Exception as exc:
        logger.error("identity_config.load_failed path=%s err=%s — falling back to defaults", path, exc)
        _active_config = IdentityConfig()
    return _active_config


def get_identity_config() -> IdentityConfig:
    """Return the process-wide active config, loading it on first call."""
    global _active_config
    if _active_config is None:
        return load_identity_config()
    return _active_config


def reset_identity_config_cache() -> None:
    """Test/ablation-harness hook: force the next get_identity_config() call
    to reload from the environment. Not used by the running application."""
    global _active_config
    _active_config = None
