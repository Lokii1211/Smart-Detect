"""
recognition/face_pose.py
─────────────────────────
Head-pose estimation from InsightFace's 5-point landmarks, and the
registration quality gate built on it.

WHY
───
Identity is minted on the first frame whose face clears the size/detection
quality gate. That gate says nothing about POSE, so a code can be created
from a profile or a bowed head — a viewpoint whose embedding matches poorly
against later frontal views of the same person. The result is a duplicate
identity for someone already enrolled.

Observed in this project: the same woman was registered twice
(SDT-0013 frontal, SDT-0014 head-bowed) from one continuous recording.

The landmarks are already returned by InsightFace on every detection, so
this costs no extra inference — it is pure geometry on data we discard today.

SCOPE — registration only
─────────────────────────
This gate must gate MINTING, never MATCHING. Refusing to match a profile
view would lose recall for someone already enrolled; refusing to *enrol*
from a profile view merely defers enrolment to a better frame, and tracks
persist for many frames. Enforced at the single call site in
SmartIdentifier.identify()'s Method 5.

GEOMETRY
────────
Landmarks are [left_eye, right_eye, nose, left_mouth, right_mouth] in image
coordinates.

    yaw   = (nose.x - eye_centre.x) / inter_ocular_distance
            0 facing camera; sign gives turn direction; |yaw| grows with turn.

    pitch = (nose.y - eye_centre.y) / (mouth_centre.y - eye_centre.y)
            ~0.5 facing camera; >0.5 head bowed DOWN; <0.5 chin raised.

    roll  = angle of the inter-ocular line, degrees.

Calibrated against demo_videos/05_head_pose_two_people.mp4 (a deliberate
head-pose sweep), 138 samples, verified by eye against the frames:

    yaw    p05 -0.416   median -0.007   p95  0.479
    pitch  p05  0.398   median  0.495   p95  0.744

Frames at pitch ~1.0 are visibly bowed; |yaw| ~0.6 is visibly profile.
These are ratios, not degrees — they are scale-invariant but NOT a
calibrated angular measurement. Do not report them as degrees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Reference values for a face looking straight at the camera.
FRONTAL_YAW = 0.0
FRONTAL_PITCH = 0.5


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    inter_ocular_px: float

    @property
    def frontality(self) -> float:
        """
        0..1 score, 1 = perfectly frontal. Used to rank frames by pose
        quality (e.g. choosing the best frame of a tracklet).
        """
        y = min(abs(self.yaw - FRONTAL_YAW) / 0.6, 1.0)
        p = min(abs(self.pitch - FRONTAL_PITCH) / 0.4, 1.0)
        return float(max(0.0, 1.0 - 0.5 * (y + p)))


def estimate_pose(kps: Optional[Sequence]) -> Optional[HeadPose]:
    """
    HeadPose from 5-point landmarks, or None when they are missing or
    degenerate. Never raises: a pose failure must not break identification.
    """
    if kps is None:
        return None
    try:
        pts = np.asarray(kps, dtype=np.float32)
        if pts.shape[0] < 5:
            return None
        le, re, nose, lm, rm = pts[0], pts[1], pts[2], pts[3], pts[4]

        iod = float(np.linalg.norm(re - le))
        if iod < 1e-3:
            return None                      # eyes coincident — unusable

        eye_c = (le + re) / 2.0
        mouth_c = (lm + rm) / 2.0
        span = float(mouth_c[1] - eye_c[1])
        if abs(span) < 1e-3:
            return None                      # no vertical extent — unusable

        return HeadPose(
            yaw=float((nose[0] - eye_c[0]) / iod),
            pitch=float((nose[1] - eye_c[1]) / span),
            roll=float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))),
            inter_ocular_px=iod,
        )
    except Exception:
        return None


def pose_ok_for_registration(pose: Optional[HeadPose], cfg) -> tuple:
    """
    (ok, reason). Applied ONLY when minting a new identity.

    Fails OPEN when pose is unavailable: a missing landmark set must not
    block enrolment, or a detector change silently stops registration.
    Returns (True, "no_pose") in that case.
    """
    if not getattr(cfg, "enable_pose_gate_at_registration", False):
        return True, "gate_disabled"
    if pose is None:
        return True, "no_pose"              # fail open, deliberately

    if abs(pose.yaw) > cfg.registration_max_abs_yaw:
        return False, f"yaw {pose.yaw:+.2f} exceeds ±{cfg.registration_max_abs_yaw}"
    if not (cfg.registration_min_pitch <= pose.pitch <= cfg.registration_max_pitch):
        return False, (f"pitch {pose.pitch:.2f} outside "
                       f"[{cfg.registration_min_pitch}, {cfg.registration_max_pitch}]")
    if abs(pose.roll) > cfg.registration_max_abs_roll_deg:
        return False, f"roll {pose.roll:+.1f}° exceeds ±{cfg.registration_max_abs_roll_deg}°"
    return True, "ok"
