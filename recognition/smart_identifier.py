"""
recognition/smart_identifier.py
─────────────────────────────────
4-method smart person identification pipeline for SmartDetect.

Priority order:
  1. Face Recognition  (FaceRecognizer — ArcFace embedding, threshold 0.72)
  2. Dress Color       (K-means torso crop, HSV distance ≤ 30)
  3. Body Re-ID        (OSNet embedding, threshold 0.78)
  4. Multi-feature     (weighted combination, threshold 0.65)
  5. New Registration  (auto-assign SDT-XXXX if all methods fail)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config.identity_config import IdentityConfig, get_identity_config
from database.models import Person
from database.queries import (
    find_person_by_embedding,
    find_by_dress_color,
    get_next_sdt_number,
    update_person_last_seen,
)
from recognition.face_recognizer import FaceRecognizer
from recognition.reid_model import PersonReID

logger = logging.getLogger(__name__)

# Singletons — lazy loaded
_face_recognizer: Optional[FaceRecognizer] = None
_reid_model: Optional[PersonReID] = None


def _get_face_recognizer() -> FaceRecognizer:
    global _face_recognizer
    if _face_recognizer is None:
        _face_recognizer = FaceRecognizer()
        _face_recognizer.load_model()
    return _face_recognizer


def _get_reid_model() -> PersonReID:
    global _reid_model
    if _reid_model is None:
        _reid_model = PersonReID()  # auto-loads model in __init__
    return _reid_model


# ─── Color helpers ─────────────────────────────────────────────────────────────

def _dominant_color_hsv(bgr_crop: np.ndarray, k: int = 3) -> Optional[Dict]:
    """K-means dominant color extraction from a BGR crop. Returns HSV dict + hex."""
    try:
        import cv2
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        h, w = bgr_crop.shape[:2]
        if h < 10 or w < 10:
            return None

        # Reshape to pixel list and run K-means
        pixels = bgr_crop.reshape(-1, 3).astype(np.float32)
        k = min(k, len(pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant_bgr = centers[np.argmax(counts)].astype(np.uint8)

        # Convert to HSV
        hsv = cv2.cvtColor(np.array([[dominant_bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        r, g, b = int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0])
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        return {
            "hue":        int(hsv[0]),
            "saturation": int(hsv[1]),
            "value":      int(hsv[2]),
            "hex_color":  hex_color,
        }
    except Exception as exc:
        logger.debug("dominant_color_hsv failed: %s", exc)
        return None


def _hsv_distance(a: Dict, b: Dict) -> float:
    """Euclidean distance in HSV space (hue wrapped)."""
    dh = min(abs(a["hue"] - b["hue"]), 180 - abs(a["hue"] - b["hue"]))
    ds = abs(a["saturation"] - b["saturation"])
    dv = abs(a["value"] - b["value"])
    return float(np.sqrt(dh**2 + ds**2 + dv**2))


def _face_veto(query_emb: Optional[np.ndarray], unique_code: str, db, threshold: float) -> bool:
    """
    Return True if the colour/Re-ID match to `unique_code` should be REJECTED:
    we have a face for the query AND the candidate has a stored face AND they
    clearly differ. Without this, colour matching hands one person's SDT code
    to a different person whose face is plainly visible.
    """
    if query_emb is None:
        return False
    try:
        person = db.query(Person).filter(Person.unique_code == unique_code).first()
        if person is None or not person.face_embedding:
            return False
        stored = np.array(json.loads(person.face_embedding), dtype=np.float32)
        na, nb = np.linalg.norm(query_emb), np.linalg.norm(stored)
        if na == 0 or nb == 0:
            return False
        sim = float(np.dot(query_emb, stored) / (na * nb))
        if sim < threshold:
            logger.debug("face veto: %s rejected (face sim %.3f < %.2f)", unique_code, sim, threshold)
            return True
    except Exception as exc:
        logger.debug("face veto check failed: %s", exc)
    return False


# ─── SmartIdentifier ──────────────────────────────────────────────────────────

class SmartIdentifier:
    """
    Multi-method person identification.

    Usage:
        si = SmartIdentifier()
        result = si.identify(frame, bbox, db)
        # result: {unique_code, method, confidence, color_hex, ...}

    All thresholds and feature flags come from config/identity_config.py
    (IdentityConfig) — pass one explicitly, or the process-wide config loaded
    from SMARTDETECT_IDENTITY_CONFIG (or defaults) is used automatically.
    """

    def __init__(self, config: Optional[IdentityConfig] = None) -> None:
        self.config = config if config is not None else get_identity_config()

    def _update_face_template(
        self,
        unique_code: str,
        query_emb: np.ndarray,
        db,
        similarity: float = 1.0,
        person_crop: Optional[np.ndarray] = None,
    ) -> None:
        try:
            person = db.query(Person).filter(Person.unique_code == unique_code).first()
            if person is None or not person.face_embedding:
                return
            stored = np.array(json.loads(person.face_embedding), dtype=np.float32)
            if stored.shape != query_emb.shape:
                return
            blend = self.config.template_blend_weight
            blended = (1.0 - blend) * stored + blend * query_emb
            norm = np.linalg.norm(blended)
            if norm > 0:
                blended = blended / norm * np.linalg.norm(stored)
            person.face_embedding = json.dumps(blended.tolist())

            # Gallery append: this view matched but looks different enough to
            # be worth remembering separately (new lighting/angle)
            if self.config.gallery_add_threshold <= similarity < self.config.gallery_add_max_similarity:
                try:
                    templates = json.loads(person.face_templates) if person.face_templates else []
                except Exception:
                    templates = []
                templates.append(query_emb.tolist())
                if len(templates) > self.config.gallery_max_templates:
                    templates = templates[-self.config.gallery_max_templates:]
                person.face_templates = json.dumps(templates)

            # Re-ID template refresh: face just confirmed identity, so store
            # TODAY's clothing/body features (re-ID is clothing-dependent)
            if person_crop is not None and person_crop.size > 0:
                try:
                    reid = _get_reid_model()
                    if not getattr(reid, "is_stub", False):
                        reid_emb = reid.extract_features(person_crop)
                        if reid_emb is not None and len(reid_emb) > 0:
                            person.reid_embedding = json.dumps(reid_emb.tolist())
                except Exception:
                    pass

            db.commit()
        except Exception as exc:
            logger.debug("face template update failed: %s", exc)
            try:
                db.rollback()
            except Exception:
                pass

    def identify(
        self,
        frame: np.ndarray,
        bbox: list,            # [x, y, w, h]
        db,
        location_id: str = "",
        zone_id: str = "",
        allow_new: bool = True,
        face_embedding: Optional[np.ndarray] = None,
        exclude_codes: Optional[set] = None,
        extract_face_if_missing: bool = True,
        face_pose=None,
    ) -> Dict:
        """
        Run the full 4-method pipeline and return result dict.

        allow_new=False skips new-person registration when no method matches
        and returns {"unique_code": "Detecting...", "method": "pending"} —
        callers gate registration on track age to avoid one ghost SDT row
        per analysis cycle.

        face_embedding: pre-computed ArcFace embedding for THE face matched to
        this person box (e.g. from the caller's full-frame scan). Passing it
        skips a per-person InsightFace run and guarantees identity comes from
        the right face when boxes overlap.

        face_pose: HeadPose for THIS person's face (recognition/face_pose.py),
        used only by the registration pose gate. None disables the gate for
        this call (fail open) — matching is never affected by pose.

        exclude_codes: SDT codes already claimed by other people visible right
        now — colour/Re-ID may not hand these to a second person. New
        registration requires a face; a box with no visible face stays
        "Detecting..." (clothing colour alone must never mint an identity).
        """
        import cv2

        exclude_codes = exclude_codes or set()
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        person_crop = frame[
            max(0, y): min(fh, y + h),
            max(0, x): min(fw, x + w),
        ]
        torso_crop = frame[
            max(0, y): min(fh, y + int(h * 0.45)),
            max(0, x): min(fw, x + w),
        ]

        color_info: Optional[Dict] = None
        reid_emb                   = None
        query_face_emb: Optional[np.ndarray] = None

        # ── Method 1: Face ───────────────────────────────────────────────────
        try:
            if face_embedding is not None:
                query_face_emb = np.asarray(face_embedding, dtype=np.float32)
            elif extract_face_if_missing and person_crop.size > 0:
                recognizer = _get_face_recognizer()
                embs = recognizer.extract_embedding(person_crop) or []
                if embs:
                    query_face_emb = embs[0]
            if query_face_emb is not None:
                match = find_person_by_embedding(
                    query_face_emb, db=db, threshold=self.config.face_match_threshold,
                    exclude_codes=exclude_codes,
                )
                if match:
                    # Blend only confident matches into the stored template —
                    # a borderline match blended in drags the template toward
                    # a lookalike and invites future merges
                    if match["similarity"] >= self.config.template_blend_threshold:
                        self._update_face_template(
                            match["unique_code"], query_face_emb, db,
                            similarity=match["similarity"], person_crop=person_crop,
                        )
                    return {
                        "unique_code": match["unique_code"],
                        "method":      "face",
                        "confidence":  round(match["similarity"], 3),
                        "color_hex":   None,
                        "embedding":   query_face_emb.tolist(),
                    }
        except Exception as exc:
            logger.debug("SmartIdentifier.face failed: %s", exc)

        # A gate-passing face that matched NOBODY is decisive: this is not a
        # person we know. Clothing colour / body shape must not overrule the
        # face and re-associate them (that is exactly how three different
        # people ended up sharing one SDT code). Colour/Re-ID below exist
        # only for boxes where no usable face is visible.
        # enable_face_anchor=False reverts to the pre-hardening behavior:
        # colour/re-ID run and match regardless of what the face says.
        face_says_stranger = self.config.enable_face_anchor and query_face_emb is not None

        # ── Method 2: Dress Color (recent persons only + face veto) ─────────
        try:
            if self.config.enable_colour_fallback:
                color_info = _dominant_color_hsv(torso_crop)
                if color_info and not face_says_stranger:
                    color_match = find_by_dress_color(
                        color_info, threshold=self.config.colour_match_threshold, db=db,
                        recent_minutes=self.config.colour_reassoc_window_minutes,
                    )
                    if (color_match
                            and color_match["unique_code"] not in exclude_codes
                            and not (self.config.enable_face_anchor and _face_veto(
                                query_face_emb, color_match["unique_code"], db,
                                self.config.face_veto_threshold,
                            ))):
                        return {
                            "unique_code": color_match["unique_code"],
                            "method":      "dress_color",
                            "confidence":  round(color_match["score"], 3),
                            "color_hex":   color_info["hex_color"],
                            "embedding":   None,
                        }
        except Exception as exc:
            logger.debug("SmartIdentifier.dress_color failed: %s", exc)

        # ── Method 3: Body Re-ID (skipped in stub mode — histogram ≠ identity)
        #
        # PERF: OSNet costs ~190 ms/call — 20% of per-frame time after the
        # CoreML change. It used to run unconditionally here and then be
        # discarded whenever face_says_stranger was true (i.e. every time a
        # visible face matched nobody, which is exactly the stranger case).
        # It is now computed lazily, at most once, and only when a consumer
        # actually needs the vector: a re-ID match attempt, or storing the
        # template at registration. Semantics are unchanged — including that
        # reid_emb stays None when enable_reid_fallback is off, so
        # registration keeps writing a NULL reid_embedding in that config.
        _reid_cache: Dict[str, Optional[np.ndarray]] = {}

        def _reid_features() -> Optional[np.ndarray]:
            if "v" not in _reid_cache:
                try:
                    _reid_cache["v"] = _get_reid_model().extract_features(person_crop)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("reid extract failed: %s", exc)
                    _reid_cache["v"] = None
            return _reid_cache["v"]

        try:
            if self.config.enable_reid_fallback:
                reid = _get_reid_model()
                # Only pay for the forward pass if this match can actually run.
                if not face_says_stranger and not getattr(reid, "is_stub", False):
                    reid_emb = _reid_features()
                else:
                    reid_emb = None
                if (not face_says_stranger
                        and not getattr(reid, "is_stub", False)
                        and reid_emb is not None and len(reid_emb) > 0):
                    reid_match = find_person_by_embedding(
                        reid_emb, db=db,
                        threshold=self.config.reid_match_threshold,
                        embedding_field="reid_embedding",
                        recent_minutes=self.config.reid_reassoc_window_hours * 60,
                    )
                    if (reid_match
                            and reid_match["unique_code"] not in exclude_codes
                            and not (self.config.enable_face_anchor and _face_veto(
                                query_face_emb, reid_match["unique_code"], db,
                                self.config.face_veto_threshold,
                            ))):
                        return {
                            "unique_code": reid_match["unique_code"],
                            "method":      "body_structure",
                            "confidence":  round(reid_match["similarity"], 3),
                            "color_hex":   color_info["hex_color"] if color_info else None,
                            "embedding":   None,
                        }
        except Exception as exc:
            logger.debug("SmartIdentifier.reid failed: %s", exc)

        # ── No match: register only when the caller confirms the track AND a
        # face is visible — identity is face-anchored; a faceless box (person
        # turned away, or a YOLO false positive) stays "Detecting..." ────────
        if not allow_new or query_face_emb is None:
            return {
                "unique_code": "Detecting...",
                "method":      "pending",
                "confidence":  0.0,
                "color_hex":   color_info["hex_color"] if color_info else None,
                "embedding":   None,
            }

        # ── Registration pose gate ──────────────────────────────────────────
        # Last check before minting: an extreme viewpoint produces an
        # embedding that will not match this person's later frontal frames,
        # so it creates a duplicate identity rather than a new person.
        # Returning "Detecting..." defers enrolment to a better frame of the
        # SAME track — it does not lose the person.
        from recognition.face_pose import pose_ok_for_registration
        pose_ok, pose_reason = pose_ok_for_registration(face_pose, self.config)
        if not pose_ok:
            logger.debug("registration deferred: pose gate — %s", pose_reason)
            return {
                "unique_code": "Detecting...",
                "method":      "pending_pose",
                "confidence":  0.0,
                "color_hex":   color_info["hex_color"] if color_info else None,
                "embedding":   None,
                "pose_reason": pose_reason,
            }

        # ── Method 5: New Registration (reuses embeddings computed above) ───
        seq = get_next_sdt_number(db)
        new_code = f"SDT-{seq:04d}"

        # Save the registration photo — evidence behind the code
        photo_path = None
        try:
            if person_crop.size > 0:
                snap_dir = Path("snapshots") / new_code
                snap_dir.mkdir(parents=True, exist_ok=True)
                photo_file = snap_dir / "registered.jpg"
                if cv2.imwrite(str(photo_file), person_crop):
                    photo_path = photo_file.as_posix()
        except Exception as exc:
            logger.debug("registration snapshot failed: %s", exc)

        try:
            face_emb_json = json.dumps(query_face_emb.tolist())
            # Registration is the one place the vector is needed even when
            # Method 3 above short-circuited (face_says_stranger). Compute it
            # now if the lazy helper has not already — still gated on
            # enable_reid_fallback so the stored value matches the old
            # behaviour exactly in every config.
            if reid_emb is None and self.config.enable_reid_fallback:
                reid_emb = _reid_features()
            reid_emb_json = json.dumps(reid_emb.tolist()) if reid_emb is not None else None

            height_ratio = round(h / max(fh, 1), 4)

            person = Person(
                unique_code      = new_code,
                face_embedding   = face_emb_json,
                face_templates   = json.dumps([query_face_emb.tolist()]),
                photo_path       = photo_path,
                reid_embedding   = reid_emb_json,
                dress_color_hsv  = json.dumps(color_info) if color_info else None,
                body_height_ratio= height_ratio,
                entry_zone       = zone_id,
                location_id      = location_id,
                person_type      = "unknown",
                created_at       = datetime.now(timezone.utc).replace(tzinfo=None),
                first_seen_at    = datetime.now(timezone.utc).replace(tzinfo=None),
                last_seen_at     = datetime.now(timezone.utc).replace(tzinfo=None),
                total_sightings  = 1,
            )
            db.add(person)
            db.commit()
            logger.info("SmartIdentifier: new person registered as %s", new_code)
        except Exception as exc:
            logger.error("SmartIdentifier: DB save failed for %s: %s", new_code, exc)
            try:
                db.rollback()
            except Exception:
                pass

        return {
            "unique_code": new_code,
            "method":      "new_registration",
            "confidence":  1.0,
            "color_hex":   color_info["hex_color"] if color_info else None,
            "embedding":   None,
        }
