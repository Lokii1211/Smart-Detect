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
    """

    FACE_THRESHOLD  = 0.56   # ArcFace cosine sim — 0.35 cross-matched
                             # strangers; 0.50 still cross-matched two pairs
                             # of different people on the 4-video eval set
    COLOR_THRESHOLD = 30.0
    REID_THRESHOLD  = 0.68   # OSNet cosine sim — 0.60 cross-matched different
                             # people in similar clothing; templates refresh to
                             # today's outfit on face confirm, so real
                             # re-associations score well above this
    MULTI_THRESHOLD = 0.65
    # Colour/Re-ID may only re-associate someone seen this recently (minutes)
    COLOR_RECENT_MINUTES = 10.0
    # Body features track clothing — trust them across hours, not days
    REID_RECENT_MINUTES  = 720.0
    # If the query face clearly differs from a candidate's stored face,
    # reject colour/Re-ID matches to that candidate (face veto).
    # 0.30 let 0.30-0.49 faces — already rejected by FACE_THRESHOLD — sneak
    # back in through colour (measured: 3 different people merged into one
    # SDT code on face-demographics-walking.mp4). Veto anything the face
    # matcher itself would not accept.
    FACE_VETO_THRESHOLD = 0.45

    # Running-average template update: each confident match blends the fresh
    # embedding into the stored one, so memory adapts to new lighting/angles
    # instead of staying frozen at the registration-day snapshot.
    TEMPLATE_BLEND = 0.20   # weight of the new embedding
    # Gallery: keep extra templates for usefully DIFFERENT views. A match
    # above this is near-identical to what we already store — skip it.
    GALLERY_APPEND_MAX_SIM = 0.85
    # ...but only from confident matches: the matcher scores a person by their
    # BEST template, so one marginal (≈0.50) view in the gallery turns the
    # code into a similarity magnet that attracts strangers
    GALLERY_APPEND_MIN_SIM = 0.60
    GALLERY_MAX_TEMPLATES  = 5

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
            blended = (1.0 - self.TEMPLATE_BLEND) * stored + self.TEMPLATE_BLEND * query_emb
            norm = np.linalg.norm(blended)
            if norm > 0:
                blended = blended / norm * np.linalg.norm(stored)
            person.face_embedding = json.dumps(blended.tolist())

            # Gallery append: this view matched but looks different enough to
            # be worth remembering separately (new lighting/angle)
            if self.GALLERY_APPEND_MIN_SIM <= similarity < self.GALLERY_APPEND_MAX_SIM:
                try:
                    templates = json.loads(person.face_templates) if person.face_templates else []
                except Exception:
                    templates = []
                templates.append(query_emb.tolist())
                if len(templates) > self.GALLERY_MAX_TEMPLATES:
                    templates = templates[-self.GALLERY_MAX_TEMPLATES:]
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
                    query_face_emb, db=db, threshold=self.FACE_THRESHOLD,
                    exclude_codes=exclude_codes,
                )
                if match:
                    # Blend only confident matches into the stored template —
                    # a borderline (0.50-0.55) match blended at 20% drags the
                    # template toward a lookalike and invites future merges
                    if match["similarity"] >= 0.55:
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
        face_says_stranger = query_face_emb is not None

        # ── Method 2: Dress Color (recent persons only + face veto) ─────────
        try:
            color_info = _dominant_color_hsv(torso_crop)
            if color_info and not face_says_stranger:
                color_match = find_by_dress_color(
                    color_info, threshold=self.COLOR_THRESHOLD, db=db,
                    recent_minutes=self.COLOR_RECENT_MINUTES,
                )
                if (color_match
                        and color_match["unique_code"] not in exclude_codes
                        and not _face_veto(
                            query_face_emb, color_match["unique_code"], db, self.FACE_VETO_THRESHOLD
                        )):
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
        try:
            reid = _get_reid_model()
            reid_emb = reid.extract_features(person_crop)
            if (not face_says_stranger
                    and not getattr(reid, "is_stub", False)
                    and reid_emb is not None and len(reid_emb) > 0):
                reid_match = find_person_by_embedding(
                    reid_emb, db=db,
                    threshold=self.REID_THRESHOLD,
                    embedding_field="reid_embedding",
                    recent_minutes=self.REID_RECENT_MINUTES,
                )
                if (reid_match
                        and reid_match["unique_code"] not in exclude_codes
                        and not _face_veto(
                            query_face_emb, reid_match["unique_code"], db, self.FACE_VETO_THRESHOLD
                        )):
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
