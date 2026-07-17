"""
database/queries.py
────────────────────
Core database query functions for SmartDetect.
Supports person tracking (any location), object sightings, and location lookups.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from database.models import Location, ObjectSighting, Person, Sighting
from backend.logger import get_structured_logger

logger = get_structured_logger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─── Person queries ───────────────────────────────────────────────────────────

def find_person_by_embedding(
    embedding: np.ndarray,
    db: Session,
    threshold: float = 0.6,
) -> Optional[Dict[str, Any]]:
    """
    Search ALL persons regardless of location — a person registered at a mall
    should be found when appearing at a campus camera.
    """
    logger.debug("query.match_start", message=f"Searching embeddings threshold={threshold}")
    persons = db.query(Person).filter(Person.face_embedding.isnot(None)).all()
    if not persons:
        return None

    best_code: Optional[str] = None
    best_sim: float = -1.0

    for person in persons:
        try:
            stored = np.array(json.loads(person.face_embedding), dtype=np.float32)
        except Exception:
            continue
        sim = _cosine_similarity(embedding, stored)
        if sim > best_sim:
            best_sim = sim
            best_code = person.unique_code

    if best_code is None or best_sim < threshold:
        logger.debug("query.match_none", message=f"No match above threshold (best={best_sim:.3f})")
        return None

    logger.info("query.match_found", message=f"Match: code={best_code} sim={best_sim:.3f}")
    return {"unique_code": best_code, "similarity": round(best_sim, 4)}


def get_person_trail(unique_code: str, db: Session) -> List[Dict[str, Any]]:
    """
    Fetch chronological movement trail.
    Joins Location to include location_name and location_type per entry.
    """
    from database.models import Location as Loc  # noqa: PLC0415
    logger.debug("query.trail", message=f"Trail for '{unique_code}'")
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if person is None:
        logger.warning("query.trail_not_found", message=f"Unknown code: '{unique_code}'")
        return []

    sightings = (
        db.query(Sighting)
        .filter(Sighting.person_id == person.id)
        .order_by(Sighting.seen_at.asc())
        .all()
    )

    trail = []
    for s in sightings:
        loc = db.query(Loc).filter(Loc.id == s.location_id).first() if s.location_id else None
        trail.append({
            "location_name":       loc.name       if loc else (s.location_id or "—"),
            "location_type":       loc.type       if loc else "unknown",
            "location_id":         s.location_id,
            "zone_id":             s.zone_id,
            "camera_id":           s.camera_id,
            "seen_at":             s.seen_at.isoformat() + "Z",
            "confidence":          round(s.confidence, 4),
            "frame_snapshot_path": s.frame_snapshot_path,
        })

    logger.debug("query.trail_result", message=f"Trail '{unique_code}': {len(trail)} sightings")
    return trail


def log_sighting(
    unique_code: str,
    location_id: str,
    zone_id: str,
    camera_id: str,
    confidence: float,
    db: Session,
    frame_path: Optional[str] = None,
) -> bool:
    """Record a person sighting. Returns True on success."""
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if person is None:
        logger.warning("sighting.rejected", message=f"Unknown code '{unique_code}'")
        return False

    sighting = Sighting(
        person_id=person.id,
        location_id=location_id,
        zone_id=zone_id,
        camera_id=camera_id,
        seen_at=datetime.now(timezone.utc).replace(tzinfo=None),
        confidence=confidence,
        frame_snapshot_path=frame_path,
    )
    try:
        db.add(sighting)
        db.commit()
        logger.info("sighting.logged",
                    message=f"Sighting: code={unique_code} loc={location_id} zone={zone_id} cam={camera_id} conf={confidence:.2f}")
        return True
    except Exception as exc:
        db.rollback()
        logger.error("sighting.db_error", message=f"DB error for '{unique_code}': {exc}")
        return False


def get_next_sdt_number(db: Session) -> int:
    """Return the next SDT sequence number (max existing + 1)."""
    import re  # noqa: PLC0415
    all_codes = [p.unique_code for p in db.query(Person.unique_code).all()]
    nums = []
    for code in all_codes:
        m = re.match(r"SDT-(\d+)", code)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


# ─── Object sighting queries (Change 2) ───────────────────────────────────────

def log_object_sighting(
    location_id: str,
    zone_id: str,
    camera_id: str,
    object_type: str,
    confidence: float,
    bbox: List[int],
    db: Session,
    frame_path: Optional[str] = None,
) -> bool:
    """Insert an ObjectSighting row. Returns True on success."""
    x, y, w, h = (bbox + [0, 0, 0, 0])[:4]
    obj = ObjectSighting(
        location_id=location_id,
        zone_id=zone_id,
        camera_id=camera_id,
        object_type=object_type,
        confidence=confidence,
        detected_at=datetime.now(timezone.utc).replace(tzinfo=None),
        bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
        frame_snapshot_path=frame_path,
    )
    try:
        db.add(obj)
        db.commit()
        logger.info("object.logged",
                    message=f"Object: type={object_type} loc={location_id} zone={zone_id} conf={confidence:.2f}")
        return True
    except Exception as exc:
        db.rollback()
        logger.error("object.db_error", message=str(exc))
        return False


def get_recent_objects(location_id: str, limit: int = 50, db: Session = None) -> List[Dict[str, Any]]:
    """Return last N object detections for a location, newest first."""
    rows = (
        db.query(ObjectSighting)
        .filter(ObjectSighting.location_id == location_id)
        .order_by(ObjectSighting.detected_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":           r.id,
            "object_type":  r.object_type,
            "location_id":  r.location_id,
            "zone_id":      r.zone_id,
            "camera_id":    r.camera_id,
            "confidence":   round(r.confidence, 3),
            "detected_at":  r.detected_at.isoformat() + "Z",
            "bbox":         [r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h],
        }
        for r in rows
    ]


def get_object_counts_today(location_id: str, db: Session = None) -> Dict[str, int]:
    """Return bag and vehicle counts detected today at a location."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    rows = (
        db.query(ObjectSighting.object_type)
        .filter(
            ObjectSighting.location_id == location_id,
            ObjectSighting.detected_at >= today_start,
        )
        .all()
    )
    bags     = sum(1 for (t,) in rows if t in {"backpack", "handbag", "suitcase"})
    vehicles = sum(1 for (t,) in rows if t in {"car", "motorcycle", "truck", "bus"})
    return {"bags": bags, "vehicles": vehicles}
