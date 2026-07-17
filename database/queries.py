"""
database/queries.py
────────────────────
Core database query functions for SmartDetect.
Uses pgvector for PostgreSQL, Python cosine similarity for SQLite.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.models import Location, Person, Sighting, WatchlistEntry, Alert
from backend.logger import get_structured_logger

logger = get_structured_logger(__name__)


def _is_postgres(db: Session) -> bool:
    return db.bind.dialect.name == "postgresql"


# ─── Helper ───────────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors (0-1)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─── Query 1 — Find Person by Embedding ──────────────────────────────────────

def find_person_by_embedding(
    embedding: np.ndarray,
    db: Session,
    threshold: float = 0.6,
    embedding_field: str = "face_embedding",
    exclude_codes: Optional[set] = None,
    recent_minutes: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Search persons table for closest embedding match.
    Uses pgvector cosine distance on PostgreSQL, Python cosine similarity on SQLite.

    exclude_codes: SDT codes that may not be returned — used to stop one code
    being assigned to two people visible in the same scene.

    recent_minutes: only consider persons seen within this window. Used for
    re-ID matching — body features track clothing, which changes daily.
    """
    logger.debug("query.match_start", message=f"Searching {embedding_field} with threshold={threshold}")

    if _is_postgres(db):
        return _find_person_pgvector(embedding, db, threshold, embedding_field,
                                     exclude_codes, recent_minutes)
    return _find_person_python(embedding, db, threshold, embedding_field,
                               exclude_codes, recent_minutes)


def _find_person_pgvector(
    embedding: np.ndarray,
    db: Session,
    threshold: float,
    embedding_field: str,
    exclude_codes: Optional[set] = None,
    recent_minutes: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """pgvector cosine distance search — runs entirely in the database."""
    from sqlalchemy import bindparam

    col = "reid_embedding" if embedding_field == "reid_embedding" else "face_embedding"
    vec_str = "[" + ",".join(str(float(v)) for v in embedding) + "]"

    excl_clause   = "AND unique_code NOT IN :excl" if exclude_codes else ""
    recent_clause = "AND last_seen_at >= :cutoff" if recent_minutes is not None else ""
    stmt = text(f"""
        SELECT unique_code,
               1 - ({col}::vector <=> :vec::vector) AS similarity
        FROM persons
        WHERE {col} IS NOT NULL {excl_clause} {recent_clause}
        ORDER BY {col}::vector <=> :vec::vector
        LIMIT 1
    """)
    params: Dict[str, Any] = {"vec": vec_str}
    if exclude_codes:
        stmt = stmt.bindparams(bindparam("excl", expanding=True))
        params["excl"] = list(exclude_codes)
    if recent_minutes is not None:
        from datetime import timedelta
        params["cutoff"] = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=recent_minutes)

    row = db.execute(stmt, params).fetchone()

    if row is None or row.similarity < threshold:
        return None

    logger.info("query.match_found", message=f"pgvector match: code={row.unique_code} sim={row.similarity:.3f}")
    return {"unique_code": row.unique_code, "similarity": round(float(row.similarity), 4)}


def _find_person_python(
    embedding: np.ndarray,
    db: Session,
    threshold: float,
    embedding_field: str,
    exclude_codes: Optional[set] = None,
    recent_minutes: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Python-side cosine similarity scan — works on SQLite."""
    if embedding_field == "reid_embedding":
        q = db.query(Person).filter(Person.reid_embedding.isnot(None))
        def get_stored(p): return p.reid_embedding
    else:
        q = db.query(Person).filter(Person.face_embedding.isnot(None))
        def get_stored(p): return p.face_embedding

    if recent_minutes is not None:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=recent_minutes)
        q = q.filter(Person.last_seen_at.isnot(None), Person.last_seen_at >= cutoff)
    persons = q.all()

    if exclude_codes:
        persons = [p for p in persons if p.unique_code not in exclude_codes]

    if not persons:
        return None

    best_code: Optional[str] = None
    best_sim: float = -1.0

    for person in persons:
        # Candidate vectors: the primary embedding plus (for faces) every
        # gallery template — a person is as recognisable as their BEST view
        candidates = []
        try:
            candidates.append(np.array(json.loads(get_stored(person)), dtype=np.float32))
        except Exception:
            pass
        if embedding_field != "reid_embedding" and getattr(person, "face_templates", None):
            try:
                for vec in json.loads(person.face_templates):
                    candidates.append(np.array(vec, dtype=np.float32))
            except Exception:
                pass

        for stored in candidates:
            if stored.shape != embedding.shape:
                continue
            sim = _cosine_similarity(embedding, stored)
            if sim > best_sim:
                best_sim = sim
                best_code = person.unique_code

    if best_code is None or best_sim < threshold:
        return None

    logger.info("query.match_found", message=f"Match: code={best_code} sim={best_sim:.3f}")
    return {"unique_code": best_code, "similarity": round(best_sim, 4)}


# ─── Query 2 — Find Person by Dress Color ────────────────────────────────────

def find_by_dress_color(
    hsv: Dict,
    threshold: float = 30.0,
    db: Session = None,
    recent_minutes: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Search for a person whose stored dress_color_hsv is within `threshold`
    HSV distance of the given colour dict {hue, saturation, value}.

    Clothing colour cannot distinguish strangers — pass `recent_minutes` to
    restrict candidates to persons seen within that window, so colour only
    re-associates someone who just walked out of face view, never claims a
    long-gone identity for a new person.

    Returns {unique_code, score} or None.
    """
    q = db.query(Person).filter(Person.dress_color_hsv.isnot(None))
    if recent_minutes is not None:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=recent_minutes)
        q = q.filter(Person.last_seen_at.isnot(None), Person.last_seen_at >= cutoff)
    persons = q.all()
    best_code = None
    best_dist = float("inf")

    for person in persons:
        try:
            stored = json.loads(person.dress_color_hsv)
            dh = min(abs(hsv["hue"] - stored["hue"]), 180 - abs(hsv["hue"] - stored["hue"]))
            ds = abs(hsv["saturation"] - stored["saturation"])
            dv = abs(hsv["value"] - stored["value"])
            dist = math.sqrt(dh**2 + ds**2 + dv**2)
            if dist < best_dist:
                best_dist = dist
                best_code = person.unique_code
        except Exception:
            continue

    if best_code is None or best_dist > threshold:
        return None

    confidence = round(1.0 - best_dist / (threshold * 2), 3)
    return {"unique_code": best_code, "score": max(0.0, confidence)}


# ─── Query 3 — Get Full Movement Trail ───────────────────────────────────────

def get_person_trail(unique_code: str, db: Session) -> List[Dict[str, Any]]:
    """Fetch chronological movement trail. Returns [] if person not found."""
    logger.debug("query.trail", message=f"Trail for '{unique_code}'")
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if person is None:
        logger.warning("query.trail_not_found", message=f"Unknown: '{unique_code}'")
        return []

    sightings = (
        db.query(Sighting)
        .filter(Sighting.person_id == person.id)
        .order_by(Sighting.seen_at.asc())
        .all()
    )

    trail: List[Dict[str, Any]] = []
    for s in sightings:
        loc = db.query(Location).filter(Location.id == s.location_id).first() if s.location_id else None
        trail.append({
            "location_name":       loc.name if loc else (s.location_id or "—"),
            "location_type":       loc.type if loc else "unknown",
            "location_id":         s.location_id,
            "zone_id":             s.zone_id,
            "camera_id":           s.camera_id,
            "seen_at":             s.seen_at.isoformat() + "Z",
            "confidence":          round(s.confidence, 4),
            "frame_snapshot_path": s.frame_snapshot_path,
        })

    logger.debug("query.trail_result", message=f"Trail '{unique_code}': {len(trail)} sightings")
    return trail


# ─── Query 4 — Log a New Sighting ────────────────────────────────────────────

def log_sighting(
    unique_code: str,
    location_id: str,
    zone_id: str,
    camera_id: str,
    confidence: float,
    db: Session,
    frame_path: Optional[str] = None,
) -> bool:
    """Record a sighting. Returns True on success, False if person not found."""
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if person is None:
        logger.warning("sighting.rejected", message=f"Unknown code '{unique_code}'")
        return False

    sighting = Sighting(
        person_id=person.id,
        unique_code=unique_code,
        location_id=location_id,
        zone_id=zone_id,
        camera_id=camera_id,
        seen_at=datetime.now(timezone.utc).replace(tzinfo=None),
        confidence=confidence,
        frame_snapshot_path=frame_path,
    )
    try:
        db.add(sighting)
        # Update last_seen_at and total_sightings
        person.last_seen_at    = datetime.now(timezone.utc).replace(tzinfo=None)
        person.total_sightings = (person.total_sightings or 0) + 1
        db.commit()
        logger.info("sighting.logged",
            message=f"Sighting: code={unique_code} loc={location_id} zone={zone_id} conf={confidence:.2f}")
        return True
    except Exception as exc:
        db.rollback()
        logger.error("sighting.db_error", message=str(exc))
        return False


# ─── Query 5 — Update Person Last Seen ───────────────────────────────────────

def update_person_last_seen(unique_code: str, db: Session) -> None:
    """Bump last_seen_at and total_sightings for a person."""
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if person:
        person.last_seen_at    = datetime.now(timezone.utc).replace(tzinfo=None)
        person.total_sightings = (person.total_sightings or 0) + 1
        try:
            db.commit()
        except Exception:
            db.rollback()


# ─── Query 6 — SDT Sequence Number ───────────────────────────────────────────

def get_next_sdt_number(db: Session) -> int:
    codes = [p.unique_code for p in db.query(Person).all()]
    nums  = [int(m.group(1)) for c in codes for m in [re.match(r"SDT-(\d+)", c)] if m]
    return max(nums, default=0) + 1


# ─── Query 7 — Recent Detections (for camera feed) ────────────────────────────

def get_recent_detections(limit: int = 20, db: Session = None) -> List[Dict[str, Any]]:
    """
    Return the last `limit` sightings with person info and color if available.
    Used by GET /camera/detections/recent.
    """
    rows = (
        db.query(Sighting, Person)
        .join(Person, Sighting.person_id == Person.id)
        .order_by(Sighting.seen_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for sighting, person in rows:
        color_hex = None
        try:
            if person.dress_color_hsv:
                color_info = json.loads(person.dress_color_hsv)
                color_hex  = color_info.get("hex_color")
        except Exception:
            pass
        result.append({
            "unique_code": person.unique_code,
            "method":      "face",   # stored method not tracked in Sighting yet — default to face
            "confidence":  round(sighting.confidence, 3),
            "color_hex":   color_hex,
            "detected_at": sighting.seen_at.isoformat(),
            "zone_id":     sighting.zone_id,
            "camera_id":   sighting.camera_id,
        })
    return result


# ─── Query 8 — Watchlist Alert Check ─────────────────────────────────────────

_watchlist_alert_cache: Dict[str, float] = {}

def check_watchlist_alert(
    unique_code: str,
    camera_id: str,
    location_id: str,
    db: Session,
) -> None:
    """Create an alert if this person is on the active watchlist. Deduped per 5 minutes."""
    now = datetime.now(timezone.utc).timestamp()
    cache_key = f"{unique_code}:{camera_id}"
    if now - _watchlist_alert_cache.get(cache_key, 0) < 300:
        return

    entry = db.query(WatchlistEntry).filter(
        WatchlistEntry.unique_code == unique_code,
        WatchlistEntry.is_active == True,
    ).first()
    if not entry:
        return

    _watchlist_alert_cache[cache_key] = now
    alert = Alert(
        alert_type="watchlist",
        severity="critical",
        title=f"Watchlisted person {unique_code} detected",
        message=f"Detected on camera {camera_id} at location {location_id}."
                + (f" Reason: {entry.reason}" if entry.reason else ""),
        unique_code=unique_code,
        camera_id=camera_id,
        location_id=location_id,
        is_read=False,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    try:
        db.add(alert)
        db.commit()
        logger.info("alert.watchlist", message=f"Alert created for {unique_code} on {camera_id}")
    except Exception as exc:
        db.rollback()
        logger.error("alert.watchlist_error", message=str(exc))
