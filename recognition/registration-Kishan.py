"""
recognition/registration.py
────────────────────────────
Universal person registration for SmartDetect.
Works for any location type — mall, campus, airport, office.

ID format changed from MET-YYYYMMDD-XXXX  →  SDT-XXXX (auto-increment).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from sqlalchemy.orm import Session

from database.models import Person
from database.queries import find_person_by_embedding, get_next_sdt_number
from recognition.face_recognizer import FaceRecognizer
from backend.logger import get_structured_logger

logger = get_structured_logger(__name__)

_recognizer: Optional[FaceRecognizer] = None


def _get_recognizer() -> FaceRecognizer:
    global _recognizer  # noqa: PLW0603
    if _recognizer is None:
        logger.info("model.load", message="Loading FaceRecognizer…")
        _recognizer = FaceRecognizer()
        _recognizer.load_model()
        logger.info("model.ready", message=f"FaceRecognizer stub_mode={_recognizer._stub_mode}")
    return _recognizer


def register_person(
    frame: np.ndarray,
    zone_id: str,
    location_id: str,
    db: Session,
    person_type: str = "unknown",
    similarity_threshold: float = 0.6,
) -> Dict[str, object]:
    """
    Full registration pipeline — works for any location.

    Parameters
    ----------
    frame               : BGR image (numpy array)
    zone_id             : Zone within the location (e.g. "Entrance A")
    location_id         : Location ID (e.g. "LOC-001")
    db                  : SQLAlchemy session
    person_type         : "visitor", "staff", or "unknown"
    similarity_threshold: cosine threshold for matching existing persons

    Returns
    -------
    dict with unique_code, person_type, location_name, is_new_registration, message
    """
    logger.info("registration.start",
                message=f"Registration: zone='{zone_id}' loc='{location_id}' type='{person_type}'")

    recognizer = _get_recognizer()
    embeddings = recognizer.extract_embedding(frame)

    if not embeddings:
        logger.warning("registration.no_face", message="No face detected in frame")
        raise ValueError("No face detected in the provided image.")

    query_embedding: np.ndarray = embeddings[0]

    # Check for existing match (across ALL locations)
    match = find_person_by_embedding(query_embedding, db=db, threshold=similarity_threshold)

    if match is not None:
        existing = db.query(Person).filter(Person.unique_code == match["unique_code"]).first()
        location_name = existing.location.name if (existing and existing.location) else location_id
        logger.info("registration.match",
                    message=f"Existing person: code={match['unique_code']} sim={match['similarity']:.3f}")
        return {
            "unique_code":         match["unique_code"],
            "person_type":         existing.person_type if existing else person_type,
            "location_name":       location_name,
            "is_new_registration": False,
            "message":             f"Welcome back! Identified as {match['unique_code']}.",
        }

    # Generate SDT-XXXX auto-increment code
    seq  = get_next_sdt_number(db)
    code = f"SDT-{seq:04d}"
    logger.info("registration.new", message=f"New code: {code} loc='{location_id}' type='{person_type}'")

    person = Person(
        unique_code=code,
        face_embedding=json.dumps(query_embedding.tolist()),
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        entry_zone=zone_id,
        location_id=location_id,
        person_type=person_type,
    )

    try:
        db.add(person)
        db.commit()
        db.refresh(person)
        location_name = person.location.name if person.location else location_id
        logger.info("registration.saved", message=f"Saved new person code={code}")
    except Exception as exc:
        db.rollback()
        logger.error("registration.db_error", message=str(exc))
        raise RuntimeError(f"Failed to save person: {exc}") from exc

    return {
        "unique_code":         code,
        "person_type":         person_type,
        "location_name":       location_id,
        "is_new_registration": True,
        "message":             f"New person registered as {code}.",
    }
