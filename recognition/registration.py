"""
recognition/registration.py
────────────────────────────
Person registration pipeline for SmartDetect.
Generates SDT-XXXX codes and stores embeddings with location/zone/person_type.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from sqlalchemy.orm import Session

from database.models import Location, Person
from database.queries import find_person_by_embedding, get_next_sdt_number
from recognition.face_recognizer import FaceRecognizer
from backend.logger import get_structured_logger

logger = get_structured_logger(__name__)

# Module-level singleton so the model is only loaded once
_recognizer: Optional[FaceRecognizer] = None


def _get_recognizer() -> FaceRecognizer:
    """Return (and lazily initialise) the shared FaceRecognizer instance."""
    global _recognizer  # noqa: PLW0603
    if _recognizer is None:
        logger.info("model.load", message="Loading FaceRecognizer model…")
        _recognizer = FaceRecognizer()
        _recognizer.load_model()
        logger.info("model.load", message=f"FaceRecognizer ready (stub_mode={_recognizer._stub_mode})")
    return _recognizer


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def register_person(
    frame: np.ndarray,
    zone_id: str,
    location_id: str,
    db: Session,
    person_type: str = "unknown",
    similarity_threshold: float = 0.6,
) -> Dict[str, object]:
    """Run the full registration pipeline for a person detected in a video frame."""
    logger.info("registration.start", message=f"Registration at location='{location_id}' zone='{zone_id}'")
    recognizer = _get_recognizer()

    # 1. Face detection + embedding extraction
    embeddings = recognizer.extract_embedding(frame)
    if not embeddings:
        logger.warning("registration.no_face", message="No face detected in frame")
        raise ValueError("No face detected in the provided image.")

    query_embedding: np.ndarray = embeddings[0]
    logger.debug("registration.embedding", message=f"Extracted embedding shape={query_embedding.shape}")

    # 2. Resolve location name for the response
    loc = db.query(Location).filter(Location.id == location_id).first()
    location_name = loc.name if loc else location_id

    # 3. Check database for existing person
    match = find_person_by_embedding(query_embedding, db=db, threshold=similarity_threshold)

    if match is not None:
        logger.info(
            "registration.match",
            message=f"Existing person recognised: code={match['unique_code']} similarity={match['similarity']:.3f}",
        )
        # Fetch their stored person_type
        existing = db.query(Person).filter(Person.unique_code == match["unique_code"]).first()
        return {
            "unique_code":         match["unique_code"],
            "person_type":         existing.person_type if existing else person_type,
            "location_name":       location_name,
            "is_new_registration": False,
            "message":             f"Welcome back! Identified as {match['unique_code']}.",
            "face_found":          True,
        }

    # 4. Generate a new SDT-XXXX code
    seq = get_next_sdt_number(db)
    new_code = f"SDT-{seq:04d}"
    logger.info("registration.new", message=f"New code: {new_code} at location='{location_id}'")

    # 5. Persist new Person record
    person = Person(
        unique_code=new_code,
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
        logger.info("registration.saved", message=f"Saved person code={new_code} loc='{location_id}'")
    except Exception as exc:
        db.rollback()
        logger.error("registration.db_error", message=f"DB commit failed for code={new_code}: {exc}")
        raise RuntimeError(f"Failed to save new person to database: {exc}") from exc

    return {
        "unique_code":         new_code,
        "person_type":         person_type,
        "location_name":       location_name,
        "is_new_registration": True,
        "message":             f"New person registered with ID {new_code}.",
        "face_found":          True,
    }
