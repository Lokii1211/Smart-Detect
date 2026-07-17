"""
database/models.py
───────────────────
SQLAlchemy ORM models for SmartDetect — Universal Camera Detection System.
SQLite-compatible: embeddings stored as JSON Text.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid():
    return str(uuid.uuid4())


# ─── Location  ────────────────────────────────────────────────────────────────

class Location(Base):
    """Physical location where SmartDetect cameras are deployed."""
    __tablename__ = "locations"

    id         = Column(String(64),  primary_key=True)
    name       = Column(String(128), nullable=False)
    type       = Column(String(64),  nullable=False, default="other")
    address    = Column(String(256), nullable=True)
    created_at = Column(DateTime,    default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Location id={self.id!r} name={self.name!r} type={self.type!r}>"


# ─── Person ───────────────────────────────────────────────────────────────────

class Person(Base):
    """Uniquely identified individual — tracked across any SmartDetect location."""
    __tablename__ = "persons"

    id             = Column(String(36),  primary_key=True, default=_uuid)
    unique_code    = Column(String(32),  nullable=False, unique=True, index=True)
    face_embedding = Column(Text,        nullable=True)
    created_at     = Column(DateTime,    default=datetime.utcnow, nullable=False)
    entry_zone     = Column(String(64),  nullable=True)
    location_id    = Column(String(64),  ForeignKey("locations.id", ondelete="SET NULL"), nullable=True, index=True)
    person_type    = Column(String(32),  nullable=False, default="unknown")

    sightings = relationship("Sighting", back_populates="person", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Person code={self.unique_code!r} type={self.person_type!r}>"


# ─── Sighting ─────────────────────────────────────────────────────────────────

class Sighting(Base):
    """Single observation of a Person at a zone/camera within a Location."""
    __tablename__ = "sightings"

    id                   = Column(String(36), primary_key=True, default=_uuid)
    person_id            = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    location_id          = Column(String(64), nullable=True, index=True)   # plain string, no FK to avoid join ambiguity
    zone_id              = Column(String(64), nullable=True, index=True)
    camera_id            = Column(String(64), nullable=False)
    seen_at              = Column(DateTime,   default=datetime.utcnow, nullable=False, index=True)
    confidence           = Column(Float,      nullable=False)
    frame_snapshot_path  = Column(Text,       nullable=True)

    # Only keep person relationship — location is looked up manually in queries.py
    person = relationship("Person", back_populates="sightings", foreign_keys=[person_id])

    def __repr__(self) -> str:
        return f"<Sighting person={self.person_id!r} zone={self.zone_id!r} at={self.seen_at}>"


# ─── ObjectSighting ───────────────────────────────────────────────────────────

class ObjectSighting(Base):
    """YOLO-detected object (bag, vehicle, etc.) captured by a camera."""
    __tablename__ = "object_sightings"

    id                  = Column(String(36),  primary_key=True, default=_uuid)
    location_id         = Column(String(64),  nullable=True, index=True)   # plain string, no FK
    zone_id             = Column(String(64),  nullable=True)
    camera_id           = Column(String(64),  nullable=False)
    object_type         = Column(String(64),  nullable=False)
    confidence          = Column(Float,       nullable=False)
    detected_at         = Column(DateTime,    default=datetime.utcnow, nullable=False, index=True)
    bbox_x              = Column(Integer,     nullable=True)
    bbox_y              = Column(Integer,     nullable=True)
    bbox_w              = Column(Integer,     nullable=True)
    bbox_h              = Column(Integer,     nullable=True)
    frame_snapshot_path = Column(Text,        nullable=True)

    def __repr__(self) -> str:
        return f"<ObjectSighting type={self.object_type!r} loc={self.location_id!r} at={self.detected_at}>"
