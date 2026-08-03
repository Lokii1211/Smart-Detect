"""
database/models.py
───────────────────
SQLAlchemy ORM models for SmartDetect — Universal Camera Detection System.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
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

    cameras = relationship("Camera", back_populates="location", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Location id={self.id!r} name={self.name!r} type={self.type!r}>"


# ─── Camera ───────────────────────────────────────────────────────────────────

class Camera(Base):
    """A camera device installed at a Location zone."""
    __tablename__ = "cameras"

    id          = Column(String(64),  primary_key=True)                          # e.g. "CAM-001"
    location_id = Column(String(64),  ForeignKey("locations.id", ondelete="CASCADE"), nullable=False, index=True)
    zone_id     = Column(String(64),  nullable=False, default="main")            # e.g. "entrance"
    label       = Column(String(128), nullable=False, default="Camera")          # human-readable name
    source      = Column(String(256), nullable=False, default="0")               # webcam index or RTSP URL
    is_active   = Column(Boolean,     nullable=False, default=False)
    created_at  = Column(DateTime,    default=datetime.utcnow, nullable=False)

    location = relationship("Location", back_populates="cameras")

    def __repr__(self) -> str:
        return f"<Camera id={self.id!r} zone={self.zone_id!r} active={self.is_active}>"


# ─── Person ───────────────────────────────────────────────────────────────────

class Person(Base):
    """Uniquely identified individual — tracked across any SmartDetect location."""
    __tablename__ = "persons"

    id                = Column(String(36),  primary_key=True, default=_uuid)
    unique_code       = Column(String(32),  nullable=False, unique=True, index=True)
    display_name      = Column(String(128), nullable=True)   # human name for the code
    face_embedding    = Column(Text,        nullable=True)   # primary template (flat vector, pgvector-castable)
    face_templates    = Column(Text,        nullable=True)   # JSON list of vectors — multi-view gallery
    photo_path        = Column(Text,        nullable=True)   # registration crop, served under /snapshots
    reid_embedding    = Column(Text,        nullable=True)
    dress_color_hsv   = Column(Text,        nullable=True)
    body_height_ratio = Column(Float,       nullable=True)
    created_at        = Column(DateTime,    default=datetime.utcnow, nullable=False)
    first_seen_at     = Column(DateTime,    nullable=True)
    last_seen_at      = Column(DateTime,    nullable=True)
    total_sightings   = Column(Integer,     nullable=False, default=0)
    entry_zone        = Column(String(64),  nullable=True)
    location_id       = Column(String(64),  nullable=True, index=True)   # plain string, no FK
    person_type       = Column(String(32),  nullable=False, default="unknown")

    # ── Data governance (see docs/DATA_GOVERNANCE.md) ────────────────────────
    # A face embedding is biometric data. Under India's DPDP Act 2023, GDPR
    # Art. 9 and BIPA it may only be processed on a stated lawful basis, so
    # every identity must carry one. Default 'unknown' is deliberately the
    # WORST case: an identity auto-registered from a live camera has given no
    # consent, and the purge job treats 'unknown' most aggressively.
    consent_status    = Column(String(16),  nullable=False, default="unknown",
                               index=True)   # consented | dataset | unknown
    consent_ref       = Column(String(128), nullable=True)   # signed-form ID / dataset licence
    consent_recorded_at = Column(DateTime,  nullable=True)
    consent_recorded_by = Column(String(64), nullable=True)  # operator username

    # Retention. retain_until is computed from consent_status when the row is
    # created and re-computed on each sighting; the purge job deletes rows
    # past it. legal_hold blocks purging regardless (e.g. active investigation).
    retain_until      = Column(DateTime,    nullable=True, index=True)
    legal_hold        = Column(Boolean,     nullable=False, default=False)

    sightings = relationship("Sighting", back_populates="person", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Person code={self.unique_code!r} type={self.person_type!r} consent={self.consent_status!r}>"


# ─── Sighting ─────────────────────────────────────────────────────────────────

class Sighting(Base):
    """Single observation of a Person at a zone/camera."""
    __tablename__ = "sightings"

    id                  = Column(String(36), primary_key=True, default=_uuid)
    person_id           = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    unique_code         = Column(String(32), nullable=True, index=True)          # denormalised for fast lookup
    location_id         = Column(String(64), nullable=True, index=True)
    zone_id             = Column(String(64), nullable=True, index=True)
    camera_id           = Column(String(64), nullable=False)
    seen_at             = Column(DateTime,   default=datetime.utcnow, nullable=False, index=True)
    confidence          = Column(Float,      nullable=False)
    frame_snapshot_path = Column(Text,       nullable=True)

    person = relationship("Person", back_populates="sightings")

    def __repr__(self) -> str:
        return f"<Sighting person={self.person_id!r} zone={self.zone_id!r} at={self.seen_at}>"


# ─── ObjectSighting ───────────────────────────────────────────────────────────

class ObjectSighting(Base):
    """Detection of a non-person object (bag, vehicle, etc.) by a camera."""
    __tablename__ = "object_sightings"

    id          = Column(String(36),  primary_key=True, default=_uuid)
    location_id = Column(String(64),  nullable=True, index=True)
    zone_id     = Column(String(64),  nullable=True)
    camera_id   = Column(String(64),  nullable=False)
    object_type = Column(String(64),  nullable=False)
    confidence  = Column(Float,       nullable=False)
    detected_at = Column(DateTime,    default=datetime.utcnow, nullable=False, index=True)
    bbox_x      = Column(Integer,     nullable=True)
    bbox_y      = Column(Integer,     nullable=True)
    bbox_w      = Column(Integer,     nullable=True)
    bbox_h      = Column(Integer,     nullable=True)

    def __repr__(self) -> str:
        return f"<ObjectSighting type={self.object_type!r} at={self.detected_at}>"


# ─── Watchlist ───────────────────────────────────────────────────────────────

class WatchlistEntry(Base):
    """A person on the watchlist — triggers alerts when detected."""
    __tablename__ = "watchlist"

    id          = Column(String(36),  primary_key=True, default=_uuid)
    unique_code = Column(String(32),  nullable=False, index=True)
    label       = Column(String(128), nullable=True)
    reason      = Column(Text,        nullable=True)
    is_active   = Column(Boolean,     nullable=False, default=True)
    created_at  = Column(DateTime,    default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<WatchlistEntry code={self.unique_code!r} active={self.is_active}>"


# ─── Alert ───────────────────────────────────────────────────────────────────

class Alert(Base):
    """Triggered when a watchlisted person is detected or a system event occurs."""
    __tablename__ = "alerts"

    id          = Column(String(36),  primary_key=True, default=_uuid)
    # watchlist | camera_offline. 'crowd' was listed here but never
    # implemented and has been removed: it needs a per-zone occupancy
    # policy that does not exist, and a guessed threshold produces alerts
    # nobody can act on. Re-add it WITH a policy, not before.
    alert_type  = Column(String(32),  nullable=False, index=True)
    severity    = Column(String(16),  nullable=False, default="info")  # info | warning | critical
    title       = Column(String(256), nullable=False)
    message     = Column(Text,        nullable=True)
    unique_code = Column(String(32),  nullable=True, index=True)
    camera_id   = Column(String(64),  nullable=True)
    location_id = Column(String(64),  nullable=True)
    is_read     = Column(Boolean,     nullable=False, default=False)
    created_at  = Column(DateTime,    default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<Alert type={self.alert_type!r} severity={self.severity!r} at={self.created_at}>"


# ─── AuditLog ─────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Immutable record of every access to biometric data: who looked up whom,
    when, from where, and what came back.

    Required for accountability under DPDP Act s.8(4)-(5) (reasonable
    security safeguards + demonstrable compliance) and GDPR Art. 30. Without
    it there is no way to answer "who searched for this person", which is the
    question that matters after a misuse complaint.

    APPEND-ONLY by policy: nothing in the application updates or deletes rows
    here except the retention purge (audit_retention_days). Note the log
    itself contains personal data — the subject_code — so it carries its own
    retention period rather than being kept forever.
    """
    __tablename__ = "audit_log"

    id           = Column(String(36),  primary_key=True, default=_uuid)
    occurred_at  = Column(DateTime,    default=datetime.utcnow, nullable=False, index=True)
    actor        = Column(String(64),  nullable=False, index=True)   # username from the JWT
    actor_role   = Column(String(32),  nullable=True)
    actor_ip     = Column(String(64),  nullable=True)
    action       = Column(String(48),  nullable=False, index=True)
    # e.g. search.by_photo | person.read | person.trail | person.erase |
    #      person.consent_update | purge.run | person.list
    subject_code = Column(String(32),  nullable=True, index=True)    # SDT code acted on
    outcome      = Column(String(32),  nullable=False)               # matched | no_match | denied | ok | error
    detail       = Column(Text,        nullable=True)                # JSON: confidence, counts, reason

    def __repr__(self) -> str:
        return (f"<AuditLog {self.occurred_at} {self.actor} {self.action} "
                f"{self.subject_code} -> {self.outcome}>")


# ─── ErasureReceipt ───────────────────────────────────────────────────────────

class ErasureReceipt(Base):
    """
    Proof that a data-subject erasure actually completed.

    DPDP Act s.12(3) gives a Data Principal the right to erasure, and s.8(7)
    requires the Data Fiduciary to erase when consent is withdrawn. A verbal
    "we deleted it" is not evidence. This row records exactly what was
    destroyed and survives the person it refers to — it deliberately holds NO
    biometric data, only counts and the code, so it is safe to retain as an
    audit artefact after the identity itself is gone.
    """
    __tablename__ = "erasure_receipts"

    id                 = Column(String(36), primary_key=True, default=_uuid)
    unique_code        = Column(String(32), nullable=False, index=True)
    erased_at          = Column(DateTime,   default=datetime.utcnow, nullable=False, index=True)
    erased_by          = Column(String(64), nullable=False)
    reason             = Column(String(64), nullable=False)   # subject_request | consent_withdrawn | retention_expiry | operator
    sightings_deleted  = Column(Integer,    nullable=False, default=0)
    snapshots_deleted  = Column(Integer,    nullable=False, default=0)
    embeddings_cleared = Column(Integer,    nullable=False, default=0)
    verified           = Column(Boolean,    nullable=False, default=False)  # post-delete re-check passed
    detail             = Column(Text,       nullable=True)

    def __repr__(self) -> str:
        return f"<ErasureReceipt {self.unique_code} at={self.erased_at} verified={self.verified}>"
