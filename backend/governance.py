"""
backend/governance.py
──────────────────────
Data-governance layer for biometric data: retention, erasure, audit, consent.

See docs/DATA_GOVERNANCE.md for the legal mapping. This module is the
enforcement mechanism behind it — the document is only true if this code runs.

Design notes
────────────
* Retention is driven by CONSENT STATUS, not one global TTL. A volunteer who
  signed a form and a stranger auto-registered from a live camera cannot
  lawfully be kept for the same period, so they are not.
* 'unknown' is the default and the most aggressive. Anything the system
  registered by itself has no lawful basis for long retention.
* Erasure is VERIFIED: after deleting we re-query and re-stat the filesystem,
  and record the result in an ErasureReceipt. "We deleted it" without a check
  is not evidence.
* Snapshot paths are validated against the snapshots root before unlinking —
  the code comes from a URL path parameter and must never be able to escape.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import AuditLog, ErasureReceipt, Person, Sighting

# ─── Consent vocabulary ──────────────────────────────────────────────────────

CONSENT_CONSENTED = "consented"   # signed form on file; consent_ref identifies it
CONSENT_DATASET = "dataset"       # licensed research corpus (ChokePoint etc.)
CONSENT_UNKNOWN = "unknown"       # auto-registered from a camera; NO lawful basis recorded
CONSENT_VALUES = (CONSENT_CONSENTED, CONSENT_DATASET, CONSENT_UNKNOWN)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class RetentionPolicy:
    """
    Retention periods in days, per consent status. Configurable, but the
    defaults are deliberately short: for biometric data the safe failure mode
    is deleting too early, not too late.

        SMARTDETECT_RETAIN_DAYS_UNKNOWN    (default 7)
        SMARTDETECT_RETAIN_DAYS_CONSENTED  (default 365)
        SMARTDETECT_RETAIN_DAYS_DATASET    (default 3650)
        SMARTDETECT_RETAIN_DAYS_AUDIT      (default 730)

    0 means "purge immediately at next run"; a negative value disables purging
    for that class (requires an explicit, documented decision — it is not the
    default, because "keep forever" is exactly what the DPDP Act's storage
    limitation principle forbids).
    """

    def __init__(self) -> None:
        self.unknown_days = _int_env("SMARTDETECT_RETAIN_DAYS_UNKNOWN", 7)
        self.consented_days = _int_env("SMARTDETECT_RETAIN_DAYS_CONSENTED", 365)
        self.dataset_days = _int_env("SMARTDETECT_RETAIN_DAYS_DATASET", 3650)
        self.audit_days = _int_env("SMARTDETECT_RETAIN_DAYS_AUDIT", 730)

    def days_for(self, consent_status: str) -> int:
        return {
            CONSENT_CONSENTED: self.consented_days,
            CONSENT_DATASET: self.dataset_days,
        }.get(consent_status, self.unknown_days)

    def retain_until(self, consent_status: str,
                     from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Expiry timestamp, or None when retention is disabled for this class."""
        days = self.days_for(consent_status)
        if days < 0:
            return None
        base = from_time or datetime.now(timezone.utc).replace(tzinfo=None)
        return base + timedelta(days=days)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unknown_days": self.unknown_days,
            "consented_days": self.consented_days,
            "dataset_days": self.dataset_days,
            "audit_days": self.audit_days,
        }


POLICY = RetentionPolicy()


# ─── Audit ───────────────────────────────────────────────────────────────────

def audit(db: Session, *, actor: str, action: str, outcome: str,
          subject_code: Optional[str] = None, actor_role: Optional[str] = None,
          actor_ip: Optional[str] = None, detail: Optional[Dict] = None) -> None:
    """
    Append an audit row. Never raises: a logging failure must not break the
    request, but it is surfaced in the application log so it is not silent.
    """
    try:
        db.add(AuditLog(
            occurred_at=datetime.now(timezone.utc).replace(tzinfo=None),
            actor=actor or "unknown", actor_role=actor_role, actor_ip=actor_ip,
            action=action, subject_code=subject_code, outcome=outcome,
            detail=json.dumps(detail) if detail else None,
        ))
        db.commit()
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).error("AUDIT WRITE FAILED %s/%s: %s",
                                          action, subject_code, exc)
        try:
            db.rollback()
        except Exception:
            pass


def client_ip(request) -> Optional[str]:
    """Client IP, honouring X-Forwarded-For when behind the documented proxy."""
    if request is None:
        return None
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


# ─── Snapshot helpers ────────────────────────────────────────────────────────

def _snapshots_root() -> Path:
    return Path("snapshots").resolve()


def person_snapshot_dir(unique_code: str) -> Optional[Path]:
    """
    Resolved snapshot directory for a code, or None if it would escape the
    snapshots root. `unique_code` arrives from a URL path parameter, so a
    value like '../../etc' must not be able to reach shutil.rmtree.
    """
    root = _snapshots_root()
    candidate = (root / unique_code).resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate


def count_snapshots(unique_code: str) -> int:
    d = person_snapshot_dir(unique_code)
    if not d or not d.is_dir():
        return 0
    return sum(1 for _ in d.rglob("*") if _.is_file())


# ─── Erasure ─────────────────────────────────────────────────────────────────

def erase_person(db: Session, unique_code: str, *, actor: str,
                 reason: str = "subject_request",
                 actor_ip: Optional[str] = None) -> Dict[str, Any]:
    """
    Erase every trace of one identity, then VERIFY and record a receipt.

    Removes: Person row (embeddings, templates, re-ID vector, dress colour),
    all Sighting rows, the entire snapshots/<code>/ directory, and any
    in-memory reference cached in a running LiveStream (tracker->code cache,
    live/recent-detections buffers) so a deleted identity cannot keep
    surfacing on a live camera view or GET /persons/live purely because a
    stream resolved it earlier this session.

    Retains: the ErasureReceipt and the AuditLog trail, both of which hold the
    SDT code but no biometric data. Keeping proof-of-deletion is compatible
    with erasure rights and is required to demonstrate compliance.

    Raises LookupError if the code does not exist, PermissionError if the
    person is under legal hold.
    """
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if person is None:
        raise LookupError(f"No person with code {unique_code}")
    if person.legal_hold:
        raise PermissionError(
            f"{unique_code} is under legal hold and cannot be erased. "
            f"Clear the hold explicitly first, recording who authorised it.")

    snap_dir = person_snapshot_dir(unique_code)
    snapshots_before = count_snapshots(unique_code)

    embeddings_cleared = sum(1 for f in (person.face_embedding, person.face_templates,
                                         person.reid_embedding, person.dress_color_hsv)
                             if f)

    sightings_deleted = db.query(Sighting).filter(
        Sighting.person_id == person.id).delete(synchronize_session=False)

    db.delete(person)
    db.commit()

    snapshot_error = None
    if snap_dir and snap_dir.is_dir():
        try:
            shutil.rmtree(snap_dir)
        except OSError as exc:
            snapshot_error = str(exc)

    # Purge any running LiveStream's in-memory cache of this code (tracker ->
    # code map, live/recent-detections buffers) — lazy import to avoid a
    # governance <-> cameras import cycle, and never fatal: a cache-purge
    # failure must not undo a completed database/filesystem erasure.
    live_cache_entries_removed = 0
    try:
        from cameras.live_stream import purge_identity as _purge_live_identity
        live_cache_entries_removed = _purge_live_identity(unique_code)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).error(
            "live-cache purge failed for %s: %s", unique_code, exc)

    # ── Verify: re-query and re-stat rather than trusting the writes ────────
    person_gone = db.query(Person).filter(
        Person.unique_code == unique_code).first() is None
    sightings_gone = db.query(Sighting).filter(
        Sighting.unique_code == unique_code).count() == 0
    snapshots_gone = count_snapshots(unique_code) == 0
    verified = bool(person_gone and sightings_gone and snapshots_gone)

    receipt = ErasureReceipt(
        unique_code=unique_code,
        erased_at=datetime.now(timezone.utc).replace(tzinfo=None),
        erased_by=actor, reason=reason,
        sightings_deleted=int(sightings_deleted or 0),
        snapshots_deleted=snapshots_before,
        embeddings_cleared=embeddings_cleared,
        verified=verified,
        detail=json.dumps({
            "person_row_removed": person_gone,
            "sighting_rows_removed": sightings_gone,
            "snapshot_files_removed": snapshots_gone,
            "snapshot_error": snapshot_error,
            "live_cache_entries_removed": live_cache_entries_removed,
        }),
    )
    db.add(receipt)
    db.commit()

    audit(db, actor=actor, action="person.erase", subject_code=unique_code,
          outcome="ok" if verified else "incomplete", actor_ip=actor_ip,
          detail={"reason": reason, "sightings": int(sightings_deleted or 0),
                  "snapshots": snapshots_before, "verified": verified})

    return {
        "unique_code": unique_code,
        "erased_at": receipt.erased_at.isoformat() + "Z",
        "erased_by": actor,
        "reason": reason,
        "sightings_deleted": int(sightings_deleted or 0),
        "snapshots_deleted": snapshots_before,
        "embeddings_cleared": embeddings_cleared,
        "live_cache_entries_removed": live_cache_entries_removed,
        "verified": verified,
        "receipt_id": receipt.id,
        "snapshot_error": snapshot_error,
    }


# ─── Consent ─────────────────────────────────────────────────────────────────

def set_consent(db: Session, unique_code: str, status: str, *, actor: str,
                consent_ref: Optional[str] = None,
                actor_ip: Optional[str] = None) -> Dict[str, Any]:
    """
    Record the lawful basis for one identity and re-derive its retention date.

    Raising the status (unknown -> consented) extends retention; lowering it
    (consent withdrawn) shortens it, and the next purge run will act on that.
    """
    if status not in CONSENT_VALUES:
        raise ValueError(f"consent_status must be one of {CONSENT_VALUES}")
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if person is None:
        raise LookupError(f"No person with code {unique_code}")

    previous = person.consent_status
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    person.consent_status = status
    person.consent_ref = consent_ref
    person.consent_recorded_at = now
    person.consent_recorded_by = actor
    person.retain_until = POLICY.retain_until(
        status, from_time=person.last_seen_at or person.created_at or now)
    db.commit()

    audit(db, actor=actor, action="person.consent_update", subject_code=unique_code,
          outcome="ok", actor_ip=actor_ip,
          detail={"from": previous, "to": status, "consent_ref": consent_ref,
                  "retain_until": person.retain_until.isoformat() + "Z"
                                  if person.retain_until else None})

    return {
        "unique_code": unique_code,
        "consent_status": status,
        "consent_ref": consent_ref,
        "previous_status": previous,
        "retain_until": person.retain_until.isoformat() + "Z" if person.retain_until else None,
    }


def refresh_retention(db: Session, person: Person) -> None:
    """Recompute retain_until from last_seen_at. Called after a sighting so a
    person who is still being seen does not expire mid-visit."""
    try:
        person.retain_until = POLICY.retain_until(
            person.consent_status or CONSENT_UNKNOWN,
            from_time=person.last_seen_at or person.created_at)
    except Exception:
        pass


# ─── Purge ───────────────────────────────────────────────────────────────────

def find_expired(db: Session, now: Optional[datetime] = None) -> List[Person]:
    """Persons past retention and not under legal hold."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    rows = db.query(Person).filter(Person.legal_hold == False).all()  # noqa: E712
    expired = []
    for p in rows:
        until = p.retain_until
        if until is None:
            # Never stamped (pre-governance row) — derive it now so legacy
            # data is not immortal simply because it predates this feature.
            until = POLICY.retain_until(
                p.consent_status or CONSENT_UNKNOWN,
                from_time=p.last_seen_at or p.created_at or now)
            if until is None:
                continue
        if until <= now:
            expired.append(p)
    return expired


def purge(db: Session, *, actor: str = "retention-job", dry_run: bool = False,
          now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Delete every identity past its retention date, plus expired audit rows.

    dry_run reports what would go without touching anything — always run that
    first on real data.
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    expired = find_expired(db, now=now)

    planned = [{
        "unique_code": p.unique_code,
        "consent_status": p.consent_status,
        "last_seen_at": p.last_seen_at.isoformat() + "Z" if p.last_seen_at else None,
        "retain_until": p.retain_until.isoformat() + "Z" if p.retain_until else None,
        "snapshots": count_snapshots(p.unique_code),
    } for p in expired]

    result: Dict[str, Any] = {
        "dry_run": dry_run,
        "policy": POLICY.as_dict(),
        "ran_at": now.isoformat() + "Z",
        "candidates": len(expired),
        "persons": planned,
        "erased": [],
        "failed": [],
        "audit_rows_purged": 0,
    }
    if dry_run:
        return result

    for p in expired:
        code = p.unique_code
        try:
            result["erased"].append(
                erase_person(db, code, actor=actor, reason="retention_expiry"))
        except Exception as exc:  # noqa: BLE001
            result["failed"].append({"unique_code": code, "error": str(exc)})

    # Audit log carries personal data (subject_code) so it expires too.
    if POLICY.audit_days >= 0:
        cutoff = now - timedelta(days=POLICY.audit_days)
        result["audit_rows_purged"] = db.query(AuditLog).filter(
            AuditLog.occurred_at < cutoff).delete(synchronize_session=False)
        db.commit()

    audit(db, actor=actor, action="purge.run", outcome="ok",
          detail={"candidates": len(expired), "erased": len(result["erased"]),
                  "failed": len(result["failed"]),
                  "audit_rows_purged": result["audit_rows_purged"]})
    return result


def retention_status(db: Session) -> Dict[str, Any]:
    """Dashboard/compliance summary: population by consent status and how
    many are already past retention."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    out: Dict[str, Any] = {"policy": POLICY.as_dict(), "by_consent": {}, "expired_now": 0,
                           "legal_hold": 0, "total": 0}
    for p in db.query(Person).all():
        out["total"] += 1
        s = p.consent_status or CONSENT_UNKNOWN
        out["by_consent"][s] = out["by_consent"].get(s, 0) + 1
        if p.legal_hold:
            out["legal_hold"] += 1
    out["expired_now"] = len(find_expired(db, now=now))
    return out
