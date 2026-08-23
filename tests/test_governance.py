"""
tests/test_governance.py
─────────────────────────
Tests for the biometric data-governance layer.

The claims in docs/DATA_GOVERNANCE.md are only true if this code behaves as
described, so each test pins one documented obligation. Erasure is checked by
re-reading the database and the filesystem, not by trusting return values —
an erasure routine that reports success without deleting is the worst
possible failure here.

Hermetic: temp DB + temp snapshots dir, no camera, no models (conftest.py).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from conftest import add_person, make_embedding

from backend import governance as gov
from database.models import AuditLog, ErasureReceipt, Person, Sighting


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    """Redirect the snapshots root into tmp so no real photo is ever touched."""
    root = tmp_path / "snapshots"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    return root


def make_snapshots(root: Path, code: str, n: int = 3) -> Path:
    d = root / code
    d.mkdir(parents=True, exist_ok=True)
    (d / "registered.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    for i in range(n - 1):
        (d / f"sighting_{i}.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    return d


def add_sightings(db, person, n: int = 4):
    for i in range(n):
        db.add(Sighting(person_id=person.id, unique_code=person.unique_code,
                        location_id="LOC-1", zone_id="z", camera_id="CAM-1",
                        seen_at=datetime.now(timezone.utc).replace(tzinfo=None),
                        confidence=0.9))
    db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Retention policy
# ═══════════════════════════════════════════════════════════════════════════

class TestRetentionPolicy:

    def test_unknown_consent_has_the_shortest_retention(self):
        """Anyone auto-registered from a camera has no recorded lawful basis,
        so they must expire soonest — not be kept like a consenting volunteer."""
        p = gov.RetentionPolicy()
        assert p.days_for(gov.CONSENT_UNKNOWN) < p.days_for(gov.CONSENT_CONSENTED)
        assert p.days_for(gov.CONSENT_CONSENTED) < p.days_for(gov.CONSENT_DATASET)

    def test_defaults_are_documented_values(self):
        p = gov.RetentionPolicy()
        assert p.unknown_days == 7
        assert p.consented_days == 365
        assert p.dataset_days == 3650
        assert p.audit_days == 730

    def test_policy_is_env_configurable(self, monkeypatch):
        monkeypatch.setenv("SMARTDETECT_RETAIN_DAYS_UNKNOWN", "3")
        monkeypatch.setenv("SMARTDETECT_RETAIN_DAYS_CONSENTED", "90")
        p = gov.RetentionPolicy()
        assert p.unknown_days == 3 and p.consented_days == 90

    def test_negative_days_disables_expiry(self, monkeypatch):
        """Opt-out exists but must be explicit — never the default."""
        monkeypatch.setenv("SMARTDETECT_RETAIN_DAYS_DATASET", "-1")
        p = gov.RetentionPolicy()
        assert p.retain_until(gov.CONSENT_DATASET) is None

    def test_retain_until_is_computed_from_the_reference_time(self):
        p = gov.RetentionPolicy()
        base = datetime(2026, 1, 1)
        assert p.retain_until(gov.CONSENT_UNKNOWN, from_time=base) == base + timedelta(days=7)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Erasure — verified and complete
# ═══════════════════════════════════════════════════════════════════════════

class TestErasure:

    def test_erasure_removes_person_sightings_and_snapshots(self, db, snapshots):
        p = add_person(db, "SDT-0001", face_emb=make_embedding(1))
        p.reid_embedding = json.dumps([0.1] * 512)
        p.dress_color_hsv = json.dumps({"hue": 1, "saturation": 2, "value": 3})
        db.commit()
        add_sightings(db, p, 4)
        make_snapshots(snapshots, "SDT-0001", 3)

        res = gov.erase_person(db, "SDT-0001", actor="admin")

        # Re-read from the database rather than trusting the return value
        assert db.query(Person).filter(Person.unique_code == "SDT-0001").first() is None
        assert db.query(Sighting).filter(Sighting.unique_code == "SDT-0001").count() == 0
        assert not (snapshots / "SDT-0001").exists()
        assert res["verified"] is True
        assert res["sightings_deleted"] == 4
        assert res["snapshots_deleted"] == 3

    def test_erasure_writes_a_verified_receipt(self, db, snapshots):
        p = add_person(db, "SDT-0002", face_emb=make_embedding(2))
        add_sightings(db, p, 2)
        make_snapshots(snapshots, "SDT-0002", 2)

        res = gov.erase_person(db, "SDT-0002", actor="admin", reason="consent_withdrawn")

        r = db.query(ErasureReceipt).filter(
            ErasureReceipt.unique_code == "SDT-0002").first()
        assert r is not None, "an erasure with no receipt cannot be evidenced"
        assert r.verified is True
        assert r.reason == "consent_withdrawn"
        assert r.erased_by == "admin"
        assert r.id == res["receipt_id"]

    def test_receipt_survives_the_person(self, db, snapshots):
        """Proof of deletion must outlive the deleted record, or an erasure
        request cannot be demonstrated as honoured."""
        p = add_person(db, "SDT-0003", face_emb=make_embedding(3))
        gov.erase_person(db, "SDT-0003", actor="admin")
        assert db.query(Person).filter(Person.unique_code == "SDT-0003").first() is None
        assert db.query(ErasureReceipt).filter(
            ErasureReceipt.unique_code == "SDT-0003").count() == 1

    def test_receipt_contains_no_biometric_data(self, db, snapshots):
        """
        The receipt is retained AFTER erasure, so if it carried the embedding
        it would defeat the erasure entirely. Assert no stored value looks
        like a feature vector.
        """
        import re
        emb = make_embedding(4)
        p = add_person(db, "SDT-0004", face_emb=emb)
        p.reid_embedding = json.dumps([float(x) for x in emb[:64]])
        db.commit()
        gov.erase_person(db, "SDT-0004", actor="admin")

        r = db.query(ErasureReceipt).filter(
            ErasureReceipt.unique_code == "SDT-0004").first()
        values = " ".join(str(getattr(r, c.name)) for c in r.__table__.columns)

        # A vector would show up as a run of comma-separated decimals.
        assert not re.search(r"-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+", values), (
            "erasure receipt appears to contain a feature vector")
        # And no fragment of the actual embedding survives.
        assert f"{float(emb[0]):.6f}" not in values
        # It should retain only counts + provenance.
        assert r.embeddings_cleared >= 1
        assert r.unique_code == "SDT-0004"

    def test_legal_hold_blocks_erasure(self, db, snapshots):
        p = add_person(db, "SDT-0005", face_emb=make_embedding(5))
        p.legal_hold = True
        db.commit()
        with pytest.raises(PermissionError):
            gov.erase_person(db, "SDT-0005", actor="admin")
        assert db.query(Person).filter(Person.unique_code == "SDT-0005").first() is not None

    def test_unknown_code_raises_lookup_error(self, db, snapshots):
        with pytest.raises(LookupError):
            gov.erase_person(db, "SDT-9999", actor="admin")

    def test_erasure_is_audited(self, db, snapshots):
        add_person(db, "SDT-0006", face_emb=make_embedding(6))
        gov.erase_person(db, "SDT-0006", actor="alice", actor_ip="10.0.0.9")
        row = db.query(AuditLog).filter(AuditLog.action == "person.erase",
                                        AuditLog.subject_code == "SDT-0006").first()
        assert row is not None and row.actor == "alice" and row.outcome == "ok"

    def test_path_traversal_in_code_cannot_escape_snapshots_root(self, snapshots):
        """`unique_code` reaches this from a URL path parameter. It must never
        resolve outside snapshots/, or erasure becomes arbitrary file deletion."""
        for evil in ("../../etc", "..", "/etc", "a/../../..", ""):
            assert gov.person_snapshot_dir(evil) is None, evil
        assert gov.person_snapshot_dir("SDT-0001") is not None


# ═══════════════════════════════════════════════════════════════════════════
# 3. Consent register
# ═══════════════════════════════════════════════════════════════════════════

class TestConsent:

    def test_new_identities_default_to_unknown(self, db):
        p = add_person(db, "SDT-0010", face_emb=make_embedding(10))
        assert p.consent_status == gov.CONSENT_UNKNOWN, (
            "auto-registered people must not default to a consented state")

    def test_recording_consent_extends_retention(self, db):
        p = add_person(db, "SDT-0011", face_emb=make_embedding(11))
        p.retain_until = None
        db.commit()
        res = gov.set_consent(db, "SDT-0011", gov.CONSENT_CONSENTED,
                              actor="admin", consent_ref="FORM-2026-014")
        db.refresh(p)
        assert p.consent_status == "consented"
        assert p.consent_ref == "FORM-2026-014"
        assert p.consent_recorded_by == "admin"
        assert p.retain_until is not None
        assert res["previous_status"] == "unknown"

    def test_withdrawing_consent_shortens_retention(self, db):
        """Consent withdrawal must bring the expiry forward so the next purge
        acts on it — DPDP s.8(7)."""
        p = add_person(db, "SDT-0012", face_emb=make_embedding(12))
        gov.set_consent(db, "SDT-0012", gov.CONSENT_CONSENTED, actor="admin")
        db.refresh(p)
        long_expiry = p.retain_until
        gov.set_consent(db, "SDT-0012", gov.CONSENT_UNKNOWN, actor="admin")
        db.refresh(p)
        assert p.retain_until < long_expiry

    def test_invalid_status_rejected(self, db):
        add_person(db, "SDT-0013", face_emb=make_embedding(13))
        with pytest.raises(ValueError):
            gov.set_consent(db, "SDT-0013", "yes-probably", actor="admin")

    def test_consent_change_is_audited(self, db):
        add_person(db, "SDT-0014", face_emb=make_embedding(14))
        gov.set_consent(db, "SDT-0014", gov.CONSENT_DATASET, actor="bob")
        row = db.query(AuditLog).filter(
            AuditLog.action == "person.consent_update",
            AuditLog.subject_code == "SDT-0014").first()
        assert row is not None and row.actor == "bob"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Purge job
# ═══════════════════════════════════════════════════════════════════════════

class TestPurge:

    def _expired(self, db, code, days_ago=30, consent=gov.CONSENT_UNKNOWN):
        p = add_person(db, code, face_emb=make_embedding(hash(code) % 1000))
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)
        p.consent_status = consent
        p.last_seen_at = past
        p.retain_until = past + timedelta(days=1)
        db.commit()
        return p

    def test_dry_run_deletes_nothing(self, db, snapshots):
        self._expired(db, "SDT-0020")
        make_snapshots(snapshots, "SDT-0020", 2)
        res = gov.purge(db, dry_run=True)
        assert res["dry_run"] is True and res["candidates"] == 1
        assert res["erased"] == []
        assert db.query(Person).filter(Person.unique_code == "SDT-0020").first() is not None
        assert (snapshots / "SDT-0020").exists()

    def test_execute_erases_expired(self, db, snapshots):
        self._expired(db, "SDT-0021")
        make_snapshots(snapshots, "SDT-0021", 2)
        res = gov.purge(db, dry_run=False)
        assert len(res["erased"]) == 1 and res["erased"][0]["verified"] is True
        assert db.query(Person).filter(Person.unique_code == "SDT-0021").first() is None
        assert not (snapshots / "SDT-0021").exists()

    def test_unexpired_survives(self, db, snapshots):
        p = add_person(db, "SDT-0022", face_emb=make_embedding(22))
        p.retain_until = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=5)
        db.commit()
        gov.purge(db, dry_run=False)
        assert db.query(Person).filter(Person.unique_code == "SDT-0022").first() is not None

    def test_legal_hold_survives_purge(self, db, snapshots):
        p = self._expired(db, "SDT-0023")
        p.legal_hold = True
        db.commit()
        res = gov.purge(db, dry_run=False)
        assert all(e["unique_code"] != "SDT-0023" for e in res["erased"])
        assert db.query(Person).filter(Person.unique_code == "SDT-0023").first() is not None

    def test_legacy_rows_without_retain_until_are_still_expired(self, db, snapshots):
        """Rows created before governance existed must not be immortal simply
        because they have no retain_until stamp."""
        p = add_person(db, "SDT-0024", face_emb=make_embedding(24))
        p.retain_until = None
        p.last_seen_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
        db.commit()
        assert any(x.unique_code == "SDT-0024" for x in gov.find_expired(db))

    def test_purge_trims_the_audit_log(self, db, snapshots, monkeypatch):
        monkeypatch.setenv("SMARTDETECT_RETAIN_DAYS_AUDIT", "30")
        monkeypatch.setattr(gov, "POLICY", gov.RetentionPolicy())
        old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=60)
        db.add(AuditLog(occurred_at=old, actor="x", action="person.read",
                        subject_code="SDT-OLD", outcome="ok"))
        db.commit()
        res = gov.purge(db, dry_run=False)
        assert res["audit_rows_purged"] >= 1
        assert db.query(AuditLog).filter(AuditLog.subject_code == "SDT-OLD").count() == 0

    def test_purge_run_is_audited(self, db, snapshots):
        gov.purge(db, dry_run=False, actor="cron")
        assert db.query(AuditLog).filter(AuditLog.action == "purge.run").count() >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Audit log
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditLog:

    def test_audit_records_actor_subject_and_outcome(self, db):
        gov.audit(db, actor="alice", actor_role="operator", actor_ip="192.0.2.5",
                  action="search.by_photo", subject_code="SDT-0030",
                  outcome="matched", detail={"confidence": 0.91})
        r = db.query(AuditLog).filter(AuditLog.subject_code == "SDT-0030").first()
        assert r.actor == "alice" and r.actor_role == "operator"
        assert r.actor_ip == "192.0.2.5" and r.outcome == "matched"
        assert json.loads(r.detail)["confidence"] == 0.91

    def test_audit_never_raises_into_the_request(self, db):
        """A logging failure must not break the endpoint it is logging."""
        class Broken:
            def add(self, *a): raise RuntimeError("db down")
            def commit(self): raise RuntimeError("db down")
            def rollback(self): raise RuntimeError("still down")
        gov.audit(Broken(), actor="x", action="person.read", outcome="ok")

    def test_client_ip_prefers_forwarded_header(self):
        class Req:
            headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
            client = type("C", (), {"host": "10.0.0.1"})()
        assert gov.client_ip(Req()) == "203.0.113.7"

    def test_client_ip_falls_back_to_socket(self):
        class Req:
            headers = {}
            client = type("C", (), {"host": "10.0.0.4"})()
        assert gov.client_ip(Req()) == "10.0.0.4"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Retention status reporting
# ═══════════════════════════════════════════════════════════════════════════

class TestErasureEndToEnd:
    """
    Full-stack proof for the CONSENT_FORM.md promise: register a person
    through the REAL identification pipeline (not the add_person() bypass
    used elsewhere in this file), populate a live camera's in-memory
    identity cache exactly as a running LiveStream worker thread would, then
    delete and verify every item traced in the deletion audit is gone —
    including the in-memory cache, which erase_person() did not touch before
    this fix (see cameras.live_stream.purge_identity).
    """

    def test_full_deletion_removes_every_trace(self, db, snapshots):
        import time as _time

        import cameras.live_stream as live_stream_mod
        from cameras.live_stream import LiveStream
        from database.queries import find_person_by_embedding, log_sighting
        from recognition.smart_identifier import SmartIdentifier

        # ── Register a person from a real frame through the real pipeline ──
        frame = np.full((240, 120, 3), 128, dtype=np.uint8)
        bbox = [0, 0, 120, 240]
        face_emb = make_embedding(777)

        identifier = SmartIdentifier()
        result = identifier.identify(
            frame, bbox, db, allow_new=True, face_embedding=face_emb,
            face_pose=None, extract_face_if_missing=False,
        )
        assert result["method"] == "new_registration"
        code = result["unique_code"]

        person = db.query(Person).filter(Person.unique_code == code).first()
        assert person is not None
        assert person.photo_path, "registration must write a snapshot"
        assert Path(person.photo_path).is_file(), "registration snapshot missing on disk"

        # A per-sighting snapshot too — registration alone is not the whole
        # picture the promise covers.
        sighting_photo = snapshots / code / "sighting_0.jpg"
        sighting_photo.write_bytes(b"\xff\xd8fake-jpeg")
        assert log_sighting(code, "LOC-1", "z", "CAM-TEST", 0.9, db,
                            frame_path=str(sighting_photo))

        # ── Confirm searchable before deletion ──────────────────────────────
        match = find_person_by_embedding(face_emb, db=db, threshold=0.56)
        assert match is not None and match["unique_code"] == code

        # ── Populate a running LiveStream's in-memory caches, as the worker
        # thread does after resolving this person on camera (constructed but
        # never start()ed: no capture thread, no cv2.VideoCapture) ──────────
        stream = LiveStream(source=0, location_id="LOC-1", zone_id="z",
                            camera_id="CAM-TEST")
        stream._track_codes[42] = {"code": code, "method": "face", "conf": 0.9}
        stream.active_tracks[42] = code
        stream._seen_cache[code] = _time.time()
        stream._detections.append({"unique_code": code, "method": "face",
                                   "confidence": 0.9, "color_hex": None})
        live_stream_mod._REGISTRY["CAM-TEST"] = stream
        try:
            # ── Delete ────────────────────────────────────────────────────
            res = gov.erase_person(db, code, actor="admin")
            assert res["verified"] is True
            assert res["live_cache_entries_removed"] >= 2  # track + seen_cache

            # ── Part A checklist, verified one item at a time ───────────────
            assert db.query(Person).filter(Person.unique_code == code).first() is None, \
                "person row"
            assert db.query(Sighting).filter(Sighting.unique_code == code).count() == 0, \
                "sighting rows"
            assert not (snapshots / code).exists(), \
                "snapshot files (registration + per-sighting)"
            assert all(v.get("code") != code for v in stream._track_codes.values()), \
                "LiveStream track cache"
            assert 42 not in stream.active_tracks
            assert code not in stream._seen_cache
            assert all(d.get("unique_code") != code for d in stream._detections)

            # ── The promise in eval/data/CONSENT_FORM.md: a photo search for
            # this person, after deletion, returns nothing ──────────────────
            assert find_person_by_embedding(face_emb, db=db, threshold=0.56) is None, \
                "deleted person is still searchable by face"
        finally:
            live_stream_mod._REGISTRY.pop("CAM-TEST", None)


class TestRetentionStatus:

    def test_status_counts_population_by_consent(self, db):
        add_person(db, "SDT-0040", face_emb=make_embedding(40))
        p2 = add_person(db, "SDT-0041", face_emb=make_embedding(41))
        gov.set_consent(db, "SDT-0041", gov.CONSENT_CONSENTED, actor="admin")
        st = gov.retention_status(db)
        assert st["total"] == 2
        assert st["by_consent"]["unknown"] == 1
        assert st["by_consent"]["consented"] == 1

    def test_status_reports_expired_count(self, db):
        p = add_person(db, "SDT-0042", face_emb=make_embedding(42))
        p.retain_until = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        db.commit()
        assert gov.retention_status(db)["expired_now"] == 1
