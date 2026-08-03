"""
tests/conftest.py
──────────────────
Hermetic test environment for the identity-arbitration suite.

NO CAMERA, NO EXTERNAL DATABASE, NO NETWORK, NO MODEL WEIGHTS.

  * DATABASE_URL is redirected to a throwaway SQLite file inside pytest's
    tmp dir BEFORE database.db is imported (it builds its engine at import
    time from that env var). The real smartdetect.db is never opened.
    An on-disk temp file is used rather than :memory: because database/db.py
    creates its own engine without StaticPool, so an in-memory URL would give
    each connection a separate empty database.
  * SMARTDETECT_STUB_MODE=1 makes FaceRecognizer a no-op stub, so InsightFace
    is never loaded. Every embedding in these tests is synthetic and supplied
    explicitly via identify(face_embedding=...).
  * Tests never construct an ObjectDetector or open a VideoCapture.

Identity arbitration is pure decision logic over embeddings; that is exactly
what these tests exercise, with the models removed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Must happen before ANY project import that touches database.db ──────────
_TMPDIR = tempfile.TemporaryDirectory(prefix="smartdetect-tests-")
_DB_PATH = Path(_TMPDIR.name) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SMARTDETECT_STUB_MODE"] = "1"
os.environ["SMARTDETECT_NO_AUTOSTART"] = "1"
os.environ.pop("SMARTDETECT_IDENTITY_CONFIG", None)   # tests set configs explicitly

import numpy as np  # noqa: E402
from database.db import SessionLocal, init_db  # noqa: E402
from database.models import Person, Sighting  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    _TMPDIR.cleanup()


# ─── Synthetic embedding helpers ────────────────────────────────────────────

DIM = 512


def unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def make_embedding(seed: int) -> np.ndarray:
    """A deterministic unit vector. Two different seeds are near-orthogonal
    in 512-D (E[cos] = 0, sd ~ 1/sqrt(512) ~ 0.044), so distinct seeds model
    distinct people far below any match threshold."""
    rng = np.random.default_rng(seed)
    return unit(rng.standard_normal(DIM))


def embedding_at_similarity(base: np.ndarray, target_cos: float,
                            seed: int = 12345) -> np.ndarray:
    """
    Construct a unit vector with EXACTLY `target_cos` cosine similarity to
    `base`, by mixing base with an orthogonal component:

        v = target*base + sqrt(1-target^2)*orth,  orth ⟂ base, |orth| = 1

    This lets a test sit precisely either side of a threshold instead of
    hoping a random vector lands there.
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(DIM).astype(np.float32)
    orth = unit(r - np.dot(r, base) * base)
    t = float(np.clip(target_cos, -1.0, 1.0))
    return unit(t * base + np.sqrt(max(0.0, 1.0 - t * t)) * orth)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _schema():
    init_db()


@pytest.fixture
def db():
    """A session against the temp DB, wiped before each test."""
    s = SessionLocal()
    s.query(Sighting).delete()
    s.query(Person).delete()
    s.commit()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture
def blank_frame():
    """A neutral mid-grey frame. Grey has near-zero saturation, so the
    K-means dress-colour extractor produces a stable, low-saturation value
    that will not accidentally match a colourful stored person."""
    return np.full((240, 120, 3), 128, dtype=np.uint8)


@pytest.fixture
def full_bbox(blank_frame):
    h, w = blank_frame.shape[:2]
    return [0, 0, w, h]


def add_person(db, code: str, face_emb=None, *, templates=None,
               dress_hsv=None, reid_emb=None, minutes_ago: float = 0.0):
    """Insert a Person directly. Bypasses the pipeline on purpose: these
    tests exercise arbitration against a known gallery, not registration."""
    import json
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    seen = now - timedelta(minutes=minutes_ago)
    p = Person(
        unique_code=code,
        face_embedding=json.dumps(face_emb.tolist()) if face_emb is not None else None,
        face_templates=json.dumps([t.tolist() for t in templates]) if templates else None,
        reid_embedding=json.dumps(reid_emb.tolist()) if reid_emb is not None else None,
        dress_color_hsv=json.dumps(dress_hsv) if dress_hsv else None,
        person_type="unknown",
        created_at=now, first_seen_at=seen, last_seen_at=seen,
        total_sightings=1,
    )
    db.add(p)
    db.commit()
    return p
