"""
database/db.py
──────────────
SQLAlchemy engine setup — supports both SQLite (local dev) and PostgreSQL (Supabase).

Set DATABASE_URL env var to switch:
  - SQLite (default):  sqlite:///./smartdetect.db
  - Supabase:          postgresql://postgres.[ref]:[pass]@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

load_dotenv()

# Default: SQLite file inside the project root
_project_root = Path(__file__).resolve().parent.parent
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_project_root / 'smartdetect.db'}",
)

_is_sqlite = DATABASE_URL.startswith("sqlite")

# ── Engine configuration ─────────────────────────────────────────────────────
if _is_sqlite:
    # SQLite needs check_same_thread=False for FastAPI's thread-per-request model
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Enable WAL mode + busy timeout for SQLite so concurrent requests don't deadlock
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")   # 30 s
        cursor.close()

else:
    # PostgreSQL (Supabase / Railway / local pg) — connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,    # recycle connections every 30 min
        pool_pre_ping=True,   # verify connections before using
        echo=False,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─── FastAPI Dependency ───────────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Table Initialisation ─────────────────────────────────────────────────────

# Columns added after the first release — create_all() does not alter existing
# tables, so add them here. (name, SQL type) pairs; types valid on both
# SQLite and PostgreSQL.
_PERSONS_NEW_COLUMNS = [
    ("display_name",   "VARCHAR(128)"),
    ("face_templates", "TEXT"),
    ("photo_path",     "TEXT"),
]


def _migrate_new_columns() -> None:
    from sqlalchemy import inspect, text  # noqa: PLC0415
    insp = inspect(engine)
    if not insp.has_table("persons"):
        return
    existing = {c["name"] for c in insp.get_columns("persons")}
    with engine.begin() as conn:
        for name, sql_type in _PERSONS_NEW_COLUMNS:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE persons ADD COLUMN {name} {sql_type}"))


def init_db() -> None:
    """Create all tables. Call once at application startup."""
    from database.models import Base  # noqa: PLC0415
    Base.metadata.create_all(bind=engine)
    _migrate_new_columns()
