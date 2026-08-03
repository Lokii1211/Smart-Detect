"""
scripts/migrate_governance.py
──────────────────────────────
Add the data-governance schema to an EXISTING SmartDetect database.

SQLAlchemy's create_all() creates missing tables but never adds columns to
tables that already exist, so a database created before governance shipped
will fail at startup with "no such column: persons.consent_status". This
script closes that gap.

Idempotent: safe to run repeatedly, and safe to run on a fresh database.

    python scripts/migrate_governance.py           # migrate
    python scripts/migrate_governance.py --check   # report only, change nothing

BACKFILL POLICY — read before running on real data
──────────────────────────────────────────────────
Existing identities were registered with no consent recorded, so they are
backfilled as 'unknown', which carries the SHORTEST retention (7 days by
default from last_seen_at). That is deliberate and is the legally defensible
default: the system has no evidence of a lawful basis for them.

Consequence: on a database whose people were last seen more than 7 days ago,
the very next purge run will erase them. Run

    python scripts/purge_expired.py --status

immediately after migrating to see what is now expiring, and record consent
(PUT /persons/{code}/consent) for anyone who should be kept BEFORE running a
purge with --execute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect, text  # noqa: E402

# column name -> DDL type (SQLite-compatible; also valid on PostgreSQL)
_PERSON_COLUMNS = {
    "consent_status":      "VARCHAR(16) NOT NULL DEFAULT 'unknown'",
    "consent_ref":         "VARCHAR(128)",
    "consent_recorded_at": "DATETIME",
    "consent_recorded_by": "VARCHAR(64)",
    "retain_until":        "DATETIME",
    "legal_hold":          "BOOLEAN NOT NULL DEFAULT 0",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    from database.db import engine, init_db, DATABASE_URL
    from database.models import Base

    print(f"database: {DATABASE_URL}")
    insp = inspect(engine)

    if "persons" not in insp.get_table_names():
        print("No 'persons' table — this is a fresh database.")
        if not args.check:
            init_db()
            print("Created full schema including governance tables.")
        return

    existing = {c["name"] for c in insp.get_columns("persons")}
    missing = [c for c in _PERSON_COLUMNS if c not in existing]

    tables = set(insp.get_table_names())
    missing_tables = [t for t in ("audit_log", "erasure_receipts") if t not in tables]

    print(f"persons: {len(missing)} governance column(s) missing "
          f"{missing if missing else ''}")
    print(f"tables : {len(missing_tables)} missing "
          f"{missing_tables if missing_tables else ''}")

    if args.check:
        print("\n--check: no changes made.")
        if missing or missing_tables:
            print("Run without --check to migrate.")
            sys.exit(1)
        print("Schema is up to date.")
        return

    if not missing and not missing_tables:
        print("\nNothing to do — schema already current.")
        return

    # New tables first (create_all skips existing ones).
    if missing_tables:
        Base.metadata.create_all(bind=engine)
        print(f"Created tables: {', '.join(missing_tables)}")

    if missing:
        with engine.begin() as conn:
            for col in missing:
                ddl = _PERSON_COLUMNS[col]
                conn.execute(text(f"ALTER TABLE persons ADD COLUMN {col} {ddl}"))
                print(f"  + persons.{col}")

        # Stamp retention on rows that predate the feature so they are not
        # immortal. Uses last_seen_at (fallback created_at) + unknown TTL.
        from datetime import timedelta
        from backend.governance import POLICY
        days = POLICY.unknown_days
        if days >= 0:
            with engine.begin() as conn:
                conn.execute(text(
                    "UPDATE persons SET retain_until = "
                    "  datetime(COALESCE(last_seen_at, created_at), :delta) "
                    "WHERE retain_until IS NULL"
                ), {"delta": f"+{days} days"})
            print(f"  backfilled retain_until = last_seen + {days}d "
                  f"for pre-existing rows (consent_status='unknown')")

    print("\nMigration complete.")
    print("\nNEXT: these identities now have consent_status='unknown' and the")
    print("shortest retention. Check what that means before purging:")
    print("    python scripts/purge_expired.py --status")
    print("Record consent for anyone who should be retained BEFORE running")
    print("a purge with --execute.")


if __name__ == "__main__":
    main()
