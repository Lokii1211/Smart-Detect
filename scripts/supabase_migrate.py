"""
scripts/supabase_migrate.py
────────────────────────────
Migrate SmartDetect from SQLite to Supabase PostgreSQL.

Usage:
    1. Set SUPABASE_DATABASE_URL in .env or as env var
    2. python scripts/supabase_migrate.py --create-tables
    3. python scripts/supabase_migrate.py --migrate-data  (optional: copy SQLite data)

Example .env:
    SUPABASE_DATABASE_URL=postgresql://postgres.xxxx:password@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def get_supabase_url():
    url = os.getenv("SUPABASE_DATABASE_URL") or os.getenv("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        print("ERROR: Set SUPABASE_DATABASE_URL in .env to your Supabase PostgreSQL connection string.")
        print("  Example: postgresql://postgres.xxxx:password@aws-0-ap-south-1.pooler.supabase.com:6543/postgres")
        sys.exit(1)
    return url


def create_tables():
    """Create all SmartDetect tables in Supabase."""
    from sqlalchemy import create_engine
    from database.models import Base

    url = get_supabase_url()
    print(f"Connecting to Supabase: {url[:50]}...")

    engine = create_engine(
        url,
        pool_pre_ping=True,
        echo=False,
    )

    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables created successfully: {tables}")
    engine.dispose()
    return True


def migrate_data():
    """Copy data from local SQLite to Supabase PostgreSQL."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database.models import Location, Camera, Person, Sighting, ObjectSighting

    # Source: local SQLite
    project_root = Path(__file__).resolve().parent.parent
    sqlite_url = f"sqlite:///{project_root / 'smartdetect.db'}"
    sqlite_engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    SqliteSession = sessionmaker(bind=sqlite_engine)

    # Target: Supabase PostgreSQL
    pg_url = get_supabase_url()
    pg_engine = create_engine(pg_url, pool_pre_ping=True)
    PgSession = sessionmaker(bind=pg_engine)

    sqlite_db = SqliteSession()
    pg_db = PgSession()

    try:
        # Migrate locations
        locations = sqlite_db.query(Location).all()
        print(f"Migrating {len(locations)} locations...")
        for loc in locations:
            from sqlalchemy import text
            existing = pg_db.execute(
                text("SELECT id FROM locations WHERE id = :id"), {"id": loc.id}
            ).first()
            if not existing:
                new_loc = Location(
                    id=loc.id, name=loc.name, type=loc.type,
                    address=loc.address, created_at=loc.created_at,
                )
                pg_db.add(new_loc)
        pg_db.commit()
        print(f"  -> {len(locations)} locations migrated")

        # Migrate cameras
        cameras = sqlite_db.query(Camera).all()
        print(f"Migrating {len(cameras)} cameras...")
        for cam in cameras:
            existing = pg_db.execute(
                text("SELECT id FROM cameras WHERE id = :id"), {"id": cam.id}
            ).first()
            if not existing:
                new_cam = Camera(
                    id=cam.id, location_id=cam.location_id, zone_id=cam.zone_id,
                    label=cam.label, source=cam.source, is_active=False,
                    created_at=cam.created_at,
                )
                pg_db.add(new_cam)
        pg_db.commit()
        print(f"  -> {len(cameras)} cameras migrated")

        # Migrate persons
        persons = sqlite_db.query(Person).all()
        print(f"Migrating {len(persons)} persons...")
        for p in persons:
            existing = pg_db.execute(
                text("SELECT id FROM persons WHERE id = :id"), {"id": p.id}
            ).first()
            if not existing:
                new_p = Person(
                    id=p.id, unique_code=p.unique_code,
                    face_embedding=p.face_embedding,
                    reid_embedding=p.reid_embedding,
                    dress_color_hsv=p.dress_color_hsv,
                    body_height_ratio=p.body_height_ratio,
                    created_at=p.created_at,
                    first_seen_at=p.first_seen_at,
                    last_seen_at=p.last_seen_at,
                    total_sightings=p.total_sightings,
                    entry_zone=p.entry_zone,
                    location_id=p.location_id,
                    person_type=p.person_type,
                )
                pg_db.add(new_p)
        pg_db.commit()
        print(f"  -> {len(persons)} persons migrated")

        # Migrate sightings (last 500 only to stay within free tier)
        sightings = sqlite_db.query(Sighting).order_by(Sighting.seen_at.desc()).limit(500).all()
        print(f"Migrating {len(sightings)} recent sightings...")
        for s in sightings:
            existing = pg_db.execute(
                text("SELECT id FROM sightings WHERE id = :id"), {"id": s.id}
            ).first()
            if not existing:
                new_s = Sighting(
                    id=s.id, person_id=s.person_id,
                    unique_code=s.unique_code,
                    location_id=s.location_id, zone_id=s.zone_id,
                    camera_id=s.camera_id, seen_at=s.seen_at,
                    confidence=s.confidence,
                    frame_snapshot_path=s.frame_snapshot_path,
                )
                pg_db.add(new_s)
        pg_db.commit()
        print(f"  -> {len(sightings)} sightings migrated")

        print("\n[OK] Migration complete!")

    except Exception as e:
        pg_db.rollback()
        print(f"\n[ERROR] Migration failed: {e}")
        raise
    finally:
        sqlite_db.close()
        pg_db.close()
        sqlite_engine.dispose()
        pg_engine.dispose()


def verify():
    """Verify Supabase connection and table contents."""
    from sqlalchemy import create_engine, text
    url = get_supabase_url()
    engine = create_engine(url, pool_pre_ping=True)

    with engine.connect() as conn:
        for table in ["locations", "cameras", "persons", "sightings"]:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table}: {count} rows")

    engine.dispose()
    print("\n[OK] Supabase connection verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SmartDetect to Supabase")
    parser.add_argument("--create-tables", action="store_true", help="Create tables in Supabase")
    parser.add_argument("--migrate-data", action="store_true", help="Copy SQLite data to Supabase")
    parser.add_argument("--verify", action="store_true", help="Verify Supabase connection")
    parser.add_argument("--all", action="store_true", help="Create tables + migrate data + verify")
    args = parser.parse_args()

    if args.all or args.create_tables:
        create_tables()
    if args.all or args.migrate_data:
        migrate_data()
    if args.all or args.verify:
        verify()

    if not any([args.create_tables, args.migrate_data, args.verify, args.all]):
        parser.print_help()
