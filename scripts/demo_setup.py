"""
scripts/demo_setup.py
──────────────────────
SmartDetect demo data seeder — STANDALONE (no backend imports needed).

Seeds the SQLite database with:
  • 3 locations (mall, campus, airport)
  • 6 cameras (2 per location — seeds CAM-001 through CAM-006)
  • 10 persons (SDT-0001 … SDT-0010) across those locations
  • Realistic sightings across multiple zones/cameras
  • 5 bag + 3 vehicle object detections per location per day

Usage:
    cd C:\\Users\\lalit\\OneDrive\\Desktop\\Project
    python scripts/demo_setup.py
    python scripts/demo_setup.py --days 3
    python scripts/demo_setup.py --wipe
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import uuid
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import SessionLocal, init_db
from database.models import Camera, Location, ObjectSighting, Person, Sighting

# ─── Demo configuration ───────────────────────────────────────────────────────

DEMO_LOCATIONS = [
    {"id": "LOC-001", "name": "Phoenix Mall",          "type": "mall",    "address": "123 Mall Rd"},
    {"id": "LOC-002", "name": "City Campus",           "type": "campus",  "address": "456 University Ave"},
    {"id": "LOC-003", "name": "International Airport", "type": "airport", "address": "789 Airport Blvd"},
]

DEMO_CAMERAS = [
    # LOC-001 Phoenix Mall
    {"id": "CAM-001", "location_id": "LOC-001", "zone_id": "entrance",   "label": "Main Entrance Camera", "source": "0"},
    {"id": "CAM-002", "location_id": "LOC-001", "zone_id": "food-court", "label": "Food Court Camera",    "source": "1"},
    # LOC-002 City Campus
    {"id": "CAM-003", "location_id": "LOC-002", "zone_id": "gate-a",     "label": "Gate A Camera",        "source": "2"},
    {"id": "CAM-004", "location_id": "LOC-002", "zone_id": "library",    "label": "Library Camera",       "source": "3"},
    # LOC-003 Airport
    {"id": "CAM-005", "location_id": "LOC-003", "zone_id": "terminal",   "label": "Terminal Camera",      "source": "4"},
    {"id": "CAM-006", "location_id": "LOC-003", "zone_id": "parking",    "label": "Parking Camera",       "source": "5"},
]

DEMO_PERSONAS = [
    {"name": "Alice",    "type": "visitor", "location": "LOC-001", "zones": ["entrance", "food-court", "west-wing"]},
    {"name": "Bob",      "type": "visitor", "location": "LOC-001", "zones": ["entrance", "east-wing", "parking"]},
    {"name": "Charlie",  "type": "staff",   "location": "LOC-001", "zones": ["staff-room", "control-room"]},
    {"name": "Diana",    "type": "visitor", "location": "LOC-002", "zones": ["gate-a", "library", "cafeteria"]},
    {"name": "Eve",      "type": "staff",   "location": "LOC-002", "zones": ["admin-block", "lab-a", "library"]},
    {"name": "Frank",    "type": "visitor", "location": "LOC-002", "zones": ["gate-a", "sports-hall"]},
    {"name": "Grace",    "type": "visitor", "location": "LOC-003", "zones": ["check-in", "gate-a", "departure"]},
    {"name": "Henry",    "type": "visitor", "location": "LOC-003", "zones": ["arrivals", "gate-b", "exit"]},
    {"name": "Isabelle", "type": "staff",   "location": "LOC-003", "zones": ["security", "gate-a", "gate-b"]},
    {"name": "Jack",     "type": "unknown", "location": "LOC-001", "zones": ["entrance", "east-wing"]},
]

BAG_TYPES     = ["backpack", "handbag", "suitcase"]
VEHICLE_TYPES = ["car", "motorcycle", "truck"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_embedding(seed: int) -> list:
    rng  = np.random.default_rng(seed=seed * 997 + 1)
    vec  = rng.standard_normal(512).astype(np.float32)
    vec /= float(np.linalg.norm(vec)) + 1e-8
    return vec.tolist()


def _random_ts(base: datetime, h_start: int, h_end: int) -> datetime:
    lo  = h_start * 3600
    hi  = max(lo + 60, h_end * 3600 - 60)
    off = random.randint(lo, hi)
    return base.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=off)


# ─── Main ─────────────────────────────────────────────────────────────────────

def setup_demo(days: int = 1, wipe: bool = False) -> None:
    init_db()
    db = SessionLocal()

    try:
        # ── Wipe ─────────────────────────────────────────────────────────
        if wipe:
            print("  Wiping demo data…")
            for p in db.query(Person).filter(Person.unique_code.like("SDT-%")).all():
                db.delete(p)
            db.query(ObjectSighting).delete()
            db.query(Camera).delete()
            db.commit()
            print("  Done.\n")

        # ── Locations ────────────────────────────────────────────────────
        print("  Ensuring 3 demo locations…")
        for loc in DEMO_LOCATIONS:
            if not db.query(Location).filter(Location.id == loc["id"]).first():
                db.add(Location(
                    id=loc["id"], name=loc["name"],
                    type=loc["type"], address=loc["address"],
                    created_at=datetime.utcnow(),
                ))
        db.commit()
        print("  ✓ Locations ready\n")

        # ── Cameras ──────────────────────────────────────────────────────
        print("  Ensuring 6 demo cameras (2 per location)…")
        for cam in DEMO_CAMERAS:
            if not db.query(Camera).filter(Camera.id == cam["id"]).first():
                db.add(Camera(
                    id=cam["id"],
                    location_id=cam["location_id"],
                    zone_id=cam["zone_id"],
                    label=cam["label"],
                    source=cam["source"],
                    is_active=False,
                    created_at=datetime.utcnow(),
                ))
                print(f"    ✓  {cam['id']} — {cam['label']} @ {cam['location_id']}/{cam['zone_id']}")
            else:
                print(f"    ↩  {cam['id']} — {cam['label']} (already exists)")
        db.commit()
        print("  ✓ Cameras ready\n")

        # ── Register 10 demo persons ──────────────────────────────────────
        today = datetime.utcnow()
        registered: list[tuple[str, dict]] = []

        print(f"  Registering {len(DEMO_PERSONAS)} demo persons…")
        for idx, persona in enumerate(DEMO_PERSONAS):
            code = f"SDT-{idx+1:04d}"
            if db.query(Person).filter(Person.unique_code == code).first():
                print(f"    ↩  {code} — {persona['name']} (already exists)")
            else:
                db.add(Person(
                    id=str(uuid.uuid4()),
                    unique_code=code,
                    face_embedding=json.dumps(_make_embedding(idx)),
                    created_at=today - timedelta(days=days),
                    entry_zone=persona["zones"][0],
                    location_id=persona["location"],
                    person_type=persona["type"],
                ))
                print(f"    ✓  {code} — {persona['name']} ({persona['type']}) @ {persona['location']}")
            registered.append((code, persona))

        db.commit()

        # ── Simulate sightings ────────────────────────────────────────────
        print(f"\n  Simulating {days} day(s) of movements…")
        total_s = 0
        total_o = 0

        for day_off in range(days):
            base = today - timedelta(days=day_off)

            for code, persona in registered:
                p = db.query(Person).filter(Person.unique_code == code).first()
                if not p:
                    continue
                for zone in persona["zones"]:
                    h  = random.randint(7, 18)
                    ts = _random_ts(base, h, h + 2) + timedelta(seconds=random.randint(-120, 120))
                    # Pick a camera from this location/zone if available
                    cam_match = next(
                        (c["id"] for c in DEMO_CAMERAS
                         if c["location_id"] == persona["location"] and c["zone_id"] == zone),
                        f"CAM-{persona['location']}-{persona['zones'].index(zone)+1:02d}",
                    )
                    db.add(Sighting(
                        id=str(uuid.uuid4()),
                        person_id=p.id,
                        unique_code=code,
                        location_id=persona["location"],
                        zone_id=zone,
                        camera_id=cam_match,
                        seen_at=ts,
                        confidence=round(random.uniform(0.72, 0.98), 3),
                    ))
                    total_s += 1

            for loc in DEMO_LOCATIONS:
                for _ in range(5):
                    ts = _random_ts(base, 8, 20)
                    db.add(ObjectSighting(
                        id=str(uuid.uuid4()),
                        location_id=loc["id"],
                        zone_id=random.choice(["entrance", "main-hall", "parking"]),
                        camera_id=f"CAM-{loc['id']}-01",
                        object_type=random.choice(BAG_TYPES),
                        confidence=round(random.uniform(0.65, 0.97), 3),
                        detected_at=ts,
                        bbox_x=random.randint(50, 400), bbox_y=random.randint(50, 300),
                        bbox_w=random.randint(60, 200), bbox_h=random.randint(60, 200),
                    ))
                    total_o += 1
                for _ in range(3):
                    ts = _random_ts(base, 6, 22)
                    db.add(ObjectSighting(
                        id=str(uuid.uuid4()),
                        location_id=loc["id"],
                        zone_id="parking",
                        camera_id=f"CAM-{loc['id']}-02",
                        object_type=random.choice(VEHICLE_TYPES),
                        confidence=round(random.uniform(0.70, 0.99), 3),
                        detected_at=ts,
                        bbox_x=random.randint(10, 300), bbox_y=random.randint(50, 250),
                        bbox_w=random.randint(150, 400), bbox_h=random.randint(100, 300),
                    ))
                    total_o += 1

            db.commit()
            print(f"    Day -{day_off}: {total_s} person sightings, {total_o} object detections")

        print(f"\n  ✓ Demo complete!")
        print(f"    Locations         : {len(DEMO_LOCATIONS)}")
        print(f"    Cameras           : {len(DEMO_CAMERAS)} (2 per location)")
        print(f"    Persons           : {len(DEMO_PERSONAS)}")
        print(f"    Person sightings  : {total_s}")
        print(f"    Object detections : {total_o}")
        print(f"\n  Camera IDs: CAM-001 through CAM-006")
        print(f"  Start via: POST /camera/start {{\"camera_id\": \"CAM-001\"}}")

    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartDetect Demo Setup")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--wipe", action="store_true")
    args = parser.parse_args()

    print("\n══════════════════════════════════════")
    print("  SmartDetect — Demo Data Setup")
    print("══════════════════════════════════════\n")

    setup_demo(days=args.days, wipe=args.wipe)
    print("\nMULTI-CAMERA COMPLETE — SmartDetect is ready\n")


if __name__ == "__main__":
    main()
