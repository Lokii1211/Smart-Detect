"""
scripts/seed_db.py
───────────────────
Seed locations directly into the SQLite database.
Run this ONCE from the project root:
    python scripts/seed_db.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.db import SessionLocal, init_db
from database.models import Location

LOCATIONS = [
    {"id": "LOC-001", "name": "Central Station",   "type": "metro", "address": "City Centre, Line 1 & 2"},
    {"id": "LOC-002", "name": "Airport Terminal",  "type": "metro", "address": "International Airport, Line 3"},
    {"id": "LOC-003", "name": "North Junction",    "type": "metro", "address": "North District, Line 1"},
    {"id": "LOC-004", "name": "South Gate",        "type": "metro", "address": "South District, Line 2"},
    {"id": "LOC-005", "name": "East Plaza",        "type": "metro", "address": "East Commercial Zone, Line 2"},
    {"id": "LOC-006", "name": "West Terminal",     "type": "metro", "address": "West Residential, Line 1"},
    {"id": "LOC-007", "name": "University Stop",   "type": "metro", "address": "University District, Line 3"},
    {"id": "LOC-008", "name": "Market Square",     "type": "metro", "address": "Old Town Market, Line 1 & 3"},
]

def main():
    init_db()
    db = SessionLocal()
    try:
        added = 0
        for loc in LOCATIONS:
            exists = db.query(Location).filter(Location.id == loc["id"]).first()
            if exists:
                print(f"  exists : {loc['id']} — {loc['name']}")
            else:
                db.add(Location(**loc))
                added += 1
                print(f"  added  : {loc['id']} — {loc['name']}")
        db.commit()
        print(f"\nDone — {added} new locations added.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
