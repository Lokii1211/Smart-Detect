"""
scripts/seed_stations.py
─────────────────────────
Seed demo locations into the database via the FastAPI /stations endpoint.
Run once after starting the backend:
    python scripts/seed_stations.py
"""

import json
import sys
import urllib.error
import urllib.request

API_BASE = "http://localhost:8000"

STATIONS = [
    {"id": "STA-001", "name": "Main Lobby",         "location": "Building A, Ground Floor"},
    {"id": "STA-002", "name": "Parking Entrance",   "location": "Building A, Basement Level"},
    {"id": "STA-003", "name": "North Wing",         "location": "Building B, Floor 1"},
    {"id": "STA-004", "name": "South Gate",          "location": "Perimeter, South Side"},
    {"id": "STA-005", "name": "East Plaza",          "location": "Outdoor Area, East Side"},
    {"id": "STA-006", "name": "West Corridor",       "location": "Building A, Floor 2"},
    {"id": "STA-007", "name": "Loading Dock",        "location": "Building C, Rear"},
    {"id": "STA-008", "name": "Reception",           "location": "Building A, Front Desk"},
]


def post_station(station: dict) -> None:
    data = json.dumps(station).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/stations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  ✓ Created  : {station['id']} — {station['name']}  [{resp.status}]")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        if "already exists" in body or exc.code == 409:
            print(f"  ↩ Exists   : {station['id']} — {station['name']}")
        else:
            print(f"  ✗ Error    : {station['id']} — HTTP {exc.code}: {body}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ Failed   : {station['id']} — {exc}", file=sys.stderr)


if __name__ == "__main__":
    print(f"Seeding {len(STATIONS)} stations to {API_BASE} ...\n")
    for s in STATIONS:
        post_station(s)
    print("\nDone. Refresh the dashboard to see the stations.")
