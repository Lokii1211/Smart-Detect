"""
scripts/demo_videos.py
──────────────────────
One-command live demo: uploads every clip in demo_videos/ as a file-source
camera and starts the first few playing, so a fresh clone can show the full
detection → identity → tracking pipeline without any manual clicking.

Prereq: backend running (see README). Then:
    python scripts/demo_videos.py            # upload all, start first 4
    python scripts/demo_videos.py --start 0  # upload only, start none
    python scripts/demo_videos.py --reset     # stop+delete existing demo cams first

Open http://localhost:5173/live and click ⊞ Wall to watch them side-by-side.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _credentials import credentials

BASE = "http://localhost:8000"
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_videos"
LOCATION = {"id": "LOC-001", "name": "Demo Site", "type": "building", "address": "Live Demo"}


def login(role: str = "operator") -> str:
    user, pw = credentials(role)
    try:
        r = requests.post(f"{BASE}/auth/login",
                          json={"username": user, "password": pw}, timeout=10)
    except requests.RequestException:
        sys.exit(f"Cannot reach backend at {BASE}. Start it first (see README).")
    if not r.ok:
        sys.exit(f"Login failed for '{user}' ({r.status_code}). "
                 f"Check the credentials in .env match the running server.")
    return r.json()["access_token"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=4, help="how many videos to auto-start (default 4)")
    ap.add_argument("--reset", action="store_true", help="stop all streams before loading")
    args = ap.parse_args()

    clips = sorted(DEMO_DIR.glob("*.mp4"))
    if not clips:
        sys.exit(f"No videos in {DEMO_DIR}. This folder ships with the repo.")

    token = login()
    hdr = {"Authorization": f"Bearer {token}"}

    # Admin token needed to create the location
    admin_token = login("admin")
    requests.post(f"{BASE}/locations", json=LOCATION,
                  headers={"Authorization": f"Bearer {admin_token}"}, timeout=10)

    if args.reset:
        # stop-all only halts streams; it leaves the camera rows behind, so a
        # second run without this loop kept uploading fresh CAM-XXX rows for
        # the same 7 clips every time instead of replacing them.
        requests.post(f"{BASE}/camera/stop-all", json={}, headers=hdr, timeout=30)
        r = requests.get(f"{BASE}/cameras", headers=hdr, timeout=10)
        if r.ok:
            for loc in r.json():
                for cam in loc.get("cameras", []):
                    if cam["label"] in {c.stem for c in clips}:
                        requests.delete(f"{BASE}/cameras/{cam['id']}", headers=hdr, timeout=10)
                        print(f"  - removed old {cam['id']} ({cam['label']})")

    print(f"Uploading {len(clips)} demo clips…")
    cam_ids = []
    for clip in clips:
        with open(clip, "rb") as fh:
            r = requests.post(
                f"{BASE}/cameras/upload",
                headers=hdr,
                files={"file": (clip.name, fh, "video/mp4")},
                data={"location_id": "LOC-001", "zone_id": "entrance", "label": clip.stem},
                timeout=120,
            )
        if r.ok:
            cam = r.json()
            cam_ids.append(cam["id"])
            print(f"  ✓ {clip.name:34s} → {cam['id']}  ({cam.get('duration_seconds')}s)")
        else:
            print(f"  ✗ {clip.name}: {r.text[:120]}")

    for cam_id in cam_ids[: args.start]:
        requests.post(f"{BASE}/camera/start", json={"camera_id": cam_id}, headers=hdr, timeout=30)
        print(f"  ▶ started {cam_id}")
        time.sleep(0.5)

    print("\nDemo ready. Open http://localhost:5173/live and click ⊞ Wall.")
    if args.start:
        print(f"{args.start} camera(s) are playing now; start the rest from the dashboard.")


if __name__ == "__main__":
    main()
