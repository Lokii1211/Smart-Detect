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

BASE = "http://localhost:8000"
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_videos"
LOCATION = {"id": "LOC-001", "name": "Demo Site", "type": "building", "address": "Live Demo"}


def login() -> str:
    for user, pw in [("operator", "smartOp2024"), ("admin", "smartAdmin2024")]:
        try:
            r = requests.post(f"{BASE}/auth/login", json={"username": user, "password": pw}, timeout=10)
            if r.ok:
                return r.json()["access_token"]
        except requests.RequestException:
            pass
    sys.exit(f"Cannot reach backend at {BASE}. Start it first (see README).")


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
    admin = requests.post(f"{BASE}/auth/login",
                          json={"username": "admin", "password": "smartAdmin2024"}, timeout=10)
    if admin.ok:
        requests.post(f"{BASE}/locations", json=LOCATION,
                      headers={"Authorization": f"Bearer {admin.json()['access_token']}"}, timeout=10)

    if args.reset:
        requests.post(f"{BASE}/camera/stop-all", json={}, headers=hdr, timeout=30)

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
