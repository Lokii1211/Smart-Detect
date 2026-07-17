"""
scripts/e2e_test.py
End-to-end integration test for SmartDetect system.
Usage: python scripts/e2e_test.py
"""
from __future__ import annotations
import base64
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import urllib.error
import urllib.request
import json
import numpy as np
from datetime import datetime, timezone

BASE_URL = "http://localhost:8000"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def http(method, path, body=None, token=None):
    url = BASE_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": f"HTTP {e.code} — empty or non-JSON response"}
    except Exception as e:
        return 0, {"error": str(e)}

def fake_image():
    import cv2
    import os

    # Use a real photo if available in test_images folder
    test_dir = "scripts/test_images"
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir)
                  if f.endswith((".jpg",".jpeg",".png"))]
        if images:
            img = cv2.imread(os.path.join(test_dir, images[0]))
            _, buf = cv2.imencode(".jpg", img)
            return base64.b64encode(buf).decode()

    # Draw a synthetic face-like shape InsightFace can detect
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Face oval
    cv2.ellipse(img, (320, 240), (100, 130), 0, 0, 360, (200, 170, 140), -1)

    # Eyes
    cv2.ellipse(img, (280, 210), (20, 12), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (360, 210), (20, 12), 0, 0, 360, (50, 50, 50), -1)

    # Nose
    cv2.circle(img, (320, 250), 8, (160, 120, 100), -1)

    # Mouth
    cv2.ellipse(img, (320, 285), (35, 15), 0, 0, 180, (140, 80, 80), -1)

    # Eyebrows
    cv2.line(img, (260, 192), (300, 196), (60, 40, 30), 4)
    cv2.line(img, (340, 196), (380, 192), (60, 40, 30), 4)

    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()

passed = 0
failed = 0
failures = []

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}✓ PASS{RESET}  {label}")
    else:
        failed += 1
        failures.append(f"{label}: {detail}")
        print(f"  {RED}✗ FAIL{RESET}  {label}")
        if detail:
            print(f"         {detail}")

print(f"\n{BOLD}SmartDetect — E2E Integration Test{RESET}")
print(f"Target: {BASE_URL} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("─" * 60)

# ── Step 0: Health check ──────────────────────────────────────
print(f"\n{CYAN}Step 0 — Health Check{RESET}")
status, body = http("GET", "/health")
check("Backend reachable", status == 200, f"HTTP {status}: {body}")

# ── Step 1: Login and get token ───────────────────────────────
print(f"\n{CYAN}Step 1 — Login & Get JWT Token{RESET}")
status, body = http("POST", "/auth/login",
    {"username": "operator", "password": "smartOp2024"})
token = body.get("access_token", "")
check("Login successful", status == 200 and token != "",
    f"HTTP {status}: {body}")

if not token:
    print(f"\n{RED}Cannot continue — login failed.{RESET}")
    sys.exit(1)

print(f"  {GREEN}Token received{RESET}: {token[:30]}...")

# ── Step 2: Get locations ─────────────────────────────────────
print(f"\n{CYAN}Step 2 — Fetch Locations{RESET}")
status, body = http("GET", "/locations", token=token)
locations = body if isinstance(body, list) else body.get("locations", [])
check("Locations returned", status == 200 and len(locations) > 0,
    f"HTTP {status}: {body}")
location_id = locations[0]["id"] if locations else "LOC-001"
print(f"  Using location: {locations[0]['name'] if locations else 'LOC-001'}")

# ── Step 3: Register 5 test persons ──────────────────────────
print(f"\n{CYAN}Step 3 — Register 5 Test Persons{RESET}")
codes = []
for i in range(1, 6):
    status, body = http("POST", "/register", {
        "base64_image": fake_image(),
        "zone_id": "entrance",
        "location_id": location_id,
        "person_type": "visitor"
    }, token=token)
    ok = status == 200 and "unique_code" in body
    check(f"Person {i} registered",
        ok, f"HTTP {status}: {body}")
    if ok:
        codes.append(body["unique_code"])
        print(f"         → {body['unique_code']}")

if not codes:
    print(f"\n{RED}No persons registered — stopping.{RESET}")
    sys.exit(1)

# ── Step 4: Log sightings ─────────────────────────────────────
print(f"\n{CYAN}Step 4 — Log Sightings Across 4 Zones{RESET}")
zones = ["entrance", "food-court", "parking", "exit"]
for code in codes:
    for zone in zones:
        status, body = http("POST", "/sighting", {
            "unique_code": code,
            "zone_id": zone,
            "location_id": location_id,
            "camera_id": f"CAM-{zone[:3].upper()}",
            "confidence": 0.91
        }, token=token)
        check(f"{code} sighting at {zone}",
            status == 200, f"HTTP {status}: {body}")
        time.sleep(0.1)

# ── Step 5: Verify trails ─────────────────────────────────────
print(f"\n{CYAN}Step 5 — Verify Movement Trails{RESET}")
for code in codes:
    status, body = http("GET", f"/person/{code}/trail", token=token)
    trail = body if isinstance(body, list) else body.get("trail", [])
    check(f"{code} trail returned",
        status == 200 and len(trail) >= 4,
        f"HTTP {status} — {len(trail)} entries")
    if len(trail) >= 2:
        times = [t.get("seen_at","") for t in trail]
        in_order = all(times[i] <= times[i+1] for i in range(len(times)-1))
        check(f"{code} trail in order", in_order, str(times))

# ── Final report ──────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  {BOLD}FINAL REPORT{RESET}")
print(f"{'─'*60}")
total = passed + failed
print(f"  Result: {passed}/{total} checks passed")
if failures:
    print(f"\n  {RED}Failures:{RESET}")
    for f in failures:
        print(f"    • {f}")
else:
    print(f"\n  {GREEN}All checks passed!{RESET}")

print(f"\n{'─'*60}\n")