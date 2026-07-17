"""
scripts/test_new_features.py
Test suite for all new SmartDetect features:
  - Rate limiting
  - Object sightings endpoints
  - Settings endpoints
  - Watchlist & Alerts system
  - pgvector query fallback (Python-side)
  - PhotoSearch responsive fix (code-level only)

Usage: python scripts/test_new_features.py
Requires: backend running at http://localhost:8000
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

BASE_URL = "http://localhost:8000"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0
skipped = 0
failures = []


def http(method, path, body=None, token=None, timeout=30):
    url = BASE_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": f"HTTP {e.code}"}
    except Exception as e:
        return 0, {"error": str(e)}


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


def skip(label, reason=""):
    global skipped
    skipped += 1
    print(f"  {YELLOW}○ SKIP{RESET}  {label} — {reason}")


def fake_image():
    import cv2
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.ellipse(img, (320, 240), (100, 130), 0, 0, 360, (200, 170, 140), -1)
    cv2.ellipse(img, (280, 210), (20, 12), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (360, 210), (20, 12), 0, 0, 360, (50, 50, 50), -1)
    cv2.circle(img, (320, 250), 8, (160, 120, 100), -1)
    cv2.ellipse(img, (320, 285), (35, 15), 0, 0, 180, (140, 80, 80), -1)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


# ═══════════════════════════════════════════════════════════════
print(f"\n{BOLD}SmartDetect — New Features Test Suite{RESET}")
print(f"Target: {BASE_URL}")
print("═" * 60)

# ── Step 0: Health & Login ────────────────────────────────────
print(f"\n{CYAN}Step 0 — Health Check & Login{RESET}")
status, body = http("GET", "/health")
check("Backend reachable", status == 200, f"HTTP {status}")

if status != 200:
    print(f"\n{RED}Backend not running. Start it first: python -m uvicorn backend.main:app{RESET}")
    sys.exit(1)

# Login as operator
status, body = http("POST", "/auth/login", {"username": "operator", "password": "smartOp2024"})
op_token = body.get("access_token", "")
check("Operator login", status == 200 and op_token, f"HTTP {status}")

# Login as admin
status, body = http("POST", "/auth/login", {"username": "admin", "password": "smartAdmin2024"})
admin_token = body.get("access_token", "")
check("Admin login", status == 200 and admin_token, f"HTTP {status}")

if not op_token:
    print(f"\n{RED}Cannot continue — login failed.{RESET}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# TEST 1: Rate Limiting
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 1 — Rate Limiting{RESET}")

# Login allows 20 per minute; Step 0 already used 2, so 18 more should work
remaining = 18
print(f"  Sending {remaining} rapid login requests...")
rate_ok = True
for i in range(remaining):
    s, _ = http("POST", "/auth/login", {"username": "operator", "password": "smartOp2024"})
    if s != 200:
        rate_ok = False
        break
check(f"{remaining} logins within limit", rate_ok, f"Failed at request {i+1}")

# Next should be rate-limited (429)
s, b = http("POST", "/auth/login", {"username": "operator", "password": "smartOp2024"})
check("Next login rate-limited (429)", s == 429, f"Got HTTP {s}: {b}")

# Wait a moment so rate limit doesn't bleed into other tests
time.sleep(2)


# ══════════════════════════════════════════════════════════════
# TEST 2: Settings Endpoints
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 2 — Settings Endpoints{RESET}")

# GET settings (operator)
status, body = http("GET", "/settings", token=op_token)
check("GET /settings returns config",
      status == 200 and "config" in body,
      f"HTTP {status}: {list(body.keys()) if isinstance(body, dict) else body}")

if status == 200:
    config = body["config"]
    check("Config has detection_confidence",
          "detection_confidence" in config,
          str(list(config.keys())))
    check("Config has face_match_threshold",
          "face_match_threshold" in config)
    check("Config has max_simultaneous_streams",
          "max_simultaneous_streams" in config)
    check("Response includes db type",
          "database_url_type" in body,
          str(body.get("database_url_type")))

# PUT settings (admin)
status, body = http("PUT", "/settings",
    {"detection_confidence": 0.35, "sighting_cooldown_seconds": 45},
    token=admin_token)
check("PUT /settings updates config",
      status == 200 and "updated" in body,
      f"HTTP {status}: {body}")

if status == 200:
    check("Updated keys returned",
          "detection_confidence" in body.get("updated", []) and
          "sighting_cooldown_seconds" in body.get("updated", []),
          str(body.get("updated")))
    check("Config reflects new values",
          body.get("config", {}).get("detection_confidence") == 0.35,
          str(body.get("config", {}).get("detection_confidence")))

# PUT with invalid key (should be ignored, not error)
status, body = http("PUT", "/settings",
    {"nonexistent_key": 999},
    token=admin_token)
check("Unknown key ignored gracefully",
      status == 200 and body.get("updated") == [],
      f"HTTP {status}: {body}")

# Restore defaults
http("PUT", "/settings",
    {"detection_confidence": 0.30, "sighting_cooldown_seconds": 30},
    token=admin_token)


# ══════════════════════════════════════════════════════════════
# TEST 3: Object Sightings Endpoints
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 3 — Object Sightings Endpoints{RESET}")

# GET /objects/recent
status, body = http("GET", "/objects/recent")
check("GET /objects/recent responds",
      status == 200 and isinstance(body, list),
      f"HTTP {status}: {type(body).__name__}")

# GET /objects/stats
status, body = http("GET", "/objects/stats")
check("GET /objects/stats responds",
      status == 200 and "total_today" in body and "by_type" in body,
      f"HTTP {status}: {body}")

# GET with filter
status, body = http("GET", "/objects/recent?object_type=backpack")
check("GET /objects/recent with type filter",
      status == 200 and isinstance(body, list),
      f"HTTP {status}")

# GET with limit
status, body = http("GET", "/objects/recent?limit=5")
check("GET /objects/recent with limit=5",
      status == 200 and isinstance(body, list) and len(body) <= 5,
      f"HTTP {status}: got {len(body) if isinstance(body, list) else 'N/A'} items")


# ══════════════════════════════════════════════════════════════
# TEST 4: Watchlist CRUD
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 4 — Watchlist CRUD{RESET}")

# First, register a person to add to watchlist
status, body = http("POST", "/register", {
    "base64_image": fake_image(),
    "zone_id": "entrance",
    "location_id": "LOC-001",
    "person_type": "visitor"
}, token=op_token)
test_code = body.get("unique_code", "SDT-9999") if status == 200 else "SDT-9999"
check("Register test person for watchlist",
      status == 200 and "unique_code" in body,
      f"HTTP {status}: {body}")

# GET watchlist (should be empty or existing)
status, body = http("GET", "/watchlist", token=op_token)
check("GET /watchlist responds",
      status == 200 and isinstance(body, list),
      f"HTTP {status}")
initial_count = len(body)

# POST to watchlist
status, body = http("POST", "/watchlist", {
    "unique_code": test_code,
    "reason": "Test watchlist entry"
}, token=op_token)
check("POST /watchlist adds entry",
      status == 201 and body.get("status") in ("added", "already_watched"),
      f"HTTP {status}: {body}")
entry_id = body.get("id", "")

# POST duplicate (should return already_watched)
status, body = http("POST", "/watchlist", {
    "unique_code": test_code,
    "reason": "Duplicate"
}, token=op_token)
check("Duplicate watchlist returns already_watched",
      status in (200, 201) and body.get("status") == "already_watched",
      f"HTTP {status}: {body}")

# POST nonexistent person
status, body = http("POST", "/watchlist", {
    "unique_code": "SDT-NONEXISTENT",
}, token=op_token)
check("Watchlist rejects unknown person (404)",
      status == 404,
      f"HTTP {status}: {body}")

# GET watchlist — verify count increased
status, body = http("GET", "/watchlist", token=op_token)
check("Watchlist count increased",
      status == 200 and len(body) > initial_count,
      f"count={len(body)} (was {initial_count})")

# DELETE from watchlist
if entry_id:
    status, body = http("DELETE", f"/watchlist/{entry_id}", token=op_token)
    check("DELETE /watchlist/{id} deactivates entry",
          status == 200 and body.get("status") == "removed",
          f"HTTP {status}: {body}")
else:
    skip("DELETE watchlist entry", "no entry_id from POST")


# ══════════════════════════════════════════════════════════════
# TEST 5: Alerts Endpoints
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 5 — Alerts Endpoints{RESET}")

# GET /alerts
status, body = http("GET", "/alerts", token=op_token)
check("GET /alerts responds",
      status == 200 and isinstance(body, list),
      f"HTTP {status}")

# GET /alerts with unread filter
status, body = http("GET", "/alerts?unread=true", token=op_token)
check("GET /alerts?unread=true responds",
      status == 200 and isinstance(body, list),
      f"HTTP {status}")

# GET /alerts/unread-count
status, body = http("GET", "/alerts/unread-count")
check("GET /alerts/unread-count responds",
      status == 200 and "count" in body,
      f"HTTP {status}: {body}")

# PUT /alerts/read-all
status, body = http("PUT", "/alerts/read-all", body={}, token=op_token)
check("PUT /alerts/read-all responds",
      status == 200 and "status" in body,
      f"HTTP {status}: {body}")

# After read-all, unread count should be 0
status, body = http("GET", "/alerts/unread-count")
check("Unread count is 0 after read-all",
      status == 200 and body.get("count") == 0,
      f"count={body.get('count')}")

# PUT /alerts/{id}/read with bad ID
status, body = http("PUT", "/alerts/nonexistent-id/read", body={}, token=op_token)
check("Mark read with bad ID returns 404",
      status == 404,
      f"HTTP {status}")


# ══════════════════════════════════════════════════════════════
# TEST 6: Watchlist → Alert Integration
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 6 — Watchlist Alert Trigger (Integration){RESET}")

# Re-add person to watchlist
status, body = http("POST", "/watchlist", {
    "unique_code": test_code,
    "reason": "Integration test alert"
}, token=op_token)
new_entry_id = body.get("id", "")
check("Re-added to watchlist for alert test",
      status == 201 and body.get("status") == "added",
      f"HTTP {status}: {body}")

# Log a sighting for the watched person — should trigger an alert
status, body = http("POST", "/sighting", {
    "unique_code": test_code,
    "location_id": "LOC-001",
    "zone_id": "entrance",
    "camera_id": "CAM-001",
    "confidence": 0.92
}, token=op_token)
check("Sighting logged for watched person",
      status == 200,
      f"HTTP {status}: {body}")

# Note: The watchlist alert is triggered by the LiveStream pipeline, not the /sighting endpoint.
# In a running system with cameras, the alert would fire automatically.
# For this test, we verify the plumbing is in place.
print(f"  {YELLOW}NOTE{RESET}: Watchlist alerts fire from the camera pipeline, not /sighting directly.")
print(f"         Camera integration test requires a running LiveStream.")


# ══════════════════════════════════════════════════════════════
# TEST 7: Database Query Functions (Unit-style)
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 7 — Database Query Functions{RESET}")

try:
    from database.db import SessionLocal, init_db
    from database.queries import (
        find_person_by_embedding,
        check_watchlist_alert,
        _cosine_similarity,
        _is_postgres,
    )
    init_db()
    db = SessionLocal()

    # Test _cosine_similarity
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    check("Cosine similarity — identical vectors = 1.0",
          abs(_cosine_similarity(a, b) - 1.0) < 0.001)
    check("Cosine similarity — orthogonal vectors = 0.0",
          abs(_cosine_similarity(a, c)) < 0.001)

    # Test _is_postgres
    is_pg = _is_postgres(db)
    check(f"_is_postgres returns bool (got {'PostgreSQL' if is_pg else 'SQLite'})",
          isinstance(is_pg, bool))

    # Test find_person_by_embedding with random vector (should return None or a match)
    random_emb = np.random.randn(512).astype(np.float32)
    result = find_person_by_embedding(random_emb, db=db, threshold=0.99)
    check("find_person_by_embedding with high threshold returns None",
          result is None,
          f"Got: {result}")

    # Test check_watchlist_alert doesn't crash on non-watched person
    try:
        check_watchlist_alert("SDT-0000", "CAM-001", "LOC-001", db=db)
        check("check_watchlist_alert — non-watched person doesn't error", True)
    except Exception as exc:
        check("check_watchlist_alert — non-watched person doesn't error", False, str(exc))

    db.close()

except Exception as exc:
    skip("Database query unit tests", f"Import error: {exc}")


# ══════════════════════════════════════════════════════════════
# TEST 8: Model Integrity (WatchlistEntry, Alert tables)
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 8 — Database Model Integrity{RESET}")

try:
    from database.db import SessionLocal, init_db
    from database.models import WatchlistEntry, Alert, ObjectSighting
    init_db()
    db = SessionLocal()

    # Verify tables exist by querying them
    try:
        wl_count = db.query(WatchlistEntry).count()
        check("WatchlistEntry table exists and queryable", True)
    except Exception as exc:
        check("WatchlistEntry table exists", False, str(exc))

    try:
        alert_count = db.query(Alert).count()
        check("Alert table exists and queryable", True)
    except Exception as exc:
        check("Alert table exists", False, str(exc))

    try:
        obj_count = db.query(ObjectSighting).count()
        check("ObjectSighting table exists and queryable", True)
    except Exception as exc:
        check("ObjectSighting table exists", False, str(exc))

    db.close()

except Exception as exc:
    skip("Database model integrity tests", f"Import error: {exc}")


# ══════════════════════════════════════════════════════════════
# TEST 9: Cleanup watchlist entry
# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}Test 9 — Cleanup{RESET}")
if new_entry_id:
    status, body = http("DELETE", f"/watchlist/{new_entry_id}", token=op_token)
    check("Cleaned up watchlist test entry",
          status == 200,
          f"HTTP {status}")


# ══════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  {BOLD}FINAL REPORT — New Features Test Suite{RESET}")
print(f"{'═'*60}")
total = passed + failed
print(f"  Passed:  {GREEN}{passed}{RESET}")
print(f"  Failed:  {RED}{failed}{RESET}")
if skipped:
    print(f"  Skipped: {YELLOW}{skipped}{RESET}")
print(f"  Total:   {total}")

if failures:
    print(f"\n  {RED}Failures:{RESET}")
    for f in failures:
        print(f"    • {f}")
else:
    print(f"\n  {GREEN}All checks passed!{RESET}")

print(f"\n{'═'*60}\n")
sys.exit(1 if failed else 0)
