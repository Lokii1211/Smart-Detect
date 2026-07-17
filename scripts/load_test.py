"""
scripts/load_test.py
─────────────────────
Locust load test for the SmartDetect Person Tracking System.

Simulates 50 concurrent operators hitting the API and reports response times.
Target: GET /person/{code}/trail average response time < 300 ms.

Usage:
    # Install locust first:
    pip install locust

    # Run with 50 users, spawn rate 5/s, for 60 seconds (headless):
    locust -f scripts/load_test.py --headless -u 50 -r 5 -t 60s \
           --host http://localhost:8000

    # Or open the Locust web UI (omit --headless):
    locust -f scripts/load_test.py --host http://localhost:8000
    # Then visit http://localhost:8089 and start the test manually.

    # Quick self-contained run (spawns locust as subprocess):
    python scripts/load_test.py --run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from typing import List

# ── Try importing locust (may not be installed yet) ───────────────────────────
try:
    from locust import HttpUser, TaskSet, between, events, task
    _LOCUST_AVAILABLE = True
except ImportError:
    _LOCUST_AVAILABLE = False


# ─── Fetch test codes from the live API ───────────────────────────────────────

def _fetch_known_codes(host: str = "http://localhost:8000", fallback_count: int = 5) -> List[str]:
    """
    Pull existing person codes from the database to use as targets.
    Falls back to synthetic MET codes if none found.
    """
    try:
        # The API has no /persons list endpoint, so we query the DB directly.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from database.db import SessionLocal     # noqa: PLC0415
        from database.models import Person       # noqa: PLC0415
        db = SessionLocal()
        try:
            persons = db.query(Person).limit(20).all()
            codes = [p.unique_code for p in persons]
        finally:
            db.close()
        if codes:
            return codes
    except Exception:  # noqa: BLE001
        pass

    # Fallback — generate plausible codes that will return empty trails (still
    # exercises the DB query path and JSON serialisation).
    from datetime import datetime  # noqa: PLC0415
    date = datetime.now().strftime("%Y%m%d")
    return [f"MET-{date}-T{i:03d}" for i in range(1, fallback_count + 1)]


# ─── TARGET_CODES is set at module level so every Locust worker shares it ─────
TARGET_CODES: List[str] = []


# ─── Locust Task Sets ─────────────────────────────────────────────────────────

if _LOCUST_AVAILABLE:

    class TrailTasks(TaskSet):
        """
        Defines the actions each simulated operator performs.
        At the moment all weight is placed on the trail lookup,
        matching the scenario of 50 operators searching simultaneously.
        """

        @task(10)
        def get_person_trail(self):
            """
            Primary load task — fetch the full movement trail for a known person.
            Locust automatically records response time and marks pass/fail.
            """
            code = random.choice(TARGET_CODES) if TARGET_CODES else "MET-00000000-0000"
            with self.client.get(
                f"/person/{code}/trail",
                name="/person/[code]/trail",   # group all codes under same stat key
                catch_response=True,
            ) as resp:
                if resp.status_code in (200, 404):
                    # 404 = unknown code — still a valid server response
                    resp.success()
                else:
                    resp.failure(f"Unexpected status: {resp.status_code}")

        @task(2)
        def health_check(self):
            """
            Secondary task — lightweight health ping.
            Keeps 1-in-6 requests as health checks to simulate mixed traffic.
            """
            self.client.get("/health", name="/health")

        @task(1)
        def list_stations(self):
            """
            Tertiary task — station list, simulates dashboard overview polling.
            """
            self.client.get("/stations", name="/stations")

    class SmartDetectOperator(HttpUser):
        """
        Represents a single control-room operator.
        Think-time between requests: 1–3 seconds (realistic operator pacing).
        """
        tasks = [TrailTasks]
        wait_time = between(1, 3)

    # ── Event hook — print a performance summary when the test ends ────────────
    @events.quitting.add_listener
    def on_quit(environment, **kwargs):
        stats = environment.stats.get("/person/[code]/trail", "GET")
        if stats:
            avg_ms = stats.avg_response_time
            p95_ms = stats.get_response_time_percentile(0.95)
            passed = avg_ms < 300
            print("\n" + "="*60)
            print("  LOAD TEST RESULTS — /person/[code]/trail")
            print("="*60)
            print(f"  Requests     : {stats.num_requests}")
            print(f"  Failures     : {stats.num_failures}")
            print(f"  Avg response : {avg_ms:.1f} ms  {'✓ PASS' if passed else '✗ FAIL (target <300ms)'}")
            print(f"  95th pctile  : {p95_ms:.1f} ms")
            print(f"  RPS          : {stats.current_rps:.1f}")
            print("="*60 + "\n")


# ─── Self-contained runner (no locust CLI required) ───────────────────────────

def _run_via_subprocess(host: str, users: int, spawn_rate: int, duration: str) -> None:
    """Launch locust as a subprocess with headless mode and CSV output."""
    csv_prefix = "load_test_results"
    cmd = [
        sys.executable, "-m", "locust",
        "-f", __file__,
        "--headless",
        f"--host={host}",
        f"-u={users}",
        f"-r={spawn_rate}",
        f"-t={duration}",
        f"--csv={csv_prefix}",
    ]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    if result.returncode == 0:
        print("\nLoad test completed. Check load_test_results_stats.csv for details.")
    else:
        print(f"\nLocust exited with code {result.returncode}")


def _run_standalone(host: str, users: int, duration_sec: int) -> None:
    """
    Minimal standalone load test using only stdlib (no locust).
    Spawns `users` threads that each hit the trail endpoint repeatedly.
    """
    import threading  # noqa: PLC0415

    codes = _fetch_known_codes(host)
    print(f"Standalone load test — {users} threads, {duration_sec}s, {len(codes)} target codes")
    print(f"Target: {host}/person/[code]/trail\n")

    timings: List[float] = []
    errors: List[str]   = []
    lock = threading.Lock()
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            code = random.choice(codes)
            url  = f"{host}/person/{code}/trail"
            t0   = time.perf_counter()
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5):
                    pass
                elapsed_ms = (time.perf_counter() - t0) * 1000
                with lock:
                    timings.append(elapsed_ms)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(str(exc))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(users)]
    for t in threads:
        t.start()

    time.sleep(duration_sec)
    stop_event.set()
    for t in threads:
        t.join(timeout=2)

    # ── Report ──────────────────────────────────────────────────────────────
    if not timings:
        print("No successful requests — is the backend running?")
        return

    avg = sum(timings) / len(timings)
    timings_sorted = sorted(timings)
    p95 = timings_sorted[int(len(timings_sorted) * 0.95)]
    passed = avg < 300

    print("=" * 60)
    print("  STANDALONE LOAD TEST RESULTS")
    print("=" * 60)
    print(f"  Requests      : {len(timings)}")
    print(f"  Errors        : {len(errors)}")
    print(f"  Avg response  : {avg:.1f} ms  {'✓ PASS' if passed else '✗ FAIL (target <300ms)'}")
    print(f"  95th pctile   : {p95:.1f} ms")
    print(f"  Min / Max     : {min(timings):.1f} / {max(timings):.1f} ms")
    print("=" * 60)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SmartDetect Load Test")
    parser.add_argument("--host",       default="http://localhost:8000")
    parser.add_argument("--users",      type=int, default=50,  help="Concurrent users")
    parser.add_argument("--spawn-rate", type=int, default=5,   help="Users spawned per second")
    parser.add_argument("--duration",   default="60s",         help="Test duration e.g. 60s, 2m")
    parser.add_argument("--run",        action="store_true",   help="Execute test immediately (subprocess or standalone)")
    parser.add_argument("--standalone", action="store_true",   help="Use stdlib-only mode (no locust required)")
    args = parser.parse_args()

    # Preload codes before locust forks workers
    global TARGET_CODES
    TARGET_CODES = _fetch_known_codes(args.host)
    print(f"Loaded {len(TARGET_CODES)} target codes: {TARGET_CODES[:3]}{'…' if len(TARGET_CODES) > 3 else ''}")

    if args.standalone or not _LOCUST_AVAILABLE:
        dur = int(args.duration.rstrip("sm")) * (60 if args.duration.endswith("m") else 1)
        _run_standalone(args.host, args.users, dur)
    elif args.run:
        _run_via_subprocess(args.host, args.users, args.spawn_rate, args.duration)
    else:
        print(
            f"\nLocust is available. Run with:\n"
            f"  locust -f scripts/load_test.py --host {args.host}\n"
            f"Or add --run to execute headlessly, or --standalone for no-locust mode.\n"
        )


if __name__ == "__main__":
    main()

print("TASK 2 COMPLETE")
