"""
eval/profile_pipeline.py
─────────────────────────
Per-stage latency profile of the identity pipeline.

FIDELITY: rather than re-creating the pipeline (which risks measuring calls
production does not make), this wraps the real production methods with timers
and then drives eval/run_eval.py's real _process_frame. Whatever the profile
shows is what the pipeline actually does — including the fact that a cached
track skips identify() entirely, so OSNet does NOT run every person every frame.

    python eval/profile_pipeline.py --video demo_videos/05_head_pose_two_people.mp4 \
        --frames 60 --label baseline --json-out eval/results/perf/baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_WORK = ROOT / "eval" / "results" / "_profile_work"
if _WORK.exists():
    shutil.rmtree(_WORK)
(_WORK / "snapshots").mkdir(parents=True)
# ultralytics resolves "yolov8n.pt" relative to cwd, and we chdir below —
# without this it tries to re-download (and silently detects nothing offline).
_w = ROOT / "yolov8n.pt"
if _w.is_file():
    try:
        (_WORK / "yolov8n.pt").symlink_to(_w)
    except OSError:
        shutil.copy(_w, _WORK / "yolov8n.pt")
# PersonReID loads models/osnet_x1_0_reid.pth relative to cwd too — without
# this it silently falls back to ImageNet weights and re-ID accuracy differs.
if (ROOT / "models").is_dir():
    try:
        (_WORK / "models").symlink_to(ROOT / "models", target_is_directory=True)
    except OSError:
        pass
os.environ["DATABASE_URL"] = f"sqlite:///{_WORK / 'profile.db'}"
os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")
os.environ.setdefault("JWT_SECRET", "profile-only-" + "x" * 40)
os.environ.setdefault("ADMIN_PASSWORD", "profile-only-not-real")
os.environ.setdefault("OPERATOR_PASSWORD", "profile-only-not-real")

import cv2          # noqa: E402
import numpy as np  # noqa: E402

STATS = defaultdict(lambda: {"t": 0.0, "n": 0})


def instrument(obj, attr: str, label: str):
    """Wrap a bound/unbound method with a timer, in place."""
    original = getattr(obj, attr)

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return original(*a, **k)
        finally:
            s = STATS[label]
            s["t"] += time.perf_counter() - t0
            s["n"] += 1
    setattr(obj, attr, timed)
    return original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="demo_videos/05_head_pose_two_people.mp4")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    video = (ROOT / args.video) if not Path(args.video).is_absolute() else Path(args.video)
    if not video.is_file():
        sys.exit(f"video not found: {video}")

    from config.identity_config import get_identity_config
    from database.db import SessionLocal, init_db
    from database.models import Location, Person, Sighting
    from cameras.live_stream import LiveStream
    from recognition.object_detector import ObjectDetector
    from recognition.face_recognizer import FaceRecognizer
    from recognition.reid_model import PersonReID
    from recognition.smart_identifier import SmartIdentifier
    import recognition.smart_identifier as si_mod
    import supervision as sv

    init_db()
    db = SessionLocal()
    db.query(Sighting).delete(); db.query(Person).delete()
    if not db.query(Location).filter(Location.id == "LOC-PROF").first():
        from datetime import datetime
        db.add(Location(id="LOC-PROF", name="Profile", type="eval",
                        address="-", created_at=datetime.utcnow()))
    db.commit(); db.close()

    # ── Instrument the real classes, before any instance is built ───────────
    instrument(ObjectDetector, "detect", "detect")
    instrument(FaceRecognizer, "extract_embedding", "face_insightface")
    instrument(PersonReID, "extract_features", "reid_osnet")
    instrument(SmartIdentifier, "identify", "arbitration")
    instrument(LiveStream, "_evidence_gate_ok", "evidence_gate")
    instrument(LiveStream, "_apply_id_switch_guard", "id_switch_guard")
    instrument(sv.ByteTrack, "update_with_detections", "track_bytetrack")

    cfg = get_identity_config()
    from eval.adapters.base import Frame, GTPerson
    from eval.run_eval import _process_frame

    prev = os.getcwd()
    os.chdir(_WORK)
    try:
        ls = LiveStream(source=0, location_id="LOC-PROF", zone_id="p", camera_id="CAM-PROF")
        ls._detector.load_model()
        ls._face_rec.load_model()

        cap = cv2.VideoCapture(str(video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ls._detector.set_imgsz_for_resolution(w, h)
        ls._face_scan_size = (1280, 720) if max(w, h) > 1280 else (w, h)

        # warm up models so first-call init is not charged to frame 1
        ok, warm = cap.read()
        if ok:
            ls._detector.detect(warm)
            ls._face_rec.extract_embedding(cv2.resize(warm, ls._face_scan_size))
        STATS.clear()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        sight_cache: dict = {}
        n = 0
        people = 0
        t_wall0 = time.perf_counter()
        while n < args.frames:
            ok, img = cap.read()
            if not ok:
                break
            fr = Frame(image=img, timestamp=n / 12.0, camera_id="CAM-PROF",
                       ground_truth=[GTPerson(person_id="gt")], frame_id=f"f{n}")
            recs = _process_frame(ls, img, fr, cfg, n, sight_cache, 0.0)
            people += sum(1 for r in recs if r.method != "no_detection")
            n += 1
        wall = time.perf_counter() - t_wall0
        cap.release()
    finally:
        os.chdir(prev)

    measured = sum(s["t"] for s in STATS.values())
    rows = []
    for name, s in STATS.items():
        rows.append({"stage": name, "total_s": s["t"], "calls": s["n"],
                     "per_frame_ms": 1000 * s["t"] / max(n, 1),
                     "per_call_ms": 1000 * s["t"] / max(s["n"], 1),
                     "calls_per_frame": s["n"] / max(n, 1),
                     "share_pct": 100 * s["t"] / max(measured, 1e-9)})
    rows.sort(key=lambda r: -r["total_s"])

    # arbitration wraps reid, so subtract to avoid double counting in the share
    reid_t = STATS["reid_osnet"]["t"]
    arb = next((r for r in rows if r["stage"] == "arbitration"), None)
    if arb:
        arb["note"] = "includes reid_osnet time (nested)"

    rep = {"label": args.label, "video": video.name, "n_frames": n,
           "wall_ms_per_frame": 1000 * wall / max(n, 1),
           "wall_fps": n / max(wall, 1e-9),
           "measured_ms_per_frame": 1000 * measured / max(n, 1),
           "avg_people_per_frame": people / max(n, 1),
           "stages": rows}

    print(f"\n{args.label} — {video.name}, {n} frames, "
          f"{rep['avg_people_per_frame']:.2f} people/frame")
    print(f"{'stage':20} {'ms/frame':>9} {'share':>7} {'calls/f':>8} {'ms/call':>9}")
    print("-" * 60)
    for r in rows:
        note = " *" if r.get("note") else ""
        print(f"{r['stage']:20} {r['per_frame_ms']:9.1f} {r['share_pct']:6.1f}% "
              f"{r['calls_per_frame']:8.2f} {r['per_call_ms']:9.2f}{note}")
    print("-" * 60)
    print(f"{'WALL CLOCK':20} {rep['wall_ms_per_frame']:9.1f}  ->  {rep['wall_fps']:.2f} fps")
    if arb:
        print("* arbitration includes nested reid_osnet "
              f"({1000*reid_t/max(n,1):.1f} ms/frame)")

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
