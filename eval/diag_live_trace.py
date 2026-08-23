"""
Live-path voter tracer: drive LiveStream._analyze_frame on a short labeled
segment and dump, per frame and per tracker, the voter buffer/commit state
and the committed codes — so the live tracklet path can be audited against
the offline run's per-track decisions.

Usage:
    .venv/bin/python eval/diag_live_trace.py config/ablation/F_tracklet_mean.json \
        P1E_S1_C1 232 330
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

cfg_path, seq, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tmp = tempfile.mkdtemp(prefix="diag-")
os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/diag.db"
os.environ["SMARTDETECT_NO_AUTOSTART"] = "1"
os.environ["SMARTDETECT_IDENTITY_CONFIG"] = str(Path(cfg_path).resolve())

from config.identity_config import get_identity_config, reset_identity_config_cache
from database.db import init_db, SessionLocal
from database.models import Person, Sighting
from cameras.live_stream import LiveStream
from eval.adapters.chokepoint import ChokePointAdapter

reset_identity_config_cache()
cfg = get_identity_config()
init_db()
db = SessionLocal()
db.query(Sighting).delete()
db.query(Person).delete()
db.commit()
db.close()

adapter = ChokePointAdapter("eval/data/chokepoint")
frames = list(adapter.frames(seq))

ls = LiveStream(source=0, location_id="LOC-EVAL", zone_id="eval", camera_id=seq)
ls._detector.load_model()
ls._face_rec.load_model()

# code -> gt person (as revealed by the offline run for comparison later)
import json
off = { (r["frame_id"], r["gt_person"]): (r["code"], r["method"])
        for r in json.load(open("eval/results/_consolidated/F_tracklet_mean/assignments.json"))
        if r["camera_id"] == seq }

print("frame gt  live_code live_method live_tid  buf ready committed_tids/codes")
for i in range(start, end + 1):
    fr = frames[i]
    img = fr.image
    fh, fw = img.shape[:2]
    ls._detector.set_imgsz_for_resolution(fw, fh)
    ls._face_scan_size = (1280, 720) if max(fw, fh) > 1280 else (fw, fh)
    ls._analyze_frame(img)
    voter = ls._voter
    for p in ls._latest_results["persons"]:
        tid = p.get("tracker_id")
        if tid is None:
            continue
        gt = [g.person_id for g in fr.ground_truth]
        off_c, off_m = off.get((fr.frame_id, gt[0]), ("?", "?")) if gt else ("-", "-")
        buf = len(voter.buffered(tid)) if voter else -1
        ready = voter.ready(tid) if voter else False
        committed = voter.committed_code(tid) if voter else None
        if gt and off_c != "?" and p["code"] != off_c:
            print(f"{i:5d} {gt} {p['code']:12s} {p['method']:16s} {tid:3d} "
                  f"{buf:2d} {str(ready):5s} committed={committed} | off={off_c}/{off_m}")
