"""
Diagnostic: run the LIVE per-frame loop and the OFFLINE orchestration on the
SAME frames in ONE process and print per-frame identity state, so the
divergence between cameras/live_stream.py and eval/run_eval.py can be pinned
to a specific mechanism.

Usage:
    .venv/bin/python eval/diag_live_vs_offline.py config/ablation/F_tracklet_mean.json
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tmp = tempfile.mkdtemp(prefix="diag-")
os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/diag.db"
os.environ["SMARTDETECT_NO_AUTOSTART"] = "1"
os.environ["SMARTDETECT_IDENTITY_CONFIG"] = str(Path(sys.argv[1]).resolve())

import numpy as np
import supervision as sv

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

adapter = ChokePointAdapter("eval/data/chokepoint")
frames = list(adapter.frames("P1E_S1_C1"))
# labeled region
start, end = 232, 320

# ── Two independent streams (one per path) ────────────────────────────────
live = LiveStream(source=0, location_id="LOC-EVAL", zone_id="eval", camera_id="P1E_S1_C1")
live._detector.load_model()
live._face_rec.load_model()

off = LiveStream(source=0, location_id="LOC-EVAL", zone_id="eval", camera_id="P1E_S1_C1")
off._detector.load_model()
off._face_rec.load_model()

print("frame  live_code live_method live_tid live_buf live_ready | off_code off_method off_tid off_buf off_ready")
for i in range(start, end + 1):
    fr = frames[i]
    img = fr.image
    fh, fw = img.shape[:2]
    live._detector.set_imgsz_for_resolution(fw, fh)
    live._face_scan_size = (1280, 720) if max(fw, fh) > 1280 else (fw, fh)
    off._detector.set_imgsz_for_resolution(fw, fh)
    off._face_scan_size = live._face_scan_size

    live._analyze_frame(img)

    # offline orchestration — copy of run_eval._process_frame's identity core
    from cameras.tiled_detect import detect_with_tiling
    detections = detect_with_tiling(off._detector, img, cfg)
    person_dets = [d for d in detections if d["label"] == "person"]
    tracked = []
    if off._tracker is not None and person_dets:
        xyxy = np.array([[d["bbox"][0], d["bbox"][1],
                          d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                         for d in person_dets], dtype=np.float32)
        conf = np.array([d["confidence"] for d in person_dets], dtype=np.float32)
        svd = sv.Detections(xyxy=xyxy, confidence=conf,
                            class_id=np.zeros(len(person_dets), dtype=int))
        svd = off._tracker.update_with_detections(svd)
        for j in range(len(svd)):
            x1, y1, x2, y2 = svd.xyxy[j].astype(int)
            tid = int(svd.tracker_id[j])
            tracked.append((tid, [x1, y1, x2 - x1, y2 - y1]))
    all_faces = []
    import cv2
    scan_w, scan_h = off._face_scan_size
    small = cv2.resize(img, (scan_w, scan_h))
    off._face_rec.extract_embedding(small)
    raw = getattr(off._face_rec, "_last_faces", []) or []
    hs, ws = fh / scan_h, fw / scan_w
    for f in raw:
        f.bbox[0] *= ws; f.bbox[2] *= ws
        f.bbox[1] *= hs; f.bbox[3] *= hs
    all_faces = raw
    from recognition.face_assign import match_faces_to_boxes
    _fboxes = [[int(f.bbox[0]), int(f.bbox[1]),
                int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1])]
               for f in all_faces]
    _assign = match_faces_to_boxes(_fboxes, [b for _t, b in tracked], cfg, frame_h=fh)

    db_o = SessionLocal()
    for _det_i, (tid, bbox) in enumerate(tracked):
        if tid is None:
            continue
        off._track_age[tid] = off._track_age.get(tid, 0) + 1
        x, y, w, h = bbox
        matched_face = None
        _fi = _assign.get(_det_i)
        if _fi is not None:
            matched_face = all_faces[_fi]
        face_emb = None
        face_pose = None
        if matched_face is not None:
            from recognition.face_pose import estimate_pose
            from recognition.face_quality import gate_passes
            face_pose = estimate_pose(getattr(matched_face, "kps", None))
            fb_q = matched_face.bbox
            face_h = float(fb_q[3] - fb_q[1])
            _box = [int(fb_q[0]), int(fb_q[1]), int(fb_q[2] - fb_q[0]), int(face_h)]
            if gate_passes(_box, float(getattr(matched_face, "det_score", 1.0) or 1.0),
                           face_pose, cfg,
                           crop=img[max(0, _box[1]):_box[1] + _box[3],
                                    max(0, _box[0]):_box[0] + _box[2]],
                           person_boxes=[b for _t, b in tracked], owner_box=bbox):
                face_emb = getattr(matched_face, "embedding", None)
        cached = off._track_codes.get(tid)
        cached = off._apply_id_switch_guard(tid, cached, face_emb, db_o)
        voter = off._voter
        o_code, o_method = "Detecting...", "pending"
        if voter is not None:
            already = voter.committed_code(tid)
            if already:
                o_code, o_method = already, "tracklet_committed"
            else:
                from recognition.tracklet_vote import quality_score
                q = 0.0
                if face_emb is not None and matched_face is not None:
                    fb_q2 = matched_face.bbox
                    q = quality_score([int(fb_q2[0]), int(fb_q2[1]),
                                       int(fb_q2[2] - fb_q2[0]), int(fb_q2[3] - fb_q2[1])],
                                      float(getattr(matched_face, "det_score", 1.0) or 1.0),
                                      face_pose, cfg)
                voter.observe(tid, face_emb, q, None, 0)
                if voter.ready(tid):
                    c, m = voter.commit(
                        tid,
                        resolve_mean=lambda e: off._identifier.identify(
                            np.zeros((8, 8, 3), dtype=np.uint8), [0, 0, 8, 8], db_o,
                            location_id=off.location_id, zone_id=off.zone_id,
                            allow_new=True, face_embedding=e, face_pose=None,
                            extract_face_if_missing=False)[:2] if False else
                            (lambda r: (r["unique_code"], r["method"]))(
                                off._identifier.identify(
                                    np.zeros((8, 8, 3), dtype=np.uint8), [0, 0, 8, 8], db_o,
                                    location_id=off.location_id, zone_id=off.zone_id,
                                    allow_new=True, face_embedding=e, face_pose=None,
                                    extract_face_if_missing=False)),
                        match_fn=None)
                    if c:
                        o_code, o_method = c, m
                        off._track_codes[tid] = {"code": c, "method": m, "conf": 1.0, "label": c}
        db_o.close()

        # live side: same tid?
        l_person = next((p for p in live._latest_results["persons"]
                         if p.get("tracker_id") == tid), None)
        l_code = l_person["code"] if l_person else "NO_BOX"
        l_method = l_person["method"] if l_person else "-"
        l_buf = len(live._voter.buffered(tid)) if live._voter else -1
        l_ready = live._voter.ready(tid) if live._voter else False
        o_buf = len(voter.buffered(tid)) if voter else -1
        o_ready = voter.ready(tid) if voter else False
        mark = "  <<< DIFF" if (l_code != o_code or l_method != o_method) else ""
        print(f"{i:5d} {l_code:12s} {l_method:18s} {l_person and l_person['tracker_id']} "
              f"{l_buf:2d} {str(l_ready):5s} | {o_code:12s} {o_method:18s} {tid} "
              f"{o_buf:2d} {str(o_ready):5s}{mark}")
