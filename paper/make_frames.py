"""
Render annotated frames from the REAL SmartDetect pipeline on ChokePoint.

Mirrors eval/run_eval.py's orchestration exactly (same production calls, same
order) but retains each detection's resolved code so it can be drawn. Every
box, code and gate verdict shown is the system's own output for that frame.
"""
from __future__ import annotations
import os, sys, shutil
from pathlib import Path

ROOT = Path("/Users/lokii/Downloads/Smart-Detect-main")
sys.path.insert(0, str(ROOT))
BASE = Path("/private/tmp/claude-501/-Users-lokii-Downloads-Smart-Detect-main/4e0d6171-becd-4766-a354-d53068b6a2ba/scratchpad")
OUT = BASE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

WORK = OUT / "_work"
if WORK.exists():
    shutil.rmtree(WORK)
(WORK / "snapshots").mkdir(parents=True)
os.environ["DATABASE_URL"] = f"sqlite:///{WORK/'frames.db'}"
os.environ["SMARTDETECT_NO_AUTOSTART"] = "1"
os.environ["SMARTDETECT_IDENTITY_CONFIG"] = str(ROOT / "config" / "ablation" / "D_full.json")
if (ROOT / "yolov8n.pt").is_file():
    try: (WORK / "yolov8n.pt").symlink_to(ROOT / "yolov8n.pt")
    except OSError: shutil.copy(ROOT / "yolov8n.pt", WORK / "yolov8n.pt")
if (ROOT / "models").is_dir():
    try: (WORK / "models").symlink_to(ROOT / "models", target_is_directory=True)
    except OSError: pass

import cv2
import numpy as np
import supervision as sv
from config.identity_config import get_identity_config, reset_identity_config_cache
from database.db import init_db, SessionLocal
from database.models import Person, Sighting, Location
from cameras.live_stream import LiveStream
from eval.adapters.chokepoint import ChokePointAdapter
from recognition.face_pose import estimate_pose

reset_identity_config_cache()
cfg = get_identity_config()
init_db()
_db = SessionLocal()
_db.query(Sighting).delete(); _db.query(Person).delete()
if not _db.query(Location).filter(Location.id == "LOC-EVAL").first():
    from datetime import datetime
    _db.add(Location(id="LOC-EVAL", name="Eval", type="eval", address="offline",
                     created_at=datetime.utcnow()))
_db.commit(); _db.close()

UNASSIGNED = "Detecting..."

# ── palette (BGR) — the paper's ink ─────────────────────────────────────
INDIGO = (104, 58, 31)     # #1f3a68  gate PASS / face
OXIDE  = (33, 47, 140)     # #8c2f21  gate FAIL
GREEN  = (60, 84, 35)      # #23543c  identified
GREY   = (140, 134, 124)   # unidentified
WHITE  = (255, 255, 255)
F = cv2.FONT_HERSHEY_DUPLEX


def analyze(ls, img, frame):
    """Production orchestration, retaining per-detection resolved identity."""
    fh, fw = img.shape[:2]
    db = SessionLocal()
    out = []
    faces_out = []
    try:
        detections = ls._detector.detect(img)
        pdets = [d for d in detections if d["label"] == "person"]

        tracked = []
        if ls._tracker is not None and pdets:
            xyxy = np.array([[d["bbox"][0], d["bbox"][1],
                              d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                             for d in pdets], dtype=np.float32)
            conf = np.array([d["confidence"] for d in pdets], dtype=np.float32)
            sd = sv.Detections(xyxy=xyxy, confidence=conf,
                               class_id=np.zeros(len(pdets), dtype=int))
            sd = ls._tracker.update_with_detections(sd)
            for i in range(len(sd)):
                x1, y1, x2, y2 = sd.xyxy[i].astype(int)
                tid = int(sd.tracker_id[i]) if sd.tracker_id is not None else None
                tracked.append((tid, [x1, y1, x2 - x1, y2 - y1]))
        else:
            tracked = [(None, list(d["bbox"])) for d in pdets]

        all_faces = []
        try:
            sw, sh = ls._face_scan_size
            small = cv2.resize(img, (sw, sh))
            ls._face_rec.extract_embedding(small)
            raw = getattr(ls._face_rec, "_last_faces", []) or []
            hs, ws = fh / sh, fw / sw
            for f_ in raw:
                f_.bbox[0] *= ws; f_.bbox[2] *= ws
                f_.bbox[1] *= hs; f_.bbox[3] *= hs
            all_faces = raw
        except Exception:
            all_faces = []

        for f_ in all_faces:
            bb = f_.bbox
            fhpx = float(bb[3] - bb[1])
            ds = float(getattr(f_, "det_score", 1.0) or 1.0)
            ok = (fhpx >= cfg.face_quality_min_height_px
                  and ds >= cfg.face_quality_min_det_score)
            faces_out.append((bb.copy(), ok, fhpx, ds))

        claimed = {ls._track_codes[t]["code"]
                   for t, _ in tracked if t is not None and t in ls._track_codes}

        for tid, bbox in tracked:
            x, y, w, h = bbox
            x2, y2 = x + w, y + h
            matched = None
            for f_ in all_faces:
                fb = f_.bbox.astype(int)
                fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                if x <= fcx <= x2 and y <= fcy <= y + (y2 - y) * 0.6:
                    matched = f_
                    break
            face_emb = None; face_pose = None; gate = None
            if matched is not None:
                fhpx = float(matched.bbox[3] - matched.bbox[1])
                ds = float(getattr(matched, "det_score", 1.0) or 1.0)
                gate = (fhpx >= cfg.face_quality_min_height_px
                        and ds >= cfg.face_quality_min_det_score)
                if gate:
                    face_emb = getattr(matched, "embedding", None)
                face_pose = estimate_pose(getattr(matched, "kps", None))

            code, method = UNASSIGNED, "pending"
            if tid is not None:
                ls._track_age[tid] = ls._track_age.get(tid, 0) + 1
                cached = ls._track_codes.get(tid)
                cached = ls._apply_id_switch_guard(tid, cached, face_emb, db)
                if cached:
                    code, method = cached["code"], cached["method"]
                else:
                    r = ls._identifier.identify(
                        img, bbox, db, location_id=ls.location_id, zone_id=ls.zone_id,
                        allow_new=ls._track_age[tid] >= cfg.min_track_age_for_registration,
                        face_embedding=face_emb,
                        face_pose=ls._pose_for_registration(tid, face_pose),
                        exclude_codes=claimed, extract_face_if_missing=False)
                    code, method = r["unique_code"], r["method"]
                    ls._note_pose_deferral(tid, method)
                    if code != UNASSIGNED:
                        ls._track_codes[tid] = {"code": code, "method": method,
                                                "conf": r["confidence"], "label": code}
                        ls.active_tracks[tid] = code
                        claimed.add(code)
            out.append({"bbox": bbox, "tid": tid, "code": code,
                        "method": method, "gate": gate})
    finally:
        db.close()
    return out, faces_out


def draw(img, dets, faces, title, subtitle):
    im = img.copy()
    h, w = im.shape[:2]
    s = w / 800.0

    for fb, ok, fhpx, ds in faces:
        x1, y1, x2, y2 = [int(v) for v in fb]
        if ok:
            cv2.rectangle(im, (x1, y1), (x2, y2), INDIGO, max(2, int(2 * s)))
        else:
            step = max(4, int(6 * s))
            for i in range(x1, x2, step):
                e = min(i + int(step * 0.5), x2)
                cv2.line(im, (i, y1), (e, y1), OXIDE, 1)
                cv2.line(im, (i, y2), (e, y2), OXIDE, 1)
            for j in range(y1, y2, step):
                e = min(j + int(step * 0.5), y2)
                cv2.line(im, (x1, j), (x1, e), OXIDE, 1)
                cv2.line(im, (x2, j), (x2, e), OXIDE, 1)
        cv2.putText(im, f"{fhpx:.0f}px", (x1, max(int(10 * s), y1 - int(4 * s))),
                    F, 0.36 * s, INDIGO if ok else OXIDE, 1, cv2.LINE_AA)

    for d in dets:
        x, y, bw, bh = d["bbox"]
        ident = d["code"] != UNASSIGNED
        col = GREEN if ident else GREY
        cv2.rectangle(im, (x, y), (x + bw, y + bh), col, max(2, int(2 * s)))
        label = d["code"] if ident else "Detecting..."
        if d["tid"] is not None:
            label += f"   t{d['tid']}"
        fs = 0.50 * s
        (tw, tht), _ = cv2.getTextSize(label, F, fs, 1)
        pad = int(6 * s)
        ly = max(tht + pad + int(2 * s), y)
        cv2.rectangle(im, (x, ly - tht - pad), (x + tw + pad * 2, ly), col, -1)
        cv2.putText(im, label, (x + pad, ly - int(pad * 0.5)), F, fs, WHITE, 1, cv2.LINE_AA)
        if ident and d["method"]:
            cv2.putText(im, d["method"], (x + int(3 * s), y + bh - int(6 * s)),
                        F, 0.40 * s, col, 1, cv2.LINE_AA)

    bh_ = int(40 * s)
    ov = im.copy()
    cv2.rectangle(ov, (0, 0), (w, bh_), (24, 20, 18), -1)
    cv2.addWeighted(ov, 0.80, im, 0.20, 0, im)
    cv2.putText(im, title, (int(10 * s), int(17 * s)), F, 0.46 * s, WHITE, 1, cv2.LINE_AA)
    cv2.putText(im, subtitle, (int(10 * s), int(32 * s)), F, 0.36 * s,
                (200, 200, 200), 1, cv2.LINE_AA)
    return im


adapter = ChokePointAdapter(ROOT / "eval" / "data" / "chokepoint")
prev = os.getcwd()
os.chdir(WORK)
saved = []
try:
    seqs = {s.seq_id: s for s in adapter.sequences()}
    for seq_id in ["P1E_S1_C1"]:
        seq = seqs[seq_id]
        ls = LiveStream(source=0, location_id="LOC-EVAL", zone_id="eval",
                        camera_id=seq.camera_id)
        ls._detector.load_model(); ls._face_rec.load_model()
        want = {252, 262, 272, 282, 296, 310, 330, 355, 385, 420, 460, 500, 540, 580}
        n = 0
        for frame in adapter.frames(seq.seq_id):
            n += 1
            if n > max(want) + 1:
                break
            img = frame.image
            fh, fw = img.shape[:2]
            ls._detector.set_imgsz_for_resolution(fw, fh)
            ls._face_scan_size = (1280, 720) if max(fw, fh) > 1280 else (fw, fh)
            dets, faces = analyze(ls, img, frame)
            if n not in want:
                continue
            gt = ", ".join(sorted(g.person_id for g in frame.ground_truth)) or "none"
            npass = sum(1 for _, ok, _, _ in faces if ok)
            im = draw(img, dets, faces,
                      f"SmartDetect   {seq.camera_id}   frame {frame.frame_id.split('/')[-1]}",
                      f"config D   GT: {gt}   persons {len(dets)}   "
                      f"faces {len(faces)} ({npass} pass gate {cfg.face_quality_min_height_px}px/"
                      f"{cfg.face_quality_min_det_score})")
            p = OUT / f"run_{n:04d}.png"
            cv2.imwrite(str(p), im)
            saved.append((n, p, len(dets), len(faces), npass,
                          [d["code"] for d in dets]))
            print(f"frame {n}: dets={len(dets)} faces={len(faces)} pass={npass} "
                  f"codes={[d['code'] for d in dets]}")
finally:
    os.chdir(prev)
print("TOTAL", len(saved))
