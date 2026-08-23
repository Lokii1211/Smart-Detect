"""
eval/run_eval.py
─────────────────
Headless offline runner: drives SmartDetect's REAL identity code path over
a labeled dataset, with no camera, no threads, no MJPEG, no FPS pacing.

Design rule: this file contains NO identity logic. Every arbitration
decision is delegated to production code —

    recognition.smart_identifier.SmartIdentifier.identify()
    cameras.live_stream.LiveStream._apply_id_switch_guard()
    cameras.live_stream.LiveStream._evidence_gate_ok()
    cameras.live_stream.LiveStream._face_sim_to_code()
    recognition.object_detector.ObjectDetector.detect()
    recognition.face_recognizer.FaceRecognizer.extract_embedding()
    supervision.ByteTrack (constructed exactly as LiveStream does)

A LiveStream instance is constructed but never .start()ed, so its real
methods and real per-track state dicts are used while its capture/analysis
threads and cv2.VideoCapture never run.

KNOWN DIVERGENCE — read before citing results
─────────────────────────────────────────────
LiveStream._analyze_frame() is a single ~380-line method that interleaves
tracking, face scanning, identity arbitration, annotation-data assembly and
sighting logging, and writes its output into thread-shared state. It cannot
be called directly here because it needs live capture state
(self._face_scan_size set by start(), self._cap, the results lock) and it
returns nothing — results go to self._latest_results for the MJPEG thread.

Rather than duplicate it, this runner re-creates only its ORCHESTRATION
(the order in which the production helpers are called) and calls each
real helper for every actual decision. Two things _analyze_frame does are
therefore NOT exercised here, and no metric below depends on them:

  * head-zoom second pass — a detection-recall booster (see the block
    comment in live_stream.py). Omitted because it needs _face_scan_size
    from start(). Effect: face recall here is a lower bound vs production.
  * annotation/overlay assembly and the MJPEG frame encode — display only.

The identity-deciding path (quality gate -> ID-switch guard -> identify()
-> evidence gate) is the real code, called in the real order.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _isolate_eval_db(out_dir: Path) -> Path:
    """Point DATABASE_URL at a throwaway DB *before* any project import, so
    smartdetect.db is never opened, let alone written."""
    db_path = out_dir / "eval.db"
    for suffix in ("", "-shm", "-wal"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")
    return db_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline identity evaluation runner")
    ap.add_argument("--config", required=True, help="path to an ablation JSON (config/ablation/*.json)")
    ap.add_argument("--adapter", required=True, choices=["folder", "chokepoint"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True, help="output directory for this config's results")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames (per sequence cap)")
    ap.add_argument(
        "--sighting-dedup-seconds", type=float, default=0.0,
        help=("Suppress repeat sightings of the same code within this many seconds "
              "of DATASET time. Default 0 = write one row per gate-passing frame. "
              "Production uses 30 s of WALL-CLOCK time, but adapter timestamps are "
              "synthetic (frame-index/fps), so applying 30 s here would be arbitrary "
              "and would collapse the evidence sample. Dedup is a write-rate limiter, "
              "not an accuracy mechanism: it does not change the correct/incorrect "
              "ratio, only the sample size behind evidence_precision."))
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # MUST happen before importing anything that touches database.db
    db_path = _isolate_eval_db(out_dir)
    # Snapshots (registration photos) are written relative to cwd by
    # SmartIdentifier — chdir into an isolated dir so the real snapshots/
    # tree is untouched.
    work_dir = out_dir / "workdir"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    (work_dir / "snapshots").mkdir(parents=True)
    # ultralytics resolves "yolov8n.pt" relative to cwd; symlink the repo's
    # copy so the chdir below doesn't trigger a fresh download per config.
    _weights = ROOT / "yolov8n.pt"
    if _weights.is_file():
        try:
            (work_dir / "yolov8n.pt").symlink_to(_weights)
        except OSError:
            shutil.copy(_weights, work_dir / "yolov8n.pt")
    # PersonReID resolves models/osnet_x1_0_reid.pth relative to cwd as well;
    # without this it silently falls back to ImageNet weights, changing re-ID
    # behaviour between the eval harness and production.
    if (ROOT / "models").is_dir() and not (work_dir / "models").exists():
        try:
            (work_dir / "models").symlink_to(ROOT / "models", target_is_directory=True)
        except OSError:
            pass

    os.environ["SMARTDETECT_IDENTITY_CONFIG"] = str(Path(args.config).resolve())

    import numpy as np
    from config.identity_config import get_identity_config, reset_identity_config_cache
    from database.db import init_db, SessionLocal
    from database.models import Person, Sighting, Location
    from cameras.live_stream import LiveStream
    from eval.scoring import Assignment, UNASSIGNED, compute_all

    reset_identity_config_cache()
    cfg = get_identity_config()

    # ── Clean slate DB ──────────────────────────────────────────────────────
    init_db()
    db = SessionLocal()
    db.query(Sighting).delete()
    db.query(Person).delete()
    if not db.query(Location).filter(Location.id == "LOC-EVAL").first():
        from datetime import datetime
        db.add(Location(id="LOC-EVAL", name="Eval", type="eval",
                        address="offline", created_at=datetime.utcnow()))
    db.commit()
    db.close()

    # ── Adapter ─────────────────────────────────────────────────────────────
    if args.adapter == "folder":
        from eval.adapters.folder import FolderAdapter
        adapter = FolderAdapter(args.data_root)
    else:
        from eval.adapters.chokepoint import ChokePointAdapter
        adapter = ChokePointAdapter(args.data_root)
    print(f"[eval] {adapter.describe()}")
    print(f"[eval] config={Path(args.config).name} db={db_path}")

    records: List[Assignment] = []
    order = 0
    frame_times: List[float] = []
    unreachable_notes: List[str] = []

    prev_cwd = os.getcwd()
    os.chdir(work_dir)   # snapshots/ written here, not in the repo
    try:
        for seq in adapter.sequences():
            # One LiveStream per sequence == one camera, mirroring production
            # (each camera owns its own tracker + per-track state).
            ls = LiveStream(source=0, location_id="LOC-EVAL",
                            zone_id="eval", camera_id=seq.camera_id)
            if ls._detector is None or ls._identifier is None or ls._face_rec is None:
                sys.exit("[eval] ML components unavailable — cannot evaluate. "
                         "Check ultralytics/insightface install.")
            ls._detector.load_model()
            ls._face_rec.load_model()
            # Production sets this in start() from the real capture size;
            # offline we set it per-frame from the frame itself (below).

            # code -> last dataset-time a sighting was written (per camera,
            # matching production's per-LiveStream _seen_cache)
            sight_cache: dict = {}

            # Tracklet voting is per camera, like the tracker it keys on.
            voter = None
            if getattr(cfg, "enable_tracklet_voting", False):
                from recognition.tracklet_vote import TrackletVoter
                voter = TrackletVoter(cfg)

            n = 0
            for frame in adapter.frames(seq.seq_id):
                if args.max_frames and n >= args.max_frames:
                    break
                n += 1
                t0 = time.perf_counter()

                img = frame.image
                fh, fw = img.shape[:2]
                # Same resolution-aware choices production makes in start()
                ls._detector.set_imgsz_for_resolution(fw, fh)
                ls._face_scan_size = (1280, 720) if max(fw, fh) > 1280 else (fw, fh)

                assignments = _process_frame(ls, img, frame, cfg, order, sight_cache,
                                             args.sighting_dedup_seconds, voter)
                records.extend(assignments)
                order += len(assignments)
                frame_times.append(time.perf_counter() - t0)

            if voter is not None:
                # Commit every still-open track from whatever it accumulated.
                # A track that ended without reaching k faces still gets its
                # one decision here; a track that never saw a face does not.
                db_f = SessionLocal()
                try:
                    voter.flush(
                        resolve_mean=lambda e: _resolve_embedding(ls, cfg, db_f, e),
                        match_fn=lambda e: _match_only(cfg, db_f, e))
                finally:
                    db_f.close()
                vs = voter.stats()
                print(f"[eval]   {seq.seq_id}: tracklet voting — "
                      f"{vs['committed']}/{vs['tracks']} tracks committed, "
                      f"{vs['never_had_a_face']} never had a gate-passing face")
            print(f"[eval]   {seq.seq_id}: {n} frames")
    finally:
        os.chdir(prev_cwd)

    if not frame_times:
        sys.exit("[eval] no frames processed — check --data-root")

    runtime = {
        "n_frames": len(frame_times),
        "mean_frame_latency_ms": 1000 * sum(frame_times) / len(frame_times),
        "median_frame_latency_ms": 1000 * sorted(frame_times)[len(frame_times) // 2],
        "throughput_fps": len(frame_times) / sum(frame_times),
        "total_wall_seconds": sum(frame_times),
        "machine": "Apple M5 (CPU inference, ONNXRuntime CPUExecutionProvider)",
    }

    metrics = compute_all(records, runtime)
    metrics["config_file"] = Path(args.config).name
    metrics["config_flags"] = {
        "enable_face_anchor": cfg.enable_face_anchor,
        "enable_colour_fallback": cfg.enable_colour_fallback,
        "enable_reid_fallback": cfg.enable_reid_fallback,
        "enable_id_switch_guard": cfg.enable_id_switch_guard,
        "enable_evidence_gating": cfg.enable_evidence_gating,
    }
    metrics["dataset"] = {"adapter": adapter.name, "root": str(Path(args.data_root).resolve())}
    metrics["sighting_dedup_seconds"] = args.sighting_dedup_seconds
    metrics["harness_notes"] = ([
        "head-zoom second pass not exercised offline (needs capture state from "
        "LiveStream.start()); face recall here is a lower bound vs production",
    ] + unreachable_notes)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "assignments.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2))
    print(f"[eval] wrote {out_dir/'metrics.json'} ({len(records)} assignment records)")



class _PendingRef:
    """
    Links a buffered tracklet observation to the Assignment record that will be
    emitted for that frame.

    The voter buffers observations DURING per-detection processing, but the
    scoring record is only built later, after detections are attributed to
    ground truth. This holder is registered with the voter immediately and
    pointed at the record once it exists, so a commit can write the code back
    into frames that were emitted as "Detecting...".
    """
    __slots__ = ("target",)

    def __init__(self):
        self.target = None

    @property
    def code(self):
        return self.target.code if self.target is not None else None

    @code.setter
    def code(self, v):
        if self.target is not None:
            self.target.code = v

    @property
    def method(self):
        return self.target.method if self.target is not None else None

    @method.setter
    def method(self, v):
        if self.target is not None:
            self.target.method = v


def _resolve_embedding(ls, cfg, db, emb):
    """Full production arbitration for one embedding (match, else enrol)."""
    import numpy as _np
    r = ls._identifier.identify(
        _np.zeros((8, 8, 3), dtype=_np.uint8), [0, 0, 8, 8], db,
        location_id=ls.location_id, zone_id=ls.zone_id,
        allow_new=True, face_embedding=emb, face_pose=None,
        extract_face_if_missing=False)
    return r["unique_code"], r["method"]


def _match_only(cfg, db, emb):
    """Match without enrolling — used by the VOTE strategy's per-embedding pass."""
    from database.queries import find_person_by_embedding
    m = find_person_by_embedding(emb, db=db, threshold=cfg.face_match_threshold)
    return m["unique_code"] if m else None


def _process_frame(ls, img, frame, cfg, order_start: int, sight_cache: dict,
                   dedup_seconds: float = 0.0, voter=None) -> List:
    """
    Re-creates LiveStream._analyze_frame's ORCHESTRATION ONLY. Every identity
    decision is made by a production method — see module docstring.

    Emits exactly ONE record per ground-truth person in this frame, including
    people the detector missed entirely (code=UNASSIGNED, method="no_detection").
    Without that, detection misses leave the denominator and coverage is
    overstated.

    sight_cache: code -> last dataset-time a sighting was written. Mirrors
    production's 30 s write-rate limiter, but keyed on DATASET time rather
    than wall-clock, since offline the whole run completes in seconds. This
    is a write-rate limiter, not identity logic.
    """
    import numpy as np
    from database.db import SessionLocal
    from database.queries import log_sighting
    from eval.scoring import Assignment, UNASSIGNED
    import supervision as sv

    fh, fw = img.shape[:2]
    db = SessionLocal()
    out: List[Assignment] = []
    try:
        # 1. Detection — production ObjectDetector
        from cameras.tiled_detect import detect_with_tiling
        detections = detect_with_tiling(ls._detector, img, cfg)
        person_dets = [d for d in detections if d["label"] == "person"]

        # 2. Tracking — production ByteTrack instance owned by LiveStream
        tracked = []
        if ls._tracker is not None and person_dets:
            xyxy = np.array([[d["bbox"][0], d["bbox"][1],
                              d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                             for d in person_dets], dtype=np.float32)
            conf = np.array([d["confidence"] for d in person_dets], dtype=np.float32)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=conf,
                                    class_id=np.zeros(len(person_dets), dtype=int))
            sv_dets = ls._tracker.update_with_detections(sv_dets)
            for i in range(len(sv_dets)):
                x1, y1, x2, y2 = sv_dets.xyxy[i].astype(int)
                tid = int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else None
                tracked.append((tid, [x1, y1, x2 - x1, y2 - y1]))
        else:
            tracked = [(None, list(d["bbox"])) for d in person_dets]

        # 3. Full-frame face scan — production FaceRecognizer
        all_faces = []
        try:
            import cv2
            scan_w, scan_h = ls._face_scan_size
            small = cv2.resize(img, (scan_w, scan_h))
            ls._face_rec.extract_embedding(small)
            raw = getattr(ls._face_rec, "_last_faces", []) or []
            hs, ws = fh / scan_h, fw / scan_w
            for f in raw:
                f.bbox[0] *= ws; f.bbox[2] *= ws
                f.bbox[1] *= hs; f.bbox[3] *= hs
            all_faces = raw
        except Exception:
            all_faces = []

        claimed = {ls._track_codes[t]["code"]
                   for t, _ in tracked if t is not None and t in ls._track_codes}

        # Face -> person association for the WHOLE frame at once. With
        # enable_optimal_face_assignment off this reproduces the shipped greedy
        # rule exactly, defects included.
        from recognition.face_assign import match_faces_to_boxes
        _fboxes = [[int(f.bbox[0]), int(f.bbox[1]),
                    int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1])]
                   for f in all_faces]
        _assign = match_faces_to_boxes(_fboxes, [b for _t, b in tracked], cfg,
                                       frame_h=fh)

        # 4. Per-detection identity resolution (production methods only)
        det_results = []
        for _det_i, (tid, bbox) in enumerate(tracked):
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            det_pending = None
            _fi = _assign.get(_det_i)
            matched_face = all_faces[_fi] if _fi is not None else None
            face_emb = None
            face_pose = None
            if matched_face is not None:
                face_h = float(matched_face.bbox[3] - matched_face.bbox[1])
                det_sc = float(getattr(matched_face, "det_score", 1.0) or 1.0)
                from recognition.face_pose import estimate_pose
                face_pose = estimate_pose(getattr(matched_face, "kps", None))
                # Quality gate. With enable_learned_quality_gate off (default)
                # this evaluates the identical two-constant test.
                from recognition.face_quality import gate_passes
                fb_q = matched_face.bbox
                _box = [int(fb_q[0]), int(fb_q[1]),
                        int(fb_q[2] - fb_q[0]), int(face_h)]
                if gate_passes(_box, det_sc, face_pose, cfg,
                               crop=img[max(0, _box[1]):_box[1] + _box[3],
                                        max(0, _box[0]):_box[0] + _box[2]],
                               person_boxes=[b for _t, b in tracked],
                               owner_box=bbox):
                    face_emb = getattr(matched_face, "embedding", None)

            code, method, fresh_face_id = UNASSIGNED, "pending", False
            if tid is not None:
                ls._track_age[tid] = ls._track_age.get(tid, 0) + 1
                cached = ls._track_codes.get(tid)

                # ID-switch guard — REAL production method
                cached = ls._apply_id_switch_guard(tid, cached, face_emb, db)

                if voter is not None:
                    # ── Tracklet voting ────────────────────────────────────
                    # Defer the decision. Until commit the track is
                    # "Detecting..." and its record is held for retroactive
                    # labelling. Buffering happens BEFORE the readiness test so
                    # the k-th face triggers commitment on its own frame.
                    already = voter.committed_code(tid)
                    if already:
                        code, method = already, "tracklet_committed"
                        fresh_face_id = face_emb is not None
                    else:
                        q = 0.0
                        if face_emb is not None:
                            from recognition.tracklet_vote import quality_score
                            fb_q2 = matched_face.bbox
                            q = quality_score(
                                [int(fb_q2[0]), int(fb_q2[1]),
                                 int(fb_q2[2] - fb_q2[0]),
                                 int(fb_q2[3] - fb_q2[1])],
                                float(getattr(matched_face, "det_score", 1.0) or 1.0),
                                face_pose, cfg)
                        pending_rec = _PendingRef()
                        voter.observe(tid, face_emb, q, pending_rec, order_start)
                        if voter.ready(tid):
                            c, m = voter.commit(
                                tid,
                                resolve_mean=lambda e: _resolve_embedding(ls, cfg, db, e),
                                match_fn=lambda e: _match_only(cfg, db, e))
                            if c:
                                code, method = c, m
                                fresh_face_id = face_emb is not None
                                ls._track_codes[tid] = {"code": c, "method": m,
                                                        "conf": 1.0, "label": c}
                                ls.active_tracks[tid] = c
                                claimed.add(c)
                        det_pending = pending_rec
                elif cached:
                    code, method = cached["code"], cached["method"]
                else:
                    # Identity arbitration — REAL production method
                    result = ls._identifier.identify(
                        img, bbox, db,
                        location_id=ls.location_id, zone_id=ls.zone_id,
                        allow_new=ls._track_age[tid] >= cfg.min_track_age_for_registration,
                        face_embedding=face_emb,
                        face_pose=ls._pose_for_registration(tid, face_pose),
                        exclude_codes=claimed,
                        extract_face_if_missing=False,
                    )
                    code, method = result["unique_code"], result["method"]
                    ls._note_pose_deferral(tid, method)
                    fresh_face_id = (face_emb is not None
                                     and method in ("face", "new_registration"))
                    if code != UNASSIGNED:
                        ls._track_codes[tid] = {"code": code, "method": method,
                                                "conf": result["confidence"], "label": code}
                        ls.active_tracks[tid] = code
                        claimed.add(code)

            # 5. Evidence gate — REAL production method. When it passes we
            # write a real sighting row via production log_sighting(), so
            # evidence_precision scores the actual stored evidence trail.
            evidence_written = False
            if code != UNASSIGNED and ls._evidence_gate_ok(tid, code, face_emb, fresh_face_id, db):
                last = sight_cache.get(code)
                if (dedup_seconds <= 0.0 or last is None
                        or (frame.timestamp - last) > dedup_seconds):
                    sight_cache[code] = frame.timestamp
                    if log_sighting(unique_code=code, location_id=ls.location_id,
                                    zone_id=ls.zone_id, camera_id=ls.camera_id,
                                    confidence=1.0, db=db):
                        evidence_written = True

            det_results.append({"bbox": bbox, "tid": tid, "code": code,
                                "method": method, "evidence_written": evidence_written,
                                "pending": det_pending})

        # 6. Attribute detections to ground truth, greedily by IoU, one
        #    detection per GT person. Every GT person yields exactly one
        #    record; unmatched GT people are detection misses.
        def _iou(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0

        used = set()
        order = order_start
        for gp in frame.ground_truth:
            chosen = None
            if gp.bbox is not None:
                best, best_iou = None, 0.0
                for i, d in enumerate(det_results):
                    if i in used:
                        continue
                    v = _iou(d["bbox"], gp.bbox)
                    if v > best_iou:
                        best, best_iou = i, v
                if best is not None and best_iou >= 0.3:
                    chosen = best
            elif len(frame.ground_truth) == 1:
                # Person-centric crop: the whole frame is this person, so the
                # largest unused detection is theirs.
                cands = [i for i in range(len(det_results)) if i not in used]
                if cands:
                    chosen = max(cands, key=lambda i: det_results[i]["bbox"][2]
                                 * det_results[i]["bbox"][3])

            if chosen is None:
                out.append(Assignment(
                    frame_id=frame.frame_id, camera_id=frame.camera_id,
                    seq_id=frame.camera_id, gt_person=gp.person_id,
                    code=UNASSIGNED, tracker_id=None, method="no_detection",
                    order=order, evidence_written=False))
            else:
                used.add(chosen)
                d = det_results[chosen]
                rec = Assignment(
                    frame_id=frame.frame_id, camera_id=frame.camera_id,
                    seq_id=frame.camera_id, gt_person=gp.person_id,
                    code=d["code"], tracker_id=d["tid"], method=d["method"],
                    order=order, evidence_written=d["evidence_written"])
                if d.get("pending") is not None:
                    # Retroactive labelling target. A later commit on this
                    # track rewrites this record's code in place.
                    d["pending"].target = rec
                out.append(rec)
            order += 1
    finally:
        db.close()
    return out


if __name__ == "__main__":
    main()
