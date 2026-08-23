"""
eval/run_eval_live.py
──────────────────────
LIVE-PATH ablation runner: drives cameras/live_stream.py's REAL per-frame
loop — LiveStream._analyze_frame() — over a labeled dataset, no camera, no
threads, no MJPEG.

WHY THIS FILE EXISTS
────────────────────
eval/run_eval.py re-creates _analyze_frame's ORCHESTRATION (it cannot call
the method itself — see its module docstring for why), which means identity
logic that lives only in _analyze_frame's body — the inline face->box
association loop, the tracklet-voting block, the raw-YOLO-box "Detecting..."
entries — is never exercised by the offline runner. The double-claim bug
existed independently in both copies. This runner closes that gap: it calls
the actual production method, frame by frame, and scores what it produced.

  * one LiveStream per camera sequence (mirrors production: each camera owns
    its tracker, its per-track state, and its tracklet voter);
  * ls._face_scan_size and the detector imgsz are set per frame exactly as
    eval/run_eval.py sets them offline;
  * after each frame the harness attributes ls._latest_results["persons"]
    (the REAL annotated output, including raw untracked boxes) to ground
    truth with the same greedy IoU / single-GT-largest-detection rule as
    run_eval's _process_frame — scoring infrastructure, not identity logic;
  * at sequence end, _commit_open_tracks() mirrors the video-EOF handler so
    open tracklet-voting tracks get their one decision, exactly as
    production does when a file source finishes;
  * evidence_written is derived from the ACTUAL sighting rows _analyze_frame
    wrote to the eval DB (per-frame per-code count delta), so
    evidence_precision scores the real stored trail, not a reconstruction.

KNOWN DIVERGENCE FROM run_eval.py — read before comparing numbers
──────────────────────────────────────────────────────────────────
Both runners call production code, but they exercise different amounts of it:

  * Head-zoom second pass: _analyze_frame runs it; run_eval does not. Face
    recall here is a ceiling vs run_eval's lower bound.
  * Raw YOLO boxes: _analyze_frame appends unconfirmed detections to its
    person list as tid=None "Detecting..." entries; run_eval does not. A GT
    person matched only to a raw box scores UNASSIGNED(method=pending) here
    vs UNASSIGNED(method=no_detection) offline — same bucket, different
    diagnostic split.
  * On-screen pre-commit frames: tracklet voting shows "Detecting..." until
    commit; run_eval retroactively relabels its records. The live on-screen
    trail therefore has MORE unassigned-detecting records on short tracks
    than the offline run. The DATABASE trail (evidence rows under the
    committed code) is the faithful comparison; see _backfill_tracklet_sightings
    in live_stream.py.
  * Evidence write-rate: _analyze_frame limits to one sighting row per code
    per 30 s of WALL-CLOCK time (production policy). run_eval uses dataset
    time with sighting_dedup_seconds (0 in the consolidated runs = one row
    per gate-passing frame). Evidence_precision RATIO is comparable; the
    sample size n_written is not.

Usage (mirrors run_eval):
    .venv/bin/python eval/run_eval_live.py \
        --config config/ablation/F_tracklet_mean.json \
        --adapter chokepoint --data-root eval/data/chokepoint \
        --out eval/results/_live/F_tracklet_mean
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
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _isolate_eval_db(out_dir: Path) -> Path:
    """Point DATABASE_URL at a throwaway DB before any project import, so
    smartdetect.db is never opened, let alone written."""
    db_path = out_dir / "eval.db"
    for suffix in ("", "-shm", "-wal"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")
    return db_path


def _sighting_counts(db) -> Dict[str, int]:
    """code -> number of sighting rows currently in the eval DB."""
    from sqlalchemy import func
    from database.models import Sighting
    rows = (db.query(Sighting.unique_code, func.count(Sighting.id))
            .group_by(Sighting.unique_code).all())
    return {c: n for c, n in rows}


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _attribute(gt_people, persons_out) -> List[Tuple[str, Optional[int], str, str, bool]]:
    """
    Greedy IoU attribution of live persons_out entries to ground truth —
    identical rule to eval/run_eval.py's _process_frame step 6.

    Returns per GT person: (gt_person_id, det_index_or_None, code, method,
    evidence_written).
    """
    from eval.scoring import UNASSIGNED

    used: set = set()
    out: List[Tuple[str, Optional[int], str, str, bool]] = []
    for gp in gt_people:
        chosen = None
        if gp.bbox is not None:
            best, best_iou = None, 0.0
            for i, d in enumerate(persons_out):
                if i in used:
                    continue
                v = _iou(d["bbox"], gp.bbox)
                if v > best_iou:
                    best, best_iou = i, v
            if best is not None and best_iou >= 0.3:
                chosen = best
        elif len(gt_people) == 1:
            cands = [i for i in range(len(persons_out)) if i not in used]
            if cands:
                chosen = max(cands, key=lambda i: persons_out[i]["bbox"][2]
                             * persons_out[i]["bbox"][3])
        if chosen is None:
            out.append((gp.person_id, None, UNASSIGNED, "no_detection", False))
        else:
            used.add(chosen)
            d = persons_out[chosen]
            out.append((gp.person_id, chosen, d["code"], d["method"],
                        bool(d.get("ev"))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LIVE-path identity evaluation runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, choices=["folder", "chokepoint"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames (per sequence cap)")
    ap.add_argument("--seq", default=None,
                    help="only this sequence (camera id); requires --keep-db to "
                         "preserve the gallery built by earlier cameras")
    ap.add_argument("--keep-db", action="store_true",
                    help="reuse the existing eval.db in --out (resumable "
                         "per-camera runs sharing one gallery)")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "eval.db"
    if not args.keep_db:
        db_path = _isolate_eval_db(out_dir)
    else:
        # Resume mode must reopen the SAME isolated eval.db, not fall back to
        # the production smartdetect.db when DATABASE_URL is unset — that
        # silently bound later cameras to the live gallery and corrupted
        # every identity decision on them.
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    work_dir = out_dir / "workdir"
    if not args.keep_db:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        (work_dir / "snapshots").mkdir(parents=True)
        _weights = ROOT / "yolov8n.pt"
        if _weights.is_file():
            try:
                (work_dir / "yolov8n.pt").symlink_to(_weights)
            except OSError:
                shutil.copy(_weights, work_dir / "yolov8n.pt")
        if (ROOT / "models").is_dir() and not (work_dir / "models").exists():
            try:
                (work_dir / "models").symlink_to(ROOT / "models", target_is_directory=True)
            except OSError:
                pass
    else:
        (work_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    os.environ["SMARTDETECT_IDENTITY_CONFIG"] = str(Path(args.config).resolve())

    from config.identity_config import get_identity_config, reset_identity_config_cache
    from database.db import init_db, SessionLocal
    from database.models import Person, Sighting, Location
    from cameras.live_stream import LiveStream
    from eval.scoring import Assignment, UNASSIGNED, compute_all

    reset_identity_config_cache()
    cfg = get_identity_config()

    init_db()
    db = SessionLocal()
    if not args.keep_db:
        db.query(Sighting).delete()
        db.query(Person).delete()
    if not db.query(Location).filter(Location.id == "LOC-EVAL").first():
        from datetime import datetime
        db.add(Location(id="LOC-EVAL", name="Eval", type="eval",
                        address="offline", created_at=datetime.utcnow()))
    db.commit()
    db.close()

    if args.adapter == "folder":
        from eval.adapters.folder import FolderAdapter
        adapter = FolderAdapter(args.data_root)
    else:
        from eval.adapters.chokepoint import ChokePointAdapter
        adapter = ChokePointAdapter(args.data_root)
    print(f"[live] {adapter.describe()}")
    print(f"[live] config={Path(args.config).name} db={db_path}")

    records: List[Assignment] = []
    if args.keep_db and (out_dir / "assignments.json").exists():
        # Resumable per-camera runs: keep the earlier cameras' records so the
        # final assignments.json is the full corpus again.
        from eval.scoring import Assignment as _A
        for d in json.loads((out_dir / "assignments.json").read_text()):
            records.append(_A(**d))
    order = len(records)
    frame_times: List[float] = []
    vote_stats_total = {"committed": 0, "tracks": 0, "never_had_a_face": 0}

    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        for seq in adapter.sequences():
            if args.seq is not None and seq.camera_id != args.seq:
                continue
            ls = LiveStream(source=0, location_id="LOC-EVAL",
                            zone_id="eval", camera_id=seq.camera_id)
            if ls._detector is None or ls._identifier is None or ls._face_rec is None:
                sys.exit("[live] ML components unavailable — cannot evaluate.")
            ls._detector.load_model()
            ls._face_rec.load_model()

            n = 0
            for frame in adapter.frames(seq.seq_id):
                if args.max_frames and n >= args.max_frames:
                    break
                n += 1
                t0 = time.perf_counter()

                img = frame.image
                fh, fw = img.shape[:2]
                ls._detector.set_imgsz_for_resolution(fw, fh)
                ls._face_scan_size = (1280, 720) if max(fw, fh) > 1280 else (fw, fh)

                db_f = SessionLocal()
                try:
                    pre = _sighting_counts(db_f)
                    # ── THE REAL PRODUCTION PER-FRAME LOOP ──────────────────
                    ls._analyze_frame(img)
                    post = _sighting_counts(db_f)

                    # Which codes gained a sighting row THIS frame: the actual
                    # stored evidence trail, written by _analyze_frame itself.
                    gained = {c for c, n2 in post.items() if n2 > pre.get(c, 0)}
                    persons_out = ls._latest_results["persons"]
                    seen_gain: set = set()
                    for p in persons_out:
                        p["ev"] = (p["code"] in gained and p["code"] not in seen_gain)
                        if p["ev"]:
                            seen_gain.add(p["code"])

                    for gt_id, det_i, code, method, ev in _attribute(
                            frame.ground_truth, persons_out):
                        tracker_id = (persons_out[det_i]["tracker_id"]
                                      if det_i is not None else None)
                        records.append(Assignment(
                            frame_id=frame.frame_id, camera_id=frame.camera_id,
                            seq_id=frame.camera_id, gt_person=gt_id,
                            code=code, tracker_id=tracker_id, method=method,
                            order=order, evidence_written=ev))
                        order += 1
                finally:
                    db_f.close()
                frame_times.append(time.perf_counter() - t0)

            # Video-EOF equivalent: commit any still-open voting tracks so a
            # person in frame when the sequence ends still earns their code
            # and their buffered frames' sighting rows are backfilled.
            if ls._voter is not None:
                # stats() BEFORE the EOF commit — the commit forgets the
                # tracks it decides, so reading after would show 0/0.
                vs = ls._voter.stats()
                vote_stats_total["committed"] += vs["committed"]
                vote_stats_total["tracks"] += vs["tracks"]
                vote_stats_total["never_had_a_face"] += vs["never_had_a_face"]
                print(f"[live]   {seq.seq_id}: {n} frames | voter: "
                      f"{vs['committed']}/{vs['tracks']} committed, "
                      f"{vs['never_had_a_face']} never had a face")
            try:
                ls._commit_open_voting_tracks()
            except Exception as exc:
                print(f"[live]   warning: voting commit at sequence end failed: {exc}")
            if ls._voter is None:
                print(f"[live]   {seq.seq_id}: {n} frames")
    finally:
        os.chdir(prev_cwd)

    if not frame_times:
        sys.exit("[live] no frames processed — check --data-root")

    runtime = {
        "n_frames": len(frame_times),
        "mean_frame_latency_ms": 1000 * sum(frame_times) / len(frame_times),
        "median_frame_latency_ms": 1000 * sorted(frame_times)[len(frame_times) // 2],
        "throughput_fps": len(frame_times) / sum(frame_times),
        "total_wall_seconds": sum(frame_times),
        "machine": "Apple M5 (CPU inference, ONNXRuntime CPUExecutionProvider)",
        "driver": "LiveStream._analyze_frame (live path)",
    }

    metrics = compute_all(records, runtime)
    metrics["config_file"] = Path(args.config).name
    metrics["config_flags"] = {
        "enable_face_anchor": cfg.enable_face_anchor,
        "enable_colour_fallback": cfg.enable_colour_fallback,
        "enable_reid_fallback": cfg.enable_reid_fallback,
        "enable_id_switch_guard": cfg.enable_id_switch_guard,
        "enable_evidence_gating": cfg.enable_evidence_gating,
        "enable_tracklet_voting": cfg.enable_tracklet_voting,
    }
    metrics["dataset"] = {"adapter": adapter.name, "root": str(Path(args.data_root).resolve())}
    metrics["harness_notes"] = [
        "DRIVER: LiveStream._analyze_frame() — the real production per-frame loop.",
        "head-zoom second pass IS exercised (it runs inside _analyze_frame).",
        "raw unconfirmed YOLO boxes appear as tid=None 'Detecting...' entries.",
        "on-screen pre-commit frames are NOT retroactively relabelled (production "
        "divergence; DB trail is backfilled under the committed code).",
        "evidence rows are limited by production's 30s wall-clock dedup, so "
        "n_written is smaller than the offline run's per-frame rows.",
    ]
    metrics["voter_stats"] = vote_stats_total

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "assignments.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2))
    print(f"[live] wrote {out_dir/'metrics.json'} ({len(records)} assignment records)")


if __name__ == "__main__":
    main()
