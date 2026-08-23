"""
eval/profile_density.py
────────────────────────
Per-stage latency as a function of PERSON COUNT.

WHY THE FRAMES ARE SYNTHETIC
────────────────────────────
No source in this repository exceeds three people in a frame (measured: max 3
on ChokePoint and on every demo clip). Densities of 5 and 10+ therefore cannot
be sampled from real footage.

This composites real person crops into a fixed-size frame. That is legitimate
for a LATENCY profile and would not be for an accuracy one: the detections and
faces are real pixels of real people, only their arrangement is synthetic, and
holding the frame size constant is what isolates per-person cost from
per-pixel cost. No accuracy number is computed here.

Usage:
    python eval/profile_density.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")

import cv2
import numpy as np

DENSITIES = [1, 3, 5, 8, 10]
REPEATS = 12


def source_person(adapter):
    """A real frame plus the crop of one real person in it, from ChokePoint."""
    from recognition.object_detector import ObjectDetector
    det = ObjectDetector(); det.load_model()
    for seq in adapter.sequences():
        for fr in adapter.frames(seq.seq_id):
            if not fr.ground_truth:
                continue
            img = fr.image
            h, w = img.shape[:2]
            det.set_imgsz_for_resolution(w, h)
            people = [d["bbox"] for d in det.detect(img) if d["label"] == "person"]
            if len(people) == 1:
                x, y, bw, bh = people[0]
                crop = img[max(0, y):y + bh, max(0, x):x + bw].copy()
                if crop.size and bh > 200:
                    return img, crop
    raise SystemExit("no suitable single-person frame found")


def compose(background, crop, n, size=(800, 600)):
    """n copies of `crop` tiled into a constant-size frame."""
    W, H = size
    out = cv2.resize(background, (W, H)).copy()
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw, ch = W // cols, H // rows
    ph = int(ch * 0.94)
    pw = max(8, int(ph * crop.shape[1] / crop.shape[0]))
    if pw > cw:
        pw = int(cw * 0.94); ph = max(8, int(pw * crop.shape[0] / crop.shape[1]))
    person = cv2.resize(crop, (pw, ph))
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= n:
                break
            x = c * cw + (cw - pw) // 2
            y = r * ch + (ch - ph) // 2
            out[y:y + ph, x:x + pw] = person
            k += 1
    return out


class _Cfg:
    """Shipped defaults for the association call."""
    enable_optimal_face_assignment = False


class Timer:
    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, k, dt):
        self.t[k] += dt
        self.n[k] += 1


def profile_one(ls, cfg, img, timer):
    """One analysis cycle, instrumented per stage. Mirrors run_eval's order."""
    import supervision as sv
    from recognition.face_pose import estimate_pose

    h, w = img.shape[:2]
    ls._detector.set_imgsz_for_resolution(w, h)
    ls._face_scan_size = (w, h)

    t0 = time.perf_counter()
    dets = ls._detector.detect(img)
    timer.add("detect", time.perf_counter() - t0)
    pdets = [d for d in dets if d["label"] == "person"]

    t0 = time.perf_counter()
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
    timer.add("track", time.perf_counter() - t0)

    t0 = time.perf_counter()
    ls._face_rec.extract_embedding(img)
    faces = getattr(ls._face_rec, "_last_faces", []) or []
    timer.add("face_scan", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for f in faces:
        estimate_pose(getattr(f, "kps", None))
    timer.add("pose", time.perf_counter() - t0)

    # Associate faces to boxes exactly as the pipeline does, so arbitration
    # receives a real face_embedding. Passing None instead forces the OSNet
    # re-ID fallback that the lazy path skips whenever a face is present —
    # measured at ~455 ms/person, which would dominate the profile with an
    # artefact of the harness rather than a cost the system pays.
    t0 = time.perf_counter()
    from recognition.face_assign import match_faces_to_boxes
    fboxes = [[int(f.bbox[0]), int(f.bbox[1]),
               int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1])] for f in faces]
    amap = match_faces_to_boxes(fboxes, [b for _t, b in tracked], _Cfg(), frame_h=h)
    timer.add("assoc", time.perf_counter() - t0)

    from database.db import SessionLocal
    db = SessionLocal()
    try:
        t0 = time.perf_counter()
        for i, (tid, bbox) in enumerate(tracked):
            fi = amap.get(i)
            emb = getattr(faces[fi], "embedding", None) if fi is not None else None
            ls._identifier.identify(
                img, bbox, db, location_id="LOC-EVAL", zone_id="eval",
                allow_new=False, face_embedding=emb, face_pose=None,
                extract_face_if_missing=False)
        timer.add("arbitration", time.perf_counter() - t0)
    finally:
        db.close()
    return len(pdets), len(faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/perf/density_profile.json")
    args = ap.parse_args()

    from cameras.live_stream import LiveStream
    from database.db import init_db
    from eval.adapters.chokepoint import ChokePointAdapter

    init_db()
    adapter = ChokePointAdapter("eval/data/chokepoint")
    bg, crop = source_person(adapter)
    print(f"source person crop: {crop.shape[1]}x{crop.shape[0]} px\n")

    ls = LiveStream(source=0, camera_id="PROFILE")
    ls._detector.load_model(); ls._face_rec.load_model()

    results = {}
    for n in DENSITIES:
        frame = compose(bg, crop, n)
        timer = Timer()
        det_n = face_n = 0
        profile_one(ls, None, frame, Timer())        # warm
        for _ in range(REPEATS):
            det_n, face_n = profile_one(ls, None, frame, timer)
        total = sum(timer.t.values()) / REPEATS * 1000
        row = {k: timer.t[k] / REPEATS * 1000 for k in timer.t}
        row["_total_ms"] = total
        row["_detected_persons"] = det_n
        row["_detected_faces"] = face_n
        results[n] = row
        print(f"requested {n:2d} persons -> detector found {det_n:2d}, "
              f"faces {face_n:2d}, total {total:7.1f} ms")

    print(f"\n{'stage':14}" + "".join(f"{('n=%d' % n):>12}" for n in DENSITIES))
    stages = ["detect", "track", "face_scan", "pose", "assoc", "arbitration"]
    for s in stages:
        print(f"{s:14}" + "".join(f"{results[n].get(s, 0):11.1f} " for n in DENSITIES))
    print(f"{'TOTAL':14}" + "".join(f"{results[n]['_total_ms']:11.1f} " for n in DENSITIES))
    print(f"{'persons found':14}" + "".join(f"{results[n]['_detected_persons']:11d} " for n in DENSITIES))
    print(f"{'faces found':14}" + "".join(f"{results[n]['_detected_faces']:11d} " for n in DENSITIES))

    print(f"\n{'stage':14}{'share @n=1':>12}{'share @max':>12}{'scaling':>12}")
    lo, hi = DENSITIES[0], DENSITIES[-1]
    for s in stages:
        a, b = results[lo].get(s, 0), results[hi].get(s, 0)
        print(f"{s:14}{100*a/results[lo]['_total_ms']:11.1f}%"
              f"{100*b/results[hi]['_total_ms']:11.1f}%"
              f"{(b/a if a > 0 else float('nan')):11.1f}x")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
