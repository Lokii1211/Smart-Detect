"""
eval/build_quality_dataset.py
──────────────────────────────
Extract a face-quality training set from ChokePoint ground truth.

TARGET — the discriminative margin
──────────────────────────────────
The fixed gate (48 px height, 0.60 det_score) is a hand-set proxy for a
quantity nobody measured: whether a face embedding can actually tell its owner
apart from everyone else. This builds that quantity directly.

For each detected face f belonging to ground-truth person p:

    genuine  = max cos(f, R_p)          R_p = p's reference embeddings
    impostor = max cos(f, R_q) q != p   over every other identity
    y        = genuine - impostor       the margin

y >> 0  : the embedding is discriminative — safe to identify from
y ~= 0  : the face is as close to a stranger as to its owner — dangerous
y <  0  : the embedding actively points at the wrong person

LEAKAGE CONTROL
───────────────
References are the best frames per identity (largest, most frontal, highest
det_score). Those exact frames are EXCLUDED from the emitted rows — a reference
face scored against itself gives genuine = 1.0 by construction, which would
teach the model to predict its own input.

Output: eval/data/face_quality.npz  (features, labels, groups, provenance)
Split by GROUP (identity) downstream — never by frame.

Usage:
    python eval/build_quality_dataset.py --data-root eval/data/chokepoint
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")

import cv2
import numpy as np

# Feature order is the model's input contract — recognition/face_quality.py
# rebuilds this exact vector at inference time.
FEATURE_NAMES = [
    "face_h_px", "face_w_px", "det_score",
    "yaw", "pitch", "roll", "frontality",
    "blur_var_laplacian", "brightness", "contrast",
    "occlusion_frac", "face_to_person_ratio",
]

# References per identity: enough to cover pose variation, few enough that
# every one is genuinely high quality.
N_REFERENCES = 3


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _crop_stats(frame: np.ndarray, box) -> Dict[str, float]:
    """Blur / brightness / contrast over the face crop."""
    x, y, w, h = [int(v) for v in box]
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return {"blur_var_laplacian": 0.0, "brightness": 0.0, "contrast": 0.0}
    crop = frame[y0:y1, x0:x1]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "blur_var_laplacian": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
    }


def _occlusion(face_box, person_boxes: List[List[int]], owner_idx: Optional[int]) -> float:
    """Fraction of the face box covered by a person box other than its owner."""
    fx, fy, fw, fh = face_box
    area = max(1.0, float(fw * fh))
    covered = 0.0
    for i, pb in enumerate(person_boxes):
        if i == owner_idx:
            continue
        px, py, pw, ph = pb
        ix = max(0, min(fx + fw, px + pw) - max(fx, px))
        iy = max(0, min(fy + fh, py + ph) - max(fy, py))
        covered += ix * iy
    return float(min(1.0, covered / area))


def extract(data_root: str, limit_per_seq: int = 0) -> Dict:
    from cameras.live_stream import LiveStream
    from eval.adapters.chokepoint import ChokePointAdapter
    from recognition.face_pose import estimate_pose

    adapter = ChokePointAdapter(data_root)
    ls = LiveStream(source=0, camera_id="QUALITY-EXTRACT")
    if ls._detector is None or ls._face_rec is None:
        sys.exit("ML components unavailable — cannot extract.")
    ls._detector.load_model()
    ls._face_rec.load_model()

    rows: List[Dict] = []
    for seq in adapter.sequences():
        n = 0
        for frame in adapter.frames(seq.seq_id):
            if not frame.ground_truth:
                continue                      # unlabelled frame — nothing to learn from
            if limit_per_seq and n >= limit_per_seq:
                break
            n += 1
            img = frame.image
            fh, fw = img.shape[:2]
            ls._detector.set_imgsz_for_resolution(fw, fh)
            ls._face_scan_size = (fw, fh)

            person_boxes = [d["bbox"] for d in ls._detector.detect(img)
                            if d["label"] == "person"]

            ls._face_rec.extract_embedding(img)
            faces = getattr(ls._face_rec, "_last_faces", []) or []

            for gp in frame.ground_truth:
                if gp.face_bbox is None:
                    continue
                # Associate a detected face to this GT person by best IoU
                best, best_iou = None, 0.0
                for f in faces:
                    b = f.bbox.astype(int)
                    fb = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                    v = _iou(fb, gp.face_bbox)
                    if v > best_iou:
                        best, best_iou = f, v
                if best is None or best_iou < 0.20:
                    continue                  # no confident association

                b = best.bbox.astype(int)
                fbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                emb = getattr(best, "embedding", None)
                if emb is None:
                    continue
                emb = np.asarray(emb, dtype=np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-9)

                pose = estimate_pose(getattr(best, "kps", None))
                stats = _crop_stats(img, fbox)

                # owner person box = the one containing the face centre
                cx, cy = fbox[0] + fbox[2] / 2, fbox[1] + fbox[3] / 2
                owner, owner_h = None, 0
                for i, pb in enumerate(person_boxes):
                    if pb[0] <= cx <= pb[0] + pb[2] and pb[1] <= cy <= pb[1] + pb[3]:
                        if pb[3] > owner_h:
                            owner, owner_h = i, pb[3]

                rows.append({
                    "person": gp.person_id,
                    "camera": frame.camera_id,
                    "frame_id": frame.frame_id,
                    "emb": emb,
                    "feat": {
                        "face_h_px": float(fbox[3]),
                        "face_w_px": float(fbox[2]),
                        "det_score": float(getattr(best, "det_score", 1.0) or 1.0),
                        "yaw": float(pose.yaw) if pose else 0.0,
                        "pitch": float(pose.pitch) if pose else 0.5,
                        "roll": float(pose.roll) if pose else 0.0,
                        "frontality": float(pose.frontality) if pose else 0.0,
                        "occlusion_frac": _occlusion(fbox, person_boxes, owner),
                        "face_to_person_ratio": (float(fbox[3]) / owner_h) if owner_h else 0.0,
                        **stats,
                    },
                })
        print(f"  {seq.seq_id}: {n} labelled frames, {len(rows)} rows so far", flush=True)
    return {"rows": rows}


def build_margins(rows: List[Dict]) -> Dict:
    """
    Pick per-identity references, hold them out, and label the rest with
    genuine - impostor.
    """
    by_person: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_person.setdefault(r["person"], []).append(i)

    # Reference quality score: big, frontal, confidently detected.
    def refscore(r):
        f = r["feat"]
        return (f["face_h_px"] / 200.0) + f["frontality"] + f["det_score"]

    ref_idx: Dict[str, List[int]] = {}
    held_out = set()
    for p, idxs in by_person.items():
        ranked = sorted(idxs, key=lambda i: -refscore(rows[i]))
        take = ranked[:N_REFERENCES]
        ref_idx[p] = take
        held_out.update(take)

    ref_mat = {p: np.stack([rows[i]["emb"] for i in idxs])
               for p, idxs in ref_idx.items() if idxs}

    X, y, groups, meta = [], [], [], []
    for i, r in enumerate(rows):
        if i in held_out:
            continue                          # leakage guard
        p = r["person"]
        if p not in ref_mat:
            continue
        e = r["emb"]
        genuine = float(np.max(ref_mat[p] @ e))
        impostor = -1.0
        for q, M in ref_mat.items():
            if q == p:
                continue
            impostor = max(impostor, float(np.max(M @ e)))
        if impostor <= -1.0:
            continue
        X.append([r["feat"][k] for k in FEATURE_NAMES])
        y.append(genuine - impostor)
        groups.append(p)
        meta.append((r["camera"], r["frame_id"], genuine, impostor))

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "groups": np.asarray(groups),
        "feature_names": np.asarray(FEATURE_NAMES),
        "cameras": np.asarray([m[0] for m in meta]),
        "frame_ids": np.asarray([m[1] for m in meta]),
        "genuine": np.asarray([m[2] for m in meta], dtype=np.float32),
        "impostor": np.asarray([m[3] for m in meta], dtype=np.float32),
        "n_reference_frames_held_out": np.int32(len(held_out)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="eval/data/chokepoint")
    ap.add_argument("--limit-per-seq", type=int, default=0)
    ap.add_argument("--out", default="eval/data/face_quality.npz")
    args = ap.parse_args()

    print("extracting faces + features ...", flush=True)
    ex = extract(args.data_root, args.limit_per_seq)
    print(f"associated faces: {len(ex['rows'])}")

    print("building margins (references held out) ...", flush=True)
    ds = build_margins(ex["rows"])
    np.savez_compressed(args.out, **ds)

    y = ds["y"]
    print(f"\nwrote {args.out}")
    print(f"  rows            : {len(y)}")
    print(f"  identities      : {len(set(ds['groups'].tolist()))}")
    print(f"  held-out refs   : {int(ds['n_reference_frames_held_out'])}")
    print(f"  margin y: min={y.min():.3f} median={np.median(y):.3f} "
          f"mean={y.mean():.3f} max={y.max():.3f}")
    print(f"  y <= 0 (dangerous): {int((y <= 0).sum())} ({100*(y<=0).mean():.1f}%)")


if __name__ == "__main__":
    main()
