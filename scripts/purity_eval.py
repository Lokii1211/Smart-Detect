"""
╔══════════════════════════════════════════════════════════════════════════╗
║  REGRESSION CHECK ONLY — NOT AN ACCURACY MEASURE.                        ║
║  Do not cite this script's output as an accuracy or purity result.       ║
║  For accuracy, use the ground-truth harness: eval/run_eval.py            ║
║  (formulas: eval/METRICS.md).                                            ║
╚══════════════════════════════════════════════════════════════════════════╝

WHY THIS IS NOT AN ACCURACY MEASURE
────────────────────────────────────
This script scores the mutual face-similarity of snapshots stored under each
SDT code. Those snapshots are not a neutral sample of the system's decisions:

  1. cameras/live_stream.py writes a sighting snapshot ONLY when
     _evidence_gate_ok() returns True.
  2. That gate requires a gate-passing face matching the assigned code at
     >= IdentityConfig.evidence_face_sim_threshold (default 0.45) cosine
     similarity — or the code to have been freshly earned from that face.
  3. This script then asks whether the faces in those snapshots are mutually
     similar, at a 0.20 threshold.

The sample is therefore FILTERED BY THE PROPERTY BEING MEASURED. Frames that
would demonstrate contamination are precisely the frames the evidence gate
refuses to save. Passing is close to guaranteed by construction, and the
result cannot support an accuracy claim.

(Partial independence remains: registration photos, `registered.jpg`, are
written by SmartIdentifier at registration time and are NOT evidence-gated.
So registration-vs-sighting comparisons carry some signal. This is why the
script still has regression value — but only that.)

WHAT IT IS STILL GOOD FOR
─────────────────────────
A fast, deterministic tripwire over an existing snapshots/ tree: it re-embeds
stored crops with no pipeline run, so a threshold or logic change that starts
merging obviously-different people will show up here in seconds. Use it as a
pre-commit smoke check, and use eval/run_eval.py for any number you report.

MECHANICS
─────────
For every SDT code, embed the faces in all its saved snapshots (registration
photo + sighting crops) and compute pairwise ArcFace cosine similarity.
Snapshots of the SAME person sit well above 0.35; a pair below 0.20 inside one
code indicates two different people share that identity.
"""
import os, sys, itertools
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recognition.face_recognizer import FaceRecognizer

SNAP = Path('snapshots')

fr = FaceRecognizer()
fr.load_model()

def embed(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    # small crops: upscale so the detector can see the face
    h, w = img.shape[:2]
    if max(h, w) < 400:
        s = 400.0 / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s)
    embs = fr.extract_embedding(img) or []
    return embs[0] if len(embs) else None

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

codes = sorted(d.name for d in SNAP.iterdir() if d.is_dir())
print(f'{len(codes)} SDT codes: {codes}')
merged = []
for code in codes:
    embs = []
    for p in sorted((SNAP / code).glob('*.jpg')):
        e = embed(p)
        if e is not None:
            embs.append((p.name, e))
    if len(embs) < 2:
        print(f'{code}: {len(embs)} usable face snapshot(s) — purity n/a')
        continue
    sims = [(a[0], b[0], cos(a[1], b[1])) for a, b in itertools.combinations(embs, 2)]
    worst = min(sims, key=lambda s: s[2])
    status = 'MERGE SUSPECTED' if worst[2] < 0.20 else 'pure'
    if worst[2] < 0.20:
        merged.append(code)
    print(f'{code}: {len(embs)} faces, min pairwise sim {worst[2]:.3f} '
          f'({worst[0]} vs {worst[1]}) -> {status}')
print(f'\nRESULT: {len(merged)}/{len(codes)} codes contain merged identities: {merged or "none"}')
