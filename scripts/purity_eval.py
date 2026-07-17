"""
Identity purity evaluation for SmartDetect.

For every SDT code, embed the faces in all its saved snapshots
(registration photo + sighting crops) and compute pairwise ArcFace cosine
similarity. Snapshots of the SAME person sit well above 0.35; a pair below
0.20 inside one code means two different people share that identity (merge).
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
