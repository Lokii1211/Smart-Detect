"""
Diagnostic: masked-face behavior through the REAL identity pipeline.
Diagnosis only — reads config, runs the same functions the live path runs,
writes NOTHING to the DB.

For each enrolled identity: baseline / surgical-mask / mask+sunglasses variants
are synthesized from the identity's own registered photo + InsightFace
landmarks, then pushed through:
  1. FaceRecognizer.extract_embedding()  (same model/det settings as live)
  2. gate_passes()                        (same quality gate)
  3. SmartIdentifier.identify()           (same matcher, face_embedding as live)
Report per condition: detection, gate, identity, cosine-sim, fallback.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2, sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from config.identity_config import get_identity_config
from recognition.face_recognizer import FaceRecognizer
from recognition.face_quality import gate_passes
from recognition.face_pose import estimate_pose
from recognition.smart_identifier import SmartIdentifier
from database.db import SessionLocal

DB = sqlite3.connect("smartdetect.db")
CFG = get_identity_config()

# ---------- mask synthesis (deterministic, landmark-driven) ----------
def _draw_surgical_mask(img, kps, cloth_bgr=(228, 196, 168), sunglasses=False):
    """Cover nose+mouth with a cloth mask; optionally add a dark eyewear band."""
    out = img.copy()
    h, w = out.shape[:2]
    # kps: [0]=left eye [1]=right eye [2]=nose [3]=left mouth [4]=right mouth
    le, re = kps[0], kps[1]
    nose, lm, rm = kps[2], kps[3], kps[4]
    eye_mid_y = (le[1] + re[1]) / 2.0
    eye_dy = abs(re[1] - le[1]) or 20.0

    if sunglasses:
        # dark band covering the eyes, ~2.2 eye-heights tall, full face width
        top = int(eye_mid_y - eye_dy * 1.1)
        bot = int(eye_mid_y + eye_dy * 1.1)
        band_top = max(0, top)
        band_bot = min(h, bot)
        if band_bot > band_top:
            band = out[band_top:band_bot, :, :]
            rng = np.random.default_rng(7)
            noise = rng.normal(0, 12, band.shape).astype(np.float32)
            band[:] = np.clip(band.astype(np.float32) * 0.25 + noise, 0, 255).astype(np.uint8)

    # mask: from just below eyes down to chin
    mid_x = (le[0] + re[0]) / 2.0
    half_w = max((re[0] - le[0]) * 0.75, 24.0)
    nose_y = nose[1]
    mouth_mid_y = (lm[1] + rm[1]) / 2.0
    chin_y = mouth_mid_y + (mouth_mid_y - nose_y) * 1.15
    top_y = nose_y + (mouth_mid_y - nose_y) * 0.15

    pts = np.array([
        [mid_x - half_w, top_y],       # top-left (under eyes)
        [mid_x + half_w, top_y],       # top-right
        [mid_x + half_w * 0.9, chin_y],# chin-right
        [mid_x - half_w * 0.9, chin_y],# chin-left
    ], dtype=np.int32)
    cv2.fillPoly(out, [pts], cloth_bgr)

    # subtle fabric fold lines so the mask is not a flat rectangle
    rng = np.random.default_rng(11)
    for _ in range(3):
        yy = int(top_y + (chin_y - top_y) * rng.uniform(0.25, 0.8))
        cv2.line(out, (int(mid_x - half_w), yy), (int(mid_x + half_w), yy),
                 (int(cloth_bgr[0]*0.85), int(cloth_bgr[1]*0.85), int(cloth_bgr[2]*0.85)), 1)
    return out

# ---------- run one condition through the real pipeline ----------
def run_condition(frame, person_bbox, rec, identifier, db, label, sim_vs_enrolled):
    dets = rec.extract_embedding(frame)   # same call the live full-frame scan makes
    faces = rec._last_faces or []
    print(f"\n── {label} ──")
    print(f"  faces detected: {len(faces)}")
    if not faces:
        # Live path: a person box with no visible face still runs identify()
        # with face_embedding=None — colour / body Re-ID fallback decide.
        result = identifier.identify(
            frame, person_bbox, db,
            location_id="LOC-001", zone_id="entrance",
            allow_new=False,
            face_embedding=None,
            face_pose=None,
            extract_face_if_missing=False,
            exclude_codes=set(),
        )
        code, method, conf = result["unique_code"], result["method"], result["confidence"]
        print(f"  → NO FACE DETECTED at all")
        print(f"  identify() → code={code}  method={method}  conf={conf:.3f}")
        if method in ("dress_color", "body_structure", "reid"):
            print(f"  !! FALLBACK fired: {method} matched {conf}")
        return
    f = faces[0]
    bbox = [float(v) for v in f.bbox]
    fh = bbox[3] - bbox[1]
    det = float(getattr(f, "det_score", 0.0))
    pose = estimate_pose(getattr(f, "kps", None))
    crop = frame[max(0,int(bbox[1])):int(bbox[3]), max(0,int(bbox[0])):int(bbox[2])]
    gate = gate_passes([int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(fh)],
                       det, pose, CFG, crop=crop,
                       person_boxes=[person_bbox], owner_box=person_bbox)
    emb = getattr(f, "embedding", None)
    print(f"  face box h={fh:.0f}px  det_score={det:.3f}  pose={pose}")
    print(f"  quality gate (48px / 0.60): {'PASS ✓' if gate else 'FAIL ✗'}")
    if emb is not None:
        sim = float(np.dot(emb, sim_vs_enrolled) /
                    (np.linalg.norm(emb) * np.linalg.norm(sim_vs_enrolled)))
        print(f"  cosine vs ENROLLED template: {sim:.4f}  (face_match_threshold={CFG.face_match_threshold})")
    # identify() exactly as live_stream calls it (extract_face_if_missing=False)
    result = identifier.identify(
        frame, person_bbox, db,
        location_id="LOC-001", zone_id="entrance",
        allow_new=False,
        face_embedding=emb if gate else None,
        face_pose=pose if gate else None,
        extract_face_if_missing=False,
        exclude_codes=set(),
    )
    code, method, conf = result["unique_code"], result["method"], result["confidence"]
    print(f"  identify() → code={code}  method={method}  conf={conf:.3f}")
    if code == "Detecting..." and method in ("dress_color", "body_structure"):
        print(f"  !! FALLBACK fired: {method} matched {conf}")

# ---------- main ----------
def main():
    rec = FaceRecognizer()
    rec.load_model()
    identifier = SmartIdentifier(CFG)
    # Diagnosis-only: never let identify() blend a match into the enrolled
    # template (that is a DB write we must not perform here).
    identifier._update_face_template = lambda *a, **k: None
    db = SessionLocal()

    codes = sys.argv[1:] or ["SDT-0001", "SDT-0017", "SDT-0049"]
    for code in codes:
        row = DB.execute("SELECT photo_path, face_embedding FROM persons WHERE unique_code=?", (code,)).fetchone()
        if not row or not row[0] or not os.path.isfile(row[0]):
            print(f"\n### {code}: no registered photo/embedding — SKIP"); continue
        img = cv2.imread(row[0])
        enrolled = np.array(json.loads(row[1]), dtype=np.float32)
        print(f"\n{'='*64}\n### ENROLLED IDENTITY {code}  (photo {row[0]})")
        # person box = whole crop (this photo IS the person crop from enrolment)
        ph, pw = img.shape[:2]
        person_bbox = [0, 0, pw, ph]

        # baseline first: confirm the pipeline self-matches
        run_condition(img, person_bbox, rec, identifier, db, f"{code} A. BASELINE (no mask)", enrolled)

        # variant B: surgical mask — detect face first to get fresh landmarks
        b_dets = rec.extract_embedding(img)
        b_faces = rec._last_faces or []
        if not b_faces:
            print("  (no landmarks from baseline — using kps=None, mask skipped)")
            b_masked = img
        else:
            kps = b_faces[0].kps
            b_masked = _draw_surgical_mask(img, kps)
        run_condition(b_masked, person_bbox, rec, identifier, db,
                      f"{code} B. SURGICAL MASK (eyes visible)", enrolled)

        # variant C: mask + sunglasses
        if b_faces:
            kps = b_faces[0].kps
            c_masked = _draw_surgical_mask(img, kps, sunglasses=True)
            run_condition(c_masked, person_bbox, rec, identifier, db,
                          f"{code} C. MASK + SUNGLASSES", enrolled)

    db.close()
    print(f"\n{'='*64}\nConfig used: gate={CFG.face_quality_min_height_px}px/{CFG.face_quality_min_det_score} "
          f"face_thr={CFG.face_match_threshold} reid_thr={CFG.reid_match_threshold} "
          f"colour_thr={CFG.colour_match_threshold} "
          f"learned_gate={getattr(CFG,'enable_learned_quality_gate',False)} "
          f"face_anchor={CFG.enable_face_anchor}")

if __name__ == "__main__":
    main()
