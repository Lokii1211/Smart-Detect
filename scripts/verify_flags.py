"""
scripts/verify_flags.py
─────────────────────────
Proves — or disproves — that every IdentityConfig feature flag gates real
control flow in the running pipeline.

Two independent techniques per flag:
  1. STATIC: walk the AST of every .py file in the repo and find every
     `Attribute` node named after the flag (e.g. `.enable_face_anchor`).
     This finds genuine reads only — comments and docstrings are not part
     of the AST, so they cannot produce a false positive here. A flag with
     zero hits is INERT and reported as a hard failure.
  2. DYNAMIC: construct a minimal scenario from REAL rows in the live
     database (smartdetect.db, copied to an isolated scratch file — the
     original is never written to) that exercises the exact code path the
     flag guards, run it with the flag on and off, and require the outcome
     to differ. If it doesn't, that's also reported as a failure — this
     script does not soften a negative result.

Run: python scripts/verify_flags.py
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Isolated scratch DB — a COPY of the real one. Never written back. ──────
SCRATCH_DB  = ROOT / "scratch_verify_flags.db"
SCRATCH_DIR = ROOT / "scratch_verify_flags_workdir"
_source_db = ROOT / "smartdetect.db"
if not _source_db.exists():
    sys.exit(f"{_source_db} not found — run the app at least once to create it.")
if SCRATCH_DB.exists():
    SCRATCH_DB.unlink()
shutil.copy(_source_db, SCRATCH_DB)
SCRATCH_DIR.mkdir(exist_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH_DB}"
os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")

import cv2
import numpy as np

from config.identity_config import IdentityConfig
from database.db import SessionLocal
from database.models import Person, Sighting
from database.queries import log_sighting
from recognition.smart_identifier import SmartIdentifier
from cameras.live_stream import LiveStream

FLAGS = [
    "enable_face_anchor",
    "enable_colour_fallback",
    "enable_reid_fallback",
    "enable_id_switch_guard",
    "enable_evidence_gating",
]

EXCLUDE_DIRS = (".venv", "node_modules", "scratch_verify_flags", ".git")


# ─── Part 1: static read-site discovery (AST-based, comment-proof) ─────────

def find_read_sites(flag: str) -> list[str]:
    hits = []
    for pyfile in sorted(ROOT.rglob("*.py")):
        rel = pyfile.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        try:
            tree = ast.parse(pyfile.read_text(), filename=str(pyfile))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == flag:
                hits.append(f"{rel}:{node.lineno}")
    return hits


# ─── Shared helpers ─────────────────────────────────────────────────────────

def bgr_for_hsv(hue: int, sat: int, val: int) -> tuple:
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)


def random_unit_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    """A face embedding matching nobody — high-dimensional random vectors
    are near-orthogonal to any fixed real embedding (expected cosine sim
    ~0, std ~1/sqrt(512)), so this reliably scores far below any of our
    match thresholds (>= 0.56) against real stored faces."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def real_people(db, n: int = 6) -> list:
    people = db.query(Person).filter(Person.face_embedding.isnot(None)).limit(n).all()
    if len(people) < n:
        sys.exit(
            f"Need >= {n} real people with stored faces in smartdetect.db to "
            f"run this harness (found {len(people)}). Run the demo pipeline "
            f"first (scripts/demo_videos.py)."
        )
    return people


results: list[dict] = []


def record(flag: str, sites: list[str], on_result: str, off_result: str,
           gates: bool, note: str = "") -> None:
    results.append(dict(flag=flag, sites=sites, on=on_result, off=off_result,
                         gates=gates, note=note))
    status = "GATES CONTROL FLOW" if gates else "DOES NOT GATE (FAILURE)"
    print(f"\n[{flag}] {status}")
    print(f"  read at {len(sites)} site(s): {sites}")
    print(f"  ON  -> {on_result}")
    print(f"  OFF -> {off_result}")
    if note:
        print(f"  note: {note}")


# ═══════════════════════════════════════════════════════════════════════════
# enable_face_anchor
#
# Scenario: a face is visible but matches nobody (a real "stranger" face —
# simulated with a random embedding, since by construction no such stranger
# exists yet in our closed test DB). The SAME frame's torso colour is
# engineered to exactly match a real, recently-seen person X's stored dress
# colour. With the anchor ON, a present-but-unmatched face must block colour
# from re-associating the box to X. With it OFF, colour matches X freely.
# ═══════════════════════════════════════════════════════════════════════════

def test_face_anchor():
    db = SessionLocal()
    people = real_people(db)
    person_x = people[0]

    hue, sat, val = 90, 200, 180
    person_x.dress_color_hsv = json.dumps(
        {"hue": hue, "saturation": sat, "value": val, "hex_color": "#5aa0b4"})
    person_x.last_seen_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    frame = np.full((200, 200, 3), bgr_for_hsv(hue, sat, val), dtype=np.uint8)
    bbox = [0, 0, 200, 200]
    stranger_face = random_unit_embedding(seed=1)

    si_on  = SmartIdentifier(IdentityConfig(enable_face_anchor=True))
    si_off = SmartIdentifier(IdentityConfig(enable_face_anchor=False))
    r_on  = si_on.identify(frame, bbox, db, allow_new=False,
                            face_embedding=stranger_face, extract_face_if_missing=False)
    r_off = si_off.identify(frame, bbox, db, allow_new=False,
                             face_embedding=stranger_face, extract_face_if_missing=False)
    db.close()

    gates = (r_on["unique_code"] != person_x.unique_code
             and r_off["unique_code"] == person_x.unique_code
             and r_off["method"] == "dress_color")
    record("enable_face_anchor", find_read_sites("enable_face_anchor"),
           on_result=f"{r_on['method']} -> {r_on['unique_code']}",
           off_result=f"{r_off['method']} -> {r_off['unique_code']}",
           gates=gates,
           note=(f"stranger face (random embedding) + torso colour engineered to "
                 f"match {person_x.unique_code}'s stored dress colour, "
                 f"last_seen_at=now (inside the 10-min re-association window)"))


# ═══════════════════════════════════════════════════════════════════════════
# enable_colour_fallback
#
# Same colour-match scenario as above, but with enable_face_anchor pinned
# OFF on both sides — isolates the master switch for Method 2 itself from
# the anchor's independent gate.
# ═══════════════════════════════════════════════════════════════════════════

def test_colour_fallback():
    db = SessionLocal()
    people = real_people(db)
    person_x = people[1]

    hue, sat, val = 40, 210, 160
    person_x.dress_color_hsv = json.dumps(
        {"hue": hue, "saturation": sat, "value": val, "hex_color": "#8caa3c"})
    person_x.last_seen_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    frame = np.full((200, 200, 3), bgr_for_hsv(hue, sat, val), dtype=np.uint8)
    bbox = [0, 0, 200, 200]
    stranger_face = random_unit_embedding(seed=2)

    cfg_on  = IdentityConfig(enable_face_anchor=False, enable_colour_fallback=True)
    cfg_off = IdentityConfig(enable_face_anchor=False, enable_colour_fallback=False)
    r_on  = SmartIdentifier(cfg_on).identify(frame, bbox, db, allow_new=False,
                            face_embedding=stranger_face, extract_face_if_missing=False)
    r_off = SmartIdentifier(cfg_off).identify(frame, bbox, db, allow_new=False,
                            face_embedding=stranger_face, extract_face_if_missing=False)
    db.close()

    gates = (r_on["method"] == "dress_color" and r_on["unique_code"] == person_x.unique_code
             and r_off["unique_code"] != person_x.unique_code)
    record("enable_colour_fallback", find_read_sites("enable_colour_fallback"),
           on_result=f"{r_on['method']} -> {r_on['unique_code']}",
           off_result=f"{r_off['method']} -> {r_off['unique_code']}",
           gates=gates,
           note=(f"enable_face_anchor pinned False on both sides to isolate this flag; "
                 f"torso colour engineered to match {person_x.unique_code}"))


# ═══════════════════════════════════════════════════════════════════════════
# enable_reid_fallback
#
# Uses a REAL registration photo and a REAL OSNet embedding extracted from
# it (not synthetic) — stores that embedding on a person, then re-submits
# the SAME crop as the query so Method 3 scores near-1.0 similarity.
# enable_face_anchor and enable_colour_fallback are pinned OFF on both sides
# to isolate Method 3's own master switch.
# ═══════════════════════════════════════════════════════════════════════════

def test_reid_fallback():
    from recognition.reid_model import PersonReID

    db = SessionLocal()
    people = real_people(db)
    person_x = people[2]

    photo_path = ROOT / "snapshots" / person_x.unique_code / "registered.jpg"
    if not photo_path.exists():
        db.close()
        record("enable_reid_fallback", find_read_sites("enable_reid_fallback"),
               on_result="SKIPPED", off_result="SKIPPED", gates=False,
               note=f"no registration photo at {photo_path} — cannot build a real crop")
        return

    crop = cv2.imread(str(photo_path))
    reid = PersonReID()
    if getattr(reid, "is_stub", False):
        db.close()
        record("enable_reid_fallback", find_read_sites("enable_reid_fallback"),
               on_result="SKIPPED", off_result="SKIPPED", gates=False,
               note="PersonReID is in stub mode on this machine — cannot test real re-ID matching")
        return

    real_reid_emb = reid.extract_features(crop)
    person_x.reid_embedding = json.dumps(real_reid_emb.tolist())
    person_x.last_seen_at = datetime.now(timezone.utc).replace(tzinfo=None)
    # Avoid the colour method incidentally matching (it's the same photo, so
    # its colour trivially matches its own stored colour too) — blank it.
    person_x.dress_color_hsv = None
    db.commit()

    bbox = [0, 0, crop.shape[1], crop.shape[0]]
    stranger_face = random_unit_embedding(seed=3)

    cfg_on  = IdentityConfig(enable_face_anchor=False, enable_colour_fallback=False,
                              enable_reid_fallback=True)
    cfg_off = IdentityConfig(enable_face_anchor=False, enable_colour_fallback=False,
                              enable_reid_fallback=False)
    r_on  = SmartIdentifier(cfg_on).identify(crop, bbox, db, allow_new=False,
                            face_embedding=stranger_face, extract_face_if_missing=False)
    r_off = SmartIdentifier(cfg_off).identify(crop, bbox, db, allow_new=False,
                            face_embedding=stranger_face, extract_face_if_missing=False)
    db.close()

    gates = (r_on["method"] == "body_structure" and r_on["unique_code"] == person_x.unique_code
             and r_off["unique_code"] != person_x.unique_code)
    record("enable_reid_fallback", find_read_sites("enable_reid_fallback"),
           on_result=f"{r_on['method']} -> {r_on['unique_code']} (conf {r_on['confidence']})",
           off_result=f"{r_off['method']} -> {r_off['unique_code']}",
           gates=gates,
           note=(f"real OSNet embedding of {person_x.unique_code}'s own registration "
                 f"photo stored, then the same photo re-submitted as the query crop"))


# ═══════════════════════════════════════════════════════════════════════════
# enable_id_switch_guard
#
# Simulates exactly the scenario in cameras/live_stream.py: a track cached
# as person A is, for two consecutive analysis cycles, shown a gate-passing
# face that is actually person B's REAL stored face (an occlusion hand-off).
# Calls the real, extracted LiveStream._apply_id_switch_guard() — not a
# reimplementation of its logic.
# ═══════════════════════════════════════════════════════════════════════════

def pick_low_similarity_pair(people: list) -> tuple:
    """Pick the pair among `people` with the LOWEST face similarity, so the
    contradiction test doesn't depend on luck."""
    best_pair, best_sim = None, 2.0
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            ea = np.asarray(json.loads(people[i].face_embedding), dtype=np.float32)
            eb = np.asarray(json.loads(people[j].face_embedding), dtype=np.float32)
            sim = cosine(ea, eb)
            if sim < best_sim:
                best_sim, best_pair = sim, (people[i], people[j])
    return best_pair, best_sim


def test_id_switch_guard():
    db = SessionLocal()
    people = real_people(db, n=6)
    (person_a, person_b), sim = pick_low_similarity_pair(people)
    emb_b = np.asarray(json.loads(person_b.face_embedding), dtype=np.float32)
    db.close()

    threshold = 0.35
    if sim >= threshold:
        record("enable_id_switch_guard", find_read_sites("enable_id_switch_guard"),
               on_result="SKIPPED", off_result="SKIPPED", gates=False,
               note=(f"lowest-similarity real pair still scores {sim:.3f} >= "
                     f"{threshold} — cannot construct a guaranteed contradiction "
                     f"from this DB's population"))
        return

    def run(guard_on: bool):
        db2 = SessionLocal()
        ls = LiveStream(source=0)  # never .start()ed — no camera/device opened
        ls._identity_config = IdentityConfig(
            enable_id_switch_guard=guard_on,
            id_switch_contradiction_limit=2,
            id_switch_similarity_threshold=threshold,
        )
        tid = 4242
        cached = {"code": person_a.unique_code, "method": "face", "conf": 1.0}
        ls._track_codes[tid] = cached
        for _ in range(2):  # two consecutive contradicting cycles
            cached = ls._apply_id_switch_guard(tid, cached, emb_b, db2)
        still_cached = tid in ls._track_codes
        db2.close()
        return cached, still_cached

    cached_on,  present_on  = run(True)
    cached_off, present_off = run(False)

    on_desc  = "cache DROPPED" if cached_on is None else f"cache KEPT ({cached_on['code']})"
    off_desc = "cache DROPPED" if cached_off is None else f"cache KEPT ({cached_off['code']})"
    gates = (cached_on is None and not present_on
             and cached_off is not None and present_off)
    record("enable_id_switch_guard", find_read_sites("enable_id_switch_guard"),
           on_result=on_desc, off_result=off_desc, gates=gates,
           note=(f"track cached as {person_a.unique_code} (real code), contradicted "
                 f"twice by {person_b.unique_code}'s real stored face (sim={sim:.3f})"))


# ═══════════════════════════════════════════════════════════════════════════
# enable_evidence_gating
#
# A faceless box (face_emb=None) carrying a code it did NOT just earn this
# cycle (fresh_face_id=False) — exactly "an inherited code" per the ask.
# Calls the real, extracted LiveStream._evidence_gate_ok(), then actually
# calls the real log_sighting()/_save_sighting_snapshot() so the proof is a
# genuine DB row / file, not just a boolean.
# ═══════════════════════════════════════════════════════════════════════════

def test_evidence_gating():
    db = SessionLocal()
    people = real_people(db)
    code = people[3].unique_code
    db.close()

    def run(gating_on: bool):
        db2 = SessionLocal()
        ls = LiveStream(source=0)
        ls._identity_config = IdentityConfig(enable_evidence_gating=gating_on)
        tid = 9191
        gate_ok = ls._evidence_gate_ok(tid, code, None, False, db2)  # face_emb=None, not freshly earned
        wrote_sighting = wrote_snapshot = False
        before = db2.query(Sighting).filter(Sighting.unique_code == code).count()
        if code and code != "Detecting..." and gate_ok:
            wrote_sighting = log_sighting(unique_code=code, location_id="LOC-VERIFY",
                                           zone_id="test", camera_id="CAM-VERIFY",
                                           confidence=0.5, db=db2)
            crop = np.full((80, 80, 3), 120, dtype=np.uint8)
            cwd = os.getcwd()
            try:
                os.chdir(SCRATCH_DIR)
                wrote_snapshot = ls._save_sighting_snapshot(code, crop) is not None
            finally:
                os.chdir(cwd)
        after = db2.query(Sighting).filter(Sighting.unique_code == code).count()
        db2.close()
        return gate_ok, wrote_sighting, wrote_snapshot, after - before

    gate_on,  wrote_sight_on,  wrote_snap_on,  delta_on  = run(True)
    gate_off, wrote_sight_off, wrote_snap_off, delta_off = run(False)

    gates = (gate_on is False and delta_on == 0 and not wrote_snap_on
             and gate_off is True and delta_off == 1 and wrote_snap_off)
    record("enable_evidence_gating", find_read_sites("enable_evidence_gating"),
           on_result=f"gate_ok={gate_on}, sighting_rows_added={delta_on}, snapshot_written={wrote_snap_on}",
           off_result=f"gate_ok={gate_off}, sighting_rows_added={delta_off}, snapshot_written={wrote_snap_off}",
           gates=gates,
           note=f"faceless box (face_emb=None) carrying inherited code {code}, not freshly earned this cycle")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("Flag verification harness — scratch DB:", SCRATCH_DB)
    print("=" * 78)
    try:
        test_face_anchor()
        test_colour_fallback()
        test_reid_fallback()
        test_id_switch_guard()
        test_evidence_gating()
    finally:
        try:
            for suffix in ("", "-shm", "-wal"):
                Path(str(SCRATCH_DB) + suffix).unlink(missing_ok=True)
            shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
        except Exception:
            pass

    print("\n" + "=" * 100)
    print(f"{'flag':<26} {'read sites':<11} {'ON result':<38} {'OFF result':<28} {'GATES?'}")
    print("-" * 100)
    any_failed = False
    for r in results:
        gates_str = "YES" if r["gates"] else "NO — FAILURE"
        if not r["gates"]:
            any_failed = True
        print(f"{r['flag']:<26} {len(r['sites']):<11} {r['on'][:37]:<38} {r['off'][:27]:<28} {gates_str}")
    print("=" * 100)

    for flag in FLAGS:
        sites = next((r["sites"] for r in results if r["flag"] == flag), None)
        if sites is not None and len(sites) == 0:
            print(f"\n*** INERT FLAG: {flag} is never read anywhere in the codebase. ***")
            any_failed = True

    if any_failed:
        print("\nRESULT: at least one flag failed to demonstrate gated control flow. See above.")
        sys.exit(1)
    else:
        print("\nRESULT: all 5 flags independently verified to gate real control flow"
              " using real DB data.")


if __name__ == "__main__":
    main()
