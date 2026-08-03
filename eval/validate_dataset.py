"""
eval/validate_dataset.py
─────────────────────────
Dataset adequacy gate for the identity ablation.

An ablation run on an underpowered dataset produces numbers that look like
results but cannot support a conclusion. This module computes corpus
statistics from ground-truth labels alone (no pixel decoding) and REFUSES
to let the sweep proceed below the minimum thresholds.

Thresholds (see MINIMUMS): 15 identities, 1000 labelled frames,
10 cross-camera transitions.

Run standalone:
    python eval/validate_dataset.py --adapter folder --data-root <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Minimum corpus for a defensible ablation ────────────────────────────────
MIN_IDENTITIES = 15
MIN_FRAMES = 1000
MIN_CROSS_CAMERA_TRANSITIONS = 10

# Rationale, in short:
#   identities  — with <15 people, purity and fragmentation move in huge
#                 discrete steps and one mislabelled person dominates.
#   frames      — 1000 labelled person-frames puts the 95% Wilson half-width
#                 on a ~95% proportion near +/-1.4pp; below ~300 the interval
#                 is wider than the effect sizes being compared.
#   transitions — cross-camera re-association cannot be estimated from a
#                 handful of hand-overs; 10 is the bare floor for a rate.


@dataclass
class DatasetStats:
    n_identities: int = 0
    n_frames: int = 0                    # labelled frames (>=1 GT person)
    n_person_frames: int = 0             # sum of GT people over frames
    n_cameras: int = 0
    n_cross_camera_transitions: int = 0
    identities: List[str] = field(default_factory=list)
    cameras: List[str] = field(default_factory=list)
    frames_per_identity: Dict[str, int] = field(default_factory=dict)
    identities_per_camera: Dict[str, int] = field(default_factory=dict)
    multi_camera_identities: List[str] = field(default_factory=list)
    transitions_detail: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_identities": self.n_identities,
            "n_frames": self.n_frames,
            "n_person_frames": self.n_person_frames,
            "n_cameras": self.n_cameras,
            "n_cross_camera_transitions": self.n_cross_camera_transitions,
            "identities": self.identities,
            "cameras": self.cameras,
            "frames_per_identity": self.frames_per_identity,
            "identities_per_camera": self.identities_per_camera,
            "multi_camera_identities": self.multi_camera_identities,
            "transitions_detail": self.transitions_detail,
        }


def collect_stats(adapter) -> DatasetStats:
    """
    Walk the adapter's ground-truth index (labels only, no image decoding).

    A "cross-camera transition" is one ordered consecutive camera pair for a
    single identity, ordered by first appearance — the same construction the
    cross_camera_reassociation metric scores. An identity on 3 cameras
    contributes 2 transitions.
    """
    st = DatasetStats()
    per_identity_cams: Dict[str, Dict[str, int]] = defaultdict(dict)  # id -> cam -> first index
    frames_per_identity: Counter = Counter()
    cams_seen: List[str] = []
    idx = 0
    labelled_frames = 0

    for _seq_id, camera_id, _frame_id, person_ids in adapter.ground_truth_index():
        idx += 1
        if camera_id not in cams_seen:
            cams_seen.append(camera_id)
        if person_ids:
            labelled_frames += 1
        for pid in person_ids:
            frames_per_identity[pid] += 1
            st.n_person_frames += 1
            if camera_id not in per_identity_cams[pid]:
                per_identity_cams[pid][camera_id] = idx

    st.n_frames = labelled_frames
    st.cameras = sorted(cams_seen)
    st.n_cameras = len(cams_seen)
    st.identities = sorted(frames_per_identity)
    st.n_identities = len(st.identities)
    st.frames_per_identity = dict(sorted(frames_per_identity.items()))

    per_cam_ids: Dict[str, set] = defaultdict(set)
    for pid, cams in per_identity_cams.items():
        for c in cams:
            per_cam_ids[c].add(pid)
    st.identities_per_camera = {c: len(v) for c, v in sorted(per_cam_ids.items())}

    for pid, cams in sorted(per_identity_cams.items()):
        if len(cams) < 2:
            continue
        st.multi_camera_identities.append(pid)
        ordered = sorted(cams, key=lambda c: cams[c])
        for a, b in zip(ordered, ordered[1:]):
            st.n_cross_camera_transitions += 1
            st.transitions_detail.append({"identity": pid, "from_cam": a, "to_cam": b})
    return st


def check(st: DatasetStats) -> Dict:
    """Evaluate every gate. Returns a report; `passed` is the go/no-go."""
    gates = [
        {"name": "identities", "actual": st.n_identities,
         "required": MIN_IDENTITIES,
         "why": "purity/fragmentation are coarse and outlier-dominated below this"},
        {"name": "labelled frames", "actual": st.n_frames,
         "required": MIN_FRAMES,
         "why": "95% Wilson intervals stay wider than the effects being compared below this"},
        {"name": "cross-camera transitions", "actual": st.n_cross_camera_transitions,
         "required": MIN_CROSS_CAMERA_TRANSITIONS,
         "why": "a re-association rate cannot be estimated from a few hand-overs"},
    ]
    for g in gates:
        g["passed"] = g["actual"] >= g["required"]
        g["shortfall"] = max(0, g["required"] - g["actual"])
    return {"passed": all(g["passed"] for g in gates), "gates": gates,
            "stats": st.to_dict()}


def format_report(report: Dict, adapter_name: str, root: str) -> str:
    st = report["stats"]
    L = []
    L.append("=" * 74)
    L.append("DATASET VALIDATION REPORT")
    L.append("=" * 74)
    L.append(f"adapter     : {adapter_name}")
    L.append(f"data root   : {root}")
    L.append("")
    if report.get("load_error"):
        L.append("DATASET COULD NOT BE LOADED")
        for ln in report["load_error"].splitlines():
            L.append(f"  {ln}")
        L.append("")
        L.append("RESULT: FAIL — ablation BLOCKED (no dataset).")
        L.append("=" * 74)
        return "\n".join(L)
    L.append("CORPUS")
    L.append(f"  identities              : {st['n_identities']}")
    L.append(f"  labelled frames         : {st['n_frames']}")
    L.append(f"  ground-truth person-frames: {st['n_person_frames']}")
    L.append(f"  cameras                 : {st['n_cameras']}  {st['cameras']}")
    L.append(f"  cross-camera transitions: {st['n_cross_camera_transitions']}")
    L.append(f"  multi-camera identities : {len(st['multi_camera_identities'])} "
             f"{st['multi_camera_identities'][:10]}")
    if st["identities_per_camera"]:
        L.append(f"  identities per camera   : {st['identities_per_camera']}")
    if st["frames_per_identity"]:
        vals = list(st["frames_per_identity"].values())
        L.append(f"  frames/identity         : min={min(vals)} max={max(vals)} "
                 f"mean={sum(vals)/len(vals):.1f}")
    L.append("")
    L.append("GATES")
    for g in report["gates"]:
        mark = "PASS" if g["passed"] else "FAIL"
        line = f"  [{mark}] {g['name']:<26} {g['actual']:>6} / {g['required']:<6}"
        if not g["passed"]:
            line += f"  SHORT BY {g['shortfall']}"
        L.append(line)
        if not g["passed"]:
            L.append(f"         reason: {g['why']}")
    L.append("")
    if report["passed"]:
        L.append("RESULT: PASS — dataset meets minimum adequacy for the ablation.")
    else:
        L.append("RESULT: FAIL — ablation BLOCKED. Missing:")
        for g in report["gates"]:
            if not g["passed"]:
                L.append(f"  - {g['shortfall']} more {g['name']} "
                         f"(have {g['actual']}, need {g['required']})")
        L.append("")
        L.append("  Add data with:  python eval/label_tool.py --video <file> \\")
        L.append("                      --camera-id <cam> --out <data-root>")
        L.append("  Or use ChokePoint: python eval/run_sweep.py --adapter chokepoint \\")
        L.append("                      --data-root <chokepoint-root>")
        L.append("")
        L.append("  Override ONLY for pipeline smoke-testing, never for reported")
        L.append("  results:  --allow-underpowered")
    L.append("=" * 74)
    return "\n".join(L)


def build_adapter(name: str, data_root: str):
    if name == "folder":
        from eval.adapters.folder import FolderAdapter
        return FolderAdapter(data_root)
    from eval.adapters.chokepoint import ChokePointAdapter
    return ChokePointAdapter(data_root)


def validate(adapter_name: str, data_root: str) -> Dict:
    """
    Returns a report dict. A dataset that cannot be loaded at all fails the
    gate with `load_error` set — never a traceback, and never a synthesised
    stand-in corpus.
    """
    try:
        adapter = build_adapter(adapter_name, data_root)
    except FileNotFoundError as exc:
        empty = DatasetStats()
        rep = check(empty)
        rep["passed"] = False
        rep["load_error"] = str(exc)
        return rep
    st = collect_stats(adapter)
    return check(st)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=["folder", "chokepoint"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    report = validate(args.adapter, args.data_root)
    print(format_report(report, args.adapter, str(Path(args.data_root).resolve())))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
