"""
tests/test_adapters_and_gate.py
────────────────────────────────
Adapters and the gate arithmetic that decides whether a run is reportable.

The ChokePoint adapter had 0% coverage while being the source of every number
in the results table; its only check was a standalone script pytest never
collected. These tests are hermetic — a synthetic ChokePoint tree is built in
tmp_path, so the 1.1 GB corpus is not required.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from eval.adapters.chokepoint import ChokePointAdapter
from eval.adapters.folder import FolderAdapter
from eval.validate_dataset import (MIN_CROSS_CAMERA_TRANSITIONS, MIN_FRAMES,
                                   MIN_IDENTITIES, DatasetStats, check,
                                   collect_stats, validate)

# ChokePoint's published ground-truth layout.
XML = """<?xml version="1.0" encoding="UTF-8"?>
<dataset name="{cam}">
  <frames>
    <frame number="00000001">
      <person id="{p1}"><leftEye x="100" y="120"/><rightEye x="140" y="122"/></person>
    </frame>
    <frame number="00000002">
      <person id="{p1}"><leftEye x="102" y="121"/><rightEye x="142" y="123"/></person>
      <person id="{p2}"><leftEye x="300" y="130"/><rightEye x="332" y="131"/></person>
    </frame>
    <frame number="00000003"></frame>
  </frames>
</dataset>
"""


def build_chokepoint(root: Path, cams=("P1E_S1_C1", "P1E_S1_C2")) -> Path:
    (root / "groundtruth").mkdir(parents=True)
    for cam in cams:
        seq = cam.rsplit("_C", 1)[0]          # P1E_S1_C1 -> P1E_S1
        d = root / seq / cam
        d.mkdir(parents=True, exist_ok=True)
        for i in (1, 2, 3):
            cv2.imwrite(str(d / f"{i:08d}.jpg"),
                        np.full((60, 80, 3), 40, dtype=np.uint8))
        (root / "groundtruth" / f"{cam}.xml").write_text(
            XML.format(cam=cam, p1="0001", p2="0007"))
    return root


# ═══════════════════════════════════════════════════════════════════════════
# 1. ChokePoint adapter
# ═══════════════════════════════════════════════════════════════════════════

class TestChokePointAdapter:

    def test_discovers_sequences(self, tmp_path):
        ad = ChokePointAdapter(build_chokepoint(tmp_path))
        seqs = sorted(s.seq_id for s in ad.sequences())
        assert seqs == ["P1E_S1_C1", "P1E_S1_C2"]
        assert all(s.n_frames == 3 for s in ad.sequences())

    def test_ground_truth_index_labels(self, tmp_path):
        ad = ChokePointAdapter(build_chokepoint(tmp_path, cams=("P1E_S1_C1",)))
        idx = list(ad.ground_truth_index())
        assert [sorted(p) for _s, _c, _f, p in idx] == [["0001"], ["0001", "0007"], []]

    def test_eye_points_become_a_face_box(self, tmp_path):
        """IOD 40 px x scale 2.2 -> ~88 px wide, centred on the eye midpoint."""
        ad = ChokePointAdapter(build_chokepoint(tmp_path, cams=("P1E_S1_C1",)))
        fb = next(iter(ad.frames("P1E_S1_C1"))).ground_truth[0].face_bbox
        assert fb is not None
        assert fb[2] == pytest.approx(int(40 * 2.2), abs=2)
        assert fb[0] + fb[2] / 2 == pytest.approx(120, abs=2)

    def test_index_order_matches_frames_order(self, tmp_path):
        """The gate is only meaningful if the index walks the frames the runner
        will score, in the same order."""
        ad = ChokePointAdapter(build_chokepoint(tmp_path, cams=("P1E_S1_C1",)))
        idx = [(f, sorted(p)) for _s, _c, f, p in ad.ground_truth_index()]
        frames = [(fr.frame_id, sorted(g.person_id for g in fr.ground_truth))
                  for fr in ad.frames("P1E_S1_C1")]
        assert idx == frames

    def test_cap_truncates_per_camera(self, tmp_path):
        ad = ChokePointAdapter(build_chokepoint(tmp_path))
        assert len(list(ad.ground_truth_index())) == 6        # 2 cams x 3
        assert len(list(ad.ground_truth_index(2))) == 4       # 2 cams x 2
        assert len(list(ad.ground_truth_index(1))) == 2

    def test_cap_can_remove_an_identity(self, tmp_path):
        """0007 appears only in frame 2; a cap of 1 drops that person entirely
        — the mechanism that removed 11 ChokePoint subjects."""
        ad = ChokePointAdapter(build_chokepoint(tmp_path, cams=("P1E_S1_C1",)))
        full = {p for _s, _c, _f, ps in ad.ground_truth_index() for p in ps}
        capped = {p for _s, _c, _f, ps in ad.ground_truth_index(1) for p in ps}
        assert full == {"0001", "0007"}
        assert capped == {"0001"}

    def test_missing_dataset_raises_with_instructions(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="arma.sourceforge.net"):
            ChokePointAdapter(tmp_path / "nope")

    def test_missing_ground_truth_raises_rather_than_yielding_empty(self, tmp_path):
        root = build_chokepoint(tmp_path, cams=("P1E_S1_C1",))
        (root / "groundtruth" / "P1E_S1_C1.xml").unlink()
        with pytest.raises(FileNotFoundError):
            list(ChokePointAdapter(root).frames("P1E_S1_C1"))

    def test_never_synthesises_data(self, tmp_path):
        """A fabricated frame would silently invalidate every metric."""
        (tmp_path / "groundtruth").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="no ChokePoint sequence"):
            ChokePointAdapter(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Folder adapter
# ═══════════════════════════════════════════════════════════════════════════

def build_folder(root: Path, people=2, cams=1, frames=3) -> Path:
    for c in range(cams):
        for p in range(people):
            d = root / f"cam{c}" / f"p{p}"
            d.mkdir(parents=True)
            for f in range(frames):
                cv2.imwrite(str(d / f"{f:03d}.jpg"),
                            np.full((20, 16, 3), 90, dtype=np.uint8))
    return root


class TestFolderAdapter:

    def test_labels_come_from_directory_names(self, tmp_path):
        ad = FolderAdapter(build_folder(tmp_path, people=2, frames=2))
        got = sorted((f, p[0]) for _s, _c, f, p in ad.ground_truth_index())
        assert got == [("cam0/p0/000.jpg", "p0"), ("cam0/p0/001.jpg", "p0"),
                       ("cam0/p1/000.jpg", "p1"), ("cam0/p1/001.jpg", "p1")]

    def test_visits_order_is_contiguous_per_person(self, tmp_path):
        ad = FolderAdapter(build_folder(tmp_path, people=2, frames=2), order="visits")
        ids = [p[0] for _s, _c, _f, p in ad.ground_truth_index()]
        assert ids == ["p0", "p0", "p1", "p1"]

    def test_interleave_order_alternates(self, tmp_path):
        ad = FolderAdapter(build_folder(tmp_path, people=2, frames=2), order="interleave")
        ids = [p[0] for _s, _c, _f, p in ad.ground_truth_index()]
        assert ids == ["p0", "p1", "p0", "p1"]

    def test_rejects_bad_order(self, tmp_path):
        with pytest.raises(ValueError, match="visits"):
            FolderAdapter(build_folder(tmp_path), order="sideways")

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dataset root not found"):
            FolderAdapter(tmp_path / "absent")

    def test_empty_root_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no camera subdirectories"):
            FolderAdapter(tmp_path / "empty")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Gate arithmetic
# ═══════════════════════════════════════════════════════════════════════════

class TestGateArithmetic:

    def _stats(self, ids, frames, trans):
        st = DatasetStats()
        st.n_identities, st.n_frames = ids, frames
        st.n_cross_camera_transitions = trans
        return st

    def test_thresholds_are_the_documented_values(self):
        assert (MIN_IDENTITIES, MIN_FRAMES, MIN_CROSS_CAMERA_TRANSITIONS) == (15, 1000, 10)

    def test_exactly_at_threshold_passes(self):
        """>= not >. A corpus sitting exactly on the minimum is adequate."""
        r = check(self._stats(15, 1000, 10))
        assert r["passed"] is True
        assert all(g["shortfall"] == 0 for g in r["gates"])

    def test_one_below_any_threshold_fails(self):
        for ids, fr, tr, name in [(14, 1000, 10, "identities"),
                                  (15, 999, 10, "labelled frames"),
                                  (15, 1000, 9, "cross-camera transitions")]:
            r = check(self._stats(ids, fr, tr))
            assert r["passed"] is False, name
            g = next(x for x in r["gates"] if x["name"] == name)
            assert g["passed"] is False and g["shortfall"] == 1

    def test_shortfall_arithmetic(self):
        r = check(self._stats(3, 250, 2))
        by = {g["name"]: g["shortfall"] for g in r["gates"]}
        assert by == {"identities": 12, "labelled frames": 750,
                      "cross-camera transitions": 8}

    def test_passing_requires_all_three(self):
        r = check(self._stats(100, 100000, 9))
        assert r["passed"] is False, "one failing gate must block the whole run"

    def test_every_gate_carries_a_reason(self):
        for g in check(self._stats(1, 1, 1))["gates"]:
            assert g["why"] and len(g["why"]) > 20

    def test_empty_corpus_fails_all_three(self):
        r = check(DatasetStats())
        assert r["passed"] is False
        assert all(not g["passed"] for g in r["gates"])


class TestTransitionCounting:

    def test_person_on_three_cameras_yields_two_transitions(self, tmp_path):
        build_folder(tmp_path, people=1, cams=3, frames=2)
        st = collect_stats(FolderAdapter(tmp_path))
        assert st.n_cameras == 3
        assert st.n_cross_camera_transitions == 2

    def test_single_camera_person_yields_none(self, tmp_path):
        build_folder(tmp_path, people=4, cams=1, frames=2)
        st = collect_stats(FolderAdapter(tmp_path))
        assert st.n_cross_camera_transitions == 0
        assert st.multi_camera_identities == []

    def test_totals_scale_with_people_and_cameras(self, tmp_path):
        build_folder(tmp_path, people=5, cams=3, frames=2)
        st = collect_stats(FolderAdapter(tmp_path))
        assert st.n_identities == 5
        assert st.n_cross_camera_transitions == 5 * 2
        assert st.n_frames == 5 * 3 * 2


class TestValidateLoadFailure:

    def test_unloadable_dataset_fails_the_gate_without_a_traceback(self, tmp_path):
        rep = validate("folder", str(tmp_path / "does-not-exist"))
        assert rep["passed"] is False
        assert "load_error" in rep
        assert rep["stats"]["n_identities"] == 0
