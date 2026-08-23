"""
tests/test_dataset_gate.py
───────────────────────────
The adequacy gate must describe THE FRAMES THAT GET SCORED.

Regression origin: `run_sweep.py` validated the full corpus and then ran the
sweep with `--max-frames`, so a 25-identity ChokePoint corpus produced a
results table over 14 identities while the report printed PASS — a table below
the very threshold the gate exists to enforce. These tests pin the fix:
truncation is applied inside `ground_truth_index()`, in each adapter's own
frame order, and `validate()` takes the same cap the runner will use.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

from eval.adapters.folder import FolderAdapter
from eval.validate_dataset import (MIN_CROSS_CAMERA_TRANSITIONS, MIN_FRAMES,
                                   MIN_IDENTITIES, check, collect_stats, validate)


def build_corpus(root: Path, n_people: int, n_cams: int, frames_each: int) -> Path:
    """<root>/<cam>/<person>/*.jpg — every person on every camera."""
    for c in range(n_cams):
        for p in range(n_people):
            d = root / f"cam{c}" / f"p{p:03d}"
            d.mkdir(parents=True)
            for f in range(frames_each):
                cv2.imwrite(str(d / f"{f:04d}.jpg"),
                            np.full((32, 24, 3), 40 + p, dtype=np.uint8))
    return root


# ═══════════════════════════════════════════════════════════════════════════
# 1. The cap reaches the statistics
# ═══════════════════════════════════════════════════════════════════════════

class TestTruncationIsVisibleToTheGate:

    def test_uncapped_sees_every_identity(self, tmp_path):
        build_corpus(tmp_path, n_people=20, n_cams=2, frames_each=30)
        st = collect_stats(FolderAdapter(tmp_path))
        assert st.n_identities == 20
        assert st.n_frames == 20 * 2 * 30
        assert st.max_frames == 0

    def test_cap_drops_identities_from_the_stats(self, tmp_path):
        """In 'visits' order a per-sequence cap truncates whole people off the
        tail — exactly what removed 11 ChokePoint subjects."""
        build_corpus(tmp_path, n_people=20, n_cams=2, frames_each=30)
        # 5 people * 30 frames = 150 frames keeps only the first 5 identities
        st = collect_stats(FolderAdapter(tmp_path), max_frames=150)
        assert st.n_identities == 5, "cap must reduce the identity count"
        assert st.n_frames == 300           # 150 per camera, 2 cameras
        assert st.max_frames == 150

    def test_capped_stats_match_what_frames_would_yield(self, tmp_path):
        """The index under a cap must be exactly the frames() prefix — same
        count and same identities. If these diverge the gate is measuring a
        different subset than the runner scores."""
        build_corpus(tmp_path, n_people=8, n_cams=1, frames_each=10)
        ad = FolderAdapter(tmp_path)
        cap = 35
        idx = list(ad.ground_truth_index(cap))
        frames = []
        for i, fr in enumerate(ad.frames("cam0")):
            if i >= cap:
                break
            frames.append((fr.frame_id, [g.person_id for g in fr.ground_truth]))
        assert len(idx) == len(frames) == cap
        assert [(f, p) for _s, _c, f, p in idx] == frames

    @pytest.mark.parametrize("order", ["visits", "interleave"])
    def test_index_and_frames_agree_in_both_orderings(self, tmp_path, order):
        """FolderAdapter.order changes frame ORDER. The index used to always
        group by person regardless, so under 'interleave' a cap selected a
        different subset in the gate than in the runner."""
        build_corpus(tmp_path, n_people=4, n_cams=1, frames_each=6)
        ad = FolderAdapter(tmp_path, order=order)
        cap = 9
        idx = [(f, p) for _s, _c, f, p in ad.ground_truth_index(cap)]
        got = []
        for i, fr in enumerate(ad.frames("cam0")):
            if i >= cap:
                break
            got.append((fr.frame_id, [g.person_id for g in fr.ground_truth]))
        assert idx == got, f"index/frames disagree under order={order!r}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. The gate verdict follows the scored set, not the corpus
# ═══════════════════════════════════════════════════════════════════════════

class TestGateVerdictFollowsScoredSet:

    def test_adequate_corpus_truncated_below_threshold_fails(self, tmp_path):
        """THE REGRESSION. Corpus passes; the scored subset does not."""
        build_corpus(tmp_path, n_people=20, n_cams=3, frames_each=40)
        ad = FolderAdapter(tmp_path)

        full = check(collect_stats(ad))
        assert full["passed"] is True, "corpus itself must be adequate"
        assert full["stats"]["n_identities"] == 20

        capped = check(collect_stats(ad, max_frames=400))   # 10 people
        assert capped["stats"]["n_identities"] == 10
        assert capped["passed"] is False, (
            "a scored set with 10 identities must FAIL even though the "
            "corpus on disk has 20")
        ident = next(g for g in capped["gates"] if g["name"] == "identities")
        assert ident["passed"] is False
        assert ident["shortfall"] == MIN_IDENTITIES - 10

    def test_validate_end_to_end_honours_the_cap(self, tmp_path):
        build_corpus(tmp_path, n_people=20, n_cams=3, frames_each=40)
        assert validate("folder", str(tmp_path))["passed"] is True
        rep = validate("folder", str(tmp_path), max_frames=400)
        assert rep["passed"] is False
        assert rep["stats"]["max_frames"] == 400

    def test_frames_gate_also_follows_the_cap(self, tmp_path):
        build_corpus(tmp_path, n_people=30, n_cams=1, frames_each=20)
        rep = validate("folder", str(tmp_path), max_frames=100)
        frames_gate = next(g for g in rep["gates"] if g["name"] == "labelled frames")
        assert frames_gate["actual"] == 100
        assert frames_gate["passed"] is False

    def test_transitions_gate_also_follows_the_cap(self, tmp_path):
        """Truncating to one camera's worth removes every cross-camera pair."""
        build_corpus(tmp_path, n_people=20, n_cams=3, frames_each=60)
        full = validate("folder", str(tmp_path))
        assert full["stats"]["n_cross_camera_transitions"] >= MIN_CROSS_CAMERA_TRANSITIONS
        capped = validate("folder", str(tmp_path), max_frames=60)  # 1 person/cam
        tr = next(g for g in capped["gates"] if g["name"] == "cross-camera transitions")
        assert tr["actual"] < MIN_CROSS_CAMERA_TRANSITIONS
        assert tr["passed"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. The sweep actually refuses
# ═══════════════════════════════════════════════════════════════════════════

class TestSweepRefusesTruncatedRun:

    def _run(self, root, out, extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "eval" / "run_sweep.py"),
             "--adapter", "folder", "--data-root", str(root),
             "--results", str(out)] + extra,
            cwd=str(ROOT), capture_output=True, text=True, timeout=300)

    def test_sweep_exits_nonzero_when_scored_set_fails(self, tmp_path):
        """No metrics.json, no summary.md — the table must not be emitted."""
        corpus = build_corpus(tmp_path / "data", n_people=20, n_cams=3, frames_each=40)
        out = tmp_path / "out"
        r = self._run(corpus, out, ["--max-frames", "400"])

        assert r.returncode != 0, "a truncated-below-threshold run must not succeed"
        assert "ABLATION BLOCKED" in (r.stdout + r.stderr)
        assert "SCORED frame set" in (r.stdout + r.stderr)
        assert not (out / "summary.md").exists(), "no results table may be written"
        assert not (out / "A_pre_hardening" / "metrics.json").exists()

    def test_blocked_message_names_the_cap(self, tmp_path):
        corpus = build_corpus(tmp_path / "data", n_people=20, n_cams=3, frames_each=40)
        out = tmp_path / "out"
        r = self._run(corpus, out, ["--max-frames", "400"])
        assert "--max-frames 400" in (r.stdout + r.stderr), (
            "the refusal must tell the operator the cap caused it")

    def test_validation_json_records_the_capped_stats(self, tmp_path):
        """Even on refusal the evidence of WHAT was judged is persisted."""
        import json
        corpus = build_corpus(tmp_path / "data", n_people=20, n_cams=3, frames_each=40)
        out = tmp_path / "out"
        self._run(corpus, out, ["--max-frames", "400"])
        rep = json.loads((out / "dataset_validation.json").read_text())
        assert rep["passed"] is False
        assert rep["stats"]["max_frames"] == 400
        assert rep["stats"]["n_identities"] == 10
