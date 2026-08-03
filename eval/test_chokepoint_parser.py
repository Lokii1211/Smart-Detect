"""
eval/test_chokepoint_parser.py
───────────────────────────────
Parser unit test for the ChokePoint adapter.

This builds a tiny XML fixture in the documented ChokePoint ground-truth
format and asserts the adapter reads identities and eye coordinates out of
it correctly.

IMPORTANT — this is a PARSER test, not a dataset. It produces no metrics and
must never be used as evaluation data. Its whole purpose is to verify the
adapter's XML handling on a machine that does not have the 4 GB corpus, so
that when the real data does land the failure modes are already excluded.

Run:  python eval/test_chokepoint_parser.py
"""
from __future__ import annotations

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from eval.adapters.chokepoint import ChokePointAdapter

# ChokePoint's published ground-truth layout: one XML per camera sequence,
# <frame number=..> containing <person id=..> with left/right eye points.
FIXTURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<dataset name="P1E_S1_C1">
  <frames>
    <frame number="00000001">
      <person id="0001">
        <leftEye x="100" y="120"/>
        <rightEye x="140" y="122"/>
      </person>
    </frame>
    <frame number="00000002">
      <person id="0001">
        <leftEye x="102" y="121"/>
        <rightEye x="142" y="123"/>
      </person>
      <person id="0007">
        <leftEye x="300" y="130"/>
        <rightEye x="332" y="131"/>
      </person>
    </frame>
    <frame number="00000003">
    </frame>
  </frames>
</dataset>
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cam = root / "P1E_S1" / "P1E_S1_C1"
        cam.mkdir(parents=True)
        (root / "groundtruth").mkdir()
        (root / "groundtruth" / "P1E_S1_C1.xml").write_text(FIXTURE_XML)
        for i in (1, 2, 3):
            img = np.full((480, 640, 3), 40, dtype=np.uint8)
            cv2.imwrite(str(cam / f"{i:08d}.jpg"), img)

        ad = ChokePointAdapter(root)

        seqs = list(ad.sequences())
        assert len(seqs) == 1, f"expected 1 sequence, got {len(seqs)}"
        assert seqs[0].seq_id == "P1E_S1_C1", seqs[0].seq_id
        assert seqs[0].n_frames == 3, seqs[0].n_frames
        print(f"sequences        : OK ({seqs[0].seq_id}, {seqs[0].n_frames} frames)")

        idx = list(ad.ground_truth_index())
        assert len(idx) == 3, f"expected 3 indexed frames, got {len(idx)}"
        ids_per_frame = [sorted(p) for _, _, _, p in idx]
        assert ids_per_frame == [["0001"], ["0001", "0007"], []], ids_per_frame
        print(f"ground_truth_index: OK {ids_per_frame}")

        frames = list(ad.frames("P1E_S1_C1"))
        assert len(frames) == 3, len(frames)
        assert [g.person_id for g in frames[0].ground_truth] == ["0001"]
        assert sorted(g.person_id for g in frames[1].ground_truth) == ["0001", "0007"]
        assert frames[2].ground_truth == []
        print("frames()         : OK (identities match per frame)")

        # Eye points -> face bbox: inter-ocular distance 40px, scale 2.2
        fb = frames[0].ground_truth[0].face_bbox
        assert fb is not None, "face_bbox not derived from eye points"
        exp_w = int(40 * 2.2)
        assert abs(fb[2] - exp_w) <= 2, f"face width {fb[2]} != ~{exp_w}"
        cx = fb[0] + fb[2] / 2
        assert abs(cx - 120) <= 2, f"face centre x {cx} != ~120"
        print(f"face bbox        : OK {fb} (IOD 40px -> w~{exp_w}, centred on eyes)")

        # Images decode at native resolution
        assert frames[0].image.shape == (480, 640, 3), frames[0].image.shape
        print("image loading    : OK (480x640x3, native)")

        # Absent ground truth must raise, not silently yield empty labels
        (root / "groundtruth" / "P1E_S1_C1.xml").unlink()
        try:
            list(ChokePointAdapter(root).frames("P1E_S1_C1"))
        except FileNotFoundError:
            print("missing GT       : OK (raises rather than yielding empty labels)")
        else:
            raise AssertionError("missing ground truth did NOT raise")

    print("\nALL PARSER TESTS PASSED")
    print("NOTE: this validates XML/geometry handling only. It is not a dataset "
          "and yields no metrics.")


if __name__ == "__main__":
    main()
