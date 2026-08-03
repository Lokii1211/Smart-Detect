"""
eval/adapters/chokepoint.py
────────────────────────────
Adapter for the ChokePoint dataset (Wong et al., CVPRW 2011) —
person identification under real-world surveillance conditions at portals,
with 3 cameras per portal and per-subject ground-truth identity labels.

    http://arma.sourceforge.net/chokepoint/

Expected on-disk layout (the dataset's own native layout after extraction):

    <root>/
      P1E_S1/                     # portal 1, entering, sequence 1
        P1E_S1_C1/                # camera 1
          00000001.jpg
          00000002.jpg
          ...
        P1E_S1_C2/
        P1E_S1_C3/
      P1E_S2/
      ...
      groundtruth/                # XML ground truth shipped with the dataset
        P1E_S1_C1.xml
        ...

Ground truth: ChokePoint ships per-frame XML giving each visible subject's
id plus eye coordinates. This adapter reads the XML and converts the eye
points into a face bbox (inter-ocular distance scaled), which is what
SmartDetect's face-quality gate reasons about. Person bboxes are not
provided by the dataset and are left as None — the runner falls back to
matching detections against ground truth by frame-level identity when no
person box exists (see eval/run_eval.py for how unlabeled-box frames are
scored).

If the dataset is not present, this adapter raises with placement
instructions rather than degrading to synthetic data.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2

from .base import DatasetAdapter, Frame, GTPerson, SequenceMeta

_SEQ_DIR_RE = re.compile(r"^P\d[EL]_S\d+$")          # e.g. P1E_S1
_CAM_DIR_RE = re.compile(r"^P\d[EL]_S\d+_C\d+$")     # e.g. P1E_S1_C1

_DOWNLOAD_HINT = (
    "ChokePoint is not bundled with SmartDetect (it is ~4 GB and has its own\n"
    "license terms — you must obtain it yourself).\n"
    "  1. Download from http://arma.sourceforge.net/chokepoint/\n"
    "  2. Extract so that the layout is:\n"
    "        <data-root>/P1E_S1/P1E_S1_C1/00000001.jpg\n"
    "        <data-root>/groundtruth/P1E_S1_C1.xml\n"
    "  3. Re-run with --data-root <data-root>\n"
    "This adapter will not synthesise stand-in data: fabricated frames would\n"
    "silently invalidate every metric in the results table."
)


class ChokePointAdapter(DatasetAdapter):
    name = "chokepoint"

    def __init__(self, root: str | Path, face_box_scale: float = 2.2):
        """
        face_box_scale: face bbox width as a multiple of inter-ocular
        distance. 2.2 approximates the crop convention used by common face
        detectors; only affects the ground-truth face box, never SmartDetect.
        """
        self.root = Path(root).expanduser().resolve()
        self.face_box_scale = face_box_scale
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"ChokePointAdapter: data root not found: {self.root}\n\n{_DOWNLOAD_HINT}"
            )

        self._cam_dirs: List[Path] = []
        for seq_dir in sorted(self.root.iterdir()):
            if seq_dir.is_dir() and _SEQ_DIR_RE.match(seq_dir.name):
                for cam_dir in sorted(seq_dir.iterdir()):
                    if cam_dir.is_dir() and _CAM_DIR_RE.match(cam_dir.name):
                        self._cam_dirs.append(cam_dir)

        if not self._cam_dirs:
            raise FileNotFoundError(
                f"ChokePointAdapter: no ChokePoint sequence directories under {self.root}\n"
                f"(looked for e.g. {self.root}/P1E_S1/P1E_S1_C1/)\n\n{_DOWNLOAD_HINT}"
            )

        self._gt_dir = self.root / "groundtruth"
        if not self._gt_dir.is_dir():
            raise FileNotFoundError(
                f"ChokePointAdapter: ground-truth directory not found: {self._gt_dir}\n"
                f"Identity metrics are meaningless without labels.\n\n{_DOWNLOAD_HINT}"
            )

    # ── ground truth ─────────────────────────────────────────────────────────

    def _load_gt(self, cam_name: str) -> Dict[str, List[GTPerson]]:
        """frame-number-string → [GTPerson]. ChokePoint XML: <frame number=..>
        <person id=..><leftEye x= y=/><rightEye x= y=/></person></frame>"""
        xml_path = self._gt_dir / f"{cam_name}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"ChokePointAdapter: missing ground truth {xml_path}\n\n{_DOWNLOAD_HINT}"
            )
        out: Dict[str, List[GTPerson]] = {}
        root = ET.parse(xml_path).getroot()
        for frame_el in root.iter("frame"):
            fnum = frame_el.get("number")
            if fnum is None:
                continue
            people: List[GTPerson] = []
            for person_el in frame_el.iter("person"):
                pid = person_el.get("id")
                if not pid:
                    continue
                face_bbox = None
                le, re_ = person_el.find("leftEye"), person_el.find("rightEye")
                if le is not None and re_ is not None:
                    try:
                        lx, ly = float(le.get("x")), float(le.get("y"))
                        rx, ry = float(re_.get("x")), float(re_.get("y"))
                        iod = max(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5, 1.0)
                        w = h = iod * self.face_box_scale
                        cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
                        face_bbox = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]
                    except (TypeError, ValueError):
                        face_bbox = None
                people.append(GTPerson(person_id=pid, bbox=None, face_bbox=face_bbox))
            # zero-padded and bare keys both, so frame lookup is robust
            out[fnum] = people
            out[fnum.lstrip("0") or "0"] = people
        return out

    # ── DatasetAdapter interface ────────────────────────────────────────────

    def sequences(self) -> Iterator[SequenceMeta]:
        for cam_dir in self._cam_dirs:
            n = len([f for f in cam_dir.iterdir() if f.suffix.lower() == ".jpg"])
            yield SequenceMeta(seq_id=cam_dir.name, camera_id=cam_dir.name,
                               n_frames=n, note=cam_dir.parent.name)

    def ground_truth_index(self) -> Iterator[tuple]:
        """Label-only pass — parses XML, decodes no JPEGs. ChokePoint has
        ~64k frames, so the pixel-decoding fallback would take minutes."""
        for cam_dir in self._cam_dirs:
            gt = self._load_gt(cam_dir.name)
            for path in sorted(f for f in cam_dir.iterdir() if f.suffix.lower() == ".jpg"):
                stem = path.stem
                people = gt.get(stem) or gt.get(stem.lstrip("0") or "0") or []
                yield (cam_dir.name, cam_dir.name, f"{cam_dir.name}/{path.name}",
                       [p.person_id for p in people])

    def frames(self, seq_id: str, fps: float = 30.0) -> Iterator[Frame]:
        cam_dir = next((c for c in self._cam_dirs if c.name == seq_id), None)
        if cam_dir is None:
            raise FileNotFoundError(f"ChokePointAdapter: unknown sequence '{seq_id}'")
        gt = self._load_gt(seq_id)

        jpgs = sorted(f for f in cam_dir.iterdir() if f.suffix.lower() == ".jpg")
        for i, path in enumerate(jpgs):
            img = cv2.imread(str(path))
            if img is None:
                continue
            stem = path.stem
            people = gt.get(stem) or gt.get(stem.lstrip("0") or "0") or []
            yield Frame(
                image=img,
                timestamp=i / fps,
                camera_id=seq_id,
                ground_truth=people,
                frame_id=f"{seq_id}/{path.name}",
            )
