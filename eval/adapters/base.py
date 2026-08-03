"""
eval/adapters/base.py
──────────────────────
Abstract dataset adapter for the offline identity-evaluation harness.

An adapter's only job is to yield frames with ground-truth identity labels.
It must not know anything about SmartDetect's identity logic — the runner
(eval/run_eval.py) drives the real production code path and compares its
output against whatever ground truth the adapter supplies.

Coordinate convention: bboxes are [x, y, w, h] in NATIVE pixel coordinates
of the image as yielded (no normalisation, no resizing). This matches what
recognition/object_detector.py and cameras/live_stream.py use internally.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np


@dataclass
class GTPerson:
    """One ground-truth person present in a frame."""
    person_id: str                              # dataset-global identity label
    bbox:      Optional[List[int]] = None       # [x, y, w, h], native px
    face_bbox: Optional[List[int]] = None       # [x, y, w, h], native px


@dataclass
class Frame:
    """One frame plus its ground truth."""
    image:        np.ndarray                    # BGR, native resolution
    timestamp:    float                         # seconds; monotonic within a sequence
    camera_id:    str
    ground_truth: List[GTPerson] = field(default_factory=list)
    frame_id:     str = ""                      # for traceability in reports


@dataclass
class SequenceMeta:
    """One contiguous recording from one camera."""
    seq_id:    str
    camera_id: str
    n_frames:  Optional[int] = None
    note:      str = ""


class DatasetAdapter(ABC):
    """
    Base class for evaluation datasets.

    Implementations MUST fail loudly (FileNotFoundError with an actionable
    message) when their data is not present on disk. Never synthesise or
    fabricate frames or labels — a fabricated dataset silently invalidates
    every metric downstream.
    """

    #: short name used in report filenames / tables
    name: str = "base"

    @abstractmethod
    def sequences(self) -> Iterator[SequenceMeta]:
        """Yield each sequence (one camera's contiguous recording)."""
        raise NotImplementedError

    @abstractmethod
    def frames(self, seq_id: str) -> Iterator[Frame]:
        """Yield frames of one sequence in temporal order."""
        raise NotImplementedError

    def ground_truth_index(self) -> Iterator[tuple]:
        """
        Yield (seq_id, camera_id, frame_id, [person_id, ...]) for every
        labelled frame, WITHOUT decoding pixels.

        Dataset validation needs identity/frame/transition counts over the
        whole corpus; decoding ~64k JPEGs to count labels would take minutes.
        Adapters should override this with a label-only path. The default
        implementation falls back to frames() and is correct but slow.
        """
        for seq in self.sequences():
            for fr in self.frames(seq.seq_id):
                yield (seq.seq_id, fr.camera_id, fr.frame_id,
                       [g.person_id for g in fr.ground_truth])

    def describe(self) -> str:
        seqs = list(self.sequences())
        cams = sorted({s.camera_id for s in seqs})
        return (f"{self.name}: {len(seqs)} sequence(s) across "
                f"{len(cams)} camera(s) {cams}")
