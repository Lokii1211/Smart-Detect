"""
eval/adapters/folder.py
────────────────────────
Fallback adapter for custom recorded footage laid out as:

    <root>/<camera_id>/<person_id>/*.jpg

Each leaf directory is one ground-truth identity as seen by one camera.
A sequence is one CAMERA.

FRAME ORDERING (`order=` ctor arg) — this materially affects results:

  "visits" (DEFAULT): each person's frames are emitted as one contiguous
      run, people in folder-name order. Models the realistic case of
      separate visits to a camera: A walks through, leaves, then B
      arrives. The hand-over from A's last frame to B's first is exactly
      the ID-switch scenario the guard exists for.

  "interleave": round-robin across people, frame by frame. AVOID for
      person-centric crops: two identities alternating at the same screen
      position every frame is not physically realisable, and ByteTrack
      (correctly) reads it as one continuous track, so identity caching
      pins both people to one code and every metric collapses. Retained
      only because it is the right choice for datasets where the crops
      genuinely are simultaneous views.

Ground truth is folder-derived: every frame under <camera>/<person>/ is
labelled with exactly that person_id and no bbox (the whole frame is the
person). That is the correct labelling for cropped/person-centric footage.
For full-scene footage with multiple people per frame, this adapter is the
wrong tool — use a dataset with per-frame boxes (see chokepoint.py).
"""
from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterator, List

import cv2

from .base import DatasetAdapter, Frame, GTPerson, SequenceMeta

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class FolderAdapter(DatasetAdapter):
    name = "folder"

    def __init__(self, root: str | Path, fps: float = 10.0, order: str = "visits"):
        if order not in ("visits", "interleave"):
            raise ValueError(f"order must be 'visits' or 'interleave', got {order!r}")
        self.root = Path(root).expanduser().resolve()
        self.fps = fps
        self.order = order
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"FolderAdapter: dataset root not found: {self.root}\n"
                f"Expected layout:\n"
                f"    {self.root}/<camera_id>/<person_id>/*.jpg\n"
                f"Create it, or pass a different --data-root."
            )
        self._cameras = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not self._cameras:
            raise FileNotFoundError(
                f"FolderAdapter: no camera subdirectories under {self.root}.\n"
                f"Expected at least one: {self.root}/<camera_id>/<person_id>/*.jpg"
            )

    def _person_dirs(self, cam_dir: Path) -> List[Path]:
        return sorted(d for d in cam_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("."))

    def sequences(self) -> Iterator[SequenceMeta]:
        for cam_dir in self._cameras:
            n = sum(len([f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS])
                    for p in self._person_dirs(cam_dir))
            yield SequenceMeta(seq_id=cam_dir.name, camera_id=cam_dir.name,
                               n_frames=n, note=f"{len(self._person_dirs(cam_dir))} identities")

    def ground_truth_index(self) -> Iterator[tuple]:
        """Label-only pass — reads directory structure, decodes no images."""
        for cam_dir in self._cameras:
            for person_dir in self._person_dirs(cam_dir):
                for f in sorted(person_dir.iterdir()):
                    if f.suffix.lower() in _IMAGE_EXTS:
                        yield (cam_dir.name, cam_dir.name,
                               f"{cam_dir.name}/{person_dir.name}/{f.name}",
                               [person_dir.name])

    def frames(self, seq_id: str) -> Iterator[Frame]:
        cam_dir = self.root / seq_id
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"FolderAdapter: no such camera sequence: {cam_dir}")

        # (filename, person_id, path) — ordering per self.order, see class docstring
        per_person: List[List[tuple]] = []
        for person_dir in self._person_dirs(cam_dir):
            files = [(f.name, person_dir.name, f) for f in sorted(person_dir.iterdir())
                     if f.suffix.lower() in _IMAGE_EXTS]
            if files:
                per_person.append(files)

        items: List[tuple] = []
        if self.order == "visits":
            for files in per_person:          # contiguous run per identity
                items.extend(files)
        else:                                  # round-robin
            for row in zip_longest(*per_person):
                items.extend(x for x in row if x is not None)

        for i, (fname, person_id, path) in enumerate(items):
            img = cv2.imread(str(path))
            if img is None:
                continue
            yield Frame(
                image=img,
                timestamp=i / self.fps,
                camera_id=seq_id,
                ground_truth=[GTPerson(person_id=person_id)],
                frame_id=f"{seq_id}/{person_id}/{fname}",
            )
