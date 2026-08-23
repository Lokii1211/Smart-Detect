"""
cameras/tiled_detect.py
────────────────────────
Sliced (tiled) inference for small / distant person detection.

THE PROBLEM
───────────
A detector is fed a fixed input size. `ObjectDetector.set_imgsz_for_resolution`
picks 416/640/960 px from the source resolution, so an 800 px-wide frame is
downscaled ~2x and a 4K frame ~4x before inference. Downscaling is what makes
CPU operation affordable, and it is also what destroys distant pedestrians: a
person 40 px tall in the source is ~20 px at the network input, below the scale
at which the backbone has usable stride-8 features.

THE FIX
───────
Cut the frame into overlapping tiles, run the detector on each tile at (near)
native resolution, translate the boxes back to frame coordinates, and merge.
A person who was 40 px in a downscaled frame is 40 px in a tile — recoverable.

SCOPE — DETECTION ONLY
──────────────────────
Nothing here touches identity. Tiling changes which person BOXES exist; it does
not change how a box acquires an identifier. A recovered person whose face is
12 px will still fail the face-quality gate and stay "Detecting..." — that is
correct behaviour, not a shortfall (see the measurement note in the docstring
of `detect_with_tiling`).

EDGE TILES ARE NOT PADDED
─────────────────────────
Padding a short edge tile to a square puts a hard synthetic border inside the
receptive field, and detectors hallucinate boxes against it. Instead the last
tile on each axis is SHIFTED BACK so it ends exactly at the frame edge. Tiles
then overlap more than the nominal ratio near the edges, which is harmless —
the merge step is idempotent under extra overlap. When a dimension is already
smaller than the tile, one tile spans it whole at its true size.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Box = Sequence[int]          # [x, y, w, h] in frame coordinates


# ─── Geometry ────────────────────────────────────────────────────────────────

def _axis_starts(dim: int, tile: int, stride: int) -> List[int]:
    """
    Tile start offsets along one axis, with the final tile shifted back to the
    edge rather than padded. Never returns a start that would run past `dim`.
    """
    if dim <= tile:
        return [0]                                  # one tile, true size, no pad
    starts = list(range(0, dim - tile + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] + tile < dim:
        starts.append(dim - tile)                   # shift back, do not pad
    return starts


def slice_frame(frame: np.ndarray, tile_size: int = 640,
                overlap_ratio: float = 0.2) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Cut `frame` into overlapping tiles.

    Returns [(tile_view, (x_offset, y_offset)), ...] where adding the offset to
    a tile-local coordinate gives the frame coordinate. Tiles are numpy VIEWS,
    not copies — slicing is free; the detector copies internally if it needs to.

    tile_size      : nominal square tile edge, px
    overlap_ratio  : fraction of the tile shared with its neighbour, [0, 1)

    A tile is never padded: an edge tile is shifted back to end at the frame
    boundary, and a frame dimension smaller than tile_size yields one tile of
    that true dimension.
    """
    if frame is None or frame.size == 0:
        return []
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")

    h, w = frame.shape[:2]
    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))

    out: List[Tuple[np.ndarray, Tuple[int, int]]] = []
    for y0 in _axis_starts(h, tile_size, stride):
        y1 = min(y0 + tile_size, h)
        for x0 in _axis_starts(w, tile_size, stride):
            x1 = min(x0 + tile_size, w)
            out.append((frame[y0:y1, x0:x1], (x0, y0)))
    return out


def translate(dets: List[Dict], x_off: int, y_off: int) -> List[Dict]:
    """Shift tile-local detections into frame coordinates."""
    moved = []
    for d in dets:
        x, y, w, h = d["bbox"]
        e = dict(d)
        e["bbox"] = [x + x_off, y + y_off, w, h]
        moved.append(e)
    return moved


# ─── Merging ─────────────────────────────────────────────────────────────────

def _inter(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return float(ix * iy)


def iou(a: Box, b: Box) -> float:
    i = _inter(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def containment(a: Box, b: Box) -> float:
    """
    Intersection over the SMALLER box ("intersection over smaller", IoS).

    Pure IoU cannot merge a tile-edge fragment into the whole person. A subject
    straddling a seam is seen complete by one tile and as a narrow sliver by
    its neighbour; if the sliver is under half the full box, IoU < 0.5 and NMS
    keeps BOTH — one person, two boxes, which is exactly the artefact tiling is
    supposed to avoid. IoS of a fragment against its parent is ~1.0, so a
    containment test suppresses it. This is why merging uses IoU OR IoS rather
    than IoU alone.
    """
    i = _inter(a, b)
    smaller = min(a[2] * a[3], b[2] * b[3])
    return i / smaller if smaller > 0 else 0.0


def merge_detections(dets: List[Dict], iou_threshold: float = 0.5,
                     containment_threshold: float = 0.70) -> List[Dict]:
    """
    Greedy non-maximum suppression over detections already in frame
    coordinates, highest confidence first.

    A candidate is suppressed by a kept box when EITHER
        IoU >= iou_threshold                  (the usual duplicate case), or
        IoS >= containment_threshold          (a tile-edge fragment of it)
    and both carry the same class label.

    Set containment_threshold >= 1.0 to disable the containment rule and get
    plain IoU NMS.
    """
    if not dets:
        return []
    order = sorted(dets, key=lambda d: -float(d.get("confidence", 0.0)))
    kept: List[Dict] = []
    for cand in order:
        cb, cl = cand["bbox"], cand.get("label")
        drop = False
        for k in kept:
            if k.get("label") != cl:
                continue
            if (iou(cb, k["bbox"]) >= iou_threshold
                    or containment(cb, k["bbox"]) >= containment_threshold):
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept


# ─── Orchestration ───────────────────────────────────────────────────────────

def should_tile(frame_w: int, frame_h: int, cfg) -> bool:
    """
    Tiling a frame the detector already sees at native scale is pure cost, so
    it is skipped below `tiled_min_source_resolution` (compared on the LONGER
    edge). A 640 px webcam frame gains nothing and would pay 2-4x detection.
    """
    if not getattr(cfg, "enable_tiled_detection", False):
        return False
    return max(frame_w, frame_h) >= getattr(cfg, "tiled_min_source_resolution", 1280)


def detect_with_tiling(detector, frame: np.ndarray, cfg) -> List[Dict]:
    """
    Full-frame pass + tiled passes, merged.

    The full-frame pass is retained deliberately. Tiles are good at small
    subjects and bad at large ones: a person taller than a tile cannot be
    wholly contained in any tile, so tiles alone fragment near-camera subjects.
    Running both and merging keeps the full-frame box for the large person and
    adds the tile-only boxes for distant ones.

    Returns detections in frame coordinates. When tiling is disabled or the
    source is below the resolution floor this is exactly `detector.detect()`,
    so the call site does not branch.

    NOTE ON IDENTITY: this returns BOXES. Whether a recovered box earns an
    identifier is decided downstream by the unchanged face-quality gate and
    arbitration. Recovering a 40 px-tall person is expected to increase
    detection coverage and NOT to increase identities minted.
    """
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]

    full = detector.detect(frame)
    if not should_tile(w, h, cfg):
        return full

    tile_size = int(getattr(cfg, "tile_size", 640))
    overlap = float(getattr(cfg, "tile_overlap", 0.2))

    merged: List[Dict] = list(full)
    for tile, (x0, y0) in slice_frame(frame, tile_size, overlap):
        if tile.size == 0:
            continue
        try:
            merged.extend(translate(detector.detect(tile), x0, y0))
        except Exception as exc:                     # noqa: BLE001
            logger.debug("tiled_detect: tile at (%d,%d) failed: %s", x0, y0, exc)

    return merge_detections(
        merged,
        iou_threshold=float(getattr(cfg, "tile_merge_iou", 0.5)),
        containment_threshold=float(getattr(cfg, "tile_merge_containment", 0.70)),
    )
