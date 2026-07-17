"""
tracker/deepsort_tracker.py
────────────────────────────
DeepSORT multi-object tracker wrapper for the SmartDetect Person Tracking System.
Provides stable track IDs across frames for consistent identity assignment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias: bounding box in pixel coordinates [x1, y1, x2, y2]
BBox = Tuple[int, int, int, int]


class CameraTracker:
    """
    Wraps ``deep_sort_realtime`` to provide persistent track IDs per camera.

    Each ``CameraTracker`` instance manages one camera stream. Call
    :meth:`update` on every frame with the YOLO/detector bounding boxes and
    it returns stable ``track_id`` values across frames.

    Example
    -------
    >>> tracker = CameraTracker()
    >>> tracks = tracker.update(frame, detections)
    >>> for t in tracks:
    ...     crop = tracker.get_crop(frame, t["bbox"])
    """

    def __init__(
        self,
        max_age: int = 10,
        n_init: int = 3,
        max_iou_distance: float = 0.6,
        max_cosine_distance: float = 0.3,
        nn_budget: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_age : int
            Maximum frames to keep a track alive without a detection match.
        n_init : int
            Frames of consecutive detections needed before a track is confirmed.
        max_iou_distance : float
            Maximum IoU distance for association (higher = more permissive).
        max_cosine_distance : float
            Maximum cosine distance for re-association from appearance features.
        nn_budget : int | None
            Maximum size of the appearance descriptor gallery per track.
        """
        self._max_age = max_age
        self._n_init = n_init
        self._max_iou_distance = max_iou_distance
        self._max_cosine_distance = max_cosine_distance
        self._nn_budget = nn_budget
        self._tracker = None
        self._load_tracker()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _load_tracker(self) -> None:
        """Initialise the DeepSort tracker instance."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # noqa: PLC0415

            self._tracker = DeepSort(
                max_age=self._max_age,
                n_init=self._n_init,
                max_iou_distance=self._max_iou_distance,
                embedder=None,  # No CNN = IoU-only tracking = ~0ms vs ~300ms
            )
            logger.info("CameraTracker: DeepSORT initialised (IoU-only, no embedder)")
        except ImportError as exc:
            raise RuntimeError(
                "deep_sort_realtime is not installed. "
                "Run: pip install deep-sort-realtime"
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        detections: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """
        Update the tracker with detections from the current frame.

        Parameters
        ----------
        frame : np.ndarray
            Current video frame (BGR, as from OpenCV).
        detections : List[List[float]]
            Each detection is ``[x, y, w, h, confidence]`` where (x, y) is
            the top-left corner of the bounding box.

        Returns
        -------
        List of dicts, one per confirmed active track::

            {
                "track_id": int,
                "bbox": (x1, y1, x2, y2),   # pixel coordinates
                "confidence": float,
            }
        """
        # deep_sort_realtime expects [[x,y,w,h], confidence, "class"]
        # Accept either format:
        #   - flat list: [x, y, w, h, confidence]
        #   - pre-formatted tuple: ([x,y,w,h], confidence, "class")
        ds_input = []
        for d in (detections or []):
            if isinstance(d, (list, np.ndarray)) and len(d) >= 5:
                # Flat format: [x, y, w, h, confidence]
                ds_input.append(([d[0], d[1], d[2], d[3]], d[4], "person"))
            elif isinstance(d, tuple) and len(d) == 3:
                # Already formatted: ([x,y,w,h], conf, "class")
                ds_input.append(d)
            else:
                logger.debug("Skipping invalid detection format: %s", type(d))
                continue

        # Always call update_tracks — even with empty ds_input DeepSORT will
        # predict existing track positions (Kalman filter), keeping them alive.
        # When embedder=None, we must supply embeddings ourselves.
        # Use lightweight bbox-derived embeddings (~0ms vs ~300ms for CNN).
        embeds = []
        for d in ds_input:
            bbox = d[0]  # [x, y, w, h]
            # Simple 128-dim embedding from bbox geometry (position + aspect ratio)
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            ar = bbox[2] / max(bbox[3], 1.0)
            area = bbox[2] * bbox[3]
            seed = np.array([cx, cy, ar, area, bbox[2], bbox[3]], dtype=np.float32)
            # Tile to 128-dim as DeepSORT expects
            embed = np.tile(seed, 128 // len(seed) + 1)[:128]
            # Normalize
            norm = np.linalg.norm(embed)
            if norm > 0:
                embed = embed / norm
            embeds.append(embed)

        tracks = self._tracker.update_tracks(
            ds_input, frame=frame,
            embeds=embeds,
        )

        results: List[Dict[str, Any]] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = (int(v) for v in ltrb)
            results.append(
                {
                    "track_id": track.track_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": track.det_conf if track.det_conf is not None else 0.0,
                }
            )

        return results

    def get_crop(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """
        Crop out the person region from a frame given a bounding box.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame from OpenCV.
        bbox : tuple
            ``(x1, y1, x2, y2)`` in pixel coordinates.

        Returns
        -------
        Cropped BGR numpy array, or ``None`` if the crop area is invalid.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]
