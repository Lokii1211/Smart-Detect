"""
recognition/object_detector.py
────────────────────────────────
YOLOv8 object detection for SmartDetect (Change 2).

Detects and classifies:
  - People      → face recognition pipeline
  - Bags        → backpack, handbag, suitcase
  - Vehicles    → car, motorcycle, truck, bus

Install: pip install ultralytics

Usage:
    detector = ObjectDetector()
    detector.load_model()
    detections = detector.detect(frame)
    annotated  = detector.draw_boxes(frame, detections)
    counts     = detector.get_counts(detections)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from backend.logger import get_structured_logger

logger = get_structured_logger(__name__)

# Classes we care about (COCO class names)
PERSON_CLASSES  = {"person"}
BAG_CLASSES     = {"backpack", "handbag", "suitcase"}
VEHICLE_CLASSES = {"car", "motorcycle", "truck", "bus"}
TARGET_CLASSES  = PERSON_CLASSES | BAG_CLASSES | VEHICLE_CLASSES

# Bounding-box colours (BGR)
_COLOR_PERSON  = (80, 200, 80)    # green
_COLOR_BAG     = (50, 50, 220)    # red
_COLOR_VEHICLE = (200, 120, 30)   # blue
_COLOR_DEFAULT = (200, 200, 200)  # grey


def _class_color(label: str) -> tuple:
    if label in PERSON_CLASSES:  return _COLOR_PERSON
    if label in BAG_CLASSES:     return _COLOR_BAG
    if label in VEHICLE_CLASSES: return _COLOR_VEHICLE
    return _COLOR_DEFAULT


class _StubDetector:
    """No-op detector used when ultralytics is not installed."""
    def predict(self, *a, **kw):
        return []


class ObjectDetector:
    """
    Wraps YOLOv8n for inference.
    Falls back gracefully to a stub when ultralytics is not installed.
    """

    # Measured on a real 4K street clip (same frame): imgsz 416 → 2 persons,
    # 640 → 4, 960 → 5, 1280 → 5. Bigger inputs keep distant pedestrians
    # above YOLO's minimum object size; past 960 there is no further gain.
    DEFAULT_IMGSZ = 416   # webcam baseline — cheap, low heat

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.40):
        self._model_name = model_name
        self._conf       = confidence
        self._model      = None
        self._stub_mode  = False
        self.imgsz       = self.DEFAULT_IMGSZ

    def set_imgsz_for_resolution(self, frame_w: int, frame_h: int) -> None:
        """
        Pick the YOLO input size from the source resolution. High-res video
        squeezed to 416px turns distant people into ~5px smudges YOLO cannot
        see — scale the network input up with the source instead.
        """
        longest = max(frame_w, frame_h)
        if longest > 2560:
            self.imgsz = 960
        elif longest > 1280:
            self.imgsz = 640
        else:
            self.imgsz = self.DEFAULT_IMGSZ
        logger.info("yolo.imgsz",
                    message=f"Detection input size set to {self.imgsz} for {frame_w}x{frame_h} source")

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load YOLOv8n. Falls back to stub on ImportError."""
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            logger.info("yolo.load", message=f"Loading YOLO model: {self._model_name}")
            self._model = YOLO(self._model_name)
            logger.info("yolo.ready", message="YOLOv8 model loaded successfully")
        except ImportError:
            logger.warning(
                "yolo.stub",
                message="ultralytics not installed — ObjectDetector running in stub mode. "
                        "Run: pip install ultralytics",
            )
            self._model     = _StubDetector()
            self._stub_mode = True
        except Exception as exc:  # noqa: BLE001
            logger.error("yolo.load_error", message=f"Failed to load YOLO model: {exc}")
            self._model     = _StubDetector()
            self._stub_mode = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8 inference on a frame.

        Returns a list of dicts:
            {label: str, confidence: float, bbox: [x, y, w, h]}

        Only TARGET_CLASSES are returned. Empty list in stub mode.
        """
        if self._model is None:
            self.load_model()

        if self._stub_mode:
            return []

        try:
            results = self._model.predict(
                source=frame,
                conf=0.30,        # low enough to catch more people
                iou=0.45,
                imgsz=self.imgsz, # resolution-aware — see set_imgsz_for_resolution
                classes=[0],      # person class only
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("yolo.infer_error", message=f"Inference failed: {exc}")
            return []

        detections: List[Dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                if label not in TARGET_CLASSES:
                    continue
                conf  = float(box.conf[0])
                # xyxy → xywh
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label":      label,
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2 - x1, y2 - y1],
                })

        return detections

    # ── Annotation ────────────────────────────────────────────────────────────

    def draw_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw coloured bounding boxes on a copy of the frame.
          - Green  → person
          - Red    → bag (backpack / handbag / suitcase)
          - Blue   → vehicle (car / motorcycle / truck / bus)
        """
        import cv2  # noqa: PLC0415
        annotated = frame.copy()
        for det in detections:
            label = det["label"]
            conf  = det["confidence"]
            x, y, w, h = det["bbox"]
            color = _class_color(label)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(
                annotated, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
        return annotated

    # ── Count helpers ─────────────────────────────────────────────────────────

    def get_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Return summary counts by category.

        Example: {"persons": 3, "bags": 1, "vehicles": 2}
        """
        persons  = sum(1 for d in detections if d["label"] in PERSON_CLASSES)
        bags     = sum(1 for d in detections if d["label"] in BAG_CLASSES)
        vehicles = sum(1 for d in detections if d["label"] in VEHICLE_CLASSES)
        return {"persons": persons, "bags": bags, "vehicles": vehicles}
