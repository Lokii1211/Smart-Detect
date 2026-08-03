"""
recognition/face_recognizer.py
───────────────────────────────
InsightFace-based face recognition module.
Uses the buffalo_l model for 512-dimensional ArcFace embeddings.
Falls back gracefully to a stub when insightface is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from config.identity_config import get_identity_config

logger = logging.getLogger(__name__)

# Set SMARTDETECT_STUB_MODE=1 to bypass real InsightFace (useful for E2E tests)
_FORCE_STUB = os.getenv("SMARTDETECT_STUB_MODE", "0") == "1"


def _resolve_providers() -> List[str]:
    """
    ONNX Runtime execution providers, fastest-available first, CPU last.

    Hardware acceleration is opportunistic: we ask onnxruntime what it
    actually has rather than assuming, so the same code runs on Apple
    silicon (CoreML -> ANE/GPU), NVIDIA (CUDA), and plain CPU boxes without
    branching on platform.

    SMARTDETECT_ONNX_PROVIDERS overrides entirely, e.g.
        SMARTDETECT_ONNX_PROVIDERS=CPUExecutionProvider
    to pin the pre-2026-08 behaviour.
    """
    override = os.getenv("SMARTDETECT_ONNX_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]

    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider"]
    chosen = [p for p in preferred if p in available]
    chosen.append("CPUExecutionProvider")   # always keep a fallback
    return chosen


# ─── Lightweight stub when insightface is not installed ──────────────────────

class _FaceStub:
    """
    Dummy face app — returns no detections, keeping the pipeline running
    without insightface installed (useful for pure Re-ID or backend-only runs).
    """

    def get(self, image_frame: np.ndarray):  # noqa: D102
        return []


class _SyntheticFace:
    """
    Synthetic face object that mimics InsightFace's Face with bbox, kps, gender.
    Used in stub mode so the annotation pipeline can draw face boxes for testing.
    """

    def __init__(self, frame_w: int, frame_h: int):
        # Place a synthetic face centred on the eyes/nose/mouth region
        # For a person crop, this is roughly 30-50% down from top
        cx = frame_w // 2
        cy = int(frame_h * 0.38)
        fw = int(frame_w * 0.22)
        fh = int(frame_h * 0.20)
        self.bbox = np.array([cx - fw, cy - fh, cx + fw, cy + fh], dtype=np.float32)
        # 5-point landmarks: left eye, right eye, nose, left mouth, right mouth
        self.kps = np.array([
            [cx - fw * 0.45, cy - fh * 0.30],   # left eye
            [cx + fw * 0.45, cy - fh * 0.30],   # right eye
            [cx,             cy + fh * 0.05],    # nose tip
            [cx - fw * 0.30, cy + fh * 0.45],   # left mouth corner
            [cx + fw * 0.30, cy + fh * 0.45],   # right mouth corner
        ], dtype=np.float32)
        self.gender = 1  # stub default
        self.embedding = None  # not used directly


# ─────────────────────────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    Wraps InsightFace's buffalo_l model for face detection and embedding
    extraction. Falls back to a no-op stub when insightface is not installed.

    Example
    -------
    >>> recognizer = FaceRecognizer()
    >>> recognizer.load_model()
    >>> embeddings = recognizer.extract_embedding(frame)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        # None => pulled from config/identity_config.py (face_detector_size /
        # face_detector_min_score) so this class's defaults track the single
        # source of truth. Pass explicit values (as cameras/camera_processor.py
        # does) to override the config for that instance.
        det_size: Optional[Tuple[int, int]] = None,
        det_thresh: Optional[float] = None,
    ) -> None:
        cfg = get_identity_config()
        self.model_name = model_name
        self.det_size = det_size if det_size is not None else cfg.face_detector_size
        self.det_thresh = det_thresh if det_thresh is not None else cfg.face_detector_min_score
        self._app = None
        self._stub_mode = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Download (if necessary) and load the InsightFace buffalo_l model.
        Falls back to a no-op stub if:
          - insightface is not installed, OR
          - SMARTDETECT_STUB_MODE=1 is set (useful for E2E tests).
        In stub mode, extract_embedding() returns a random 512-dim vector
        so the full registration/sighting pipeline can be tested without
        real face images.
        """
        if _FORCE_STUB:
            logger.info("FaceRecognizer: SMARTDETECT_STUB_MODE=1 — using stub")
            self._app = _FaceStub()
            self._stub_mode = True
            return

        try:
            from insightface.app import FaceAnalysis  # lazy import

            # ── Execution provider selection ────────────────────────────────
            # InsightFace is by far the most expensive stage in the pipeline
            # (83% of per-frame time, measured — eval/profile_pipeline.py).
            # onnxruntime ships CoreMLExecutionProvider on Apple silicon, which
            # runs the ONNX graphs on the ANE/GPU: measured 182 -> 38 ms/frame
            # (4.8x) on an M5 with identical detections.
            #
            # CPU is always appended as a fallback so any op CoreML cannot
            # handle still executes rather than failing the whole session.
            # Override with SMARTDETECT_ONNX_PROVIDERS (comma-separated) —
            # set it to "CPUExecutionProvider" to force the old behaviour if a
            # platform's CoreML implementation ever produces different output.
            providers = _resolve_providers()
            self._app = FaceAnalysis(
                name=self.model_name,
                providers=providers,
            )
            self._app.prepare(
                ctx_id=0,
                det_size=self.det_size,
                det_thresh=self.det_thresh,
            )
            logger.info("FaceRecognizer: loaded model '%s'", self.model_name)
        except ImportError:
            logger.warning(
                "insightface not installed — face recognition disabled. "
                "Install with: pip install insightface onnxruntime"
            )
            self._app = _FaceStub()
            self._stub_mode = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load InsightFace model '%s' (%s) — using stub.",
                self.model_name,
                exc,
            )
            self._app = _FaceStub()
            self._stub_mode = True

    def extract_embedding(self, image_frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in a BGR frame and return ArcFace embedding vectors.

        Also stores the raw face objects in ``self._last_faces`` so callers
        can access ``face.bbox``, ``face.kps``, ``face.gender``, etc.

        In stub mode (SMARTDETECT_STUB_MODE=1 or InsightFace unavailable),
        returns one random 512-dim embedding so the registration pipeline
        can be exercised without a real face in the image.
        """
        self._ensure_loaded()
        self._last_faces = []
        if self._stub_mode:
            # Return a random unit-normalised embedding for testing
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            # Create a synthetic face object so the annotation pipeline works
            h, w = image_frame.shape[:2]
            stub_face = _SyntheticFace(w, h)
            self._last_faces = [stub_face]
            return [emb]
        try:
            faces = self._app.get(image_frame)
            self._last_faces = faces  # expose for annotation pipeline
            return [face.embedding for face in faces if face.embedding is not None]
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_embedding error: %s", exc)
            return []

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.

        Returns
        -------
        float in [0.0, 1.0] — higher means more similar.
        """
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        return float(np.clip(np.dot(e1, e2), 0.0, 1.0))

    def find_match(
        self,
        query_embedding: np.ndarray,
        db_embeddings: List[np.ndarray],
        threshold: float = 0.6,
    ) -> int:
        """
        Find the best-matching embedding from a gallery list.

        Returns
        -------
        Index of the best match in ``db_embeddings``, or ``-1`` if no match
        exceeds ``threshold``.
        """
        if not db_embeddings:
            return -1

        best_idx = -1
        best_score = -1.0

        for idx, db_emb in enumerate(db_embeddings):
            score = self.compare(query_embedding, db_emb)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx if best_score >= threshold else -1

    # ──────────────────────────────────────────────────────────────────────────
    # Private helper
    # ──────────────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._app is None:
            self.load_model()
