"""
cameras/camera_processor.py
─────────────────────────────
Real-time camera feed processor for SmartDetect.

Processing pipeline per frame:
  1. YOLOv8 person detection        — GREEN bounding boxes
  2. Face detection inside crop     — CYAN face bbox overlay
  3. Face recognition / Re-ID       — SDT code identification
  4. Throttled sighting calls       — max once per 30 s per track
  5. HUD overlay                    — camera info + frame stats

Auto-reconnect (up to MAX_RECONNECT_ATTEMPTS retries).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import requests

from recognition.face_recognizer import FaceRecognizer
from recognition.reid_model import PersonReID
from tracker.deepsort_tracker import CameraTracker
from backend.logger import get_structured_logger

logger = get_structured_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SIGHTING_COOLDOWN       = 30    # seconds between sightings for same person
CONFIDENCE_THRESHOLD    = 0.6
API_BASE_URL            = "http://localhost:8000"
MAX_RECONNECT_ATTEMPTS  = 5
RECONNECT_INTERVAL_SEC  = 10

# BGR colours
_GREEN  = (0,   210, 80)    # person bounding box
_CYAN   = (255, 220, 0)     # face bounding box
_WHITE  = (255, 255, 255)
_BLACK  = (15,  15,  15)
_ORANGE = (0,   160, 255)   # "Detecting..." label
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


class CameraProcessor:
    """
    Manages a single camera stream with real-time person tracking,
    recognition, sighting reporting, and automatic stream reconnection.

    Parameters
    ----------
    camera_id    : Unique identifier (e.g. "CAM-001").
    location_id  : Location this camera belongs to.
    zone_id      : Zone label (e.g. "entrance").
    video_source : Webcam index or RTSP URL.
    api_base_url : Base URL of the FastAPI backend.
    debug        : If True, opens an OpenCV preview window.
    """

    def __init__(
        self,
        camera_id:    str,
        location_id:  str,
        zone_id:      str       = "main",
        video_source: int | str = 0,
        api_base_url: str       = API_BASE_URL,
        debug:        bool      = True,
    ) -> None:
        self.camera_id    = camera_id
        self.location_id  = location_id
        self.zone_id      = zone_id
        self.video_source = video_source
        self.api_base_url = api_base_url
        self.debug        = debug

        # ML handles (initialised in start())
        self._tracker:  Optional[CameraTracker]  = None
        self._face_rec: Optional[FaceRecognizer] = None
        self._reid:     Optional[PersonReID]     = None
        self._detector  = None   # YOLO ObjectDetector — lazy loaded

        # Track registry: track_id → {unique_code, last_sighting}
        self._track_registry: Dict[int, Dict[str, Any]] = {}

        # Track frame counter: track_id → number of frames seen
        self._track_frame_count: Dict[int, int] = {}

        # Cached face results per track_id — updated by background thread
        self._cached_faces: Dict[int, Dict[str, Any]] = {}

        # Re-ID gallery (shared with reid worker)
        self._reid_gallery_features: List = []
        self._reid_gallery_codes:    List = []

        # Frame stats
        self._frame_count      = 0
        self._fps_window: deque = deque(maxlen=30)

        # SDT code counter for new registrations
        self._next_sdt_number  = 1

        # Auth token for sighting posts
        self._auth_headers: Dict[str, str] = {}

        # Background worker queues (non-blocking)
        self._face_queue: queue.Queue = queue.Queue(maxsize=1)
        self._reid_queue: queue.Queue = queue.Queue(maxsize=2)
        self._detect_queue: queue.Queue = queue.Queue(maxsize=2)

        # Cached track results from background detect+track worker
        self._track_result: List = []
        self._track_lock = threading.Lock()

        # Control flags
        self._stop_event      = threading.Event()
        self._is_online       = False
        self._reconnect_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Background Workers
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_track_worker(self) -> None:
        """Background thread: YOLO + DeepSORT. Updates cached tracks."""
        import traceback as tb
        print("[DETECT] Worker thread started")
        while not self._stop_event.is_set():
            try:
                frame = self._detect_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                track_input = []
                if self._detector is not None:
                    detections = self._detector.detect(frame)
                    for d in detections:
                        if d.get("label") == "person":
                            bx, by, bw, bh = d["bbox"]
                            track_input.append(([bx, by, bw, bh], 0.8, "person"))
                tracks = self._tracker.update(frame, track_input)
                with self._track_lock:
                    self._track_result = tracks
            except Exception as exc:
                print(f"[DETECT] Worker error: {exc}")
                tb.print_exc()
                continue

    def _face_worker(self) -> None:
        """Background thread: runs InsightFace on resized frame, scales coords back."""
        import traceback as tb
        print("[FACE] Worker thread started")
        while not self._stop_event.is_set():
            try:
                item = self._face_queue.get(timeout=1)
                if len(item) == 4:
                    small_frame, track_list, orig_w, orig_h = item
                else:
                    # Fallback for old format
                    small_frame, track_list = item
                    orig_w, orig_h = small_frame.shape[1], small_frame.shape[0]
            except queue.Empty:
                continue
            except Exception:
                continue

            try:
                sh, sw = small_frame.shape[:2]
                sx = orig_w / sw  # scale factor x
                sy = orig_h / sh  # scale factor y

                self._face_rec.extract_embedding(small_frame)
                all_faces = getattr(self._face_rec, "_last_faces", []) or []

                # Match faces to persons and update cache
                for tinfo in track_list:
                    tid = tinfo["track_id"]
                    bbox = tinfo["bbox"]
                    # Normalise to (x1,y1,x2,y2) in ORIGINAL coords
                    if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        px1, py1, px2, py2 = [int(v) for v in bbox]
                    else:
                        bx, by, bw, bh = [int(v) for v in bbox]
                        px1, py1, px2, py2 = bx, by, bx + bw, by + bh

                    # Scale person box to small frame coords for matching
                    spx1, spy1 = px1 / sx, py1 / sy
                    spx2, spy2 = px2 / sx, py2 / sy

                    best_face = None
                    best_score = 0.0
                    for face_obj in all_faces:
                        fb = face_obj.bbox.astype(float)
                        fc_x = (fb[0] + fb[2]) / 2
                        fc_y = (fb[1] + fb[3]) / 2
                        if not (spx1 < fc_x < spx2):
                            continue
                        upper_limit = spy1 + (spy2 - spy1) * 0.65
                        if not (spy1 < fc_y < upper_limit):
                            continue
                        person_cx = (spx1 + spx2) / 2
                        score = 1.0 / (abs(fc_x - person_cx) + 1)
                        if score > best_score:
                            best_score = score
                            best_face = face_obj

                    if best_face is not None:
                        bfb = best_face.bbox.astype(float)
                        # Scale face bbox back to ORIGINAL frame coords
                        self._cached_faces[tid] = {
                            "bbox": [bfb[0] * sx, bfb[1] * sy, bfb[2] * sx, bfb[3] * sy],
                            "kps": [[kp[0] * sx, kp[1] * sy] for kp in best_face.kps[:5].tolist()] if getattr(best_face, "kps", None) is not None else None,
                            "person_center": [
                                (px1 + px2) / 2.0,
                                (py1 + py2) / 2.0,
                            ],
                        }
            except Exception as exc:
                print(f"[FACE] Worker error: {exc}")
                tb.print_exc()

    def _reid_worker(self) -> None:
        """Background thread: runs Re-ID embedding + SDT assignment."""
        while not self._stop_event.is_set():
            try:
                track_id, crop, track_frames = self._reid_queue.get(timeout=1)
            except queue.Empty:
                continue
            except Exception:
                continue

            # Already identified?
            if self._track_registry.get(track_id, {}).get("unique_code"):
                continue

            try:
                query_feat = self._reid.extract_features(crop)
                if self._reid_gallery_features:
                    idx = self._reid.match(query_feat, self._reid_gallery_features, threshold=0.7)
                    if idx >= 0:
                        code = self._reid_gallery_codes[idx]
                        if code:
                            self._track_registry[track_id] = {"unique_code": code, "last_sighting": 0.0}
                            continue
                self._reid_gallery_features.append(query_feat)
                self._reid_gallery_codes.append(None)
            except Exception:
                pass

            if track_frames >= 3:
                new_code = f"SDT-{self._next_sdt_number:04d}"
                self._next_sdt_number += 1
                self._track_registry[track_id] = {"unique_code": new_code, "last_sighting": 0.0}
                for i, c in enumerate(self._reid_gallery_codes):
                    if c is None:
                        self._reid_gallery_codes[i] = new_code
                        break

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open stream, load models, process frames. Blocks — call in a thread."""
        logger.info("camera.init", message=f"[{self.camera_id}] Initialising ML models…")
        self._tracker  = CameraTracker()
        self._face_rec = FaceRecognizer(det_size=(160, 160), det_thresh=0.30)
        self._face_rec.load_model()
        self._reid     = PersonReID()

        # Lazy-load YOLO
        try:
            from recognition.object_detector import ObjectDetector
            self._detector = ObjectDetector()
            self._detector.load_model()
        except Exception as exc:
            logger.warning("camera.yolo_fail", message=f"YOLO load failed: {exc}")

        # Start background workers
        threading.Thread(target=self._face_worker, daemon=True, name="face-worker").start()
        threading.Thread(target=self._reid_worker, daemon=True, name="reid-worker").start()
        threading.Thread(target=self._detect_track_worker, daemon=True, name="detect-worker").start()
        print("[INIT] Background workers launched (detect + face + reid)")

        # Auth: get JWT token for sighting posts
        self._get_auth_token()

        # Backend health check (non-blocking)
        try:
            resp = requests.get(f"{self.api_base_url}/health", timeout=3)
            if resp.ok:
                logger.info("camera.backend_ok", message="Backend is reachable")
            else:
                logger.warning("camera.backend_warn", message=f"Backend returned {resp.status_code}")
        except requests.RequestException:
            logger.warning("camera.backend_unreachable",
                           message=f"Backend not reachable at {self.api_base_url}/health — "
                                   "camera will run locally, sightings will retry in background")

        # ── Outer reconnect loop ───────────────────────────────────────────
        while not self._stop_event.is_set():
            cap = self._open_stream()
            if cap is None:
                break

            self._is_online       = True
            self._reconnect_count = 0
            logger.info("camera.stream_open",
                        message=f"[{self.camera_id}] Stream opened on source={self.video_source}")

            stream_lost = False
            t_last = time.time()
            try:
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("camera.read_error",
                                       message=f"[{self.camera_id}] Frame read failed — stream dropped")
                        stream_lost = True
                        break

                    # FPS tracking
                    now = time.time()
                    self._fps_window.append(now - t_last)
                    t_last = now
                    self._frame_count += 1

                    annotated = self._process_frame(frame)

                    if self.debug:
                        cv2.imshow(f"SmartDetect — {self.camera_id}", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self._stop_event.set()
                            break
            finally:
                cap.release()
                if self.debug:
                    cv2.destroyAllWindows()

            if stream_lost and not self._stop_event.is_set():
                if not self._handle_stream_loss():
                    break
            else:
                break

        logger.info("camera.stopped", message=f"[{self.camera_id}] Processor stopped.")

    def stop(self) -> None:
        """Signal the processing loop to stop gracefully."""
        logger.info("camera.stop", message=f"[{self.camera_id}] Stop requested.")
        self._stop_event.set()

    @property
    def fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        avg = sum(self._fps_window) / len(self._fps_window)
        return round(1.0 / max(avg, 0.001), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Auto-Reconnect Logic
    # ─────────────────────────────────────────────────────────────────────────

    def _open_stream(self) -> Optional[cv2.VideoCapture]:
        attempts = 0
        while attempts <= MAX_RECONNECT_ATTEMPTS and not self._stop_event.is_set():
            if isinstance(self.video_source, int):
                cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.video_source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prevent stale frame buildup
                # Use MJPEG codec for faster USB webcam reads
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
                if attempts > 0:
                    logger.info("camera.reconnected",
                                message=f"[{self.camera_id}] Reconnected after {attempts} attempt(s).")
                return cap
            cap.release()
            attempts += 1
            if attempts <= MAX_RECONNECT_ATTEMPTS:
                logger.warning("camera.retry",
                               message=f"[{self.camera_id}] Cannot open source={self.video_source}. "
                                       f"Retry {attempts}/{MAX_RECONNECT_ATTEMPTS} in {RECONNECT_INTERVAL_SEC}s…")
                self._stop_event.wait(timeout=RECONNECT_INTERVAL_SEC)
        self._mark_offline()
        return None

    def _handle_stream_loss(self) -> bool:
        self._reconnect_count += 1
        self._is_online = False
        if self._reconnect_count > MAX_RECONNECT_ATTEMPTS:
            logger.error("camera.offline",
                         message=f"[{self.camera_id}] Exceeded max reconnect attempts. Marking OFFLINE.")
            self._mark_offline()
            return False
        logger.warning("camera.stream_lost",
                       message=f"[{self.camera_id}] Stream lost. "
                               f"Reconnect {self._reconnect_count}/{MAX_RECONNECT_ATTEMPTS} in {RECONNECT_INTERVAL_SEC}s…")
        self._stop_event.wait(timeout=RECONNECT_INTERVAL_SEC)
        return True

    def _mark_offline(self) -> None:
        self._is_online = False
        threading.Thread(target=self._post_camera_offline, daemon=True).start()

    def _post_camera_offline(self) -> None:
        try:
            requests.post(f"{self.api_base_url}/camera/status",
                          json={"camera_id": self.camera_id, "status": "offline"}, timeout=5)
        except requests.RequestException:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Frame Processing Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Per-frame pipeline: read cached tracks, annotate, push to workers."""
        annotated = frame  # draw in-place
        h, w = frame.shape[:2]

        # ── 1. Push frame to detect worker (drop old if full) ────────────
        try:
            self._detect_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._detect_queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._detect_queue.put_nowait(frame)
            except queue.Full:
                pass

        # ── 2. Read cached tracks from background thread (instant) ──────
        with self._track_lock:
            tracks = list(self._track_result)

        # ── 3. Deduplicate overlapping tracks (keep oldest track_id) ─────
        if len(tracks) > 1:
            sorted_tracks = sorted(tracks, key=lambda t: t.get("track_id", 0))
            kept = []
            for t in sorted_tracks:
                tb = t["bbox"]
                if len(tb) == 4 and tb[2] > tb[0] and tb[3] > tb[1]:
                    bx1, by1, bx2, by2 = tb
                else:
                    bx1, by1, bw, bht = tb
                    bx2, by2 = bx1 + bw, by1 + bht
                is_dup = False
                for k in kept:
                    kb = k["bbox"]
                    if len(kb) == 4 and kb[2] > kb[0] and kb[3] > kb[1]:
                        kx1, ky1, kx2, ky2 = kb
                    else:
                        kx1, ky1, kw, kht = kb
                        kx2, ky2 = kx1 + kw, ky1 + kht
                    ix1 = max(bx1, kx1); iy1 = max(by1, ky1)
                    ix2 = min(bx2, kx2); iy2 = min(by2, ky2)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    a1 = max(1, (bx2-bx1)*(by2-by1))
                    a2 = max(1, (kx2-kx1)*(ky2-ky1))
                    smaller_area = min(a1, a2)
                    overlap_ratio = inter / smaller_area
                    if overlap_ratio > 0.70:
                        is_dup = True
                        break
                if not is_dup:
                    kept.append(t)
            tracks = kept

        person_count = len(tracks)

        # ── 4. Push to face worker (every 2nd frame, resized for speed) ───
        if self._frame_count % 2 == 0 and tracks:
            if not self._face_queue.full():
                try:
                    # Resize to 320x240 for faster InsightFace inference
                    small = cv2.resize(frame, (320, 240))
                    self._face_queue.put_nowait((small, list(tracks), w, h))
                except queue.Full:
                    pass

        # ── 2. Per-track annotation ───────────────────────────────────────
        for track in (tracks or []):
            track_id:   int   = track["track_id"]
            bbox              = track["bbox"]
            confidence: float = track.get("confidence", 0.8)

            # Normalise to (x1,y1,x2,y2)
            if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            else:
                bx, by, bw, bh = [int(v) for v in bbox]
                x1, y1, x2, y2 = bx, by, bx + bw, by + bh

            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            # Skip garbage oversized boxes (>85% of frame = not a real person box)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 0.85 * w * h:
                continue

            # ── 3. Identify person (push to background, never block) ─────
            self._track_frame_count[track_id] = self._track_frame_count.get(track_id, 0) + 1
            track_frames = self._track_frame_count[track_id]
            unique_code = self._track_registry.get(track_id, {}).get("unique_code")

            if not unique_code and self._frame_count % 10 == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    try:
                        self._reid_queue.put_nowait((track_id, crop.copy(), track_frames))
                    except queue.Full:
                        pass

            # Force-register on main thread if 5+ frames and no code yet
            if not unique_code and track_frames >= 5:
                new_code = f"SDT-{self._next_sdt_number:04d}"
                self._next_sdt_number += 1
                self._track_registry[track_id] = {"unique_code": new_code, "last_sighting": 0.0}
                unique_code = new_code

            if unique_code and confidence >= CONFIDENCE_THRESHOLD:
                self._maybe_log_sighting(unique_code, confidence)

            # ── 4. Draw GREEN person box ──────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _GREEN, 2)

            # ── 5. SDT label above box ────────────────────────────────────
            label  = unique_code if unique_code else "Detecting..."
            lcolor = _GREEN if unique_code else _ORANGE
            (tw, th_t), _ = cv2.getTextSize(label, _FONT, 0.52, 2)
            cv2.rectangle(annotated, (x1, y1 - th_t - 8), (x1 + tw + 6, y1), lcolor, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        _FONT, 0.52, _BLACK, 2)

            # ── 6. Draw face from InsightFace cache with motion offset ───
            cached = self._cached_faces.get(track_id)
            if cached:
                try:
                    dx, dy = 0, 0
                    old_center = cached.get("person_center")
                    if old_center:
                        old_cx, old_cy = old_center
                        cur_cx = (x1 + x2) / 2
                        cur_cy = (y1 + y2) / 2
                        dx = int(cur_cx - old_cx)
                        dy = int(cur_cy - old_cy)
                        if abs(dx) > 80 or abs(dy) > 80:
                            self._cached_faces.pop(track_id, None)
                            continue
                    fb = cached["bbox"]
                    fx1 = int(fb[0]) + dx
                    fy1 = int(fb[1]) + dy
                    fx2 = int(fb[2]) + dx
                    fy2 = int(fb[3]) + dy
                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), _CYAN, 2)
                    if cached.get("kps"):
                        lcolors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
                        for i, kp in enumerate(cached["kps"]):
                            kx = int(kp[0]) + dx
                            ky = int(kp[1]) + dy
                            cv2.circle(annotated, (kx, ky), 3, lcolors[i], -1)
                except Exception:
                    pass

        # ── 7. HUD — top bar ──────────────────────────────────────────────
        status_str = "LIVE" if self._is_online else f"RECONNECTING ({self._reconnect_count}/{MAX_RECONNECT_ATTEMPTS})"
        hud_text = f"SmartDetect | Cam: {self.camera_id} | Zone: {self.zone_id} | {status_str}"
        (hw, hh), _ = cv2.getTextSize(hud_text, _FONT, 0.55, 2)
        cv2.rectangle(annotated, (0, 0), (hw + 14, hh + 10), (0, 0, 0), -1)
        cv2.putText(annotated, hud_text, (7, hh + 4), _FONT, 0.55, _WHITE, 1)

        # ── 8. HUD — bottom-left stats ────────────────────────────────────
        stats = f"Persons: {person_count} | FPS: {self.fps:.0f}"
        (sw, sh), _ = cv2.getTextSize(stats, _FONT, 0.52, 2)
        by = h - 10
        cv2.rectangle(annotated, (0, by - sh - 6), (sw + 12, h), (0, 0, 0), -1)
        cv2.putText(annotated, stats, (6, by - 2), _FONT, 0.52, _WHITE, 1)

        return annotated

    # ─────────────────────────────────────────────────────────────────────────
    # Auth + Sighting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_auth_token(self) -> bool:
        """Login to backend and get JWT token for sighting posts."""
        try:
            resp = requests.post(
                f"{self.api_base_url}/auth/login",
                json={"username": "operator", "password": "smartOp2024"},
                timeout=5,
            )
            logger.debug("camera.auth_response",
                         message=f"Auth response: {resp.status_code} {resp.text[:200]}")
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token")
                if token:
                    self._auth_headers = {"Authorization": f"Bearer {token}"}
                    logger.info("camera.auth_ok",
                                message=f"JWT token acquired (role={data.get('role', '?')})")
                    return True
                else:
                    logger.warning("camera.auth_no_token",
                                   message=f"No access_token in response: {list(data.keys())}")
            else:
                logger.warning("camera.auth_fail",
                               message=f"Login failed: {resp.status_code} {resp.text[:100]}")
        except Exception as exc:
            logger.debug("camera.auth_fail", message=f"Auth exception: {exc}")
        self._auth_headers = {}
        return False

    def _maybe_log_sighting(self, unique_code: str, confidence: float) -> None:
        """Post a sighting with cooldown enforcement."""
        reg_key = next(
            (k for k, v in self._track_registry.items() if v.get("unique_code") == unique_code),
            None,
        )
        registry = self._track_registry.get(reg_key) if reg_key is not None else None
        now  = time.time()
        last = registry["last_sighting"] if registry else 0.0
        if now - last < SIGHTING_COOLDOWN:
            return
        if registry:
            registry["last_sighting"] = now
        payload = {
            "unique_code": unique_code,
            "location_id": self.location_id,
            "zone_id":     self.zone_id,
            "camera_id":   self.camera_id,
            "confidence":  confidence,
        }
        threading.Thread(
            target=self._post_sighting_with_auth,
            args=(payload,),
            daemon=True,
        ).start()

    def _post_sighting_with_auth(self, payload: Dict[str, Any]) -> None:
        """POST sighting with JWT auth and auto-refresh on 401."""
        # Ensure we have a token before posting
        if not self._auth_headers:
            self._get_auth_token()

        url = f"{self.api_base_url}/sighting"
        try:
            resp = requests.post(url, json=payload, headers=self._auth_headers, timeout=3)
            if resp.status_code == 401:
                # Token expired — re-login and retry once
                logger.debug("camera.auth_refresh", message="Token expired, refreshing...")
                if self._get_auth_token():
                    resp = requests.post(url, json=payload, headers=self._auth_headers, timeout=3)
            if resp.status_code in (200, 201):
                logger.info("camera.sighting",
                            message=f"[{self.camera_id}] Sighting: {payload.get('unique_code')} conf={payload.get('confidence', 0):.2f}")
            else:
                logger.warning("camera.sighting_error",
                               message=f"[{self.camera_id}] Sighting API: {resp.status_code} {resp.text[:100]}")
        except requests.RequestException as exc:
            logger.debug("camera.sighting_fail",
                         message=f"[{self.camera_id}] Sighting post failed: {exc}")
