"""
cameras/live_stream.py
───────────────────────
Live camera capture + real-time MJPEG streaming for SmartDetect.

Architecture (capture and ML are decoupled — same pattern as camera_processor):

  Capture thread (~30 fps):
    read frame → push to analysis queue (drop-oldest)
    → annotate frame from CACHED analysis results
    → encode JPEG → self._latest_frame
  Analysis worker thread (runs at its own pace, ~1-3 Hz on CPU):
    YOLO persons/objects → ByteTrack (persistent track ids)
    → SmartIdentifier once per track (registration gated on track age)
    → InsightFace full-frame, dress colour, height, bag linking
    → sighting logging (deduped) → cached results for the capture thread
  GET /camera/stream/{id} yields MJPEG from self._latest_frame
"""

from __future__ import annotations

import json
import logging
import math
import queue
import sys
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.identity_config import get_identity_config
from database.db import SessionLocal
from database.models import Person
from database.queries import log_sighting, check_watchlist_alert

logger = logging.getLogger(__name__)

# supervision provides ByteTrack + PolygonZone (deprecation warning is about
# the v0.30 move to the separate `trackers` package — pinned <0.30 in requirements)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import supervision as sv
    _SV_AVAILABLE = True
except Exception as _sv_exc:  # pragma: no cover
    sv = None
    _SV_AVAILABLE = False
    logger.warning("supervision unavailable (%s) — tracking disabled, no new registrations", _sv_exc)


# ─── BGR Colours ──────────────────────────────────────────────────────────────
_GREEN   = (80,  200, 80)    # face-identified person
_BLUE    = (200, 130, 50)    # body/Re-ID identified person
_WHITE   = (230, 230, 230)   # new registration
_CYAN    = (255, 220, 0)     # face bounding box
_ORANGE  = (0,   160, 255)   # bag / carried object
_YELLOW  = (0,   230, 255)   # bottle
_RED     = (60,  60,  220)   # legend border
_BLACK   = (15,  15,  15)
_GREY    = (160, 160, 160)
_FONT    = cv2.FONT_HERSHEY_SIMPLEX

# Method → box colour mapping
_BOX_COLORS = {
    "face":             _GREEN,
    "dress_color":      _BLUE,
    "body_structure":   _BLUE,
    "new_registration": _WHITE,
}

# Blank 480×640 placeholder frame (shown while camera is starting)
_BLANK = np.zeros((480, 640, 3), dtype=np.uint8)

# Object classes we track
_BAG_CLASSES    = {"backpack", "handbag", "suitcase"}
_BOTTLE_CLASSES = {"bottle"}
_OBJECT_LABELS  = {
    "backpack": "Bag",
    "handbag":  "Handbag",
    "suitcase": "Suitcase",
    "bottle":   "Bottle",
}

# Movement-trail length (analysis cycles) drawn behind each tracked person
_TRAIL_LENGTH = 30


class _BufferedFrameRecord:
    """
    One buffered frame held by the tracklet voter while a track is undecided.

    The voter patches `.code` / `.method` on commit (mirroring the offline
    harness's record objects, where this is how scoring records get
    relabelled). In the live path those patches are inert — the on-screen
    history is deliberately NOT rewritten — but `.when` is read by
    _backfill_tracklet_sightings() so the retroactive sighting rows carry the
    frame's actual processing time, not the commit time.
    """
    __slots__ = ("when", "code", "method")

    def __init__(self, when: float):
        self.when = when
        self.code = None
        self.method = None

# Identity-arbitration parameters previously hardcoded here — min track age
# for registration, face-quality gate, head-zoom pass, ID-switch contradiction
# guard, evidence-gating similarity — now live in config/identity_config.py
# (IdentityConfig) as the single source of truth. See self._identity_config,
# set in LiveStream.__init__. Defaults are unchanged; override via the
# SMARTDETECT_IDENTITY_CONFIG env var (config/ablation/*.json for ready-made
# ablation configs).


# ── Live-stream registry ────────────────────────────────────────────────────
# camera_id -> running LiveStream instance. Lets governance.erase_person()
# reach into every active stream's in-memory identity cache without
# backend/main.py's `active_streams` dict — importing that here would be
# circular (main.py imports governance, governance would import main).
# Populated in start(), cleared in stop() — same lifetime as active_streams.
_REGISTRY: Dict[str, "LiveStream"] = {}


def purge_identity(unique_code: str) -> int:
    """
    Remove every in-memory reference to `unique_code` from every running
    stream: the tracker_id -> code cache (`_track_codes`, `active_tracks`),
    the sighting dedup cache (`_seen_cache`), and the recent-detections
    buffer (`_detections`, surfaced verbatim by GET /camera/detections/recent
    and GET /persons/live).

    Called by governance.erase_person() so a deleted identity cannot keep
    appearing as "live" or in recent activity purely because a stream
    resolved it earlier this session and never re-queries the DB per frame.
    Returns the number of cache entries removed, across all streams.

    These dicts are otherwise written only by each stream's analysis worker
    thread, but already read from other threads elsewhere in this class
    (e.g. _on_video_finished reads _track_codes from the capture thread) —
    this follows that same informal safety level rather than adding new
    locking to the per-frame hot path for a rare, admin-triggered event.
    """
    removed = 0
    for stream in list(_REGISTRY.values()):
        stale_tids = [tid for tid, v in list(stream._track_codes.items())
                      if v.get("code") == unique_code]
        for tid in stale_tids:
            stream._track_codes.pop(tid, None)
            stream.active_tracks.pop(tid, None)
            removed += 1
        if stream._seen_cache.pop(unique_code, None) is not None:
            removed += 1
        kept = [d for d in stream._detections if d.get("unique_code") != unique_code]
        if len(kept) != len(stream._detections):
            removed += len(stream._detections) - len(kept)
            stream._detections = deque(kept, maxlen=100)
    return removed


class LiveStream:
    """
    Manages one camera source with capture and ML analysis on separate threads.

    Parameters
    ----------
    source        : int (webcam index) or str (RTSP / file path)
    location_id   : SmartDetect location ID
    zone_id       : zone label (e.g. "entrance")
    camera_id     : human-readable camera identifier
    target_fps    : capture/stream rate (default 30 — analysis runs independently)
    """

    def __init__(
        self,
        source,
        location_id: str = "LOC-001",
        zone_id:     str  = "main",
        camera_id:   str  = "CAM-001",
        target_fps:  int  = 30,
    ):
        self.source      = source
        self._source     = source  # kept for status API
        self.location_id = location_id
        self.zone_id     = zone_id
        self.camera_id   = camera_id
        self._interval   = 1.0 / max(target_fps, 1)

        # ── File-source mode (uploaded surveillance video) ──────────────
        # Detected in start(); files play at native FPS and finish at EOF
        # instead of retrying like a dropped live connection.
        self._is_file      = isinstance(source, str) and Path(source).is_file()
        self._finished     = False
        self._frame_total  = 0   # CAP_PROP_FRAME_COUNT for files (0 = unknown)
        self._face_scan_size = (640, 480)  # raised for high-res files in start()

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread:  Optional[threading.Thread] = None
        self._worker:  Optional[threading.Thread] = None
        self._running  = False
        self._lock     = threading.Lock()
        self._latest_frame: bytes = self._encode_frame(_BLANK)

        # ── Analysis handoff (capture → worker, drop-oldest) ───────────
        self._analysis_queue: queue.Queue = queue.Queue(maxsize=1)
        self._results_lock = threading.Lock()
        # persons: [{bbox, code, method, conf, color_hex, gender, height_label,
        #            face_bbox, face_kps, carrying, tracker_id}], objects: [{bbox, label}]
        self._latest_results: Dict = {"persons": [], "objects": []}

        # ── Tracking state (worker thread only) ────────────────────────
        self._tracker = None
        self._zone    = None            # sv.PolygonZone — built on first frame
        self.zone_count: int = 0        # persons currently in the camera zone
        self._track_codes: Dict[int, Dict] = {}   # tracker_id → {code, method, conf}
        self._track_age:   Dict[int, int]  = {}   # tracker_id → analysis cycles seen
        self._track_face_mismatch: Dict[int, int] = {}  # tracker_id → consecutive face contradictions
        self._track_pose_defer: Dict[int, int] = {}     # tracker_id → consecutive pose-gate deferrals
        self._track_trails: Dict[int, deque] = {} # tracker_id → recent bbox centers

        # ── Stats ──────────────────────────────────────────────────────────
        self.persons_today: int = 0
        self.active_tracks: Dict[int, str] = {}   # track_id → SDT code
        self._detections: deque = deque(maxlen=100)

        # ── FPS counter (frame count + time window) ────────────────────
        self._fps_count     = 0
        self._fps_last_time = time.time()
        self._fps_value     = 0.0
        self._analysis_fps  = 0.0   # analysis cycles per second
        self._total_frames  = 0
        self._analyzed_frames = 0   # frames the ML worker fully processed

        # ── Frame-level counters (updated per analysis cycle) ──────────
        self._frame_persons = 0
        self._frame_faces   = 0
        self._frame_bags    = 0

        # ── Face bbox smoothing (temporal averaging, keyed by tracker_id) ──
        self._face_bbox_history: Dict[int, list] = {}
        self._face_kps_history:  Dict[int, list] = {}

        # ── Dedup cache  ───────────────────────────────────────────────────
        # unique_code → last log timestamp; skip re-log within 30s
        self._seen_cache: Dict[str, float] = {}

        # ── Identity-arbitration config (config/identity_config.py) ────────
        self._identity_config = get_identity_config()

        # ── Tracklet voting (recognition/tracklet_vote.py) ─────────────────
        # OFF by default (IdentityConfig.enable_tracklet_voting). When on,
        # identity commits once per TRACK from up to `tracklet_buffer_size`
        # quality-weighted gate-passing face embeddings instead of on the
        # first gate-passing frame; until commit the track shows
        # "Detecting..." (see the voting block in _analyze_frame).
        self._voter = None
        if self._identity_config.enable_tracklet_voting:
            from recognition.tracklet_vote import TrackletVoter
            self._voter = TrackletVoter(self._identity_config)

        # Last analysed frame — used by track-end commits, which can fire
        # after the frame loop has moved on (cleanup / camera stop).
        self._last_frame = None

        # ── ML components (lazy-loaded to avoid startup crash) ─────────────
        self._detector   = None
        self._identifier = None
        self._face_rec   = None

        try:
            from recognition.object_detector import ObjectDetector
            from recognition.smart_identifier import SmartIdentifier
            from recognition.face_recognizer import FaceRecognizer
            self._detector   = ObjectDetector()
            self._identifier = SmartIdentifier()
            self._face_rec   = FaceRecognizer()
        except Exception as exc:
            logger.warning("ML models unavailable (%s) — stream will show raw frames", exc)

        if _SV_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                # activation 0.25 so conf-0.3 detections start tracks on the
                # first cycle — analysis runs ~1-2 Hz, waiting frames = seconds
                self._tracker = sv.ByteTrack(track_activation_threshold=0.25)

    # ── Public interface ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the camera/video and start the capture + analysis threads."""
        if isinstance(self.source, int):
            # CAP_DSHOW is a Windows-only backend; on macOS it fails immediately
            # without falling back to AVFoundation (or reaching the TCC camera
            # check), so use the platform default everywhere else.
            if sys.platform.startswith("win"):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")

        if self._is_file:
            # Uploaded video: play at its recorded speed; webcam props don't apply
            native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0
            if not (1 <= native_fps <= 120):
                native_fps = 25.0
            self._interval    = 1.0 / native_fps
            self._frame_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            # Resolution-aware detection: a 4K frame squeezed to YOLO's 416px
            # webcam default loses every distant pedestrian (measured: 2 vs 5)
            src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if self._detector is not None and src_w and src_h:
                try:
                    self._detector.set_imgsz_for_resolution(src_w, src_h)
                except Exception:
                    pass
            # Face scan resolution follows the source too (see _analyze_frame).
            # Sources at or below 1280px are scanned at native size — shrinking
            # a 768px file to 640x480 pushed borderline faces under the
            # detector's floor and cost real recall.
            if max(src_w, src_h) > 1280:
                self._face_scan_size = (1280, 720)
            elif src_w and src_h:
                self._face_scan_size = (src_w, src_h)
            else:
                self._face_scan_size = (640, 480)
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # MJPG codec: USB webcams deliver 30 fps MJPG vs ~8 fps uncompressed YUY2
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        if self._detector is not None:
            try:
                self._detector.load_model()
            except Exception as exc:
                logger.warning("YOLO load failed (%s) — streaming raw frames", exc)
                self._detector = None
        if self._face_rec is not None:
            try:
                self._face_rec.load_model()
            except Exception as exc:
                logger.warning("FaceRecognizer load failed (%s)", exc)
                self._face_rec = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"LiveStream-{self.camera_id}")
        self._thread.start()
        self._worker = threading.Thread(target=self._analysis_loop, daemon=True,
                                        name=f"LiveStream-ML-{self.camera_id}")
        self._worker.start()
        _REGISTRY[self.camera_id] = self
        logger.info("LiveStream %s started — source=%s", self.camera_id, self.source)

    def stop(self) -> None:
        """Stop both threads and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._worker:
            self._worker.join(timeout=3)
        if self._cap:
            self._cap.release()
        # Stopping the stream ends every open track: commit any pending
        # voting decisions so the DB trail keeps the last seconds' evidence.
        try:
            self._commit_open_voting_tracks()
        except Exception as exc:
            logger.debug("voting commit at stop failed: %s", exc)
        if _REGISTRY.get(self.camera_id) is self:
            _REGISTRY.pop(self.camera_id, None)
        logger.info("LiveStream %s stopped", self.camera_id)

    def is_connected(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    def get_status(self) -> Dict:
        progress = 0.0
        if self._is_file and self._frame_total > 0:
            progress = min(1.0, self._total_frames / self._frame_total)
        return {
            "connected":             self.is_connected(),
            "camera_id":             self.camera_id,
            "fps":                   round(self._fps_value, 1),
            "analysis_fps":          round(self._analysis_fps, 1),
            "persons_detected_today": self.persons_today,
            "active_tracks":         len(self.active_tracks),
            # raw YOLO person count from the LAST analysis cycle — active_tracks
            # only counts CONFIRMED identities, this shows detection is firing
            # even before any face/registration gate passes
            "frame_persons":         self._frame_persons,
            "zone_count":            self.zone_count,
            "is_file":               self._is_file,
            "finished":              self._finished,
            "progress":              round(progress, 3),
            "analyzed_frames":       self._analyzed_frames,
        }

    def get_mjpeg_frame(self) -> bytes:
        with self._lock:
            return self._latest_frame

    def get_recent_detections(self, limit: int = 20) -> list:
        return list(self._detections)[-limit:]

    def get_live_persons(self) -> list:
        """Return persons seen in the last 30 seconds."""
        now = time.time()
        return [
            {"unique_code": code, "last_seen": ts}
            for code, ts in self._seen_cache.items()
            if now - ts < 30
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Capture loop — lightweight: read, hand off, annotate from cache, encode
    # ─────────────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                if self._is_file:
                    # End of uploaded video — finish cleanly, don't retry
                    self._on_video_finished()
                    break
                logger.warning("LiveStream %s: frame read failed — retrying", self.camera_id)
                time.sleep(0.5)
                continue

            self._total_frames += 1

            # Hand the frame to the analysis worker — only copy when the
            # worker is ready for one (it consumes ~1/s; copying every frame
            # wastes capture-loop time)
            if self._analysis_queue.empty():
                try:
                    self._analysis_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            annotated = self._annotate_frame(frame)
            jpg       = self._encode_frame(annotated)
            with self._lock:
                self._latest_frame = jpg

            # ── FPS calculation ───────────────────────────────────────────
            self._fps_count += 1
            now = time.time()
            elapsed_since_reset = now - self._fps_last_time
            if elapsed_since_reset >= 1.0:
                self._fps_value = self._fps_count / elapsed_since_reset
                self._fps_count = 0
                self._fps_last_time = now

            elapsed = now - t0
            sleep_for = max(0, self._interval - elapsed)
            time.sleep(sleep_for)

    def _on_video_finished(self) -> None:
        """Uploaded video reached EOF: final frame, DB offline, stop threads."""
        self._finished = True
        # A person still in frame at EOF is a track end: commit any open
        # voting tracks so their buffered frames are not lost.
        try:
            self._commit_open_voting_tracks()
        except Exception as exc:
            logger.debug("voting commit at EOF failed: %s", exc)
        persons_found = len({c["code"] for c in self._track_codes.values()}) or self.persons_today

        # Final "VIDEO ENDED" frame stays visible in the MJPEG stream
        end_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(end_frame, "VIDEO ENDED", (170, 210), _FONT, 1.3, _WHITE, 2)
        cv2.putText(end_frame, f"{persons_found} person(s) identified | {self._analyzed_frames} frames analysed",
                    (95, 260), _FONT, 0.55, _GREY, 1)
        cv2.putText(end_frame, f"{self.camera_id} — processing complete", (170, 300), _FONT, 0.5, _GREY, 1)
        with self._lock:
            self._latest_frame = self._encode_frame(end_frame)

        # Mark the camera offline in the DB (user chose: finish = offline)
        try:
            db = SessionLocal()
            try:
                from database.models import Camera
                cam = db.query(Camera).filter(Camera.id == self.camera_id).first()
                if cam is not None:
                    cam.is_active = False
                    db.commit()
                # Operational alert — the dashboard already renders
                # alert_type='camera_offline'; nothing used to emit it, so a
                # camera could die silently behind a stale last frame.
                from database.queries import create_camera_offline_alert
                create_camera_offline_alert(
                    camera_id=self.camera_id, location_id=self.location_id,
                    reason="video source reached end of stream", db=db)
            finally:
                db.close()
        except Exception as exc:
            logger.debug("video-finished DB update failed: %s", exc)

        self._running = False
        if self._cap:
            self._cap.release()
        logger.info("LiveStream %s: video finished — %d person(s), %d frames analysed",
                    self.camera_id, persons_found, self._analyzed_frames)

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis worker — YOLO + ByteTrack + identify + sightings (own pace)
    # ─────────────────────────────────────────────────────────────────────────

    def _analysis_loop(self) -> None:
        cycle_times: deque = deque(maxlen=10)
        while self._running:
            try:
                frame = self._analysis_queue.get(timeout=1)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                self._analyze_frame(frame)
                self._analyzed_frames += 1
            except Exception as exc:
                logger.debug("LiveStream analysis error: %s", exc)
            cycle_times.append(time.time() - t0)
            if cycle_times:
                avg = sum(cycle_times) / len(cycle_times)
                self._analysis_fps = 1.0 / max(avg, 0.001)

    def _analyze_frame(self, frame: np.ndarray) -> None:
        """Heavy ML pipeline. Writes results into self._latest_results."""
        fh, fw = frame.shape[:2]

        if not self._detector:
            return

        # Detection only — tiling changes which boxes exist, never how a box
        # acquires an identity. Falls through to a plain detect() when
        # enable_tiled_detection is off or the source is below the floor.
        from cameras.tiled_detect import detect_with_tiling
        detections = detect_with_tiling(self._detector, frame, self._identity_config)
        person_dets = [d for d in detections if d["label"] == "person"]
        bag_dets    = [d for d in detections if d["label"] in _BAG_CLASSES]
        bottle_dets = [d for d in detections if d["label"] in _BOTTLE_CLASSES]
        object_dets = bag_dets + bottle_dets

        self._frame_persons = len(person_dets)
        self._frame_bags    = len(bag_dets) + len(bottle_dets)

        # ── ByteTrack: persistent tracker_id per person ────────────────────
        tracked: List[Tuple[Optional[int], List[int]]] = []  # (tracker_id, [x,y,w,h])
        if self._tracker is not None and person_dets:
            xyxy = np.array(
                [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                 for d in person_dets],
                dtype=np.float32,
            )
            conf = np.array([d["confidence"] for d in person_dets], dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros(len(person_dets), dtype=int),
            )
            sv_dets = self._tracker.update_with_detections(sv_dets)
            for i in range(len(sv_dets)):
                x1, y1, x2, y2 = sv_dets.xyxy[i].astype(int)
                tid = int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else None
                tracked.append((tid, [x1, y1, x2 - x1, y2 - y1]))

            # ByteTrack only returns confirmed tracks — also show raw YOLO
            # detections it hasn't confirmed yet (as "Detecting..."), so a
            # second person appears instantly instead of cycles later
            for d in person_dets:
                dx, dy, dw, dh = d["bbox"]
                covered = False
                for _, tb in tracked:
                    tx, ty, tw_, th_ = tb
                    ix = max(0, min(dx + dw, tx + tw_) - max(dx, tx))
                    iy = max(0, min(dy + dh, ty + th_) - max(dy, ty))
                    inter = ix * iy
                    union = dw * dh + tw_ * th_ - inter
                    if union > 0 and inter / union > 0.5:
                        covered = True
                        break
                if not covered:
                    tracked.append((None, [dx, dy, dw, dh]))

            # Zone occupancy (full-frame polygon by default — customise later)
            if self._zone is None and _SV_AVAILABLE:
                self._zone = sv.PolygonZone(
                    polygon=np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]])
                )
            if self._zone is not None:
                try:
                    self._zone.trigger(sv_dets)
                    self.zone_count = int(self._zone.current_count)
                except Exception:
                    self.zone_count = len(tracked)
        else:
            # No tracker: annotate without ids; registration stays gated off
            tracked = [(None, list(d["bbox"])) for d in person_dets]
            self.zone_count = len(tracked)

        # ── Full-frame face detection (resized for CPU speed) ──────────────
        all_faces: list = []
        if self._face_rec:
            try:
                scan_w, scan_h = self._face_scan_size
                small_frame = cv2.resize(frame, (scan_w, scan_h))
                self._face_rec.extract_embedding(small_frame)
                raw_faces = getattr(self._face_rec, "_last_faces", []) or []
                h_scale = fh / scan_h
                w_scale = fw / scan_w
                for face_obj in raw_faces:
                    face_obj.bbox[0] *= w_scale
                    face_obj.bbox[1] *= h_scale
                    face_obj.bbox[2] *= w_scale
                    face_obj.bbox[3] *= h_scale
                    kps = getattr(face_obj, "kps", None)
                    if kps is not None:
                        for kp in kps:
                            kp[0] *= w_scale
                            kp[1] *= h_scale
                all_faces = raw_faces
            except Exception as exc:
                logger.debug("Full-frame face detection error: %s", exc)

        # ── Head-zoom second pass ────────────────────────────────────────────
        # WHAT: the full-frame face scan above runs on a downscaled copy of
        # the whole frame (self._face_scan_size — e.g. 640x480) for CPU speed.
        # That downscale can shrink a real, close-enough face below what
        # InsightFace's detector can find, even though the SAME face would be
        # detectable if scanned at higher resolution. This pass gives tracked
        # persons who came up faceless in the full-frame scan one more look,
        # at native crop resolution, digitally zoomed in on just their head.
        #
        # WHEN it triggers: only for person boxes taller than
        # zoom_min_person_height_px (default 140px — shorter boxes are too
        # far away to hold a gate-passing face at all, not worth the CPU) AND
        # not already matched to a face from the full-frame scan
        # (_box_has_face). Candidates are sorted tallest/nearest first and
        # capped at zoom_max_crops_per_frame (default 4) per analysis cycle —
        # a CPU budget limit, not an accuracy one.
        #
        # HOW: crops the top ~45% of the person box (head/shoulders, plus a
        # 20% horizontal margin), upscales it by 1.5x-zoom_max_factor
        # (default up to 4x) toward roughly zoom_target_width_px pixels wide,
        # and re-runs InsightFace detection on just that crop. Any face found
        # has its bbox/landmarks mapped back from crop-and-zoom space into
        # the ORIGINAL frame's native pixel coordinates, then appended to
        # all_faces exactly as if the full-frame scan had found it directly.
        #
        # INTERACTION WITH THE FACE QUALITY GATE (face_quality_min_height_px /
        # face_quality_min_det_score, applied later in this method): this
        # pass only improves DETECTION RECALL — it does not weaken the
        # identity-decision quality bar. Because recovered face coordinates
        # are mapped back to native frame size before the gate ever sees
        # them, a face that's genuinely tiny in the source video still
        # measures as tiny after zoom-recovery and still fails the height
        # gate; the zoom only helps InsightFace's detector NOTICE a face that
        # was already native-resolution large enough to pass the gate, but
        # got lost in the full-frame scan's downscale. A face too small to
        # ever pass the gate gains nothing from being zoomed in on.
        if self._face_rec and tracked:
            try:
                zoom_min_h   = self._identity_config.zoom_min_person_height_px
                zoom_max_n   = self._identity_config.zoom_max_crops_per_frame
                zoom_target  = self._identity_config.zoom_target_width_px
                zoom_max     = self._identity_config.zoom_max_factor

                def _box_has_face(bx, by, bw, bh):
                    for f in all_faces:
                        fcx = (f.bbox[0] + f.bbox[2]) / 2
                        fcy = (f.bbox[1] + f.bbox[3]) / 2
                        if bx <= fcx <= bx + bw and by <= fcy <= by + bh * 0.6:
                            return True
                    return False

                faceless = [b for _, b in tracked
                            if b[3] >= zoom_min_h and not _box_has_face(*b)]
                faceless.sort(key=lambda b: -b[3])
                for bx, by, bw, bh in faceless[:zoom_max_n]:
                    mx  = int(bw * 0.2)
                    cx0 = max(0, bx - mx)
                    cy0 = max(0, by - mx)
                    cx1 = min(fw, bx + bw + mx)
                    cy1 = min(fh, by + int(bh * 0.45))  # head = top of the box
                    crop = frame[cy0:cy1, cx0:cx1]
                    if crop.size == 0:
                        continue
                    zoom = min(zoom_max, max(1.5, zoom_target / crop.shape[1]))
                    zoomed = cv2.resize(crop, None, fx=zoom, fy=zoom)
                    self._face_rec.extract_embedding(zoomed)
                    for f in (getattr(self._face_rec, "_last_faces", []) or []):
                        f.bbox[0] = f.bbox[0] / zoom + cx0
                        f.bbox[2] = f.bbox[2] / zoom + cx0
                        f.bbox[1] = f.bbox[1] / zoom + cy0
                        f.bbox[3] = f.bbox[3] / zoom + cy0
                        kps = getattr(f, "kps", None)
                        if kps is not None:
                            for kp in kps:
                                kp[0] = kp[0] / zoom + cx0
                                kp[1] = kp[1] / zoom + cy0
                        all_faces.append(f)
            except Exception as exc:
                logger.debug("Head-zoom face pass error: %s", exc)
        self._frame_faces = len(all_faces)

        # ── Per-person: identify (once per track), face match, colour, log ──
        persons_out: List[Dict] = []
        seen_tids: set = set()
        db = SessionLocal()
        try:
            now = time.time()
            stale = [c for c, t in self._seen_cache.items() if now - t > 300]
            for c in stale:
                self._seen_cache.pop(c, None)

            # Codes worn by people visible RIGHT NOW — colour/Re-ID must not
            # hand one of these to a second person in the same scene
            current_tids  = {t for t, _ in tracked if t is not None}
            claimed_codes = {
                self._track_codes[t]["code"]
                for t in current_tids if t in self._track_codes
            }
            # A face already matched to an earlier box this frame must not also
            # be handed to a later, overlapping box — see
            # recognition/face_assign.py's docstring for the failure this
            # causes (two person boxes both identified from one stolen face).
            # This loop is a separate implementation from face_assign.py (it
            # also extracts the pose/embedding for identify() inline), so the
            # same `taken`-set fix is applied here directly rather than shared.
            claimed_faces: set = set()

            for tid, bbox in tracked:
                x, y, w, h = bbox
                x2, y2 = x + w, y + h

                # ── Face match FIRST (center inside upper 60% of person box) —
                # its embedding feeds identify(), so identity always comes from
                # THIS person's face and InsightFace runs once per cycle ──────
                face_bbox, face_kps, gender_str = None, None, ""
                matched_face = None
                for face_idx, face_obj in enumerate(all_faces):
                    if face_idx in claimed_faces:
                        continue
                    try:
                        fb = face_obj.bbox.astype(int)
                        fc_x = (fb[0] + fb[2]) / 2
                        fc_y = (fb[1] + fb[3]) / 2
                        head_bottom = y + (y2 - y) * 0.6
                        if x <= fc_x <= x2 and y <= fc_y <= head_bottom:
                            matched_face = face_obj
                            claimed_faces.add(face_idx)
                            break
                    except Exception:
                        continue

                # Quality gate: tiny or low-confidence faces give noisy
                # embeddings that cross-match strangers — draw them, but never
                # let them decide identity (match OR register)
                face_emb = None
                face_pose = None
                if matched_face is not None:
                    try:
                        fb_q = matched_face.bbox
                        face_h = float(fb_q[3] - fb_q[1])
                        det_sc = float(getattr(matched_face, "det_score", 1.0) or 1.0)
                        from recognition.face_pose import estimate_pose
                        from recognition.face_quality import gate_passes
                        _pose = estimate_pose(getattr(matched_face, "kps", None))
                        _box = [int(fb_q[0]), int(fb_q[1]),
                                int(fb_q[2] - fb_q[0]), int(face_h)]
                        # With enable_learned_quality_gate off (default) this is
                        # the identical two-constant test.
                        if gate_passes(_box, det_sc, _pose, self._identity_config,
                                       crop=frame[max(0, _box[1]):_box[1] + _box[3],
                                                  max(0, _box[0]):_box[0] + _box[2]],
                                       person_boxes=[b for _t, b in tracked],
                                       owner_box=bbox):
                            face_emb = getattr(matched_face, "embedding", None)
                    except Exception:
                        face_emb = getattr(matched_face, "embedding", None)
                    # Head pose from the landmarks InsightFace already returned.
                    # Costs no inference; used only by the registration gate.
                    from recognition.face_pose import estimate_pose
                    face_pose = estimate_pose(getattr(matched_face, "kps", None))

                # ── Identify: cached per tracker_id, registration age-gated ──
                code, method, conf = "Detecting...", "pending", 0.0
                label              = "Detecting..."
                color_hex_from_id  = None
                fresh_face_id      = False  # code earned from THIS face this cycle
                if tid is not None:
                    seen_tids.add(tid)
                    self._track_age[tid] = self._track_age.get(tid, 0) + 1
                    cached = self._track_codes.get(tid)

                    # ID-switch guard — see _apply_id_switch_guard() docstring
                    # and IdentityConfig.enable_id_switch_guard.
                    cached = self._apply_id_switch_guard(tid, cached, face_emb, db)

                    if cached:
                        code, method, conf = cached["code"], cached["method"], cached["conf"]
                        label = cached.get("label", code)
                    elif self._voter is not None:
                        # ── Tracklet voting — the live port of the mean
                        # strategy proven offline (recognition/tracklet_vote.py,
                        # ablation config F). Identity commits ONCE per track
                        # from up to `tracklet_buffer_size` quality-weighted
                        # gate-passing face embeddings, at `tracklet_commit_k`
                        # faces or track end (whichever first), instead of on
                        # the first gate-passing frame — a single mediocre
                        # first view can no longer mint a duplicate that lasts
                        # the whole track.
                        #
                        # ONLINE DIVERGENCE FROM THE OFFLINE HARNESS
                        # (inherent, not accidental — documented here): the
                        # harness relabels its scoring records retroactively
                        # on commit, but frames ALREADY SHOWN to an operator
                        # cannot be rewritten — a live track legitimately
                        # displays "Detecting..." until its decision lands.
                        # Database correctness is still restored:
                        # _backfill_tracklet_sightings() below writes the
                        # buffered gate-passing frames' sighting rows under
                        # the committed code (evidence gate re-tested per
                        # frame), so the stored trail is right even though
                        # the on-screen history is not.
                        committed = self._voter.committed_code(tid)
                        if committed:
                            # The ID-switch guard just dropped this track's
                            # committed cache — that decision was wrong, so
                            # the track re-buffers from scratch.
                            self._voter.reset(tid)
                        # Buffer this frame's gate-passing face (if any) with
                        # its quality weight; hold a frame record so the
                        # commit can backfill the database trail.
                        q = 0.0
                        if face_emb is not None and matched_face is not None:
                            try:
                                from recognition.tracklet_vote import quality_score
                                fb_q = matched_face.bbox
                                q = quality_score(
                                    [int(fb_q[0]), int(fb_q[1]),
                                     int(fb_q[2] - fb_q[0]),
                                     int(fb_q[3] - fb_q[1])],
                                    float(getattr(matched_face, "det_score", 1.0) or 1.0),
                                    face_pose, self._identity_config)
                            except Exception:
                                q = 0.0
                        self._voter.observe(tid, face_emb, q,
                                            _BufferedFrameRecord(when=now))
                        if self._voter.ready(tid):
                            # The k-th gate-passing face triggers the commit
                            # on its own frame.
                            c, m = self._voter.commit(
                                tid, self._tracklet_resolve(
                                    tid, frame, bbox, db, claimed_codes))
                            if c and c != "Detecting...":
                                code, method = c, m
                                conf = 1.0
                                fresh_face_id = face_emb is not None
                                label = code
                                try:
                                    prow = db.query(Person).filter(Person.unique_code == code).first()
                                    if prow is not None and getattr(prow, "display_name", None):
                                        label = f"{prow.display_name} ({code})"
                                except Exception:
                                    pass
                                self._track_codes[tid] = {"code": code, "method": method,
                                                          "conf": conf, "label": label}
                                self.active_tracks[tid] = code
                                claimed_codes.add(code)
                                # Retroactive DB correctness: the buffered
                                # frames were displayed as "Detecting..." but
                                # the stored trail must reflect the committed
                                # code. On-screen history is NOT rewritten.
                                self._backfill_tracklet_sightings(tid, code, db)
                    elif self._identifier and self._track_has_face_evidence(tid, face_emb):
                        try:
                            result = self._identifier.identify(
                                frame, bbox, db,
                                location_id=self.location_id,
                                zone_id=self.zone_id,
                                allow_new=self._track_age[tid] >= self._identity_config.min_track_age_for_registration,
                                face_embedding=face_emb,
                                face_pose=self._pose_for_registration(tid, face_pose),
                                exclude_codes=claimed_codes,
                                # full-frame scan already found every face —
                                # a per-person InsightFace rerun just burns CPU
                                extract_face_if_missing=False,
                            )
                            code   = result["unique_code"]
                            method = result["method"]
                            conf   = result["confidence"]
                            color_hex_from_id = result.get("color_hex")
                            self._note_pose_deferral(tid, method)
                            # face/new_registration used face_emb as input, so
                            # the code is confirmed by this person's own face
                            fresh_face_id = (face_emb is not None and
                                             method in ("face", "new_registration"))
                            if code != "Detecting...":
                                # Named enrollment: label shows the person's
                                # name once an operator has set one
                                label = code
                                try:
                                    prow = db.query(Person).filter(Person.unique_code == code).first()
                                    if prow is not None and getattr(prow, "display_name", None):
                                        label = f"{prow.display_name} ({code})"
                                except Exception:
                                    pass
                                self._track_codes[tid] = {"code": code, "method": method,
                                                          "conf": conf, "label": label}
                                self.active_tracks[tid] = code
                                claimed_codes.add(code)
                        except Exception as exc:
                            logger.debug("SmartIdentifier error: %s", exc)

                    # Movement trail (one point per analysis cycle)
                    trail = self._track_trails.setdefault(tid, deque(maxlen=_TRAIL_LENGTH))
                    trail.append((x + w // 2, y2))

                if matched_face is not None and tid is not None:
                    raw_bbox = matched_face.bbox.copy()
                    hist = self._face_bbox_history.setdefault(tid, [])
                    hist.append(raw_bbox)
                    self._face_bbox_history[tid] = hist[-5:]
                    face_bbox = np.mean(self._face_bbox_history[tid], axis=0).astype(int).tolist()

                    kps = getattr(matched_face, "kps", None)
                    if kps is not None and len(kps) >= 5:
                        khist = self._face_kps_history.setdefault(tid, [])
                        khist.append(kps[:5].copy())
                        self._face_kps_history[tid] = khist[-5:]
                        face_kps = np.mean(self._face_kps_history[tid], axis=0).astype(int).tolist()

                    g_val = getattr(matched_face, "gender", None)
                    if g_val is not None:
                        gender_str = "M" if int(g_val) == 1 else "F"

                # ── Dress colour (torso crop, K-means) ──────────────────────
                color_hex = color_hex_from_id or ""
                if not color_hex:
                    try:
                        torso_crop = frame[
                            max(0, y + int(h * 0.30)): min(fh, y + int(h * 0.70)),
                            max(0, x): min(fw, x2),
                        ]
                        if torso_crop.size > 100:
                            dom = self._dominant_color(torso_crop)
                            if dom is not None:
                                r, g_c, b = dom
                                color_hex = f"#{r:02x}{g_c:02x}{b:02x}"
                    except Exception:
                        pass

                # ── Height estimate ─────────────────────────────────────────
                height_ratio = h / max(fh, 1)
                if height_ratio > 0.6:
                    height_label = "~Tall"
                elif height_ratio > 0.35:
                    height_label = "~Medium"
                else:
                    height_label = "~Short"

                # ── Link bags to this person ────────────────────────────────
                carrying = None
                for obj in object_dets:
                    ox, oy, ow, oh = obj["bbox"]
                    overlap_x = max(0, min(x2, ox + ow) - max(x, ox))
                    overlap_y = max(0, min(y2, oy + oh) - max(y, oy))
                    overlaps  = overlap_x > 0 and overlap_y > 0
                    dist = math.sqrt((ox + ow // 2 - (x + w // 2)) ** 2
                                     + (oy + oh // 2 - (y + h // 2)) ** 2)
                    if overlaps or dist < 50:
                        carrying = _OBJECT_LABELS.get(obj["label"], "object")
                        break

                # ── Log sighting (deduped) — see _evidence_gate_ok() docstring
                # and IdentityConfig.enable_evidence_gating ─────────────────
                evidence_gate_ok = self._evidence_gate_ok(tid, code, face_emb, fresh_face_id, db)
                if (code and code != "Detecting..."
                        and evidence_gate_ok):
                    if now - self._seen_cache.get(code, 0) > 30:
                        self._seen_cache[code] = now
                        self.persons_today += (1 if method == "new_registration" else 0)
                        snap_path = self._save_sighting_snapshot(
                            code, frame[max(0, y):min(fh, y2), max(0, x):min(fw, x2)]
                        )
                        try:
                            log_sighting(
                                unique_code=code,
                                location_id=self.location_id,
                                zone_id=self.zone_id,
                                camera_id=self.camera_id,
                                confidence=conf,
                                db=db,
                                frame_path=snap_path,
                            )
                            check_watchlist_alert(
                                unique_code=code,
                                camera_id=self.camera_id,
                                location_id=self.location_id,
                                db=db,
                            )
                        except Exception as exc:
                            logger.debug("log_sighting error: %s", exc)

                    self._detections.append({
                        "unique_code": code,
                        "method":      method,
                        "confidence":  conf,
                        "color_hex":   color_hex or None,
                        "gender":      gender_str or None,
                        "height":      height_label,
                        "detected_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                        "zone_id":     self.zone_id,
                        "camera_id":   self.camera_id,
                    })

                persons_out.append({
                    "tracker_id":   tid,
                    "bbox":         [x, y, w, h],
                    "code":         code,
                    "label":        label,
                    "method":       method,
                    "conf":         conf,
                    "color_hex":    color_hex,
                    "gender":       gender_str,
                    "height_label": height_label,
                    "face_bbox":    face_bbox,
                    "face_kps":     face_kps,
                    "carrying":     carrying,
                })
        finally:
            db.close()

        # ── Drop state for tracks that disappeared ──────────────────────────
        # A disappeared track is a track END. Tracklet voting must decide it
        # here (commit from whatever accumulated — a track that ended below
        # k faces still gets its one decision) BEFORE the per-track state is
        # dropped; the commit writes sighting rows under the earned code.
        dead_tids = [t for t in self._track_age if t not in seen_tids]
        if dead_tids and self._voter is not None:
            db_dead = SessionLocal()
            try:
                for tid in dead_tids:
                    try:
                        c, m = self._voter.end_track(
                            tid, self._tracklet_resolve(tid, None, None,
                                                        db_dead, None))
                        if c and c != "Detecting...":
                            # keep the code so a returning ByteTrack id
                            # re-uses it (same policy as the cached path)
                            self._track_codes[tid] = {"code": c, "method": m,
                                                      "conf": 1.0, "label": c}
                            self._backfill_tracklet_sightings(tid, c, db_dead)
                    except Exception as exc:
                        logger.debug("track-end commit failed for %s: %s",
                                     tid, exc)
                    finally:
                        self._voter.forget(tid)
            finally:
                db_dead.close()
        for tid in dead_tids:
            self._track_age.pop(tid, None)
            self._track_trails.pop(tid, None)
            self._face_bbox_history.pop(tid, None)
            self._face_kps_history.pop(tid, None)
            self.active_tracks.pop(tid, None)
            # keep self._track_codes so a returning ByteTrack id keeps its code

        objects_out = [{"bbox": o["bbox"], "label": _OBJECT_LABELS.get(o["label"], o["label"]),
                        "is_bag": o["label"] in _BAG_CLASSES} for o in object_dets]

        with self._results_lock:
            self._latest_results = {"persons": persons_out, "objects": objects_out}

    # ─────────────────────────────────────────────────────────────────────────
    # Annotation — runs in the capture loop using cached analysis results
    # ─────────────────────────────────────────────────────────────────────────

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame  # draw in place; capture loop owns this frame
        fh, fw = frame.shape[:2]

        # Box/text sizes below were tuned for a ~640x480 webcam frame. A 4K
        # upload is 6x wider — fixed 2px strokes and 0.5 font scale become
        # nearly invisible at that resolution, making working detection look
        # like nothing is happening. Scale everything to the actual frame size.
        s = max(fw, fh) / 640.0
        thick      = max(2, round(2 * s))
        thick_fill = max(2, round(3 * s))
        font_lg    = 0.50 * s
        font_md    = 0.45 * s
        font_sm    = 0.35 * s
        font_xs    = 0.32 * s
        pad        = round(6 * s)
        sq_size    = round(14 * s)
        kp_radius  = max(2, round(3 * s))

        with self._results_lock:
            results = self._latest_results
        persons = results["persons"]
        objects = results["objects"]

        # ── Bag / object boxes ────────────────────────────────────────────
        for obj in objects:
            ox, oy, ow, oh = obj["bbox"]
            color = _ORANGE if obj["is_bag"] else _YELLOW
            label = obj["label"]
            cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), color, thick)
            (tw, th), _ = cv2.getTextSize(label, _FONT, font_md, 1)
            cv2.rectangle(annotated, (ox, oy - th - pad), (ox + tw + pad, oy), color, -1)
            cv2.putText(annotated, label, (ox + 2, oy - 3), _FONT, font_md, _BLACK, 1)

        # ── Movement trails ───────────────────────────────────────────────
        for p in persons:
            tid = p["tracker_id"]
            trail = self._track_trails.get(tid) if tid is not None else None
            if trail and len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [pts], False, _BOX_COLORS.get(p["method"], _GREY), thick)

        # ── Person boxes + labels ─────────────────────────────────────────
        for p in persons:
            x, y, w, h = p["bbox"]
            x2, y2 = x + w, y + h
            code, method, conf = p["code"], p["method"], p["conf"]
            color = _BOX_COLORS.get(method, _GREY if method == "pending" else _WHITE)

            cv2.rectangle(annotated, (x, y), (x2, y2), color, thick_fill)

            # Face box + landmarks
            if p["face_bbox"]:
                fx1, fy1, fx2, fy2 = p["face_bbox"]
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), _CYAN, thick)
                cv2.putText(annotated, "Face", (fx1, int(fy1 - 8 * s)), _FONT, font_lg, _CYAN, 1)
                if p["face_kps"]:
                    landmark_colors = [
                        (255, 0,   0),    # left eye
                        (0,   255, 0),    # right eye
                        (0,   0,   255),  # nose
                        (255, 255, 0),    # left mouth
                        (0,   255, 255),  # right mouth
                    ]
                    for i, kp in enumerate(p["face_kps"][:5]):
                        cv2.circle(annotated, (int(kp[0]), int(kp[1])), kp_radius, landmark_colors[i], -1)

            # Dress colour square + hex
            if p["color_hex"]:
                try:
                    hexv = p["color_hex"].lstrip("#")
                    r, g_c, b = int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)
                    sq_x, sq_y = x + 3, y2 - sq_size - 3
                    cv2.rectangle(annotated, (sq_x, sq_y),
                                  (sq_x + sq_size, sq_y + sq_size), (b, g_c, r), -1)
                    cv2.rectangle(annotated, (sq_x, sq_y),
                                  (sq_x + sq_size, sq_y + sq_size), _WHITE, thick)
                    cv2.putText(annotated, p["color_hex"],
                                (sq_x + sq_size + 3, sq_y + sq_size - 2),
                                _FONT, font_xs, _WHITE, 1)
                except Exception:
                    pass

            # Name/SDT label + gender above box
            id_text = p.get("label") or code
            if p["gender"]:
                id_text += f" {p['gender']}"
            (tw, th_t), _ = cv2.getTextSize(id_text, _FONT, font_lg, thick)
            cv2.rectangle(annotated, (x, y - th_t - pad - 2), (x + tw + pad, y), color, -1)
            cv2.putText(annotated, id_text, (x + 3, y - 4), _FONT, font_lg, _BLACK, thick)

            # Method + confidence below box
            method_short = {
                "face": "Face", "dress_color": "Color",
                "body_structure": "Body",
                "new_registration": "New", "pending": "..."
            }.get(method, method)
            cv2.putText(annotated, f"{method_short} {int(conf * 100)}%",
                        (x, int(y2 + 14 * s)), _FONT, font_sm, color, thick)

            # Height label on right side of box
            cv2.putText(annotated, p["height_label"],
                        (int(x2 - 55 * s), y2 - 5), _FONT, font_sm, _WHITE, thick)

            if p["carrying"]:
                cv2.putText(annotated, f"carrying {p['carrying']}",
                            (x, int(y2 + 28 * s)), _FONT, font_sm, _ORANGE, thick)

        # ── HUD overlay ───────────────────────────────────────────────────
        self._draw_hud(annotated, fh, fw)
        return annotated

    # ─────────────────────────────────────────────────────────────────────────
    # HUD Overlay
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_hud(self, annotated: np.ndarray, fh: int, fw: int) -> None:
        """Draw on-screen stats overlay."""
        fps = self._fps_value
        s = max(fw, fh) / 640.0  # see _annotate_frame — same webcam-tuned baseline

        # ── Top-left: camera info ─────────────────────────────────────────
        hud_top = f"SmartDetect | {self.camera_id} | LIVE"
        (hw, hh), _ = cv2.getTextSize(hud_top, _FONT, 0.55 * s, max(1, round(2 * s)))
        cv2.rectangle(annotated, (0, 0), (hw + round(14 * s), hh + round(12 * s)), (0, 0, 0), -1)
        cv2.putText(annotated, hud_top, (round(7 * s), hh + round(5 * s)), _FONT, 0.55 * s, _WHITE, max(1, round(s)))

        # ── Bottom-left: stats ────────────────────────────────────────────
        stats = (f"Persons: {self._frame_persons} | Faces: {self._frame_faces} | "
                 f"Bags: {self._frame_bags} | FPS: {fps:.0f} | ML: {self._analysis_fps:.1f}/s")
        (sw, sh), _ = cv2.getTextSize(stats, _FONT, 0.45 * s, max(1, round(s)))
        by = fh - round(10 * s)
        cv2.rectangle(annotated, (0, by - sh - round(8 * s)), (sw + round(14 * s), fh), (0, 0, 0), -1)
        cv2.putText(annotated, stats, (round(7 * s), by - 2), _FONT, 0.45 * s, _WHITE, max(1, round(s)))

        # ── Bottom-right: timestamp ───────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (tsw, tsh), _ = cv2.getTextSize(ts, _FONT, 0.42 * s, max(1, round(s)))
        cv2.rectangle(annotated, (fw - tsw - round(14 * s), fh - tsh - round(12 * s)), (fw, fh), (0, 0, 0), -1)
        cv2.putText(annotated, ts, (fw - tsw - round(7 * s), fh - round(6 * s)), _FONT, 0.42 * s, _GREY, max(1, round(s)))

        # ── Colour legend (top-right) ─────────────────────────────────────
        legend_items = [
            (_GREEN,  "Face ID"),
            (_BLUE,   "Body ID"),
            (_WHITE,  "New"),
            (_ORANGE, "Bag/Obj"),
        ]
        lx = fw - round(100 * s)
        ly = round(8 * s)
        box = round(10 * s)
        step = round(16 * s)
        for lcolor, ltext in legend_items:
            cv2.rectangle(annotated, (lx, ly), (lx + box, ly + box), lcolor, -1)
            cv2.putText(annotated, ltext, (lx + box + round(4 * s), ly + box - 1), _FONT, 0.33 * s, _WHITE, max(1, round(s)))
            ly += step

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dominant_color(bgr_crop: np.ndarray, k: int = 3) -> Optional[Tuple[int, int, int]]:
        """Extract dominant colour from a BGR crop using K-means. Returns (R,G,B)."""
        try:
            pixels = bgr_crop.reshape(-1, 3).astype(np.float32)
            k = min(k, len(pixels))
            if k < 1:
                return None
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            dominant_bgr = centers[np.argmax(counts)].astype(np.uint8)
            return (int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0]))  # RGB
        except Exception:
            return None

    _MAX_SNAPSHOTS_PER_PERSON = 20

    def _face_sim_to_code(self, face_emb, code: str, db) -> Optional[float]:
        """Max cosine similarity between a face embedding and a person's
        stored face template + gallery. None when nothing is stored."""
        try:
            person = db.query(Person).filter(Person.unique_code == code).first()
            if person is None or not person.face_embedding:
                return None
            templates = [json.loads(person.face_embedding)]
            try:
                templates += json.loads(person.face_templates) if person.face_templates else []
            except Exception:
                pass
            q = np.asarray(face_emb, dtype=np.float32)
            qn = np.linalg.norm(q) + 1e-8
            best = -1.0
            for t in templates:
                tv = np.asarray(t, dtype=np.float32)
                if tv.shape != q.shape:
                    continue
                best = max(best, float(np.dot(q, tv) / (qn * (np.linalg.norm(tv) + 1e-8))))
            return best if best > -1.0 else None
        except Exception as exc:
            logger.debug("face-sim-to-code failed: %s", exc)
            return None

    def _pose_for_registration(self, tid, pose):
        """
        Pose to hand to identify(), honouring the starvation safety valve.

        Returns None once a track has been pose-deferred
        pose_gate_max_deferrals times in a row, which makes the gate fail open
        for that track (see pose_ok_for_registration). Without this, a person
        who is never frontal — or sparsely-sampled input — would never enrol.
        """
        limit = getattr(self._identity_config, "pose_gate_max_deferrals", 0)
        if tid is None or limit <= 0:
            return pose
        if self._track_pose_defer.get(tid, 0) >= limit:
            return None          # valve open: accept the best frame available
        return pose

    def _note_pose_deferral(self, tid, method: str) -> None:
        """Count consecutive pose deferrals; any other outcome resets."""
        if tid is None:
            return
        if method == "pending_pose":
            self._track_pose_defer[tid] = self._track_pose_defer.get(tid, 0) + 1
        else:
            self._track_pose_defer.pop(tid, None)

    def _apply_id_switch_guard(self, tid, cached: Optional[Dict], face_emb, db) -> Optional[Dict]:
        """
        ID-switch guard: a gate-passing face that contradicts the cached
        code means the tracker likely handed this track to a different
        person (occlusion swap). id_switch_contradiction_limit consecutive
        contradictions (face similarity to the cached code's stored face
        below id_switch_similarity_threshold) → drop the cache and force
        re-identification from scratch.

        Returns the (possibly None) cached dict — None means "guard
        dropped it, re-identify". Gated by IdentityConfig.enable_id_switch_guard;
        when off, `cached` is returned unchanged (trust the cache
        unconditionally — pre-guard behavior) and the mismatch-strike
        counter is not touched.

        Extracted from _analyze_frame as its own method so
        enable_id_switch_guard's effect is independently testable
        (see scripts/verify_flags.py) without needing to run a full frame
        through the capture/analysis pipeline.
        """
        if self._identity_config.enable_id_switch_guard and cached and face_emb is not None:
            sim = self._face_sim_to_code(face_emb, cached["code"], db)
            if sim is not None and sim < self._identity_config.id_switch_similarity_threshold:
                strikes = self._track_face_mismatch.get(tid, 0) + 1
                self._track_face_mismatch[tid] = strikes
                if strikes >= self._identity_config.id_switch_contradiction_limit:
                    logger.info(
                        "ID-switch suspected on track %s: face sim %.3f "
                        "to cached %s — re-identifying", tid, sim, cached["code"])
                    self._track_codes.pop(tid, None)
                    self.active_tracks.pop(tid, None)
                    self._track_face_mismatch[tid] = 0
                    cached = None
            else:
                self._track_face_mismatch[tid] = 0
        return cached

    def _track_has_face_evidence(self, tid: int, face_emb) -> bool:
        """
        May this track's box take an identity from identify() this frame?

        True when a face is visible right now, or when the track has shown a
        face at some earlier point (so a person who turns away keeps their
        colour/Re-ID re-association). False for a track that NEVER produced a
        face — a YOLO false positive such as a chair — which must stay
        "Detecting..." instead of being stamped with a stranger's code by the
        colour/Re-ID fallback.
        """
        return face_emb is not None or tid in self._face_bbox_history

    def _tracklet_resolve(self, tid, frame, bbox, db, exclude_codes):
        """
        Production arbitration for ONE tracklet-mean embedding: match, else
        enrol — the same identify() call the non-voting path makes, so a
        tracklet commit can never mint an identity the live path would not.

        The mean embedding has no single pose, so the registration pose gate
        fails open (face_pose=None), exactly as in the offline harness; the
        track-age gate on allow_new still applies, so a sub-3-cycle track
        ending early cannot mint an identity from a glimpse.

        Returns a closure (the TrackletVoter calls it with the embedding).
        """
        if self._identifier is None:
            return lambda emb: ("Detecting...", "pending")
        import numpy as _np
        frame_for_crop = frame if frame is not None \
            else _np.zeros((8, 8, 3), dtype=_np.uint8)
        bbox_for_crop = bbox if bbox is not None else [0, 0, 8, 8]
        exclude = exclude_codes or set()

        def resolve(emb):
            try:
                result = self._identifier.identify(
                    frame_for_crop, bbox_for_crop, db,
                    location_id=self.location_id,
                    zone_id=self.zone_id,
                    allow_new=(self._track_age.get(tid, 0)
                               >= self._identity_config.min_track_age_for_registration),
                    face_embedding=emb,
                    face_pose=None,          # mean embedding has no pose
                    exclude_codes=exclude,
                    extract_face_if_missing=False,
                )
                return result["unique_code"], result["method"]
            except Exception as exc:
                logger.debug("tracklet resolve failed for track %s: %s",
                             tid, exc)
                return "Detecting...", "pending"
        return resolve

    def _backfill_tracklet_sightings(self, tid, code: str, db) -> None:
        """
        Retroactive DATABASE correctness for tracklet voting.

        Frames buffered before commit were displayed as "Detecting..." and
        wrote no sighting row. On commit the stored trail must reflect the
        code the track actually earned: write one sighting row per buffered
        gate-passing frame whose embedding individually agrees with the
        committed code at the evidence threshold — the same per-frame test
        the live evidence gate applies. The write-rate limiter (30 s per
        code, the production `_seen_cache` policy) is respected using each
        frame's OWN processing time, so the backfill yields exactly the rows
        production would have written had the code been known at the time.

        On-screen history is deliberately NOT rewritten — the operator saw
        "Detecting...", and `_detections` keeps only what was displayed.
        Backfilled rows carry no photo: those frames' crops were never
        retained, and photo evidence is written only for face-confirmed
        frames from commit onward.
        """
        if self._voter is None or not self._identity_config.enable_evidence_gating:
            return
        try:
            for o in self._voter.buffered(tid):
                if o.emb is None:
                    continue
                sim = self._face_sim_to_code(o.emb, code, db)
                if sim is None or sim < self._identity_config.evidence_face_sim_threshold:
                    continue
                when = getattr(o.record, "when", None) or time.time()
                if when - self._seen_cache.get(code, 0.0) <= 30:
                    continue
                self._seen_cache[code] = when
                try:
                    log_sighting(unique_code=code,
                                 location_id=self.location_id,
                                 zone_id=self.zone_id,
                                 camera_id=self.camera_id,
                                 confidence=1.0, db=db, frame_path=None)
                except Exception as exc:
                    logger.debug("tracklet backfill sighting failed: %s", exc)
        except Exception as exc:
            logger.debug("tracklet backfill failed for track %s: %s", tid, exc)

    def _commit_open_voting_tracks(self) -> int:
        """
        Commit every still-open voting track (camera stopped / video EOF).

        The offline harness flushes its voter at sequence end; the live
        equivalent is the stream ending. Without this, a person still in
        frame when the camera stops would lose their earned code and the
        buffered frames' sighting rows entirely. Returns how many tracks
        committed.
        """
        if self._voter is None:
            return 0
        n = 0
        db = SessionLocal()
        try:
            for tid in self._voter.track_ids():
                try:
                    c, m = self._voter.end_track(
                        tid, self._tracklet_resolve(tid, None, None, db, None))
                    if c and c != "Detecting...":
                        self._track_codes[tid] = {"code": c, "method": m,
                                                  "conf": 1.0, "label": c}
                        self._backfill_tracklet_sightings(tid, c, db)
                        n += 1
                except Exception as exc:
                    logger.debug("voting commit at stream end failed for %s: %s",
                                 tid, exc)
                finally:
                    self._voter.forget(tid)
        finally:
            db.close()
        return n

    def _evidence_gate_ok(self, tid, code: str, face_emb, fresh_face_id: bool, db) -> bool:
        """
        Whether a sighting row / photo may be filed for `code` this cycle.

        A track with ID-switch contradiction strikes is mid-switch — its
        code is suspect, so no evidence until it re-verifies. Otherwise,
        evidence requires a gate-passing face agreeing with the code (or
        the code was freshly earned from that face this cycle) — a
        faceless box that inherited a code (cached track, colour/re-ID
        re-association) can keep its live on-screen label but cannot
        write evidence under someone else's code.

        Gated by IdentityConfig.enable_evidence_gating; when off, every
        non-"Detecting..." code is evidence-eligible unconditionally
        (pre-gating behavior).

        Extracted from _analyze_frame as its own method so
        enable_evidence_gating's effect is independently testable (see
        scripts/verify_flags.py).
        """
        if not self._identity_config.enable_evidence_gating:
            return True
        identity_in_doubt = tid is not None and self._track_face_mismatch.get(tid, 0) > 0
        face_confirms = fresh_face_id
        if not face_confirms and code and code != "Detecting..." and face_emb is not None:
            sim_ev = self._face_sim_to_code(face_emb, code, db)
            face_confirms = (sim_ev is not None
                             and sim_ev >= self._identity_config.evidence_face_sim_threshold)
        return not identity_in_doubt and face_confirms

    def _save_sighting_snapshot(self, code: str, crop: np.ndarray) -> Optional[str]:
        """Save a person crop for this sighting; keep only the newest files."""
        try:
            if crop is None or crop.size == 0:
                return None
            snap_dir = Path("snapshots") / code
            snap_dir.mkdir(parents=True, exist_ok=True)
            fname = snap_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            if not cv2.imwrite(str(fname), crop):
                return None
            # Prune oldest sighting shots (never registered.jpg)
            shots = sorted(p for p in snap_dir.glob("*.jpg") if p.name != "registered.jpg")
            for old in shots[:-self._MAX_SNAPSHOTS_PER_PERSON]:
                try:
                    old.unlink()
                except OSError:
                    pass
            return fname.as_posix()
        except Exception as exc:
            logger.debug("sighting snapshot failed: %s", exc)
            return None

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()
