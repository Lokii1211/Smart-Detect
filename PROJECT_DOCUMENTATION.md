# SmartDetect — Metro Person Tracking System
## Complete Project Documentation

> **Purpose**: This document contains every detail of the SmartDetect project so you can understand, run, modify, and debug it from scratch — even without conversation history.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [ML Pipeline — How Detection Works](#4-ml-pipeline--how-detection-works)
5. [Backend API (FastAPI)](#5-backend-api-fastapi)
6. [Camera Processing](#6-camera-processing)
7. [Dashboard (React)](#7-dashboard-react)
8. [Database Schema](#8-database-schema)
9. [Environment & Configuration](#9-environment--configuration)
10. [How to Run](#10-how-to-run)
11. [Key Technical Decisions & Optimization History](#11-key-technical-decisions--optimization-history)
12. [Troubleshooting](#12-troubleshooting)
13. [Git & GitHub](#13-git--github)
14. [File-by-File Reference](#14-file-by-file-reference)
15. [Dependencies](#15-dependencies)

---

## 1. Project Overview

**SmartDetect** is an AI-powered real-time person tracking and identification system designed for metro stations. It uses:

- **YOLOv8** for person detection
- **DeepSORT** for multi-object tracking (stable IDs across frames)
- **InsightFace (ArcFace)** for face recognition (512-dim embeddings)
- **OSNet (torchreid)** for person re-identification via body features
- **FastAPI** backend with REST APIs
- **React + Vite** dashboard with live MJPEG streaming
- **SQLite** database (with optional PostgreSQL + pgvector)

### What it does:
1. Opens a webcam or RTSP camera feed
2. Detects all persons in the frame (YOLO)
3. Tracks each person across frames (DeepSORT → stable track IDs)
4. Identifies each person using a 4-method cascade:
   - Face recognition (ArcFace embedding match)
   - Dress color matching (K-means torso crop, HSV distance)
   - Body Re-ID (OSNet embedding match)
   - Multi-feature weighted combination
5. If no match → auto-registers a new person as `SDT-XXXX`
6. Logs sightings to the database with location, zone, camera, confidence
7. Streams annotated video to the dashboard via MJPEG
8. Shows live analytics, person trails, photo search on the dashboard

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Dashboard (Vite)                     │
│              http://localhost:5173                            │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────┐  │
│  │Dashboard │ │Photo      │ │Live      │ │Locations      │  │
│  │(stats,   │ │Search     │ │Camera    │ │(manage        │  │
│  │ recent   │ │(upload    │ │(MJPEG    │ │ stations,     │  │
│  │ detect.) │ │ face →    │ │ stream,  │ │ cameras)      │  │
│  │          │ │ find SDT) │ │ start/   │ │               │  │
│  │          │ │           │ │ stop)    │ │               │  │
│  └──────────┘ └───────────┘ └──────────┘ └───────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / MJPEG
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend                             │
│              http://localhost:8000                            │
│                                                              │
│  /auth/login          → JWT token                            │
│  /health              → status check                         │
│  /register            → register person (base64 image)       │
│  /sighting            → log a sighting                       │
│  /person/{code}/trail → movement history                     │
│  /cameras             → list cameras by location             │
│  /camera/start        → start LiveStream (ML + MJPEG)        │
│  /camera/stream/{id}  → MJPEG video stream                   │
│  /search/by-photo     → face search across database          │
│  /analytics/count/live→ live person count                    │
│                                                              │
│  active_streams: Dict[str, LiveStream]                       │
│  (max 4 simultaneous cameras)                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ SQLAlchemy
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     SQLite Database                           │
│                     metro.db                                  │
│                                                              │
│  Tables: persons, sightings, locations, cameras              │
│  WAL mode + 30s busy timeout for concurrency                 │
└─────────────────────────────────────────────────────────────┘
```

### Two Camera Modes

The system has **two separate camera processing systems**:

#### Mode 1: `CameraProcessor` (standalone, high-FPS)
- File: `cameras/camera_processor.py`
- Run via: `python scripts/run_camera.py --camera CAM-001 --station entrance --source 0`
- Uses **background threads** for YOLO/DeepSORT (no FPS impact)
- Uses **InsightFace cached face detection** with motion offset
- Shows OpenCV window locally with annotations
- Reports sightings to the backend API via HTTP
- **Best for: local real-time monitoring with 15-20 FPS**

#### Mode 2: `LiveStream` (backend-integrated, dashboard streaming)
- File: `cameras/live_stream.py`
- Started via: `POST /camera/start` API endpoint
- Runs ALL ML models in single thread (lower FPS ~1-5)
- Streams MJPEG to dashboard via `GET /camera/stream/{id}`
- **Best for: dashboard live view**

---

## 3. Directory Structure

```
C:\Project\
├── backend/                    # FastAPI backend
│   ├── __init__.py
│   ├── main.py                 # 850 lines — all API routes
│   ├── auth.py                 # JWT authentication (operator/admin roles)
│   └── logger.py               # Structured JSON logger
│
├── cameras/                    # Camera processing
│   ├── __init__.py
│   ├── camera_processor.py     # 687 lines — standalone high-FPS processor
│   └── live_stream.py          # 615 lines — backend-integrated MJPEG stream
│
├── recognition/                # ML models
│   ├── __init__.py
│   ├── object_detector.py      # YOLOv8 person detector (180 lines)
│   ├── face_recognizer.py      # InsightFace ArcFace (213 lines)
│   ├── reid_model.py           # OSNet Re-ID (175 lines)
│   ├── smart_identifier.py     # 4-method identification cascade (251 lines)
│   └── registration.py         # Person registration logic
│
├── tracker/                    # Multi-object tracking
│   ├── __init__.py
│   └── deepsort_tracker.py     # DeepSORT wrapper (205 lines)
│
├── database/                   # Database layer
│   ├── __init__.py
│   ├── db.py                   # SQLAlchemy engine (SQLite + WAL)
│   ├── models.py               # ORM models (Person, Sighting, Location, Camera)
│   └── queries.py              # Query functions (find_by_embedding, log_sighting, etc.)
│
├── dashboard/                  # React frontend
│   ├── src/
│   │   ├── App.jsx             # Router + layout shell
│   │   ├── main.jsx            # Entry point
│   │   ├── index.css           # Global styles (TailwindCSS + custom)
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx   # Main stats + recent detections
│   │   │   ├── LiveCamera.jsx  # Camera management + MJPEG stream
│   │   │   ├── PhotoSearch.jsx # Upload photo → search by face
│   │   │   └── Locations.jsx   # Station/location management
│   │   └── components/
│   │       ├── Sidebar.jsx     # Navigation sidebar
│   │       ├── TopBar.jsx      # Page title + system status
│   │       ├── StatsPanel.jsx  # Dashboard metric cards
│   │       ├── SightingCard.jsx# Detection card component
│   │       ├── SearchBar.jsx   # Search input
│   │       ├── RegisterPerson.jsx # Person registration form
│   │       ├── PersonTrail.jsx # Movement trail viewer
│   │       ├── LiveCamera.jsx  # Camera stream component
│   │       └── CameraConnect.jsx # Camera connection manager
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js          # Vite 8 config (preserveSymlinks)
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── scripts/                    # Utility scripts
│   ├── run_camera.py           # Launch CameraProcessor standalone
│   ├── start_dashboard_cam.py  # Seed location + start camera for dashboard
│   ├── demo_setup.py           # Seed demo data (locations, persons)
│   ├── seed_db.py              # Seed database
│   ├── seed_stations.py        # Seed metro stations
│   ├── e2e_test.py             # End-to-end tests
│   ├── accuracy_test.py        # ML accuracy testing
│   └── load_test.py            # Load/stress testing
│
├── docker/                     # Docker config
│   ├── backend.Dockerfile
│   └── init.sql                # PostgreSQL init script
│
├── models/                     # ML model package (weights excluded from git)
│   └── __init__.py
│
├── .env                        # Environment variables
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker multi-service setup
├── README.md                   # Project README
└── PROJECT_DOCUMENTATION.md    # THIS FILE
```

---

## 4. ML Pipeline — How Detection Works

### 4.1 Person Detection (YOLO)

**File**: `recognition/object_detector.py`

```python
# Key settings:
model = YOLO("yolov8n.pt")  # nano model for speed
results = model.predict(
    source=frame,
    conf=0.30,       # low threshold to catch partial/sitting people
    iou=0.45,        # NMS IoU threshold
    imgsz=416,       # input resolution (higher = better boxes)
    classes=[0],     # person class only
    verbose=False,
)
```

**Output**: List of `{"label": "person", "bbox": [x, y, w, h], "confidence": 0.85}`

- Returns bounding boxes in `[x, y, width, height]` format
- Also detects bags, bottles for "carrying object" annotation
- Model file `yolov8n.pt` is ~6.5MB (excluded from git, auto-downloads on first run)

### 4.2 Multi-Object Tracking (DeepSORT)

**File**: `tracker/deepsort_tracker.py`

```python
# Initialized without embedder (IoU-only mode for speed):
DeepSort(
    max_age=10,         # frames before track is deleted
    n_init=3,           # detections before track is confirmed
    max_iou_distance=0.7,
    embedder=None,      # no CNN embedder — we provide bbox-derived embeddings
)
```

**Key fix**: When `ds_input` is empty (no detections), always pass `embeds=embeds` (empty list), never `None`. Passing `None` causes "Embedder not created" error.

```python
# CORRECT:
tracks = self._tracker.update_tracks(ds_input, frame=frame, embeds=embeds)

# WRONG (was causing crashes):
# tracks = self._tracker.update_tracks(ds_input, frame=frame, embeds=embeds if embeds else None)
```

**Output**: List of `{"track_id": 5, "bbox": [x1, y1, x2, y2], "confidence": 0.9}`

### 4.3 Face Recognition (InsightFace)

**File**: `recognition/face_recognizer.py`

```python
FaceAnalysis(name="buffalo_l")  # 512-dim ArcFace embeddings
face_app.prepare(ctx_id=-1, det_size=(160, 160), det_thresh=0.35)
# ctx_id=-1 = CPU, ctx_id=0 = GPU (requires onnxruntime-gpu)
```

- Uses `buffalo_l` model (largest InsightFace model)
- Returns: face bounding box, 5 keypoints (eyes, nose, mouth corners), gender, 512-dim embedding
- Falls back to `_FaceStub` when insightface not installed
- Set `SMARTDETECT_STUB_MODE=1` for testing without real ML

### 4.4 Person Re-ID (OSNet)

**File**: `recognition/reid_model.py`

- Uses `osnet_x1_0` from torchreid (2.1M params)
- Extracts 512-dim body appearance features
- Falls back to color histogram stub when torchreid unavailable
- Used when face is not visible/detectable

### 4.5 Smart Identification Cascade

**File**: `recognition/smart_identifier.py`

**Priority order** (stops at first confident match):

| Priority | Method | Model | Threshold | Description |
|----------|--------|-------|-----------|-------------|
| 1 | **Face** | InsightFace ArcFace | cosine ≥ 0.72 | Most reliable |
| 2 | **Dress Color** | K-means torso crop | HSV distance ≤ 30 | Fast, short-term |
| 3 | **Body Re-ID** | OSNet embedding | cosine ≥ 0.78 | When face hidden |
| 4 | **Multi-feature** | Weighted combo | combined ≥ 0.65 | Last resort |
| 5 | **New Registration** | — | — | Auto-assigns `SDT-XXXX` |

### 4.6 Annotations Drawn on Frame

| Element | Color | Description |
|---------|-------|-------------|
| Person box | GREEN `(0, 210, 80)` | YOLO bounding box |
| Face box | CYAN `(255, 220, 0)` | InsightFace face detection |
| Face keypoints | Multi-color | 5 points: eyes, nose, mouth |
| SDT label | Above person box | `SDT-0001 M` (code + gender) |
| Method label | Below person box | `Face 85%` or `Color 72%` |
| Dress color | Small square | Dominant torso color swatch |
| Height estimate | Right side | `~Tall`, `~Medium`, `~Short` |
| Bag/object | ORANGE | `carrying Bag` annotation |
| HUD | Top/bottom bars | Camera info, FPS, timestamp |

---

## 5. Backend API (FastAPI)

**File**: `backend/main.py` (850 lines)

### 5.1 Authentication

```
POST /auth/login
Body: {"username": "operator", "password": "metroOp2024"}
Response: {"access_token": "eyJ...", "role": "operator"}
```

**Credentials**:
- `operator` / `metroOp2024` — Can do everything except admin routes
- `admin` / `metroAdmin2024` — Full access

### 5.2 All API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/login` | — | Get JWT token |
| GET | `/health` | — | Health check → `{"status": "ok"}` |
| POST | `/register` | operator | Register person (base64 face image) |
| GET | `/person/{code}/trail` | operator | Movement trail for SDT code |
| POST | `/sighting` | operator | Log a sighting manually |
| GET | `/persons` | — | List all registered persons |
| GET | `/persons/live` | — | Currently visible persons |
| GET | `/locations` | operator | List all locations |
| POST | `/locations` | admin | Create a location |
| GET | `/cameras` | operator | Cameras grouped by location |
| POST | `/cameras` | operator | Create camera record |
| DELETE | `/cameras/{id}` | operator | Delete camera |
| POST | `/camera/start` | operator | Start LiveStream for a camera |
| POST | `/camera/stop` | operator | Stop a camera stream |
| POST | `/camera/stop-all` | operator | Stop all streams |
| GET | `/camera/status` | — | All camera statuses |
| GET | `/camera/stream/{id}` | — | MJPEG live stream (use as `<img src>`) |
| GET | `/camera/detections/recent` | — | Recent detections (in-memory + DB) |
| GET | `/logs` | admin | Recent log lines |
| POST | `/search/by-photo` | — | Search person by uploaded photo |
| GET | `/analytics/count/live` | — | Live person count |

### 5.3 MJPEG Streaming

The stream endpoint (`/camera/stream/{id}`) returns a `multipart/x-mixed-replace` response. The dashboard uses it as:
```html
<img src="http://localhost:8000/camera/stream/CAM-001" />
```

---

## 6. Camera Processing

### 6.1 CameraProcessor (standalone, high-FPS)

**File**: `cameras/camera_processor.py` (687 lines)

**Architecture** (decoupled ML from UI):

```
Main Thread (20+ FPS)
├── Read webcam frame
├── Draw cached person boxes (from background YOLO)
├── Draw cached face boxes (from InsightFace cache with motion offset)
├── Draw HUD overlay
└── Show OpenCV window + encode for MJPEG

Background Thread: _detect_track_worker
├── YOLO detection (imgsz=416, conf=0.30)
├── DeepSORT tracking (IoU-only, bbox embeddings)
└── Update self._cached_tracks

Background Thread: _face_worker
├── InsightFace embedding extraction (det_size=160x160)
└── Update self._cached_faces

Background Thread: _reid_worker
├── OSNet Re-ID on person crops
└── Update self._cached_reid
```

**Key settings**:
- Camera resolution: `640x480`
- Face queue size: `1` (drop old frames if face worker is slow)
- Sighting cooldown: `30 seconds` per person
- Oversized box filter: Boxes > 85% of frame area are ignored
- Max reconnect attempts: `5` (with 10s interval)

### 6.2 LiveStream (backend-integrated)

**File**: `cameras/live_stream.py` (615 lines)

- All ML runs in single thread (simpler but slower)
- InsightFace runs every 5th frame with temporal smoothing
- Stores latest JPEG in memory for MJPEG streaming
- Includes dress color detection, height estimation, bag linking
- Max 4 simultaneous streams

---

## 7. Dashboard (React)

**Stack**: React 18 + Vite 8 + TailwindCSS + Axios

**Pages**:

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | Stats cards, recent detections, live feed preview |
| Photo Search | `/search` | Upload face photo → find matching SDT code |
| Live Camera | `/live` | Camera management, start/stop, MJPEG stream |
| Locations | `/locations` | Add/manage metro stations |
| Settings | `/settings` | Placeholder |

**API connection**: `http://localhost:8000` (configurable via `VITE_API_URL` in `dashboard/.env`)

**Auto-login**: Dashboard auto-authenticates as `operator` on page load.

---

## 8. Database Schema

**Engine**: SQLite (`metro.db`) with WAL mode + 30s busy timeout

### Tables

#### `persons`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| unique_code | VARCHAR | `SDT-0001` format |
| person_type | VARCHAR | `visitor`, `staff`, `unknown` |
| face_embedding | BLOB | 512-dim ArcFace vector (JSON) |
| reid_embedding | BLOB | 512-dim OSNet vector (JSON) |
| dress_color_hex | VARCHAR | `#a52b3f` |
| gender | VARCHAR | `M` or `F` |
| location_id | VARCHAR | First seen location |
| total_sightings | INTEGER | Count |
| first_seen_at | DATETIME | |
| last_seen_at | DATETIME | |
| created_at | DATETIME | |

#### `sightings`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| unique_code | VARCHAR | FK → persons.unique_code |
| location_id | VARCHAR | |
| zone_id | VARCHAR | |
| camera_id | VARCHAR | |
| confidence | FLOAT | 0.0 – 1.0 |
| seen_at | DATETIME | |
| frame_snapshot_path | VARCHAR | Optional snapshot path |

#### `locations`
| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR PK | `LOC-001` |
| name | VARCHAR | `Central Station` |
| type | VARCHAR | `metro_station` |
| address | VARCHAR | Optional |
| created_at | DATETIME | |

#### `cameras`
| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR PK | `CAM-001` |
| location_id | VARCHAR | FK → locations.id |
| zone_id | VARCHAR | `entrance`, `main`, etc. |
| label | VARCHAR | Human-readable name |
| source | VARCHAR | `0` (webcam) or RTSP URL |
| is_active | BOOLEAN | |
| created_at | DATETIME | |

---

## 9. Environment & Configuration

### 9.1 Environment Variables

```bash
# .env file (excluded from git)
DATABASE_URL=sqlite:///./metro.db
MODEL_CACHE_DIR=.insightface
API_HOST=0.0.0.0
API_PORT=8000
SMARTDETECT_STUB_MODE=0       # 0=real ML, 1=stubs (for testing)
SMARTDETECT_NO_AUTOSTART=1    # 1=don't auto-start webcam on backend boot
```

### 9.2 PowerShell Environment (for running)

```powershell
$env:SMARTDETECT_STUB_MODE="0"
$env:SMARTDETECT_NO_AUTOSTART="1"
```

### 9.3 Workspace Path

The project lives at `C:\Project` which is junction-linked to `C:\Users\lalit\OneDrive\Desktop\Project`.

```powershell
# This junction was created with:
New-Item -ItemType Junction -Path "C:\Users\lalit\OneDrive\Desktop\Project" -Target "C:\Project"
```

### 9.4 Python Path

Python is installed at: `C:\Python314\python.exe`

---

## 10. How to Run

### 10.1 Quick Start (3 terminals)

**Terminal 1 — Backend:**
```powershell
cd C:\Project
$env:SMARTDETECT_STUB_MODE="0"
$env:SMARTDETECT_NO_AUTOSTART="1"
C:\Python314\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Start camera for dashboard:**
```powershell
cd C:\Project
C:\Python314\python.exe scripts/start_dashboard_cam.py
```

**Terminal 3 — Dashboard:**
```powershell
cd C:\Project\dashboard
npm run dev
```

Then open: **http://localhost:5173/**

### 10.2 Standalone Camera (high-FPS, OpenCV window)

```powershell
cd C:\Project
$env:SMARTDETECT_STUB_MODE="0"
$env:SMARTDETECT_NO_AUTOSTART="1"
C:\Python314\python.exe scripts/run_camera.py --camera CAM-001 --station entrance --source 0
```
- Press `Q` in the OpenCV window to stop
- Source `0` = default webcam, `1` = second webcam
- Source `rtsp://192.168.1.x:554/stream` for IP cameras

### 10.3 Seed Demo Data

```powershell
C:\Python314\python.exe scripts/demo_setup.py
```
Creates 8 metro station locations.

### 10.4 Docker (PostgreSQL + pgvector)

```powershell
docker-compose up -d
```

---

## 11. Key Technical Decisions & Optimization History

### 11.1 FPS Optimization

**Problem**: Running YOLO + InsightFace + DeepSORT in the main loop gave ~3 FPS.

**Solution**: Decoupled architecture in `camera_processor.py`:
- Main thread: frame read + UI rendering at 20+ FPS
- Background thread: YOLO detection (runs every frame, no FPS impact)
- Background thread: InsightFace (heavy, runs asynchronously)
- Background thread: Re-ID (OSNet, runs asynchronously)

### 11.2 Face Detection: Haar Cascade vs InsightFace

**Tried**: OpenCV Haar Cascade for live face detection (every frame, ~3ms).
**Result**: Face boxes were inaccurate, poor quality, didn't reliably detect faces.
**Final**: Reverted to InsightFace cached approach with motion offset compensation. Face detection runs in background, cached results are drawn on every frame with dx/dy offset based on person movement.

### 11.3 YOLO Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `imgsz` | 416 | Higher = better bounding box accuracy. Was 320 (too inaccurate) |
| `conf` | 0.30 | Low enough to catch partial/sitting people |
| `iou` | 0.45 | NMS threshold |
| `classes` | [0] | Person class only |

### 11.4 DeepSORT Embeddings Fix

DeepSORT was initialized without a built-in embedder (for speed). We provide lightweight bbox-derived embeddings:
```python
# 128-dim embedding from bbox geometry
seed = [cx, cy, aspect_ratio, area, width, height]
embed = np.tile(seed, 128 // 6 + 1)[:128]
embed = embed / np.linalg.norm(embed)
```

**Critical bug fixed**: `embeds if embeds else None` → `embeds` (empty list is falsy in Python, was passing `None` causing "Embedder not created" crash).

### 11.5 Oversized Box Filter

Boxes covering > 85% of frame area are filtered out — prevents "garbage" full-frame detections.

### 11.6 InsightFace Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Model | `buffalo_l` | Best accuracy (heaviest) |
| `det_size` | (160, 160) | Reduced from default for faster CPU inference |
| `det_thresh` | 0.35 | Lower threshold for more detections |
| `ctx_id` | -1 (CPU) | Change to 0 for GPU (requires onnxruntime-gpu) |

### 11.7 GPU Acceleration (optional)

If NVIDIA GPU available:
```powershell
pip install onnxruntime-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Then change: `face_app.prepare(ctx_id=0, ...)` — 10x faster InsightFace.

---

## 12. Troubleshooting

### "Embedder not created during init so embeddings must be given now!"
**Fix**: In `tracker/deepsort_tracker.py`, ensure `embeds=embeds` (never `None`).

### Camera shows "No Signal" in dashboard
**Fix**: Run `python scripts/start_dashboard_cam.py` after starting the backend. This creates the camera record and starts the LiveStream.

### FPS is low (< 10)
- Use `camera_processor.py` (standalone mode) instead of `live_stream.py`
- Reduce `imgsz` to 320 in `object_detector.py`
- Switch InsightFace model from `buffalo_l` to `buffalo_s`
- Enable GPU acceleration

### Dashboard won't start (Vite crash)
- Vite 5 crashes on Node.js v24. Upgrade to Vite 8:
  ```bash
  cd dashboard && npm install vite@latest @vitejs/plugin-react@latest
  ```

### "Backend unreachable" warning in camera
- Start the backend first: `uvicorn backend.main:app --port 8000`
- Camera will still work locally without backend (sightings retry in background)

### Face detection is slow/laggy
- In `camera_processor.py`, face detection uses InsightFace in a background thread
- Cached face boxes are drawn every frame with motion compensation
- Reduce `det_size` to `(128, 128)` for faster processing

### Multiple people not detected
- Lower YOLO `conf` threshold (currently 0.30)
- Increase `imgsz` (currently 416, try 640)
- Check if people are too small in frame (increase camera resolution)

### Permission denied pushing to GitHub
- Git may cache wrong credentials
- Use: `git remote set-url origin https://USERNAME:TOKEN@github.com/kishan02-sg/Smart-Detect.git`

---

## 13. Git & GitHub

**Repository**: https://github.com/kishan02-sg/Smart-Detect

**Branch**: `main`

### Commits:
1. `f673ab6` — Initial commit: full SmartDetect system (70 files, 10,361 lines)
2. `e8687f0` — Fix: upgrade Vite to v8 for Node 24 compatibility
3. `229795e` — Feat: add start_dashboard_cam.py script

### .gitignore excludes:
- `.env`, `*.pt`, `*.onnx`, `*.pth` (model weights)
- `*.db`, `*.db-shm`, `*.db-wal` (databases)
- `node_modules/`, `.venv/`, `__pycache__/`
- `logs/`, `brain/`, debug files
- `check_imports.py`, `fix_deepsort.py` (temp scripts)

### How to push new changes:
```powershell
cd C:\Project
git add -A
git commit -m "your commit message"
git push origin main
```

---

## 14. File-by-File Reference

### Backend
| File | Lines | Purpose |
|------|-------|---------|
| `backend/main.py` | 850 | All FastAPI routes, camera management, MJPEG streaming |
| `backend/auth.py` | ~200 | JWT auth, LoginRequest/Response, role-based access |
| `backend/logger.py` | ~170 | Structured JSON logging with timestamps |

### Camera
| File | Lines | Purpose |
|------|-------|---------|
| `cameras/camera_processor.py` | 687 | Standalone high-FPS processor (background ML threads) |
| `cameras/live_stream.py` | 615 | Backend-integrated processor (single-thread, MJPEG) |

### Recognition
| File | Lines | Purpose |
|------|-------|---------|
| `recognition/object_detector.py` | 180 | YOLOv8 wrapper (person + object detection) |
| `recognition/face_recognizer.py` | 213 | InsightFace ArcFace wrapper + stub fallback |
| `recognition/reid_model.py` | 175 | OSNet Re-ID wrapper + histogram stub fallback |
| `recognition/smart_identifier.py` | 251 | 4-method identification cascade |
| `recognition/registration.py` | ~140 | Person registration (embedding + DB insert) |

### Tracker
| File | Lines | Purpose |
|------|-------|---------|
| `tracker/deepsort_tracker.py` | 205 | DeepSORT wrapper with bbox embeddings |

### Database
| File | Lines | Purpose |
|------|-------|---------|
| `database/db.py` | 62 | SQLAlchemy engine, session, WAL pragma |
| `database/models.py` | ~190 | ORM models (Person, Sighting, Location, Camera) |
| `database/queries.py` | ~280 | find_person_by_embedding, log_sighting, etc. |

### Dashboard
| File | Lines | Purpose |
|------|-------|---------|
| `dashboard/src/App.jsx` | 62 | React Router + layout shell |
| `dashboard/src/pages/Dashboard.jsx` | ~220 | Stats, recent detections, live preview |
| `dashboard/src/pages/LiveCamera.jsx` | 474 | Camera CRUD, start/stop, MJPEG stream |
| `dashboard/src/pages/PhotoSearch.jsx` | ~550 | Upload face → search database |
| `dashboard/src/pages/Locations.jsx` | ~160 | Location management |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/run_camera.py` | Launch CameraProcessor standalone |
| `scripts/start_dashboard_cam.py` | Seed location + start camera for dashboard |
| `scripts/demo_setup.py` | Seed 8 demo metro stations |
| `scripts/seed_db.py` | Seed database with sample data |
| `scripts/e2e_test.py` | End-to-end API tests |
| `scripts/accuracy_test.py` | ML accuracy benchmarks |
| `scripts/load_test.py` | Load/stress tests |

---

## 15. Dependencies

### Python (`requirements.txt`)

```
# Web Framework
fastapi==0.110.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9

# Database
sqlalchemy==2.0.29
psycopg2-binary==2.9.9      # only for PostgreSQL
pgvector==0.2.5              # only for PostgreSQL
alembic==1.13.1

# ML / Vision
insightface==0.7.3           # ArcFace face recognition
onnxruntime==1.17.1          # CPU inference for insightface
torchreid==1.4.0             # Person Re-ID (OSNet)
torch>=2.0.0
torchvision>=0.15.0
opencv-python==4.9.0.80      # or opencv-contrib-python
deep-sort-realtime==1.3.2    # Multi-object tracking
ultralytics                  # YOLOv8 (auto-installs yolov8n.pt)

# Utilities
numpy==1.26.4
pillow==10.3.0
pydantic==2.7.0
python-dotenv==1.0.1
requests==2.31.0
scipy==1.13.0
```

### Node.js (`dashboard/package.json`)

```
react, react-dom, react-router-dom
axios
vite (v8), @vitejs/plugin-react
tailwindcss, postcss, autoprefixer
```

### System Requirements
- **Python**: 3.10+ (tested with 3.14)
- **Node.js**: 18+ (tested with 24.14)
- **OS**: Windows 10/11
- **GPU**: Optional (NVIDIA for CUDA acceleration)
- **Webcam**: Required for live detection

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────┐
│              SmartDetect Quick Reference               │
├──────────────────────────────────────────────────────┤
│ Backend:    http://localhost:8000                      │
│ Dashboard:  http://localhost:5173                      │
│ API Docs:   http://localhost:8000/docs                 │
│ Stream:     http://localhost:8000/camera/stream/CAM-001│
│                                                        │
│ Login:      operator / metroOp2024                     │
│ Admin:      admin / metroAdmin2024                     │
│                                                        │
│ Python:     C:\Python314\python.exe                    │
│ Project:    C:\Project                                 │
│ GitHub:     github.com/kishan02-sg/Smart-Detect        │
│                                                        │
│ Start Backend:                                         │
│   $env:SMARTDETECT_STUB_MODE="0"                      │
│   $env:SMARTDETECT_NO_AUTOSTART="1"                   │
│   python -m uvicorn backend.main:app --port 8000      │
│                                                        │
│ Start Camera (dashboard):                              │
│   python scripts/start_dashboard_cam.py               │
│                                                        │
│ Start Camera (standalone, high-FPS):                   │
│   python scripts/run_camera.py \                      │
│     --camera CAM-001 --station entrance --source 0    │
│                                                        │
│ Start Dashboard:                                       │
│   cd dashboard && npm run dev                          │
│                                                        │
│ Push to GitHub:                                        │
│   git add -A && git commit -m "msg" && git push       │
└──────────────────────────────────────────────────────┘
```

---

*Last updated: 2026-05-31*
*Project by: kishan02-sg*
*GitHub: https://github.com/kishan02-sg/Smart-Detect*
