# SmartDetect — Intelligent Person Tracking, Re-Identification & Surveillance Analytics

<div align="center">

**AI-powered real-time person detection, multi-camera tracking, face-anchored re-identification, and compliance-ready video analytics.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.0-646CFF?logo=vite)](https://vitejs.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://ultralytics.com)
[![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-FF6F00)](https://github.com/deepinsight/insightface)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 1. Overview

**SmartDetect** detects, registers, tracks, and re-identifies individuals across multi-camera feeds — including live webcams, RTSP IP cameras, and uploaded surveillance video files (`.mp4`, `.avi`, `.mov`). 

Each person passing through camera zones is assigned a persistent, unique identifier (`SDT-XXXX`). The system constructs chronological movement trails accompanied by high-resolution photographic evidence crops, all managed under strict data governance and biometric privacy controls.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SMARTDETECT SYSTEM PIPELINE                               │
│                                                                                        │
│   [ Camera Ingest / RTSP / Video Upload ]                                              │
│                     │                                                                  │
│                     ▼                                                                  │
│   [ Frame Capture Thread (30 FPS) ] ──► [ Drop-Oldest Queue ] ──► [ ML Worker Thread ] │
│             │                                                            │             │
│             │ (Cached Annotations / HUD)                                 │ (Inference) │
│             ▼                                                            ▼             │
│   [ MJPEG Live Stream ] ◄────────────────────────────────── [ Detection + Tracking +   │
│             │                                                 ArcFace + OSNet Re-ID]   │
│             ▼                                                            │             │
│   [ React 18 Dashboard ] ◄─────── [ FastAPI REST API + JWT ] ────────────┴─────────────►
│                                             │                                          │
│                                             ▼                                          │
│                             [ SQLite / PostgreSQL pgvector ]                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 2. Key System Features & Capabilities

### 🧠 Computer Vision & Identity Arbitration
- **Decoupled Asynchronous Engine**: Frame capture runs at native camera framerate (30 FPS) on dedicated threads, while deep neural networks process asynchronously via a drop-oldest buffer.
- **Resolution-Aware Scaling & Tiled Inference**: Automatically scales detector input based on source resolution (4K $\to$ 960px, 1080p $\to$ 640px, webcam $\to$ 416px), with sliced/tiled detection (`cameras/tiled_detect.py`) for high-altitude/wide-angle feeds.
- **Face-Anchored Matching**: If a face passes the quality gate but matches no known identity, the subject is tagged as a verified stranger. This prevents secondary heuristics (clothing color or body Re-ID) from incorrectly hijacking someone else's identity.
- **ByteTrack ID-Switch Guard**: If tracker occlusion causes an ID swap, 2 consecutive facial contradictions (cosine similarity $< 0.35$) immediately invalidate the cached identity and force fresh re-identification.
- **Face-Confirmed Evidence Gate**: Sighting logs and snapshot crops (`snapshots/{SDT}/`) are only saved when confirmed by a matching face ($\ge 0.45$), ensuring zero contamination in photographic archives.
- **Head-Pose Registration Gate**: Rejects extreme yaw, pitch, or roll angles during enrollment to prevent duplicate identity fragmentation.
- **Optimal Face-to-Person Assignment**: Uses the Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`) to resolve overlapping person/face bounding boxes in crowded scenes.
- **Tracklet-Level Identity Voting**: Aggregates multi-frame embeddings across tracklets to establish consensus before committing identity.

### 💻 Web Dashboard & Operations
- **Live CCTV Wall**: Grid monitor displaying up to 4 simultaneous live-annotated video streams with Start All / Stop All controls and fullscreen monitor mode.
- **Reverse Photo Search**: Upload any photograph to extract ArcFace embeddings and perform sub-second similarity searches across the enrolled database with live camera indicators.
- **Search by Code / Name**: Normalize queries (`8`, `008`, `sdt-8` $\to$ `SDT-0008`) and view complete movement trails with camera visitation history.
- **People & Gallery Management**: Directory of registered individuals, classification tags (Employee, VIP, Visitor, Contractor), and duplicate identity suggestion banners with one-click merge tooling.
- **Carried Object Analytics**: Detection and tracking of non-person objects (backpacks, suitcases, handbags, bottles) in transit zones.
- **Watchlists & Real-Time Alerts**: Instant alert generation when watchlisted individuals appear or when camera streams disconnect.

### 🛡️ Biometric Privacy & Legal Compliance
- **Consent Classification**: Classifies every subject (`consented`, `dataset`, `unknown`) to comply with GDPR (Art. 9/12/30), India's DPDP Act 2023, and Illinois BIPA.
- **Automated Retention Purge**: Configurable TTLs purge unconsented biometrics and sightings automatically (`scripts/purge_expired.py`).
- **Cryptographically Verified Erasures**: Right-to-be-forgotten requests physically delete records and snapshot files, generating an immutable `ErasureReceipt`.
- **Append-Only Audit Ledger**: Tracks every biometric query, search, and export with actor credentials and timestamps in `audit_log`.

---

## 🛠️ 3. Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend API** | **FastAPI (0.110.0)** | Asynchronous Python framework with Uvicorn & Pydantic v2 |
| **Authentication** | **PyJWT (2.8.0)** | Default-deny RBAC, constant-time verification, scoped stream tokens |
| **Object Detection** | **YOLOv8n (Ultralytics)** | High-speed person and carried-object detector |
| **Multi-Object Tracking** | **ByteTrack (Supervision)** | Temporal association across bounding boxes |
| **Face Recognition** | **InsightFace (`buffalo_l`)** | 512-dim ArcFace facial embeddings with ONNX Runtime acceleration |
| **Body Re-ID** | **OSNet x1.0** | 512-dim deep person re-identification backbone |
| **Database & ORM** | **SQLAlchemy 2.0** | SQLite (WAL mode) / PostgreSQL 16 + pgvector |
| **Frontend UI** | **React 18.3 + Vite** | SPA with React Router v7, Axios interceptors, TailwindCSS 3.4 |
| **DevOps & Containers** | **Docker & Compose** | Multi-container setup (PostgreSQL pgvector, FastAPI, Node frontend) |
| **Testing & Evaluation**| **Pytest / Vitest** | Unit/integration test suites and offline ground-truth evaluation harness |

---

## 📁 4. Project Directory Structure

```
Smart-Detect/
├── backend/                  # FastAPI Application Layer
│   ├── auth.py               # JWT authentication, RBAC, password security
│   ├── governance.py         # Retention policies, erasures, audit logs
│   ├── logger.py             # Structured JSON logger
│   └── main.py               # REST API endpoints, middleware, stream routes
├── cameras/                  # Video Processing & Streaming
│   ├── camera_processor.py   # Legacy single-thread processor (DeepSORT)
│   ├── live_stream.py        # Asynchronous multi-threaded LiveStream engine
│   └── tiled_detect.py       # Sliced / tiled detection for distant subjects
├── config/                   # Configuration & Feature Flags
│   ├── identity_config.py    # Centralized IdentityConfig dataclass & loader
│   └── ablation/             # Ablation study JSON configurations
├── dashboard/                # React 18 + Vite Web Dashboard
│   ├── src/
│   │   ├── components/       # UI components (LiveCamera, Lightbox, MergeBanner)
│   │   ├── pages/            # 8 application views (Dashboard, Search, People, etc.)
│   │   ├── auth.js           # Token management and Axios interceptors
│   │   └── App.jsx           # App shell and routing
│   └── package.json          # Frontend dependencies
├── database/                 # SQLAlchemy ORM & Query Layer
│   ├── db.py                 # SQLite / PostgreSQL engine & session setup
│   ├── models.py             # ORM models (Person, Sighting, AuditLog, etc.)
│   └── queries.py            # High-performance DB queries & vector search
├── docker/                   # Docker deployment assets
│   ├── backend.Dockerfile    # Multi-stage Python backend container
│   └── init.sql              # PostgreSQL + pgvector initialization script
├── eval/                     # Evaluation, Metrics & Ablation Harness
│   ├── adapters/             # Dataset adapters (ChokePoint, etc.)
│   ├── scoring.py            # Partition-based evaluation metrics
│   ├── METRICS.md            # Exact mathematical definitions of metrics
│   └── run_eval.py           # Offline evaluation runner
├── models/                   # Local weights & model caches (OSNet, calibration)
├── recognition/              # Core ML & Biometric Algorithms
│   ├── face_assign.py        # Hungarian optimal face-to-person matching
│   ├── face_pose.py          # Landmark-ratio head pose estimation
│   ├── face_quality.py       # Quality gate (hand-crafted and learned)
│   ├── face_recognizer.py    # InsightFace ArcFace wrapper (512-d)
│   ├── object_detector.py    # YOLOv8 person/object detection
│   ├── osnet_arch.py         # Vendored OSNet architecture
│   ├── registration.py       # Person registration helper
│   ├── reid_model.py         # Body Re-ID feature extraction
│   ├── score_calibration.py  # FMR-to-cosine threshold calibration
│   ├── smart_identifier.py   # 5-stage arbitration engine
│   └── tracklet_vote.py      # Multi-frame consensus voting
├── scripts/                  # Management, migration & benchmark scripts
├── tests/                    # Pytest unit & integration test suites
├── docker-compose.yml        # Multi-container service composition
└── requirements.txt          # Pinned Python dependencies
```

---

## ⚡ 5. Quick Start Guide

### Option A: Local Development Setup (Recommended)

#### 1. Clone the repository & enter workspace
```bash
git clone https://github.com/Lokii1211/Smart-Detect.git
cd Smart-Detect
```

#### 2. Configure Python Virtual Environment (Python 3.10 – 3.12 recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Fetch Pretrained Model Weights & Generate Environment
```bash
# Fetch OSNet Body Re-ID weights (~56MB)
python scripts/fetch_osnet.py

# Generate secure credentials in .env
python scripts/generate_env.py
```

#### 4. Run the Backend API
```bash
export DATABASE_URL="sqlite:///./smartdetect.db"   # On Windows: $env:DATABASE_URL = "sqlite:///./smartdetect.db"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*API Swagger Documentation is available at: **http://localhost:8000/docs***

#### 5. Launch the React Dashboard
```bash
cd dashboard
npm install
npm run dev
```
*Dashboard will be live at: **http://localhost:5173***

---

### Option B: Docker Compose Setup

Run the full stack (PostgreSQL with `pgvector`, FastAPI backend, and React dashboard) with a single command:

```bash
docker compose up --build
```

---

## 🎬 6. Running the Live Multi-Camera Demo

The repository includes sample public surveillance footage in `demo_videos/`. Once the backend is running, launch the automated demo:

```bash
python scripts/demo_videos.py
```

1. Open **http://localhost:5173/live** and switch to **⊞ Wall View** to observe real-time tracking across multiple simulated camera feeds.
2. Visit **People** to inspect registered identities and test the duplicate merge suggestion banner.
3. Visit **Photo Search** to test reverse face searching and trajectory trail mapping.

---

## 🔒 7. API Reference Table

All protected routes require a Bearer token (`Authorization: Bearer <token>`) from `POST /auth/login` or an ephemeral `?token=` stream token for MJPEG media endpoints.

| Method | Endpoint | Access Role | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/auth/login` | **Public** | Exchange credentials for JWT access token |
| `POST` | `/auth/stream-token` | Operator / Admin | Mint short-lived token for `<img>` MJPEG streams |
| `GET` | `/health` | **Public** | Service liveness probe |
| `GET` | `/camera/stream/{id}` | Stream Token | Live annotated MJPEG video feed |
| `POST` | `/cameras/upload` | Operator / Admin | Upload `.mp4`/`.avi` video file to analyze as a virtual camera |
| `POST` | `/camera/start` | Operator / Admin | Start live stream for webcam, RTSP, or uploaded file |
| `POST` | `/camera/stop` | Operator / Admin | Stop camera stream |
| `GET` | `/camera/status` | Operator / Admin | Query per-camera FPS, progress, and detection stats |
| `GET` | `/persons` | Operator / Admin | List all tracked individuals with pagination & filters |
| `GET` | `/persons/{code}` | Operator / Admin | Get individual appearance history and sighting crops |
| `PUT` | `/persons/{code}` | Operator / Admin | Update display name or person category (VIP/Employee) |
| `POST` | `/persons/{dup}/merge-into/{keep}` | Operator / Admin | Merge two duplicate SDT profiles (combines sightings and gallery) |
| `GET` | `/persons/duplicate-suggestions` | Operator / Admin | Cosine-similarity duplicate identity suggestions |
| `POST` | `/search/by-photo` | Operator / Admin | Reverse face image search against database |
| `GET` | `/person/{code}/trail` | Operator / Admin | Spatiotemporal zone movement trail |
| `DELETE`| `/persons/{code}` | Admin | Verified right-to-be-forgotten biometric erasure |
| `PUT` | `/persons/{code}/consent` | Operator / Admin | Update legal consent metadata (`consented`/`unknown`) |
| `POST` | `/governance/purge` | Admin | Trigger batch cleanup of expired biometrics |
| `GET` | `/governance/audit` | Admin | Retrieve immutable biometric audit log entries |

---

## 🧪 8. Test Suite & Verification

Execute the complete automated test suite:

```bash
# Run backend pytest suite
pytest tests/ -v

# Run frontend Vitest suite
cd dashboard && npm test
```

---

## 📄 9. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
