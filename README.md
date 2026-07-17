# SmartDetect — Person Tracking System

<div align="center">

**AI-powered real-time person tracking and re-identification for any camera-equipped environment**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

**SmartDetect** detects, registers, and tracks individuals across camera feeds — live webcams, RTSP cameras, or **uploaded surveillance video files** — assigning each person a persistent `SDT-XXXX` code and building a chronological movement trail with photo evidence.

Pipeline: **YOLOv8** person detection → **ByteTrack** multi-object tracking → **InsightFace** ArcFace face recognition (identity anchor) → **OSNet** body re-identification (when the face isn't visible) → adaptive multi-template face gallery that improves recognition with every visit.

---

## Project Status & AI Handoff (updated 2026-07-15)

> This section exists so that a developer — or an AI assistant — on a **new machine** can pick up exactly where the project left off. Read it before touching anything.

### What is built and verified working

| Feature | State |
|---|---|
| Live webcam detection/tracking/identity (CAM auto-start) | ✅ verified live |
| **Video file upload → full analysis pipeline** (`POST /cameras/upload`, multipart, validated) | ✅ verified live |
| Uploaded video plays at native FPS; **EOF → "VIDEO ENDED" frame + camera marked offline** | ✅ verified live |
| Resolution-aware detection (4K→960px, HD→640px, webcam→416px YOLO input) | ✅ verified: **2→5 people found** on the same 4K frame |
| Resolution-scaled overlay drawing (boxes/labels/HUD scale with source size) | ✅ verified on 4K |
| Face-anchored identity: no face → no SDT code; strict 0.50 ArcFace threshold; face veto on colour/re-ID matches; one code per visible scene | ✅ verified with 2 people |
| Adaptive face memory: 80/20 template blending + multi-view gallery (`face_templates`, cap 5) | ✅ |
| Snapshot evidence: registration photo + per-sighting crops under `snapshots/{code}/`, served at `/snapshots` | ✅ verified |
| Named enrollment: `PUT /persons/{code}`, People dashboard page (photos, rename, type, appearance gallery) | ✅ verified |
| Photo search → clickable SDT code → person's appearance frames | ✅ |
| Real OSNet re-ID (vendored arch + downloaded weights — see gotchas) | ✅ `is_stub=False`, 512-dim features |
| Progress/`frame_persons`/`analyzed_frames` diagnostics in `/camera/status` | ✅ |

### Known open items (deliberately deferred)

1. **Box lag on high-res uploads**: video plays real-time (30fps) but 4K analysis takes ~5s/frame on the dev laptop's CPU, so overlay boxes trail moving people by seconds. Decision deferred — owner is moving to a stronger CPU. Options designed: analysis-paced "thorough mode" playback, or per-upload toggle.
2. **Label overlap**: when two people walk close together their "Detecting..." labels overlap. Cosmetic.
3. **Coach checkpoints pending**: Security review of the upload endpoint (validation exists: extension whitelist, 500MB cap, content sniff via cv2, server-generated filenames, operator JWT — but the formal findings review hasn't run) and end-to-end QA with a real clip.
4. **Supabase (production DB) is paused/unreachable** — everything currently runs on local SQLite (`smartdetect.db`). Restore from the Supabase dashboard, then `scripts/supabase_migrate.py` pushes local data up. The pgvector query path (`database/queries.py`) is written but untested live.
5. Distant people in wide street footage stay "Detecting..." (grey) forever **by design** — their faces are below the quality gate (48px height, 0.60 det-score) that prevents identity cross-contamination. Identity requires footage where people pass within ~5–10m of the camera.

### Environment gotchas (will bite you if unread)

- **Python 3.14 venv** (`.venv/`). Two packages need care:
  - `torchreid` has **no installable package** — PyPI's "torchreid" is an unrelated dead project and the real deep-person-reid doesn't build on 3.14. Solution in place: the OSNet architecture is **vendored** at `recognition/osnet_arch.py` (MIT) and re-ID-trained weights are fetched by `python scripts/fetch_osnet.py` → `models/osnet_x1_0_reid.pth` (~56MB, gitignored).
  - `albumentations==1.4.15` / `albucore==0.0.16` are **pinned on purpose**: newer albucore pulls a native `stringzilla` DLL that Windows Application Control blocks, which silently kills `insightface` (symptom: "insightface not installed" when it is). Don't upgrade them.
- **The dev machine's HTTPS is intercepted** (AV/proxy): plain `requests`/`urllib` fail with `CERTIFICATE_VERIFY_FAILED`. `pip` and `git` work. For other downloads inject `truststore` first (`import truststore; truststore.inject_into_ssl()`) — `scripts/fetch_osnet.py` shows the pattern. On a normal machine, ultralytics auto-downloads `yolov8n.pt` on first run and InsightFace fetches buffalo_l into `~/.insightface`; on an SSL-intercepted machine, copy `yolov8n.pt` and `~/.insightface/models/buffalo_l/` over from the old machine (both are gitignored).
- **Backend must run with cwd = project root.** `uploads/`, `snapshots/`, and `sqlite:///./smartdetect.db` are all cwd-relative. A stray `cd` once rooted everything inside `dashboard/` and looked like data loss. Start it exactly like this:
  ```powershell
  Set-Location "<project root>"
  $env:DATABASE_URL = "sqlite:///./smartdetect.db"
  .venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
  ```
- `SMARTDETECT_NO_AUTOSTART=1` skips the CAM-001 auto-start (useful for API tests without grabbing a camera).
- The dev laptop's webcam hardware caps at **15 FPS** @ 640×480 — don't chase 30.
- Default credentials (`admin/smartAdmin2024`, `operator/smartOp2024`) and the JWT secret are development defaults hardcoded as fallbacks — **override via env vars before any real deployment** (`ADMIN_PASSWORD`, `OPERATOR_PASSWORD`, `JWT_SECRET`). The dashboard auto-logins with the operator default (see `dashboard/src/pages/Alerts.jsx` pattern).

### Architecture decisions already made (don't re-litigate)

- Capture and ML run on **separate threads** per camera (drop-oldest queue, `cameras/live_stream.py`) — this is what keeps stream FPS independent of analysis cost.
- **ByteTrack** (via `supervision`, pinned `<0.30`) replaced DeepSORT — identity resolution runs **once per track**, not per frame.
- Identity is **face-anchored**: clothing colour and body re-ID can only *re-associate* someone recently seen (10 min / 12 h windows), never mint or steal an identity; both are face-vetoed. Re-ID templates refresh to today's clothing on every face-confirmed match.
- Uploaded videos are **cameras with a file source** — same Camera row, same `/camera/start`, same stream endpoint. EOF is detected only for file sources (webcams keep retrying).

---

## Quick Start (local, no Docker)

```bash
# Backend (Python 3.10+; dev machine uses 3.14 — see gotchas above)
pip install -r requirements.txt
python scripts/fetch_osnet.py          # one-time: body re-ID weights
$env:DATABASE_URL = "sqlite:///./smartdetect.db"   # PowerShell
python -m uvicorn backend.main:app --port 8000

# Frontend
cd dashboard && npm install && npm run dev   # http://localhost:5173
```

Docker route (`docker compose up`) exists but the compose file predates the video-upload feature — local run is the tested path right now.

### Try the core flows

1. **Live camera**: the default webcam auto-starts as CAM-001 (unless its DB row points elsewhere). Stand in frame → grey "Detecting..." → green `SDT-0001` within ~3 s.
2. **Upload a surveillance video**: Live Camera page → ＋ Add Camera → source "Upload Video File" → pick an `.mp4` (≤500 MB) → Start. Watch annotated playback; card shows PROCESSING %, then FINISHED and the camera goes offline.
3. **Name someone**: People page → pencil icon → type a name → the live label becomes `Name (SDT-XXXX)`.
4. **Find someone from a photo**: Photo Search → upload a face → click the matched code → their captured frames.

---

## API Highlights

Interactive docs: **http://localhost:8000/docs** — all protected routes take `Authorization: Bearer <token>` from `POST /auth/login`.

| Method | Path | Role | Description |
|---|---|---|---|
| `POST` | `/cameras/upload` | operator+ | **Multipart video upload → creates a file-source camera** (ext whitelist, 500 MB cap, content-sniffed) |
| `POST` | `/camera/start` / `/camera/stop` | operator+ | Start/stop any camera (webcam index, RTSP URL, or uploaded file) |
| `GET` | `/camera/status` | public | Per-camera fps, `progress`, `finished`, `frame_persons`, `analyzed_frames` |
| `GET` | `/camera/stream/{id}` | public | Annotated MJPEG stream |
| `GET` | `/persons` | public | All registered people (+ `display_name`, `photo_path`) |
| `GET` | `/persons/{code}` | public | Person detail + appearance list (sighting snapshots) |
| `PUT` | `/persons/{code}` | operator+ | Named enrollment: set `display_name` / `person_type` |
| `POST` | `/search/by-photo` | operator+ | Face search from a base64 photo |
| `GET` | `/person/{code}/trail` | operator+ | Movement trail |
| `GET` | `/snapshots/...` | public (static) | Registration + sighting photos |

---

## Tech Stack

| Category | Technology |
|---|---|
| Detection | **YOLOv8n** (ultralytics), resolution-aware input size |
| Tracking | **ByteTrack** (supervision `>=0.26,<0.30`) |
| Face Recognition | InsightFace buffalo_l (ArcFace 512-dim), multi-template gallery |
| Body Re-ID | **OSNet x1.0** — vendored arch (`recognition/osnet_arch.py`) + model-zoo weights |
| Backend | FastAPI, Uvicorn, SQLAlchemy, PyJWT (RBAC operator/admin) |
| Database | SQLite (current) / PostgreSQL + pgvector (written, pending Supabase restore) |
| Frontend | React 18, Vite, Axios |

---

## Project Structure

```
smart detect/
├── backend/            # FastAPI app (main.py), auth, logger
├── cameras/            # live_stream.py — threaded capture+analysis per camera
│                       # camera_processor.py — standalone debug-window variant
├── database/           # models, queries (sqlite + pgvector paths), db setup
├── dashboard/          # React frontend (pages: Dashboard, LiveCamera, People,
│                       #   PhotoSearch, Locations, ObjectFeed, Alerts, Settings)
├── recognition/        # face_recognizer, smart_identifier (identity rules),
│                       #   reid_model + osnet_arch (vendored), object_detector
├── scripts/            # fetch_osnet, supabase_migrate, e2e_test, demo_setup…
├── models/             # osnet_x1_0_reid.pth (gitignored — run fetch_osnet.py)
├── uploads/            # uploaded surveillance videos (gitignored)
├── snapshots/          # per-person photo evidence (gitignored)
└── requirements.txt    # read the ML section comments before upgrading anything
```

---

## License

MIT © 2026 SmartDetect Team
