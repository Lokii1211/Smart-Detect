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

## Project Status & AI Handoff (updated 2026-07-30)

> This section exists so that a developer — or an AI assistant — on a **new machine** can pick up exactly where the project left off. Read it before touching anything.

### What is built and verified working

| Feature | State |
|---|---|
| Live webcam detection/tracking/identity (CAM auto-start) | ✅ verified live |
| **Video file upload → full analysis pipeline** (`POST /cameras/upload`, multipart, validated) | ✅ verified live |
| Uploaded video plays at native FPS; **EOF → "VIDEO ENDED" frame + camera marked offline** | ✅ verified live |
| Resolution-aware detection (4K→960px, HD→640px, webcam→416px YOLO input) | ✅ mechanism verified on a 1920×1080 clip (correctly selects 640px). The originally-recorded **2→5 people on a 4K frame** magnitude was not re-measured this session — no genuine 4K test clip was on hand |
| Resolution-scaled overlay drawing (boxes/labels/HUD scale with source size) | ✅ verified on 4K |
| **Identity accuracy hardened** (2026-07-30 session): face-anchored matching so a visible face that matches nobody blocks colour/re-ID fallback from re-associating strangers; thresholds tightened (face match 0.50→0.56, re-ID 0.60→0.68, face veto 0.30→0.45); ByteTrack **ID-switch guard** drops a track's cached code after 2 consecutive face contradictions; **evidence is face-confirmed only** — no sighting row/photo logs while a track's identity is in doubt | ✅ measured **0/15 merged identities** across a 9-clip public evaluation set (`scripts/purity_eval.py`), including 4 people correctly kept separate while seated in the same crowded frame at once |
| **Duplicate-identity merge tooling**: `POST /persons/{dup}/merge-into/{keep}` combines sightings/gallery/snapshots and deletes the duplicate; `GET /persons/duplicate-suggestions` auto-flags likely-duplicate code pairs (face-gallery cosine sim ≥0.50) with a one-click merge banner on the People page | ✅ verified — used to merge two duplicate codes of the same person seen at different head angles |
| **Photo Search wired to the real backend**: result cards show the camera's actual live MJPEG stream (not a mock), real camera roster polled every 5s, auto re-search every 5s while a match is live so LIVE badges follow the person between cameras, per-camera sighting rollup so every area a person visited is shown | ✅ verified live |
| **Search by SDT code** on the Photo Search page: normalizes `8` / `008` / `sdt-8` → `SDT-0008`, falls back to a name search, shows every camera/area the person was seen in with counts | ✅ verified |
| **Live Camera CCTV wall view**: grid of every camera's live annotated stream, ▶ Start All / ⏹ Stop All, ⛶ fullscreen monitor mode | ✅ verified — 4 simultaneous video streams (backend's `MAX_STREAMS` cap) run at full native fps on an Apple M5 with CPU headroom to spare |
| Adaptive face memory: template blending (only on confident ≥0.55 matches) + multi-view gallery (`face_templates`, cap 5, floor 0.60 to add a view) | ✅ |
| Snapshot evidence: registration photo + per-sighting crops under `snapshots/{code}/`, served at `/snapshots` | ✅ verified |
| Named enrollment: `PUT /persons/{code}`, People dashboard page (photos, rename, type, appearance gallery) | ✅ verified |
| Real OSNet re-ID (vendored arch + downloaded weights — see gotchas) | ✅ `is_stub=False`, 512-dim features |
| Progress/`frame_persons`/`analyzed_frames` diagnostics in `/camera/status` | ✅ |
| **Crowd-scene behavior validated** with two real public clips: overhead/distant crowd (5 shoppers, `demo_videos/06_store_aisle.mp4`) → tracked and boxed but **0 identities minted** (faces too small, by design); eye-level crowd (4 people seated together, `classroom.mp4`) → **4 identities minted, all correctly separated, 0 merges**. Deciding factor is face visibility, not headcount. | ✅ verified |
| One-command live demo: `demo_videos/` (7 public MIT clips) + `python scripts/demo_videos.py` uploads and auto-starts them | ✅ verified |

### Known open items (deliberately deferred)

1. **Box lag on high-res uploads**: video plays real-time (30fps) but 4K analysis takes ~5s/frame on a CPU-only box, so overlay boxes trail moving people by seconds. Decision deferred. Options designed: analysis-paced "thorough mode" playback, or per-upload toggle.
2. **Label overlap**: when two people walk close together their "Detecting..." labels overlap. Cosmetic.
3. **Coach checkpoints pending**: Security review of the upload endpoint (validation exists: extension whitelist, 500MB cap, content sniff via cv2, server-generated filenames, operator JWT — but the formal findings review hasn't run) and end-to-end QA with a real clip.
4. **Supabase (production DB) is paused/unreachable** — everything currently runs on local SQLite (`smartdetect.db`). Restore from the Supabase dashboard, then `scripts/supabase_migrate.py` pushes local data up. The pgvector query path (`database/queries.py`) is written but untested live.
5. Distant people in wide street/overhead footage stay "Detecting..." (grey) forever **by design** — their faces are below the quality gate (48px height, 0.60 det-score) that prevents identity cross-contamination. Identity requires footage where people pass within ~5–10m of the camera, roughly eye level. Re-confirmed with real footage in this session (see crowd-scene row above).
6. **Duplicate codes are now the accepted trade-off over merged (wrong) codes.** The identity-hardening work above intentionally makes the system split rather than merge when unsure — e.g. the same person registered twice from very different head angles/poses. The merge tooling (`/persons/duplicate-suggestions` + `/persons/{dup}/merge-into/{keep}`) exists to clean these up, but nothing auto-merges. A face-pose quality gate at registration time (skip minting a new code from an extreme head angle) would reduce duplicate creation further — not yet implemented.
7. `scripts/seed_stations.py` is stale — it posts to `/stations`, but the live API is `/locations`. Use `scripts/demo_setup.py` or `scripts/demo_videos.py` instead.

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

**macOS session notes (2026-07-30, Apple M5, no SSL interception):**
- Use **Python 3.12** (`brew install python@3.12`), not 3.9 (system default, too old) or 3.14 (untested here). `python3.12 -m venv .venv` then `pip install -r requirements.txt` installed cleanly including `insightface` built from source (Xcode CLT required).
- No SSL interception on a normal Mac — `yolov8n.pt` and InsightFace's `buffalo_l` auto-download on first use exactly as ultralytics/insightface intend; `truststore` injection in `scripts/fetch_osnet.py` is a harmless no-op fallback here.
- **Vite binds IPv6 `localhost` only** — `curl http://127.0.0.1:5173` fails (connection refused) while `http://localhost:5173` works fine. Not a bug, just don't debug the wrong host.
- ONNXRuntime runs InsightFace on **CPU** by default (`CPUExecutionProvider`) even though CoreML is available — still fast enough for single/multi-stream use on an M5 (raw YOLO: ~77 fps on a 768×576 clip). Swapping to `onnxruntime-silicon` for ANE/GPU acceleration is an open lever, not yet done.
- 4 concurrent video-file streams (`MAX_STREAMS` in `backend/main.py`) run at full native fps simultaneously on an M5 with CPU headroom to spare — raising the cap is likely feasible if you need more cameras, but each stream is a full analysis pipeline, so test before assuming it scales.
- `scripts/purity_eval.py` (added this session) embeds every snapshot per SDT code and reports pairwise face-similarity — the regression check for "did this change cause identity merges." Run it after any threshold/logic change in `recognition/smart_identifier.py` or `cameras/live_stream.py`.

### Architecture decisions already made (don't re-litigate)

- Capture and ML run on **separate threads** per camera (drop-oldest queue, `cameras/live_stream.py`) — this is what keeps stream FPS independent of analysis cost.
- **ByteTrack** (via `supervision`, pinned `<0.30`) replaced DeepSORT — identity resolution runs **once per track**, not per frame.
- Identity is **face-anchored**: clothing colour and body re-ID can only *re-associate* someone recently seen (10 min / 12 h windows), never mint or steal an identity; both are face-vetoed, and — since 2026-07-30 — **a gate-passing face that matches nobody blocks colour/re-ID fallback entirely** (a stranger's face can no longer be re-associated to an existing code via clothing). Re-ID templates refresh to today's clothing on every face-confirmed match.
- **ByteTrack ID-switch guard** (`cameras/live_stream.py`): a tracker ID can silently jump from one person to another during occlusion. If a gate-passing face contradicts a track's cached code twice in a row (cosine sim < 0.35), the cache is dropped and the track re-identifies from scratch instead of keeping the wrong code.
- **Evidence is face-confirmed only**: a sighting row / snapshot is written only when a gate-passing face agrees with the code (or the code was just earned from that face this cycle). A faceless box that inherited a code (cached track, or mid ID-switch) can keep its live on-screen label but cannot write photo evidence under someone else's code.
- **Merging is manual, by design**: nothing auto-merges two SDT codes, even when the system's own duplicate-suggestions endpoint is highly confident. An operator confirms via the People-page banner or `POST /persons/{dup}/merge-into/{keep}`. This trades a few extra duplicate codes for zero silent wrong-merges.
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

### One-command live demo

The repo ships with 7 sample clips in `demo_videos/` (public MIT test footage). With the backend running:

```bash
python scripts/demo_videos.py        # uploads all clips, starts 4 playing
```

Then open **http://localhost:5173/live** and click **⊞ Wall** to watch every camera side-by-side like a CCTV monitor wall — grey "Detecting…" labels turn into green `SDT-XXXX` codes as faces come into range. The **People** page shows each identity with photo evidence (and auto-flags likely duplicate codes for one-click merge); **Photo Search** finds a person across all cameras from a single photo, live.

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
| `POST` | `/persons/{dup}/merge-into/{keep}` | operator+ | Merge a duplicate identity: sightings, gallery, snapshots move to `keep`; `dup` is deleted |
| `GET` | `/persons/duplicate-suggestions` | operator+ | Auto-detected likely-duplicate SDT code pairs (face-gallery cosine sim ≥0.50) |
| `POST` | `/search/by-photo` | operator+ | Face search from a base64 photo — returns live + history matches across all cameras |
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
├── scripts/            # fetch_osnet, supabase_migrate, e2e_test, demo_setup,
│                       #   demo_videos (one-command demo), purity_eval (identity regression check)
├── demo_videos/        # 7 public MIT sample clips, committed for the live demo
├── models/             # osnet_x1_0_reid.pth (gitignored — run fetch_osnet.py)
├── uploads/            # uploaded surveillance videos (gitignored)
├── snapshots/          # per-person photo evidence (gitignored)
└── requirements.txt    # read the ML section comments before upgrading anything
```

---

## License

MIT © 2026 SmartDetect Team
