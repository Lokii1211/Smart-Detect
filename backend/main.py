"""
backend/main.py
────────────────
FastAPI application for SmartDetect — Universal Camera Detection System.

AUTHENTICATION — DEFAULT DENY (2026-08-01)
──────────────────────────────────────────
Every route requires a valid JWT unless listed in PUBLIC_ROUTES below. Only
two entries are public: POST /auth/login (credential exchange) and
GET /health (liveness probe). Adding a new route without a dependency now
fails CLOSED — it is protected by the middleware, not accidentally exposed.

Also enforced outside the router, where FastAPI dependencies do not reach:
  * /snapshots/**  — StaticFiles serving photographs of identifiable people
  * /docs, /redoc, /openapi.json — protected unless SMARTDETECT_PUBLIC_DOCS=1

MJPEG streams cannot send an Authorization header from <img src=...>, so
GET /camera/stream/{id} additionally accepts ?token= carrying a short-lived
token minted by POST /auth/stream-token. Those tokens are scoped 'stream'
and are rejected by every JSON endpoint.

Run scripts/audit_routes.py for the authoritative, live route/auth table.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import time as _time
from collections import defaultdict

from database.db import get_db, init_db
from database.models import Location, Person, Camera as CameraModel
from database.queries import (
    get_person_trail,
    get_recent_detections,
    log_sighting,
)
from recognition.registration import register_person

from backend.auth import (
    LoginRequest, LoginResponse, TokenData,
    login as _auth_login,
    require_operator, require_admin,
)
from backend.logger import get_structured_logger, read_recent_logs
from backend import governance as _gov

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = get_structured_logger(__name__)

# ─── App Factory ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartDetect API",
    description="Universal Camera Detection & Person Tracking System.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

import os as _os

# CORS. allow_origins=["*"] with allow_credentials=True is rejected by
# browsers and is wrong for a credentialed API, so origins are explicit.
_ALLOWED_ORIGINS = [
    o.strip() for o in _os.getenv(
        "SMARTDETECT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT-DENY AUTHENTICATION
#
# Every route requires a valid token unless its (method, path) appears in
# PUBLIC_ROUTES below. A newly added route is therefore protected by default:
# forgetting a dependency fails closed, not open. Per-route dependencies
# (require_operator / require_admin) still apply on top for role checks.
#
# Adding anything here means deliberately publishing it to the internet.
# Justify each entry.
# ─────────────────────────────────────────────────────────────────────────────
PUBLIC_ROUTES: set = {
    # Credential exchange — cannot require a token to obtain a token.
    ("POST", "/auth/login"),
    # Liveness probe for load balancers / orchestrators. Returns {"status":"ok"}
    # and nothing else: no counts, no identities, no configuration.
    ("GET", "/health"),
}

# FastAPI's own docs. Schema disclosure only (no data), but it maps the whole
# attack surface, so it is protected unless explicitly opened for development.
_PUBLIC_DOCS = _os.getenv("SMARTDETECT_PUBLIC_DOCS", "0") == "1"
_DOC_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


@app.middleware("http")
async def enforce_default_deny_auth(request: Request, call_next):
    """
    Reject unauthenticated requests to anything not explicitly public.

    Runs before routing resolves path parameters, so it matches on the raw
    path for docs/static and on the resolved route template for API routes.
    """
    method = request.method
    path = request.url.path

    # CORS preflight carries no credentials by design.
    if method == "OPTIONS":
        return await call_next(request)

    if path in _DOC_PATHS:
        if _PUBLIC_DOCS:
            return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={"detail": "API documentation requires authentication. "
                               "Set SMARTDETECT_PUBLIC_DOCS=1 for local development."},
        )

    # Snapshot images: photographs of identifiable people. Served by
    # StaticFiles, which has no dependency injection, so it is enforced here.
    if path.startswith("/snapshots"):
        if not _has_valid_token(request, ("api", "stream")):
            return JSONResponse(status_code=401,
                                content={"detail": "Authentication required"})
        return await call_next(request)

    # Resolve the route template (e.g. /persons/{unique_code}) so the
    # allowlist cannot be bypassed by a crafted concrete path.
    matched = _match_route(request)
    if matched and (method, matched) in PUBLIC_ROUTES:
        return await call_next(request)

    # Unknown paths fall through to FastAPI's 404 — but only after auth, so
    # an unauthenticated caller cannot enumerate which routes exist.
    scopes = ("api", "stream") if (matched or "").startswith("/camera/stream") else ("api",)
    if not _has_valid_token(request, scopes):
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


def _match_route(request: Request) -> Optional[str]:
    """Route template for this request, or None if nothing matches."""
    from starlette.routing import Match
    for route in app.routes:
        try:
            match, _ = route.matches(request.scope)
        except Exception:
            continue
        if match is not Match.NONE:
            return getattr(route, "path", None)
    return None


def _has_valid_token(request: Request, scopes: tuple = ("api",)) -> bool:
    """
    True when the request carries a valid JWT, via the Authorization header
    or — for stream/snapshot URLs used in <img src=...>, which cannot set
    headers — a ?token= query parameter.
    """
    from backend.auth import decode_token
    raw = None
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        raw = header.split(" ", 1)[1].strip()
    elif "token" in request.query_params:
        raw = request.query_params["token"]
    if not raw:
        return False
    try:
        decode_token(raw, expected_scopes=scopes)
        return True
    except HTTPException:
        return False

# Person snapshot photos (registration + sightings) — written by the camera
# pipeline under ./snapshots/{SDT-code}/
from pathlib import Path as _Path
from fastapi.staticfiles import StaticFiles
_Path("snapshots").mkdir(exist_ok=True)
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

# Uploaded surveillance videos, analysed like live cameras
_Path("uploads").mkdir(exist_ok=True)

# ─── Rate Limiting (in-memory, no external dependency) ───────────────────────
_rate_buckets: Dict[str, List[float]] = defaultdict(list)

def _check_rate_limit(request: Request, max_calls: int, window_seconds: int) -> None:
    client_ip = request.client.host if request.client else "unknown"
    key = f"{request.url.path}:{client_ip}"
    now = _time.time()
    hits = _rate_buckets[key]
    _rate_buckets[key] = [t for t in hits if now - t < window_seconds]
    if len(_rate_buckets[key]) >= max_calls:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    _rate_buckets[key].append(now)

# ─── Global camera state ─────────────────────────────────────────────────────
# camera_id → LiveStream instance  (max 4 simultaneous)
active_streams: Dict[str, Any] = {}
MAX_STREAMS = 4


@app.on_event("startup")
def startup_event() -> None:
    # Refuse to serve a biometric database with missing or default-valued
    # credentials. Raising here aborts startup — see backend/auth.py.
    from backend.auth import require_configured
    require_configured()
    logger.info("startup", message="Auth configuration validated.")

    try:
        init_db()
        logger.info("startup", message="Database initialised successfully.")
    except Exception as exc:
        logger.error("startup", message=f"Database initialisation failed: {exc}")
        return

    # ── Identity-arbitration config ──────────────────────────────────────────
    # Logged in full so it's obvious from the logs alone which parameters and
    # feature flags a given run used — critical for ablation reproducibility.
    import os
    from config.identity_config import get_identity_config
    identity_cfg = get_identity_config()
    logger.info("startup", message=(
        f"Identity config active (source="
        f"{os.getenv('SMARTDETECT_IDENTITY_CONFIG') or 'defaults'}): "
        f"{identity_cfg.as_loggable_dict()}"
    ))

    # ── Auto-start default webcam ────────────────────────────────────────────
    if os.getenv("SMARTDETECT_NO_AUTOSTART", "0") != "1":
        try:
            from cameras.live_stream import LiveStream
            from database.db import SessionLocal
            from datetime import datetime

            db = SessionLocal()
            try:
                # Ensure a default camera record exists
                cam = db.query(CameraModel).filter(CameraModel.id == "CAM-001").first()
                if not cam:
                    cam = CameraModel(
                        id="CAM-001",
                        location_id="LOC-001",
                        zone_id="main",
                        label="Default Webcam",
                        source="0",
                        is_active=False,
                        created_at=datetime.utcnow(),
                    )
                    db.add(cam)
                    db.commit()
                    logger.info("startup", message="Created default camera CAM-001")

                # Resolve source the same way POST /camera/start does — CAM-001
                # may since have been repointed at an RTSP URL or uploaded
                # video rather than the original default webcam
                source: Union[int, str] = cam.source
                try:
                    source = int(source)
                except (ValueError, TypeError):
                    pass

                stream = LiveStream(
                    source=source,
                    location_id=cam.location_id,
                    zone_id=cam.zone_id,
                    camera_id="CAM-001",
                )
                stream.start()
                active_streams["CAM-001"] = stream

                # Mark as active in DB
                cam.is_active = True
                db.commit()
                logger.info("startup", message=f"Auto-started CAM-001 (source={source})")
            finally:
                db.close()
        except Exception as exc:
            logger.warning("startup", message=f"Webcam auto-start skipped: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    base64_image: str  = Field(..., description="Base64-encoded face image")
    zone_id:      str  = Field(..., description="Zone where registration occurs")
    location_id:  str  = Field(..., description="Location ID (e.g. LOC-001)")
    person_type:  str  = Field("unknown", description="visitor | staff | unknown")


class RegisterResponse(BaseModel):
    unique_code:         str
    person_type:         str
    location_name:       str
    is_new_registration: bool
    message:             str


class SightingRequest(BaseModel):
    unique_code:         str
    location_id:         str
    zone_id:             str
    camera_id:           str
    confidence:          float = Field(..., ge=0.0, le=1.0)
    frame_snapshot_path: Optional[str] = None


class SightingResponse(BaseModel):
    success: bool
    message: str


class TrailItem(BaseModel):
    location_name:       str
    location_type:       str
    location_id:         Optional[str]
    zone_id:             Optional[str]
    camera_id:           str
    seen_at:             str
    confidence:          float
    frame_snapshot_path: Optional[str] = None


class LocationModel(BaseModel):
    id:      str
    name:    str
    type:    str = "other"
    address: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class LogsResponse(BaseModel):
    lines: List[str]
    count: int


class CameraStartRequest(BaseModel):
    camera_id: str = Field(..., description="Camera ID to start (e.g. CAM-001)")


class CameraStopRequest(BaseModel):
    camera_id: str = "CAM-001"


class CameraCreateRequest(BaseModel):
    location_id: str  = Field(..., description="Location ID")
    zone_id:     str  = Field("main", description="Zone label")
    label:       str  = Field("Camera", description="Human-readable camera name")
    source:      str  = Field("0", description="Webcam index or RTSP URL")


# ─────────────────────────────────────────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
def login(request: Request, payload: LoginRequest) -> LoginResponse:
    # Brute-force throttle. Keyed per client IP by _check_rate_limit.
    _check_rate_limit(request, max_calls=10, window_seconds=60)
    result = _auth_login(payload)
    logger.info("auth.login", message=f"Login by user='{payload.username}' role='{result.role}'")
    return result


@app.post("/auth/stream-token", tags=["Auth"])
def stream_token(token: TokenData = Depends(require_operator)) -> Dict[str, Any]:
    """
    Mint a short-lived token for MJPEG <img src=...> URLs, which cannot carry
    an Authorization header.

    Scoped 'stream' so it is rejected by the JSON API: these tokens end up in
    browser history, referrer headers and access logs, so they must not be
    replayable for data access. Valid 60 minutes.
    """
    from backend.auth import issue_stream_token
    return issue_stream_token(token, minutes=60.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# Person Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/register", response_model=RegisterResponse, tags=["Person"])
def register(
    request: Request,
    payload: RegisterRequest,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    _check_rate_limit(request, max_calls=20, window_seconds=60)
    logger.info("register", message=f"Registration by '{token.username}' loc='{payload.location_id}'")
    try:
        img_bytes = base64.b64decode(payload.base64_image)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode image.")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {exc}") from exc

    try:
        result = register_person(
            frame,
            zone_id=payload.zone_id,
            location_id=payload.location_id,
            db=db,
            person_type=payload.person_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return RegisterResponse(**result)


@app.get("/person/{unique_code}/trail", response_model=List[TrailItem], tags=["Person"])
def person_trail(
    request:     Request,
    unique_code: str,
    db:          Session   = Depends(get_db),
    token:       TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    """Movement history for one identity — where they were, when, on which
    camera. Audited: this is the most revealing read in the API."""
    _check_rate_limit(request, max_calls=60, window_seconds=60)
    _gov.audit(db, actor=token.username or "unknown", action="person.trail",
               subject_code=unique_code, outcome="ok", actor_role=token.role,
               actor_ip=_gov.client_ip(request))
    return get_person_trail(unique_code, db=db)


@app.post("/sighting", response_model=SightingResponse, tags=["Sighting"])
def record_sighting(
    payload: SightingRequest,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    success = log_sighting(
        unique_code=payload.unique_code,
        location_id=payload.location_id,
        zone_id=payload.zone_id,
        camera_id=payload.camera_id,
        confidence=payload.confidence,
        db=db,
        frame_path=payload.frame_snapshot_path,
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Person '{payload.unique_code}' not found.")
    return SightingResponse(success=True, message="Sighting logged.")


@app.get("/persons", tags=["Person"])
def list_persons(
    request: Request,
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    # Enumeration of every tracked identity — throttled even for operators.
    _check_rate_limit(request, max_calls=30, window_seconds=60)
    try:
        persons = db.query(Person).order_by(Person.created_at.desc()).all()
        return [
            {
                "unique_code":     p.unique_code,
                "display_name":    getattr(p, "display_name", None),
                "photo_path":      getattr(p, "photo_path", None),
                "person_type":     p.person_type,
                "location_id":     p.location_id,
                "total_sightings": getattr(p, "total_sightings", None) or 0,
                "first_seen_at":   p.first_seen_at.isoformat() if getattr(p, "first_seen_at", None) else None,
                "last_seen_at":    p.last_seen_at.isoformat()  if getattr(p, "last_seen_at",  None) else None,
                "created_at":      p.created_at.isoformat() if p.created_at else None,
            }
            for p in persons
        ]
    except Exception as exc:
        logger.error("persons.list", message=f"Error: {exc}")
        return []


class PersonUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(None, max_length=128)
    person_type:  Optional[str] = Field(None, description="visitor | staff | unknown")


@app.put("/persons/{unique_code}", tags=["Person"])
def update_person(
    unique_code: str,
    payload:     PersonUpdateRequest,
    db:          Session   = Depends(get_db),
    token:       TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """Attach a display name / person type to an SDT code (named enrollment)."""
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if not person:
        raise HTTPException(status_code=404, detail=f"Person '{unique_code}' not found.")
    if payload.display_name is not None:
        person.display_name = payload.display_name.strip() or None
    if payload.person_type is not None:
        person.person_type = payload.person_type
    db.commit()
    logger.info("person.update",
                message=f"'{token.username}' set {unique_code}: name={person.display_name!r} type={person.person_type}")
    return {
        "unique_code":  person.unique_code,
        "display_name": person.display_name,
        "person_type":  person.person_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data governance — retention, erasure, consent, audit
# See docs/DATA_GOVERNANCE.md and backend/governance.py
# ─────────────────────────────────────────────────────────────────────────────

class ConsentRequest(BaseModel):
    consent_status: str = Field(..., description="consented | dataset | unknown")
    consent_ref:    Optional[str] = Field(None, max_length=128,
                                          description="signed-form ID or dataset licence reference")


@app.delete("/persons/{unique_code}", tags=["Governance"])
def erase_person_route(
    request:     Request,
    unique_code: str,
    reason:      str       = Query("subject_request",
                                   description="subject_request | consent_withdrawn | operator"),
    db:          Session   = Depends(get_db),
    token:       TokenData = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Erase every trace of one identity: embeddings, face-template gallery,
    re-ID vector, dress colour, all sightings, and all snapshot images.

    Admin-only and irreversible. Returns a verified receipt — the server
    re-queries the database and re-stats the filesystem after deleting, so
    `verified: true` means checked, not assumed.

    Implements the DPDP Act s.12(3) right to erasure and the s.8(7) duty to
    erase on consent withdrawal.
    """
    from backend import governance as gov
    ip = gov.client_ip(request)
    try:
        return gov.erase_person(db, unique_code, actor=token.username or "unknown",
                                reason=reason, actor_ip=ip)
    except LookupError:
        gov.audit(db, actor=token.username or "unknown", action="person.erase",
                  subject_code=unique_code, outcome="not_found", actor_ip=ip)
        raise HTTPException(status_code=404, detail=f"Person '{unique_code}' not found.")
    except PermissionError as exc:
        gov.audit(db, actor=token.username or "unknown", action="person.erase",
                  subject_code=unique_code, outcome="denied_legal_hold", actor_ip=ip)
        raise HTTPException(status_code=409, detail=str(exc))


@app.put("/persons/{unique_code}/consent", tags=["Governance"])
def set_consent_route(
    request:     Request,
    unique_code: str,
    payload:     ConsentRequest,
    db:          Session   = Depends(get_db),
    token:       TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """
    Record the lawful basis for processing this identity, and re-derive its
    retention date from it.

    'unknown' (the default for anyone auto-registered from a camera) carries
    the shortest retention precisely because no lawful basis has been
    established for them.
    """
    from backend import governance as gov
    try:
        return gov.set_consent(db, unique_code, payload.consent_status,
                               actor=token.username or "unknown",
                               consent_ref=payload.consent_ref,
                               actor_ip=gov.client_ip(request))
    except LookupError:
        raise HTTPException(status_code=404, detail=f"Person '{unique_code}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/governance/retention", tags=["Governance"])
def retention_status_route(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """Active retention policy, population by consent status, and how many
    identities are already past their retention date."""
    from backend import governance as gov
    return gov.retention_status(db)


@app.post("/governance/purge", tags=["Governance"])
def purge_route(
    request: Request,
    dry_run: bool      = Query(True, description="true (default) reports without deleting"),
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Run the retention purge. Defaults to dry_run=true — pass dry_run=false to
    actually delete. Admin-only; every erasure produces its own receipt.
    """
    from backend import governance as gov
    return gov.purge(db, actor=token.username or "unknown", dry_run=dry_run)


@app.get("/governance/audit", tags=["Governance"])
def audit_log_route(
    subject_code: Optional[str] = Query(None, description="filter by SDT code"),
    actor:        Optional[str] = Query(None, description="filter by username"),
    action:       Optional[str] = Query(None),
    limit:        int           = Query(200, ge=1, le=1000),
    db:           Session       = Depends(get_db),
    token:        TokenData     = Depends(require_admin),
) -> List[Dict[str, Any]]:
    """
    Who accessed whose biometric data, when, and with what result.

    Admin-only: the log itself is personal data (it links operators to the
    people they looked up). Supports answering a data-subject's "who has
    accessed my data?" request via ?subject_code=.
    """
    from database.models import AuditLog
    q = db.query(AuditLog)
    if subject_code:
        q = q.filter(AuditLog.subject_code == subject_code)
    if actor:
        q = q.filter(AuditLog.actor == actor)
    if action:
        q = q.filter(AuditLog.action == action)
    rows = q.order_by(AuditLog.occurred_at.desc()).limit(limit).all()
    return [{
        "occurred_at": r.occurred_at.isoformat() + "Z",
        "actor": r.actor, "actor_role": r.actor_role, "actor_ip": r.actor_ip,
        "action": r.action, "subject_code": r.subject_code,
        "outcome": r.outcome,
        "detail": json.loads(r.detail) if r.detail else None,
    } for r in rows]


@app.get("/governance/erasures", tags=["Governance"])
def erasure_receipts_route(
    unique_code: Optional[str] = Query(None),
    limit:       int           = Query(100, ge=1, le=500),
    db:          Session       = Depends(get_db),
    token:       TokenData     = Depends(require_admin),
) -> List[Dict[str, Any]]:
    """
    Proof-of-deletion receipts. These survive the identities they refer to and
    contain no biometric data — they are how you demonstrate an erasure
    request was honoured.
    """
    from database.models import ErasureReceipt
    q = db.query(ErasureReceipt)
    if unique_code:
        q = q.filter(ErasureReceipt.unique_code == unique_code)
    rows = q.order_by(ErasureReceipt.erased_at.desc()).limit(limit).all()
    return [{
        "receipt_id": r.id, "unique_code": r.unique_code,
        "erased_at": r.erased_at.isoformat() + "Z", "erased_by": r.erased_by,
        "reason": r.reason, "verified": r.verified,
        "sightings_deleted": r.sightings_deleted,
        "snapshots_deleted": r.snapshots_deleted,
        "embeddings_cleared": r.embeddings_cleared,
        "detail": json.loads(r.detail) if r.detail else None,
    } for r in rows]


@app.post("/persons/{source_code}/merge-into/{target_code}", tags=["Person"])
def merge_person(
    source_code: str,
    target_code: str,
    db:          Session   = Depends(get_db),
    token:       TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """
    Merge a duplicate identity into another: sightings, face-template gallery,
    snapshots and stats all move to the target; the source code is deleted.
    Used when the same person was issued two SDT codes (e.g. frontal vs
    profile registration).
    """
    import json as _json
    import shutil
    from pathlib import Path as _Path

    if source_code == target_code:
        raise HTTPException(status_code=400, detail="Cannot merge a person into themselves.")
    src = db.query(Person).filter(Person.unique_code == source_code).first()
    dst = db.query(Person).filter(Person.unique_code == target_code).first()
    if not src or not dst:
        missing = source_code if not src else target_code
        raise HTTPException(status_code=404, detail=f"Person '{missing}' not found.")

    # 1. Re-home sightings
    from database.models import Sighting
    moved = (db.query(Sighting)
               .filter(Sighting.person_id == src.id)
               .update({Sighting.person_id: dst.id, Sighting.unique_code: dst.unique_code},
                       synchronize_session=False))

    # 2. Merge face gallery: source primary + templates become extra views
    try:
        gallery = _json.loads(dst.face_templates) if dst.face_templates else []
        if src.face_embedding:
            gallery.append(_json.loads(src.face_embedding))
        if src.face_templates:
            gallery.extend(_json.loads(src.face_templates))
        gallery = gallery[-8:]  # merged people keep a slightly larger gallery
        dst.face_templates = _json.dumps(gallery)
    except Exception:
        pass
    if not dst.reid_embedding and src.reid_embedding:
        dst.reid_embedding = src.reid_embedding

    # 3. Combine stats
    dst.total_sightings = (dst.total_sightings or 0) + (src.total_sightings or 0)
    if src.first_seen_at and (not dst.first_seen_at or src.first_seen_at < dst.first_seen_at):
        dst.first_seen_at = src.first_seen_at
    if src.last_seen_at and (not dst.last_seen_at or src.last_seen_at > dst.last_seen_at):
        dst.last_seen_at = src.last_seen_at
    if not dst.display_name and src.display_name:
        dst.display_name = src.display_name

    # 4. Move snapshot evidence into the target's folder
    try:
        src_dir = _Path("snapshots") / source_code
        dst_dir = _Path("snapshots") / target_code
        if src_dir.is_dir():
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.glob("*.jpg"):
                new_name = f"merged_{source_code}_{f.name}" if f.name == "registered.jpg" else f.name
                shutil.move(str(f), str(dst_dir / new_name))
            src_dir.rmdir()
    except Exception as exc:
        logger.warning("person.merge", message=f"Snapshot move failed: {exc}")

    db.delete(src)
    db.commit()
    logger.info("person.merge",
                message=f"'{token.username}' merged {source_code} into {target_code} ({moved} sightings moved)")
    return {"merged": source_code, "into": target_code, "sightings_moved": moved,
            "total_sightings": dst.total_sightings}


@app.get("/persons/duplicate-suggestions", tags=["Person"])
def duplicate_suggestions(
    request: Request,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    """
    Pairs of SDT codes whose stored face galleries look like the same person
    (max cross-gallery cosine similarity >= IdentityConfig.duplicate_suggestion_threshold,
    default 0.50). Sorted most-similar first; the operator confirms with the
    merge endpoint.
    """
    # O(n^2) gallery comparison — expensive and identity-revealing.
    _check_rate_limit(request, max_calls=10, window_seconds=60)
    import json as _json
    import numpy as _np

    from config.identity_config import get_identity_config
    cfg = get_identity_config()
    threshold = cfg.duplicate_suggestion_threshold

    persons = db.query(Person).filter(Person.face_embedding.isnot(None)).all()
    vecs: List[tuple] = []
    for p in persons:
        templates = []
        try:
            templates.append(_np.asarray(_json.loads(p.face_embedding), dtype=_np.float32))
            if p.face_templates:
                templates += [_np.asarray(v, dtype=_np.float32) for v in _json.loads(p.face_templates)]
        except Exception:
            continue
        vecs.append((p, templates))

    out: List[Dict[str, Any]] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            pa, ta = vecs[i]
            pb, tb = vecs[j]
            best = -1.0
            for a in ta:
                na = float(_np.linalg.norm(a)) + 1e-8
                for b in tb:
                    if a.shape != b.shape:
                        continue
                    sim = float(_np.dot(a, b)) / (na * (float(_np.linalg.norm(b)) + 1e-8))
                    best = max(best, sim)
            if best >= threshold:
                band, guidance = _merge_confidence_band(best, cfg)
                out.append({
                    "code_a": pa.unique_code, "name_a": pa.display_name,
                    "photo_a": pa.photo_path,
                    "code_b": pb.unique_code, "name_b": pb.display_name,
                    "photo_b": pb.photo_path,
                    "similarity": round(best, 3),
                    "confidence_band": band,          # high | medium | low
                    "guidance": guidance,             # what the operator should do
                    "auto_merge": False,              # ALWAYS false — see below
                })
    out.sort(key=lambda d: -d["similarity"])
    return out


def _merge_confidence_band(similarity: float, cfg) -> tuple:
    """
    Band a duplicate suggestion so the operator knows how much scrutiny it
    needs. Bands are advisory ONLY.

    NOTHING AUTO-MERGES, AT ANY CONFIDENCE.
    ───────────────────────────────────────
    Merging is irreversible and destroys one identity's code, so a wrong
    auto-merge silently fuses two real people — the exact failure the whole
    identity-hardening effort exists to prevent, and one that would be
    invisible afterwards because the evidence trails are already combined.
    A high band means "look at this first", never "this is safe to apply
    without looking". `auto_merge` is hard-coded False and there is no code
    path that merges without an explicit operator POST.
    """
    hi = getattr(cfg, "merge_band_high", 0.80)
    med = getattr(cfg, "merge_band_medium", 0.65)
    if similarity >= hi:
        return "high", ("Very likely the same person. Compare the two photos, "
                        "then merge if they match.")
    if similarity >= med:
        return "medium", ("Probably the same person. Check pose and lighting "
                          "differ rather than the face.")
    return "low", ("Possible match only — near the detection floor. Look-alikes "
                   "land here. Do not merge without clear photographic evidence.")


def _names_for(codes: List[str], db: Session) -> Dict[str, str]:
    """unique_code → display_name for the codes that have one."""
    if not codes:
        return {}
    try:
        rows = db.query(Person).filter(Person.unique_code.in_(codes)).all()
        return {p.unique_code: p.display_name for p in rows if getattr(p, "display_name", None)}
    except Exception:
        return {}


@app.get("/persons/live", tags=["Person"])
def live_persons(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    all_live: List[Dict] = []
    for stream in active_streams.values():
        try:
            for entry in stream.get_live_persons():
                recent = {d["unique_code"]: d for d in stream.get_recent_detections(100)}
                det = recent.get(entry["unique_code"], {})
                all_live.append({
                    "unique_code":     entry["unique_code"],
                    "method":          det.get("method", "unknown"),
                    "confidence":      det.get("confidence", 0.0),
                    "color_hex":       det.get("color_hex"),
                    "zone":            stream.zone_id,
                    "camera_id":       stream.camera_id,
                    "total_sightings": 0,
                })
        except Exception:
            pass
    names = _names_for([e["unique_code"] for e in all_live], db)
    for e in all_live:
        e["display_name"] = names.get(e["unique_code"])
    return all_live


# NOTE: registered after /persons/live so "live" is never captured as a code
@app.get("/persons/{unique_code}", tags=["Person"])
def person_detail(
    request:     Request,
    unique_code: str,
    db:          Session   = Depends(get_db),
    token:       TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """
    Full person detail: identity fields + every appearance (sightings with
    snapshot frames) — powers the "click the ID, see them in the video" view.
    """
    _check_rate_limit(request, max_calls=60, window_seconds=60)
    _gov.audit(db, actor=token.username or "unknown", action="person.read",
               subject_code=unique_code, outcome="ok", actor_role=token.role,
               actor_ip=_gov.client_ip(request))
    person = db.query(Person).filter(Person.unique_code == unique_code).first()
    if not person:
        raise HTTPException(status_code=404, detail=f"Person '{unique_code}' not found.")

    trail = get_person_trail(unique_code, db=db)
    appearances = [
        {
            "camera_id":     t.get("camera_id"),
            "zone_id":       t.get("zone_id"),
            "location_name": t.get("location_name"),
            "seen_at":       t.get("seen_at"),
            "confidence":    t.get("confidence"),
            "snapshot":      t.get("frame_snapshot_path"),
        }
        for t in trail
    ]
    return {
        "unique_code":     person.unique_code,
        "display_name":    getattr(person, "display_name", None),
        "person_type":     person.person_type,
        "photo_path":      getattr(person, "photo_path", None),
        "total_sightings": person.total_sightings or 0,
        "first_seen_at":   person.first_seen_at.isoformat() if person.first_seen_at else None,
        "last_seen_at":    person.last_seen_at.isoformat() if person.last_seen_at else None,
        "appearances":     appearances,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Location Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/locations", response_model=List[LocationModel], tags=["Locations"])
def list_locations(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    locs = db.query(Location).order_by(Location.name).all()
    return [{"id": l.id, "name": l.name, "type": l.type, "address": l.address} for l in locs]


@app.post("/locations", response_model=LocationModel, tags=["Locations"],
          status_code=status.HTTP_201_CREATED)
def create_location(
    payload: LocationModel,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_admin),
) -> Dict[str, Any]:
    if db.query(Location).filter(Location.id == payload.id).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Location ID already exists.")
    from datetime import datetime
    loc = Location(id=payload.id, name=payload.name, type=payload.type,
                   address=payload.address, created_at=datetime.utcnow())
    db.add(loc); db.commit(); db.refresh(loc)
    return {"id": loc.id, "name": loc.name, "type": loc.type, "address": loc.address}


# ─────────────────────────────────────────────────────────────────────────────
# Camera CRUD Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/cameras", tags=["Camera"])
def list_cameras(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    """Return all cameras grouped by location."""
    locations = db.query(Location).order_by(Location.name).all()
    result = []
    for loc in locations:
        cams = db.query(CameraModel).filter(CameraModel.location_id == loc.id).all()
        # merge with live is_active from active_streams
        cam_list = []
        for c in cams:
            stream  = active_streams.get(c.id)
            is_live = stream is not None
            # A finished video stream lingers in active_streams until evicted
            # on next start — don't report it as active
            finished = bool(is_live and getattr(stream, "_finished", False))
            cam_list.append({
                "id":        c.id,
                "zone_id":   c.zone_id,
                "label":     c.label,
                "source":    c.source,
                "is_active": (is_live and not finished) or c.is_active,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            })
        result.append({
            "location_id":   loc.id,
            "location_name": loc.name,
            "location_type": loc.type,
            "cameras":       cam_list,
        })
    return result


def _next_camera_id(db: Session) -> str:
    """Server-generated sequential camera ID (CAM-001, CAM-002, …)."""
    existing = db.query(CameraModel).count()
    cam_id = f"CAM-{existing + 1:03d}"
    while db.query(CameraModel).filter(CameraModel.id == cam_id).first():
        existing += 1
        cam_id = f"CAM-{existing + 1:03d}"
    return cam_id


@app.post("/cameras", tags=["Camera"], status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreateRequest,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """Create a new camera record."""
    loc = db.query(Location).filter(Location.id == payload.location_id).first()
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{payload.location_id}' not found.")

    cam_id = _next_camera_id(db)

    from datetime import datetime
    cam = CameraModel(
        id=cam_id,
        location_id=payload.location_id,
        zone_id=payload.zone_id,
        label=payload.label,
        source=payload.source,
        is_active=False,
        created_at=datetime.utcnow(),
    )
    db.add(cam); db.commit(); db.refresh(cam)
    logger.info("camera.create", message=f"Created {cam_id} at {payload.location_id}/{payload.zone_id}")
    return {
        "id":          cam.id,
        "location_id": cam.location_id,
        "zone_id":     cam.zone_id,
        "label":       cam.label,
        "source":      cam.source,
        "is_active":   cam.is_active,
    }


# ── Video upload: analyse a pre-recorded surveillance clip like a camera ─────
_VIDEO_EXTENSIONS   = {".mp4", ".avi", ".mov", ".mkv"}
_MAX_VIDEO_BYTES    = 500 * 1024 * 1024   # 500 MB
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@app.post("/cameras/upload", tags=["Camera"], status_code=status.HTTP_201_CREATED)
def upload_video_camera(
    request:     Request,
    file:        UploadFile = File(..., description="Surveillance video (.mp4/.avi/.mov/.mkv)"),
    location_id: str = Form(...),
    zone_id:     str = Form("main"),
    label:       str = Form("Uploaded Video"),
    db:          Session   = Depends(get_db),
    token:       TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """
    Upload a surveillance video and register it as a (file-backed) camera.
    Start it with POST /camera/start like any other camera; it plays at native
    speed, runs the full detection pipeline, and goes offline at end of video.
    """
    _check_rate_limit(request, max_calls=10, window_seconds=60)

    loc = db.query(Location).filter(Location.id == location_id).first()
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found.")

    # The client filename is used ONLY to read the extension — the stored name
    # is fully server-generated (no path traversal, no attacker-chosen names)
    ext = _Path(file.filename or "").suffix.lower()
    if ext not in _VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(_VIDEO_EXTENSIONS))}",
        )

    cam_id    = _next_camera_id(db)
    dest_path = _Path("uploads") / f"{cam_id}{ext}"

    # Stream to disk in chunks; abort past the size cap (no full read into RAM)
    written = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = file.file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_VIDEO_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video exceeds the {_MAX_VIDEO_BYTES // (1024*1024)} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    if written == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # Content sanity check: must actually decode as video (an .exe renamed to
    # .mp4 fails here) — extension alone is never trusted
    probe = cv2.VideoCapture(str(dest_path))
    try:
        ok, _frame = probe.read()
        native_fps  = probe.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        probe.release()
    if not ok:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="File is not a readable video.")

    from datetime import datetime
    cam = CameraModel(
        id=cam_id,
        location_id=location_id,
        zone_id=zone_id,
        label=label,
        source=dest_path.as_posix(),
        is_active=False,
        created_at=datetime.utcnow(),
    )
    db.add(cam); db.commit(); db.refresh(cam)
    duration_s = round(frame_count / native_fps, 1) if native_fps > 0 else None
    logger.info("camera.upload",
                message=f"'{token.username}' uploaded {written / 1e6:.1f} MB video as {cam_id} "
                        f"({duration_s or '?'} s, {native_fps:.0f} fps)")
    return {
        "id":          cam.id,
        "location_id": cam.location_id,
        "zone_id":     cam.zone_id,
        "label":       cam.label,
        "source":      cam.source,
        "is_active":   False,
        "is_file":     True,
        "size_bytes":  written,
        "fps":         round(native_fps, 1),
        "frame_count": frame_count,
        "duration_seconds": duration_s,
    }


@app.delete("/cameras/{camera_id}", tags=["Camera"])
def delete_camera(
    camera_id: str,
    db:        Session   = Depends(get_db),
    token:     TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    # Stop stream if running
    if camera_id in active_streams:
        try:
            active_streams[camera_id].stop()
        except Exception:
            pass
        active_streams.pop(camera_id, None)
    db.delete(cam); db.commit()
    logger.info("camera.delete", message=f"Deleted {camera_id}")
    return {"status": "deleted", "camera_id": camera_id}


# ─────────────────────────────────────────────────────────────────────────────
# Camera Stream Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/camera/start", tags=["Camera"])
def camera_start(
    payload: CameraStartRequest,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """
    Start a live camera stream.
    Looks up the Camera record by camera_id to get source, location_id, zone_id.
    """
    from cameras.live_stream import LiveStream

    camera_id = payload.camera_id

    # Already running? (a finished video may be restarted — replays from 0)
    existing = active_streams.get(camera_id)
    if existing is not None:
        if not getattr(existing, "_finished", False):
            return {"status": "already_running", "camera_id": camera_id}
        active_streams.pop(camera_id, None)

    # Cap is configurable via PUT /settings (max_simultaneous_streams),
    # defaulting to MAX_STREAMS when unset — read at request time so a
    # settings update applies without a restart.
    max_streams = int(_app_config.get("max_simultaneous_streams", MAX_STREAMS))
    if len(active_streams) >= max_streams:
        # Finished video streams only hold their "VIDEO ENDED" frame — evict
        # them before refusing a new stream
        for cid, s in list(active_streams.items()):
            if getattr(s, "_finished", False):
                try:
                    s.stop()
                except Exception:
                    pass
                active_streams.pop(cid, None)
        if len(active_streams) >= max_streams:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Max {max_streams} simultaneous streams already active.",
            )

    # Look up Camera record
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found. Create it first via POST /cameras.",
        )

    # Resolve source type
    source: Union[int, str] = cam.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # keep as string (RTSP URL)

    try:
        stream = LiveStream(
            source=source,
            location_id=cam.location_id,
            zone_id=cam.zone_id,
            camera_id=camera_id,
        )
        stream.start()
        active_streams[camera_id] = stream

        # Mark active in DB
        cam.is_active = True
        db.commit()

        logger.info("camera.start", message=f"Stream started: {camera_id} source={source}")
        return {
            "status":      "started",
            "camera_id":   camera_id,
            "location_id": cam.location_id,
            "zone_id":     cam.zone_id,
            "label":       cam.label,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@app.post("/camera/stop", tags=["Camera"])
def camera_stop(
    payload: CameraStopRequest,
    db:      Session = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """Stop a specific camera stream and mark is_active=False in DB."""
    camera_id = payload.camera_id
    if camera_id in active_streams:
        try:
            active_streams[camera_id].stop()
        except Exception:
            pass
        active_streams.pop(camera_id, None)

    # Update DB
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if cam:
        cam.is_active = False
        db.commit()

    logger.info("camera.stop", message=f"Stream stopped: {camera_id}")
    return {"status": "stopped", "camera_id": camera_id}


@app.post("/camera/stop-all", tags=["Camera"])
def camera_stop_all(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """Stop ALL running camera streams."""
    count = 0
    for cid in list(active_streams.keys()):
        try:
            active_streams[cid].stop()
        except Exception:
            pass
        active_streams.pop(cid, None)
        count += 1

    # Update all is_active in DB
    db.query(CameraModel).update({"is_active": False})
    db.commit()

    logger.info("camera.stop_all", message=f"Stopped {count} stream(s).")
    return {"status": "all_stopped", "count": count}


@app.get("/camera/status", tags=["Camera"])
def camera_status(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    """Return status of all cameras (active and inactive)."""
    all_cams = db.query(CameraModel).all()
    total = len(all_cams)

    cameras_list = []
    for c in all_cams:
        stream     = active_streams.get(c.id)
        is_running = stream is not None
        fps        = 0.0
        persons    = 0
        is_file    = False
        finished   = False
        progress   = 0.0
        analyzed   = 0
        frame_persons = 0
        if stream:
            try:
                st       = stream.get_status()
                fps      = st.get("fps", 0.0)
                persons  = st.get("persons_detected_today", 0)
                is_file  = st.get("is_file", False)
                finished = st.get("finished", False)
                progress = st.get("progress", 0.0)
                analyzed = st.get("analyzed_frames", 0)
                frame_persons = st.get("frame_persons", 0)
            except Exception:
                pass

        loc = db.query(Location).filter(Location.id == c.location_id).first()
        cameras_list.append({
            "camera_id":             c.id,
            "label":                 c.label,
            "location_id":           c.location_id,
            "location_name":         loc.name if loc else c.location_id,
            "zone_id":               c.zone_id,
            "source":                c.source,
            "is_active":             is_running and not finished,
            "fps":                   fps,
            "persons_detected_today": persons,
            "is_file":               is_file,
            "finished":              finished,
            "progress":              progress,
            "analyzed_frames":       analyzed,
            "frame_persons":         frame_persons,
        })

    # Count only streams that are actually live — a finished clip lingers in
    # active_streams until the next start/stop, and len() would over-report.
    active_count = sum(1 for c in cameras_list if c["is_active"])
    return {
        "total_cameras":  total,
        "active_cameras": active_count,
        "connected":      active_count > 0,
        "cameras":        cameras_list,
    }


@app.get("/camera/detections/recent", tags=["Camera"])
def camera_detections_recent(
    limit: int       = Query(20, ge=1, le=100),
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    """Recent detections — in-memory first, then DB fallback."""
    if active_streams:
        stream = next(iter(active_streams.values()))
        try:
            recents = stream.get_recent_detections(limit)
            if recents:
                names = _names_for([r["unique_code"] for r in recents], db)
                for r in recents:
                    r["display_name"] = names.get(r["unique_code"])
                return recents
        except Exception:
            pass

    try:
        from database.models import Sighting
        rows = (
            db.query(Sighting)
            .order_by(Sighting.seen_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "unique_code": r.unique_code or getattr(r, "person_id", ""),
                "zone_id":     r.zone_id,
                "camera_id":   r.camera_id,
                "confidence":  r.confidence,
                "seen_at":     r.seen_at.isoformat() if r.seen_at else None,
                "detected_at": r.seen_at.isoformat() if r.seen_at else None,
                "method":      "face",
                "color_hex":   None,
            }
            for r in rows
        ]
    except Exception:
        return []


@app.get("/camera/stream/{camera_id}", tags=["Camera"])
def camera_stream(camera_id: str) -> StreamingResponse:
    """Stream MJPEG video from an active camera. Use as <img src=...>."""
    import time

    if camera_id not in active_streams and active_streams:
        camera_id = next(iter(active_streams))

    def generate():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No Signal", (220, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 60, 60), 2)
        _, blank_jpg   = cv2.imencode(".jpg", blank)
        blank_bytes    = blank_jpg.tobytes()

        while True:
            stream = active_streams.get(camera_id)
            frame_bytes = blank_bytes
            if stream:
                try:
                    frame_bytes = stream.get_mjpeg_frame()
                except Exception:
                    pass

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(0.03)  # ~30 FPS browser stream (was 0.1 = 10 FPS cap)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Admin-only Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/logs", response_model=LogsResponse, tags=["System"])
def get_logs(
    lines: int       = 100,
    token: TokenData = Depends(require_admin),
) -> Dict[str, Any]:
    recent = read_recent_logs(n=lines)
    return {"lines": recent, "count": len(recent)}


# ─────────────────────────────────────────────────────────────────────────────
# Photo Search Route
# ─────────────────────────────────────────────────────────────────────────────

class PhotoSearchRequest(BaseModel):
    base64_image: str  = Field(..., description="Base64-encoded photo")
    scope:        str  = Field("live_and_history", description="live_and_history | live_only")
    check_only:   bool = Field(False, description="If true, only verify face detected")


@app.post("/search/by-photo", tags=["Search"])
def search_by_photo(
    request: Request,
    payload: PhotoSearchRequest,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    # Face search against the entire biometric gallery: the single most
    # abusable endpoint here. Explicit role check plus a tight throttle.
    _check_rate_limit(request, max_calls=15, window_seconds=60)
    import base64 as _b64
    from datetime import datetime, timezone, timedelta
    from recognition.face_recognizer import FaceRecognizer
    from database.queries import (
        find_person_by_embedding,
        find_person_candidates,
        get_person_trail,
    )

    try:
        img_bytes = _b64.b64decode(payload.base64_image)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode image")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    recognizer = FaceRecognizer()
    recognizer.load_model()
    embeddings   = recognizer.extract_embedding(frame)
    face_detected = bool(embeddings)

    if payload.check_only:
        return {"face_detected": face_detected}

    if not face_detected:
        return {
            "matched": False,
            "face_detected": False,
            "cameras_searched": len(active_streams) or 1,
            "message": "No face found in photo.",
        }

    query_emb = embeddings[0]
    match = find_person_by_embedding(query_emb, db=db, threshold=0.65)
    cameras_searched = max(len(active_streams), 1)
    # Read-only field ranking for the operator's "how close was the field"
    # view — never used for identity assignment (the match above is).
    candidates = find_person_candidates(query_emb, db=db, top_k=5, min_similarity=0.30)

    if not match:
        # Audit even a miss: an unsuccessful search still processed a face
        # image against the whole gallery, and the pattern of who is being
        # searched for is itself the thing that needs oversight.
        _gov.audit(db, actor=token.username or "unknown", action="search.by_photo",
                   outcome="no_match", actor_role=token.role,
                   actor_ip=_gov.client_ip(request),
                   detail={"scope": payload.scope, "cameras_searched": cameras_searched})
        return {
            "matched": False, "face_detected": True,
            "cameras_searched": cameras_searched,
            "message": "Person not found in any camera",
        }

    unique_code = match["unique_code"]
    confidence  = match["similarity"]
    matched_person = db.query(Person).filter(Person.unique_code == unique_code).first()
    _gov.audit(db, actor=token.username or "unknown", action="search.by_photo",
               subject_code=unique_code, outcome="matched", actor_role=token.role,
               actor_ip=_gov.client_ip(request),
               detail={"confidence": round(float(confidence), 4),
                       "scope": payload.scope})
    trail = get_person_trail(unique_code, db=db)

    now_ts = datetime.now(timezone.utc)
    if payload.scope == "live_only":
        cutoff_iso = (now_ts.replace(tzinfo=None) - timedelta(minutes=5)).isoformat()
        trail = [t for t in trail if t.get("seen_at", "") >= cutoff_iso]

    live_matches    = []
    history_matches = []
    live_codes      = set()

    for cam_id, stream in active_streams.items():
        recent = {d["unique_code"]: d for d in stream.get_recent_detections(100)}
        if unique_code in recent:
            det = recent[unique_code]
            live_matches.append({
                "camera_id":   cam_id,
                "camera_name": f"Camera {cam_id}",
                "zone_id":     stream.zone_id,
                "confidence":  det.get("confidence", confidence),
                "detected_at": det.get("detected_at"),
            })
            live_codes.add(cam_id)

    for t in trail[-20:]:
        cam_id = t.get("camera_id", "")
        history_matches.append({
            "camera_id":   cam_id or t.get("location_id", "—"),
            "camera_name": f"Camera {cam_id}" if cam_id else t.get("location_name", "—"),
            "zone_id":     t.get("zone_id"),
            "confidence":  t.get("confidence", confidence),
            "detected_at": t.get("seen_at"),
        })

    return {
        "matched":          True,
        "face_detected":    True,
        "unique_code":      unique_code,
        "confidence":       round(confidence, 4),
        "photo_path":       matched_person.photo_path if matched_person else None,
        "cameras_searched": cameras_searched,
        "is_live_now":      len(live_matches) > 0,
        "live_matches":     live_matches,
        "history_matches":  history_matches[:10],
        "candidates":       candidates,
        "timeline":         trail[-20:],
        "first_seen":       trail[0]["seen_at"] if trail else None,
        "last_seen":        trail[-1]["seen_at"] if trail else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Object Sightings Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/objects/recent", tags=["Objects"])
def list_object_sightings(
    limit: int = Query(50, ge=1, le=200),
    object_type: Optional[str] = Query(None, description="Filter by type: backpack, car, etc."),
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    from database.models import ObjectSighting
    q = db.query(ObjectSighting).order_by(ObjectSighting.detected_at.desc())
    if object_type:
        q = q.filter(ObjectSighting.object_type == object_type)
    rows = q.limit(limit).all()
    return [
        {
            "id":          r.id,
            "object_type": r.object_type,
            "confidence":  round(r.confidence, 3),
            "camera_id":   r.camera_id,
            "location_id": r.location_id,
            "zone_id":     r.zone_id,
            "detected_at": r.detected_at.isoformat() if r.detected_at else None,
            "bbox":        {"x": r.bbox_x, "y": r.bbox_y, "w": r.bbox_w, "h": r.bbox_h}
                           if r.bbox_x is not None else None,
        }
        for r in rows
    ]


@app.get("/objects/stats", tags=["Objects"])
def object_stats(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    from database.models import ObjectSighting
    from datetime import datetime, timezone
    from sqlalchemy import func

    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    try:
        total = db.query(ObjectSighting).filter(ObjectSighting.detected_at >= today).count()
        by_type = (
            db.query(ObjectSighting.object_type, func.count(ObjectSighting.id))
            .filter(ObjectSighting.detected_at >= today)
            .group_by(ObjectSighting.object_type)
            .all()
        )
    except Exception:
        total = 0
        by_type = []
    return {
        "total_today": total,
        "by_type": {t: c for t, c in by_type},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Settings / Config Routes
# ─────────────────────────────────────────────────────────────────────────────

_app_config: Dict[str, Any] = {
    "detection_confidence": 0.30,
    "nms_iou_threshold": 0.45,
    "face_match_threshold": 0.72,
    "reid_match_threshold": 0.78,
    "color_match_threshold": 30.0,
    "sighting_cooldown_seconds": 30,
    "max_simultaneous_streams": 4,
    "yolo_input_size": 416,
    "insightface_det_size": 160,
}


@app.get("/settings", tags=["Settings"])
def get_settings(
    token: TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    return {
        "config": _app_config,
        "active_streams": len(active_streams),
        "max_streams": int(_app_config.get("max_simultaneous_streams", MAX_STREAMS)),
        "database_url_type": "postgresql" if not _is_sqlite() else "sqlite",
    }


@app.put("/settings", tags=["Settings"])
def update_settings(
    payload: Dict[str, Any],
    token: TokenData = Depends(require_admin),
) -> Dict[str, Any]:
    updated = []
    for key, value in payload.items():
        if key in _app_config:
            _app_config[key] = value
            updated.append(key)
    logger.info("settings.update", message=f"Updated: {updated}")
    return {"updated": updated, "config": _app_config}


def _is_sqlite() -> bool:
    from database.db import DATABASE_URL
    return DATABASE_URL.startswith("sqlite")


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/analytics/count/live", tags=["Analytics"])
def analytics_live_count(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    from database.models import Sighting
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    try:
        count = db.query(Sighting).filter(Sighting.seen_at >= today).count()
    except Exception:
        count = 0
    return {"total": count}


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist Routes
# ─────────────────────────────────────────────────────────────────────────────

class WatchlistRequest(BaseModel):
    unique_code: str = Field(..., description="Person SDT code to watch")
    label:       Optional[str] = None
    reason:      Optional[str] = None


@app.get("/watchlist", tags=["Alerts"])
def list_watchlist(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> List[Dict[str, Any]]:
    from database.models import WatchlistEntry
    entries = db.query(WatchlistEntry).order_by(WatchlistEntry.created_at.desc()).all()
    return [
        {
            "id":          e.id,
            "unique_code": e.unique_code,
            "label":       e.label,
            "reason":      e.reason,
            "is_active":   e.is_active,
            "created_at":  e.created_at.isoformat() if e.created_at else None,
        }
        for e in entries
    ]


@app.post("/watchlist", tags=["Alerts"], status_code=status.HTTP_201_CREATED)
def add_to_watchlist(
    payload: WatchlistRequest,
    db:      Session   = Depends(get_db),
    token:   TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    from database.models import WatchlistEntry
    person = db.query(Person).filter(Person.unique_code == payload.unique_code).first()
    if not person:
        raise HTTPException(status_code=404, detail=f"Person '{payload.unique_code}' not found.")

    existing = db.query(WatchlistEntry).filter(
        WatchlistEntry.unique_code == payload.unique_code,
        WatchlistEntry.is_active == True,
    ).first()
    if existing:
        return {"id": existing.id, "status": "already_watched", "unique_code": payload.unique_code}

    from datetime import datetime
    entry = WatchlistEntry(
        unique_code=payload.unique_code,
        label=payload.label or person.unique_code,
        reason=payload.reason,
        is_active=True,
        created_at=datetime.utcnow(),
    )
    db.add(entry); db.commit(); db.refresh(entry)
    logger.info("watchlist.add", message=f"Added {payload.unique_code} to watchlist")
    return {"id": entry.id, "status": "added", "unique_code": payload.unique_code}


@app.delete("/watchlist/{entry_id}", tags=["Alerts"])
def remove_from_watchlist(
    entry_id: str,
    db:       Session   = Depends(get_db),
    token:    TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    from database.models import WatchlistEntry
    entry = db.query(WatchlistEntry).filter(WatchlistEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Watchlist entry not found.")
    entry.is_active = False
    db.commit()
    logger.info("watchlist.remove", message=f"Deactivated watchlist entry {entry_id}")
    return {"status": "removed", "id": entry_id}


# ─────────────────────────────────────────────────────────────────────────────
# Alert Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/alerts", tags=["Alerts"])
def list_alerts(
    limit:  int            = Query(50, ge=1, le=200),
    unread: Optional[bool] = Query(None),
    db:     Session        = Depends(get_db),
    token:  TokenData      = Depends(require_operator),
) -> List[Dict[str, Any]]:
    from database.models import Alert
    q = db.query(Alert).order_by(Alert.created_at.desc())
    if unread is True:
        q = q.filter(Alert.is_read == False)
    rows = q.limit(limit).all()
    return [
        {
            "id":          a.id,
            "alert_type":  a.alert_type,
            "severity":    a.severity,
            "title":       a.title,
            "message":     a.message,
            "unique_code": a.unique_code,
            "camera_id":   a.camera_id,
            "location_id": a.location_id,
            "is_read":     a.is_read,
            "created_at":  a.created_at.isoformat() if a.created_at else None,
        }
        for a in rows
    ]


@app.get("/alerts/unread-count", tags=["Alerts"])
def alerts_unread_count(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> Dict[str, int]:
    from database.models import Alert
    try:
        count = db.query(Alert).filter(Alert.is_read == False).count()
    except Exception:
        count = 0
    return {"count": count}


@app.put("/alerts/{alert_id}/read", tags=["Alerts"])
def mark_alert_read(
    alert_id: str,
    db:       Session   = Depends(get_db),
    token:    TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    from database.models import Alert
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    alert.is_read = True
    db.commit()
    return {"status": "read", "id": alert_id}


@app.put("/alerts/read-all", tags=["Alerts"])
def mark_all_alerts_read(
    db:    Session   = Depends(get_db),
    token: TokenData = Depends(require_operator),
) -> Dict[str, Any]:
    from database.models import Alert
    count = db.query(Alert).filter(Alert.is_read == False).update({"is_read": True})
    db.commit()
    return {"status": "all_read", "count": count}
