"""
backend/auth.py
────────────────
JWT authentication for the SmartDetect Person Tracking System.

Provides:
  - POST /auth/login  — returns a signed JWT for valid credentials
  - JWT validation dependency  — protect routes with Depends(require_auth)
  - Role checking helpers      — require_operator, require_admin

Roles:
  operator  — can search trails, log sightings, read stations
  admin     — full access including user management, log viewing

Default credentials (override via .env):
  admin    / smartAdmin2024
  operator / smartOp2024

TOKEN LIFETIME: 8 hours (configurable via JWT_EXPIRE_HOURS env var)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Try PyJWT (pip install pyjwt); if missing, fall back to a stub that
# keeps the server running without authentication enforcement.
try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False

from pydantic import BaseModel

# ─── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY   = os.getenv("JWT_SECRET",       "smartdetect-secret-key-change-in-prod")
ALGORITHM    = "HS256"
EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))

# ─── In-memory user store ──────────────────────────────────────────────────────
# Production NOTE: replace this dict with a real users table.
_USERS = {
    os.getenv("ADMIN_USERNAME",    "admin"):    {
        "password": os.getenv("ADMIN_PASSWORD",    "smartAdmin2024"),
        "role":     "admin",
    },
    os.getenv("OPERATOR_USERNAME", "operator"): {
        "password": os.getenv("OPERATOR_PASSWORD", "smartOp2024"),
        "role":     "operator",
    },
}


# ─── Pydantic models ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    role:         str
    expires_in:   int     # seconds


class TokenData(BaseModel):
    username: Optional[str] = None
    role:     Optional[str] = None


# ─── Token helpers ─────────────────────────────────────────────────────────────

def create_access_token(username: str, role: str) -> str:
    """Create and sign a JWT with an expiry timestamp."""
    if not _JWT_AVAILABLE:
        # Stub token when PyJWT is not installed — NOT secure, dev only
        import base64, json as _json  # noqa: PLC0415
        payload = {"sub": username, "role": role, "stub": True}
        return "stub." + base64.b64encode(_json.dumps(payload).encode()).decode()

    expire = datetime.now(timezone.utc) + timedelta(hours=EXPIRE_HOURS)
    payload = {
        "sub":  username,
        "role": role,
        "exp":  expire,
        "iat":  datetime.now(timezone.utc),
    }
    return pyjwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT.
    Raises HTTPException 401 on any failure.
    """
    if not _JWT_AVAILABLE:
        # Stub decode
        try:
            import base64, json as _json  # noqa: PLC0415
            _, b64 = token.split(".", 1)
            data = _json.loads(base64.b64decode(b64 + "==").decode())
            return TokenData(username=data.get("sub"), role=data.get("role"))
        except Exception:  # noqa: BLE001
            raise _unauthorized("Invalid stub token")

    try:
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(username=payload.get("sub"), role=payload.get("role"))
    except pyjwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired")
    except pyjwt.InvalidTokenError:
        raise _unauthorized("Invalid token")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


# ─── FastAPI security scheme & dependencies ────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> TokenData:
    """
    Dependency — inject into any route to require a valid Bearer token.
    Returns the decoded TokenData (username + role).
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Missing or invalid Authorization header")
    return decode_token(credentials.credentials)


def require_operator(token: TokenData = Depends(require_auth)) -> TokenData:
    """Dependency — allow operator or admin role."""
    if token.role not in ("operator", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator or admin role required",
        )
    return token


def require_admin(token: TokenData = Depends(require_auth)) -> TokenData:
    """Dependency — allow admin role only."""
    if token.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return token


# ─── Login endpoint helper (imported in main.py) ──────────────────────────────

def login(payload: LoginRequest) -> LoginResponse:
    """
    Validate credentials and return a JWT.
    Called by the POST /auth/login route in main.py.
    """
    user = _USERS.get(payload.username)
    if user is None or user["password"] != payload.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(payload.username, user["role"])
    return LoginResponse(
        access_token=token,
        role=user["role"],
        expires_in=EXPIRE_HOURS * 3600,
    )
