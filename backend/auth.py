"""
backend/auth.py
────────────────
JWT authentication for SmartDetect.

SECURITY POSTURE (changed 2026-08-01 — see PROJECT_REVIEW.md §4.1)
──────────────────────────────────────────────────────────────────
This module previously shipped hardcoded fallbacks for the JWT signing key
and for both account passwords, and silently degraded to unsigned base64
"tokens" when PyJWT was absent. All three are removed:

  * JWT_SECRET, ADMIN_PASSWORD and OPERATOR_PASSWORD are REQUIRED. The
    process refuses to start without them (see require_configured()).
  * PyJWT is a hard dependency. There is no unsigned-token fallback: that
    path let anyone mint an admin token with base64 alone.
  * Passwords are compared with hmac.compare_digest (constant time).

Local development: run `python scripts/generate_env.py` once to write a .env
with a random secret and random passwords.

Roles
    operator  — person data, streams, search, cameras
    admin     — everything, plus logs and settings

Token lifetime: JWT_EXPIRE_HOURS (default 8).
"""

from __future__ import annotations

import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

load_dotenv()   # so a local .env satisfies the required vars below

# PyJWT is REQUIRED. The previous ImportError fallback produced unsigned
# tokens that anyone could forge — a silent auth bypass. Fail loudly instead.
try:
    import jwt as pyjwt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyJWT is required for authentication but is not installed.\n"
        "    pip install PyJWT==2.8.0\n"
        "Refusing to start: the previous unsigned-token fallback allowed "
        "trivial authentication bypass."
    ) from exc


ALGORITHM = "HS256"
EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))

# Minimum entropy for the signing key. 32 chars of the generator's alphabet
# is ~190 bits; anything shorter is likely a human-typed placeholder.
_MIN_SECRET_LEN = 32

_REQUIRED_VARS = ("JWT_SECRET", "ADMIN_PASSWORD", "OPERATOR_PASSWORD")

# Values that were previously shipped as defaults. If any of them shows up in
# the environment, someone has copied the old insecure config forward.
_KNOWN_BAD = {
    "smartdetect-secret-key-change-in-prod",
    "smartdetect-docker-secret-change-in-prod",
    "smartAdmin2024",
    "smartOp2024",
    "changeme", "password", "secret",
}


class ConfigurationError(RuntimeError):
    """Raised at import/startup when required auth config is missing or weak."""


def require_configured() -> None:
    """
    Validate auth configuration. Called at application startup so the process
    dies immediately rather than serving a biometric database with guessable
    credentials.
    """
    missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
    problems: list[str] = []

    if missing:
        problems.append(f"missing required environment variables: {', '.join(missing)}")

    secret = os.getenv("JWT_SECRET", "")
    if secret and len(secret) < _MIN_SECRET_LEN:
        problems.append(
            f"JWT_SECRET is {len(secret)} chars; at least {_MIN_SECRET_LEN} required")

    for var in _REQUIRED_VARS:
        val = os.getenv(var)
        if val and val in _KNOWN_BAD:
            problems.append(
                f"{var} is set to a known default/placeholder value — it is "
                f"published in this repository's history and must be changed")

    if problems:
        raise ConfigurationError(
            "SmartDetect authentication is not configured securely.\n\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nFix:\n"
              "    python scripts/generate_env.py      # writes .env with random values\n"
              "  or export them yourself:\n"
              "    export JWT_SECRET=$(python -c \"import secrets;print(secrets.token_urlsafe(48))\")\n"
              "    export ADMIN_PASSWORD=...  OPERATOR_PASSWORD=...\n\n"
              "These are NOT optional: this service exposes face embeddings, "
              "photographs and live camera streams."
        )


def _secret() -> str:
    s = os.getenv("JWT_SECRET")
    if not s:
        raise ConfigurationError("JWT_SECRET is not set")
    return s


def _users() -> dict:
    """Built per-call so tests and startup validation see current env."""
    return {
        os.getenv("ADMIN_USERNAME", "admin"): {
            "password": os.getenv("ADMIN_PASSWORD") or "",
            "role": "admin",
        },
        os.getenv("OPERATOR_USERNAME", "operator"): {
            "password": os.getenv("OPERATOR_PASSWORD") or "",
            "role": "operator",
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

def create_access_token(username: str, role: str,
                        expire_hours: Optional[float] = None,
                        scope: str = "api") -> str:
    """
    Sign a JWT. `scope` distinguishes ordinary API tokens from the
    short-lived, query-string stream tokens issued for MJPEG <img> tags,
    which cannot carry an Authorization header.
    """
    hours = EXPIRE_HOURS if expire_hours is None else expire_hours
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  username,
        "role": role,
        "scope": scope,
        "exp":  now + timedelta(hours=hours),
        "iat":  now,
    }
    return pyjwt.encode(payload, _secret(), algorithm=ALGORITHM)


def decode_token(token: str, expected_scopes: tuple = ("api",)) -> TokenData:
    """Decode and validate a JWT. Raises 401 on any failure."""
    try:
        payload = pyjwt.decode(token, _secret(), algorithms=[ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired")
    except pyjwt.InvalidTokenError:
        raise _unauthorized("Invalid token")

    scope = payload.get("scope", "api")
    if scope not in expected_scopes:
        # A stream token must not be replayable against the JSON API.
        raise _unauthorized(f"Token scope '{scope}' not valid for this endpoint")
    return TokenData(username=payload.get("sub"), role=payload.get("role"))


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
    """Require a valid Bearer token. Returns decoded TokenData."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Missing or invalid Authorization header")
    return decode_token(credentials.credentials)


def require_operator(token: TokenData = Depends(require_auth)) -> TokenData:
    if token.role not in ("operator", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Operator or admin role required")
    return token


def require_admin(token: TokenData = Depends(require_auth)) -> TokenData:
    if token.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Admin role required")
    return token


# ─── Login ────────────────────────────────────────────────────────────────────

def login(payload: LoginRequest) -> LoginResponse:
    """
    Validate credentials and return a JWT.

    Compares with hmac.compare_digest so response time does not leak how much
    of the password was correct, and always runs a comparison even for unknown
    usernames so valid/invalid usernames are not distinguishable by timing.
    """
    users = _users()
    user = users.get(payload.username)
    supplied = payload.password.encode()
    expected = (user["password"] if user else secrets.token_hex(32)).encode()
    ok = hmac.compare_digest(supplied, expected)

    if user is None or not ok or not user["password"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password")

    return LoginResponse(
        access_token=create_access_token(payload.username, user["role"]),
        role=user["role"],
        expires_in=EXPIRE_HOURS * 3600,
    )


def issue_stream_token(token: TokenData, minutes: float = 60.0) -> dict:
    """
    Short-lived token for MJPEG stream URLs.

    An <img src="..."> cannot set an Authorization header, so the stream
    endpoint accepts ?token=. That token is deliberately scoped 'stream' and
    short-lived: it lands in browser history, referrer headers and server
    logs, so it must not be usable against the JSON API.
    """
    return {
        "stream_token": create_access_token(
            token.username or "unknown", token.role or "operator",
            expire_hours=minutes / 60.0, scope="stream"),
        "expires_in": int(minutes * 60),
    }
