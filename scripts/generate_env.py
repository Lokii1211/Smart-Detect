"""
scripts/generate_env.py
────────────────────────
Write a .env with cryptographically random auth secrets.

backend/auth.py refuses to start without JWT_SECRET, ADMIN_PASSWORD and
OPERATOR_PASSWORD (the hardcoded fallbacks were removed — they were published
in this repository's history). This makes the secure path the easy path.

    python scripts/generate_env.py            # create .env if absent
    python scripts/generate_env.py --force    # regenerate (invalidates tokens)
    python scripts/generate_env.py --print    # print to stdout, write nothing
"""
from __future__ import annotations

import argparse
import secrets
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

_ALPHABET = string.ascii_letters + string.digits


def password(n: int = 24) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(n))


def render() -> str:
    return f"""# SmartDetect environment — GENERATED, DO NOT COMMIT
# Created by scripts/generate_env.py
#
# .env is gitignored. Treat this file as a credential: it protects live
# camera streams, face embeddings and photographs of identifiable people.

# ── Auth (required — the server will not start without these) ──────────────
JWT_SECRET={secrets.token_urlsafe(48)}
JWT_EXPIRE_HOURS=8

ADMIN_USERNAME=admin
ADMIN_PASSWORD={password()}

OPERATOR_USERNAME=operator
OPERATOR_PASSWORD={password()}

# ── Database ───────────────────────────────────────────────────────────────
DATABASE_URL=sqlite:///./smartdetect.db

# ── Optional ───────────────────────────────────────────────────────────────
# Expose /docs, /redoc and /openapi.json without auth (development only).
# SMARTDETECT_PUBLIC_DOCS=1
#
# Skip auto-starting CAM-001 on boot.
# SMARTDETECT_NO_AUTOSTART=1
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing .env (invalidates issued tokens)")
    ap.add_argument("--print", dest="to_stdout", action="store_true",
                    help="print instead of writing")
    args = ap.parse_args()

    content = render()

    if args.to_stdout:
        print(content)
        return

    if ENV_PATH.exists() and not args.force:
        raise SystemExit(
            f"{ENV_PATH} already exists — refusing to overwrite.\n"
            "Use --force to regenerate (all issued JWTs become invalid), "
            "or --print to see a fresh block."
        )

    ENV_PATH.write_text(content)
    ENV_PATH.chmod(0o600)   # owner-only: this file holds credentials
    print(f"Wrote {ENV_PATH} (mode 600)")
    print("\nCredentials for logging into the dashboard:")
    for line in content.splitlines():
        if line.startswith(("ADMIN_USERNAME", "ADMIN_PASSWORD",
                            "OPERATOR_USERNAME", "OPERATOR_PASSWORD")):
            print(f"  {line}")
    print("\nStore these in a password manager. They are not recoverable from "
          "the JWT secret, and regenerating .env changes them.")


if __name__ == "__main__":
    main()
