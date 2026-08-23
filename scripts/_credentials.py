"""
scripts/_credentials.py
────────────────────────
Credentials for helper scripts, read from the environment / .env.

The hardcoded operator/admin passwords these scripts used were removed in the
2026-08-01 security work (they were published in this repo's history). Every
script now resolves credentials the same way the server does, so there is one
place to change and nothing to leak.

    python scripts/generate_env.py     # creates .env if you have none
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def credentials(role: str = "operator") -> tuple:
    """(username, password) for 'operator' or 'admin'. Exits with guidance
    if the environment is not configured."""
    if role == "admin":
        user = os.getenv("ADMIN_USERNAME", "admin")
        pw = os.getenv("ADMIN_PASSWORD")
        var = "ADMIN_PASSWORD"
    else:
        user = os.getenv("OPERATOR_USERNAME", "operator")
        pw = os.getenv("OPERATOR_PASSWORD")
        var = "OPERATOR_PASSWORD"
    if not pw:
        sys.exit(
            f"{var} is not set.\n"
            "Helper scripts no longer carry hardcoded passwords — the old\n"
            "defaults were removed for security. Create credentials with:\n"
            "    python scripts/generate_env.py\n"
            f"or export {var} yourself."
        )
    return user, pw
