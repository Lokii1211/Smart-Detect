"""
backend/logger.py
──────────────────
Structured logging for the SmartDetect Person Tracking System.

Features:
  - Writes to /logs/system.log with rotating file handler (max 10 MB × 5 backups)
  - Also outputs to console (stdout) for Docker/terminal visibility
  - Structured format: timestamp | level | name | event | message
  - read_recent_logs(n) utility used by GET /logs endpoint

Usage example:
    from backend.logger import get_structured_logger
    logger = get_structured_logger(__name__)
    logger.info("sighting", message="Person seen at CAM-001", code="MET-20260318-0001")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import List

# ─── Log directory ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR  = _PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "system.log"

# Create the logs directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)

# ─── Log format ────────────────────────────────────────────────────────────────
#  2026-03-18 18:30:00,123 | INFO     | backend.main | sighting | Person seen at CAM-001
_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(event)s | %(message)s"
_FORMATTER = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


class _EventDefaultFilter(logging.Filter):
    """Inject a default event='' into any record that lacks it.

    Third-party libraries (matplotlib, insightface, uvicorn…) use the standard
    logging API without an 'event' extra field.  Without this filter the
    RotatingFileHandler crashes with ``KeyError: 'event'`` because our format
    string requires ``%(event)s``.
    """
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        if not hasattr(record, "event"):
            record.event = ""          # type: ignore[attr-defined]
        return True

# ─── Root handler setup (done once at import time) ────────────────────────────
_root_configured = False


def _configure_root() -> None:
    """Configure the root logger with file + console handlers (called once)."""
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Rotating file handler — keeps up to 50 MB of logs (10 MB × 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,    # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    file_handler.addFilter(_EventDefaultFilter())

    # Console handler — INFO and above only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMATTER)
    console_handler.addFilter(_EventDefaultFilter())

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    _root_configured = True


_configure_root()


# ─── StructuredLogger wrapper ───────────────────────────────────────────────────

class StructuredLogger:
    """
    Thin wrapper around the standard logger that injects an 'event' field
    into every log record, enabling structured filtering in log viewers.

    Usage:
        logger = get_structured_logger("recognition.face_recognizer")
        logger.info("model.load", message="InsightFace loaded", model="buffalo_l")
    """

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _log_msg(self, level: int, event: str, message: str, **kwargs) -> None:
        """Emit one log record with the structured event field."""
        extra_msg = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._log.log(
            level,
            message + extra_msg,
            extra={"event": event},
        )

    def debug(self, event: str, message: str, **kwargs) -> None:
        self._log_msg(logging.DEBUG, event, message, **kwargs)

    def info(self, event: str, message: str, **kwargs) -> None:
        self._log_msg(logging.INFO, event, message, **kwargs)

    def warning(self, event: str, message: str, **kwargs) -> None:
        self._log_msg(logging.WARNING, event, message, **kwargs)

    def error(self, event: str, message: str, **kwargs) -> None:
        self._log_msg(logging.ERROR, event, message, **kwargs)

    def critical(self, event: str, message: str, **kwargs) -> None:
        self._log_msg(logging.CRITICAL, event, message, **kwargs)


def get_structured_logger(name: str) -> StructuredLogger:
    """Factory — returns a StructuredLogger for the given module name."""
    return StructuredLogger(name)


# ─── Log reader (used by GET /logs) ───────────────────────────────────────────

def read_recent_logs(n: int = 100) -> List[str]:
    """
    Return the last `n` lines from system.log.
    Returns an empty list if the log file doesn't exist yet.
    """
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines[-n:]]
    except OSError:
        return []
