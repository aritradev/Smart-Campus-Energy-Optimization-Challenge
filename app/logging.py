"""Structured logging utility.

Uses `python-json-logger` when available, falls back to plain text format otherwise.
Implements a deterministic, secret-safe log schema.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

try:  # pragma: no cover - import guard
    from pythonjsonlogger import jsonlogger  # type: ignore
    _HAS_JSON = True
except Exception:  # pragma: no cover
    _HAS_JSON = False


_SECRET_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
)


def _sanitize(record: logging.LogRecord) -> dict[str, Any]:
    """Extract a sanitized dictionary from a log record.

    Any attribute whose name contains a secret key is replaced with ``***``.
    """
    base: dict[str, Any] = {
        "timestamp": self_iso(getattr(record, "created", None)),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    for key, value in record.__dict__.items():
        if key.startswith("_"):
            continue
        if key in (
            "args", "msg", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno",
            "funcName", "created", "msecs", "relativeCreated", "thread",
            "threadName", "processName", "process", "name", "message",
        ):
            continue
        if any(s in key.lower() for s in _SECRET_KEYS):
            base[key] = "***"
        else:
            base[key] = value
    if record.exc_info:
        base["exc_info"] = self_format_exception(record.exc_info)
    return base


def self_format_exception(exc_info) -> str:
    import traceback
    return "".join(traceback.format_exception(*exc_info))


def self_iso(ts: float | None) -> str:
    if ts is None:
        return ""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class _SanitizingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if any(s in key.lower() for s in _SECRET_KEYS):
                record.__dict__[key] = "***"
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging.

    Idempotent — calling twice does not duplicate handlers.
    """
    root = logging.getLogger()
    # remove any existing handlers (e.g. uvicorn defaults)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler: logging.Handler
    if _HAS_JSON:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(logger)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        )
    handler.addFilter(_SanitizingFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    # quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
