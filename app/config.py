"""Configuration loaded from environment variables.

This module auto-loads a ``.env`` file (if present) at import time using
``python-dotenv`` so operators can ship config without exporting env vars
manually. Set ``GRIDWISE_SKIP_DOTENV=1`` to disable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    # Look for .env in the project root (parent of app/) and in CWD.
    _HERE = Path(__file__).resolve().parent.parent
    for _candidate in (_HERE / ".env", Path.cwd() / ".env"):
        if _candidate.is_file() and os.getenv("GRIDWISE_SKIP_DOTENV") != "1":
            load_dotenv(_candidate, override=False)
except ImportError:  # python-dotenv is optional
    pass


# Real, publicly available Google Gemini model IDs (Sep 2026).
# Ordered fastest / cheapest → most capable for the fallback chain.
# NOTE: gemini-2.5-flash / gemini-2.5-pro were deprecated by Google on the
# free tier and now return HTTP 404 ("no longer available to new users").
# Replaced with currently-serving alternatives.
_DEFAULT_GEMINI_FALLBACK = (
    "gemini-3.8-flash,"
    "gemini-3.7-flash,"
    "gemini-3.6-flash,"
    "gemini-3.5-flash,"
    "gemini-3.5-flash-lite,"
    "gemini-3.1-flash-lite,"
    "gemini-3.1-pro-preview"
)


@dataclass(frozen=True)
class Settings:
    """Application settings. All values are read once at startup."""

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # LLM (generic)
    llm_provider: str = os.getenv("LLM_PROVIDER", "deterministic")
    # When set, used for OpenAI-compatible providers
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY") or None
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url: Optional[str] = os.getenv("LLM_BASE_URL") or None
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT", "20"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Gemini (primary generative backend for the competition)
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or None
    gemini_model_fallback_list: List[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.getenv(
                "GEMINI_MODEL_FALLBACK_LIST", _DEFAULT_GEMINI_FALLBACK
            ).split(",")
            if m.strip()
        ]
    )
    llm_request_timeout_seconds: float = float(
        os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "3.0")
    )
    # In-memory LRU cache size for duplicate note sequences. Set to 0 to disable.
    gemini_cache_size: int = int(os.getenv("GEMINI_CACHE_SIZE", "256"))

    # Solver
    solver_time_limit_s: float = float(os.getenv("SOLVER_TIME_LIMIT", "15"))
    numerical_tolerance_kwh: float = float(os.getenv("NUMERICAL_TOLERANCE_KWH", "0.01"))
    numerical_tolerance_bdt: float = float(os.getenv("NUMERICAL_TOLERANCE_BDT", "0.01"))


def get_settings() -> Settings:
    return Settings()
