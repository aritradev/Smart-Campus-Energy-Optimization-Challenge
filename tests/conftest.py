"""Shared pytest configuration.

**Important**: tests must NEVER hit the real Gemini API. We force
``LLM_PROVIDER=deterministic`` for the entire test session so the
integration / adversarial tests run offline against the deterministic
interpreter (which is what they were designed for). The Gemini backend
itself is exercised only by the mock-based unit tests in
``tests/unit/test_gemini_interpreter.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app` package is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force deterministic LLM for the test session — must be set BEFORE
# any `app.*` module is imported.
os.environ.setdefault("LLM_PROVIDER", "deterministic")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("LLM_REQUEST_TIMEOUT_SECONDS", "1.0")
os.environ.setdefault("GEMINI_CACHE_SIZE", "0")  # disable cache for tests


# ---------------------------------------------------------------------------
# Per-test cache reset so unit tests can swap mock backends freely.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_gemini_cache():
    """Wipe the in-memory Gemini note cache before every test."""
    try:
        from app.llm.gemini import clear_note_cache
        clear_note_cache()
    except Exception:
        pass
    yield
    try:
        from app.llm.gemini import clear_note_cache
        clear_note_cache()
    except Exception:
        pass
