"""LLM directive interpreter.

Provides three backends:

* **GeminiInterpreter** — Google Gemini generative LLM with structured JSON
  output and a strict 2-second-per-model timeout + multi-model fallback
  loop. This is the **primary** interpreter mandated by the competition
  rules and is selected when ``LLM_PROVIDER=gemini``.
* **DeterministicInterpreter** — a rule-based, fully offline interpreter
  that recognises the six directive types without any external call. It
  serves as the **safety net**: if the Gemini backend is unavailable, every
  model times out, or all models error, the orchestrator falls through to
  the deterministic backend so the service never returns HTTP 500 due to
  LLM outage.
* **OpenAICompatibleInterpreter** — a JSON-schema-structured-output
  interpreter backed by an OpenAI-compatible HTTP API (alternate).

All backends return ``LLMDirectiveList`` objects after Pydantic validation
and guardrail enforcement.
"""
from __future__ import annotations

import json
from typing import List, Optional

from app.logging import get_logger
from app.schemas.directives import (
    LLMDirective,
    LLMDirectiveList,
)
from app.config import Settings, get_settings

logger = get_logger(__name__)


class LLMError(Exception):
    """Raised when LLM interpretation fails irrecoverably."""


def interpret_notes(
    notes: List[str],
    settings: Optional[Settings] = None,
    battery_capacity_kwh: Optional[float] = None,
) -> LLMDirectiveList:
    """Top-level entry point.

    Selects the backend according to configuration. If the selected backend
    fails irrecoverably (Gemini total outage, OpenAI HTTP error, etc.), the
    orchestrator falls through to the deterministic interpreter so the
    service never returns HTTP 500 due to LLM unavailability.

    ``battery_capacity_kwh`` is plumbed through to the deterministic
    fallback so percentage-of-capacity phrases such as "Keep at least 50%
    of battery capacity" can be resolved to absolute kWh.
    """
    settings = settings or get_settings()
    provider = (settings.llm_provider or "deterministic").lower()

    def _make_deterministic():
        from app.llm.deterministic import DeterministicInterpreter
        return DeterministicInterpreter(battery_capacity_kwh=battery_capacity_kwh)

    if provider == "gemini":
        backend_name = "gemini"
        try:
            from app.llm.gemini import GeminiInterpreter

            backend = GeminiInterpreter(settings)
        except Exception as exc:  # pragma: no cover - import / config guard
            logger.warning(
                "gemini backend unavailable, falling back to deterministic",
                extra={"error": str(exc)},
            )
            backend = _make_deterministic()
            backend_name = "deterministic"
    elif provider in ("openai", "openai-compatible"):
        backend_name = "openai"
        try:
            from app.llm.openai_compat import OpenAICompatibleInterpreter

            backend = OpenAICompatibleInterpreter(settings)
        except Exception as exc:  # pragma: no cover - import / config guard
            logger.warning(
                "openai-compat backend unavailable, falling back to deterministic",
                extra={"error": str(exc)},
            )
            backend = _make_deterministic()
            backend_name = "deterministic"
    else:
        backend = _make_deterministic()
        backend_name = "deterministic"

    try:
        raw = backend.interpret(notes)
    except LLMError as exc:
        # Total LLM outage (every model timed out / errored / rejected).
        # Fall through to deterministic so the pipeline never returns
        # HTTP 500 solely because the LLM was unreachable.
        logger.warning(
            "LLM failed entirely, falling back to deterministic",
            extra={"backend": backend_name, "error": str(exc)},
        )
        raw = _make_deterministic().interpret(notes)

    # The interpreter always returns a list of dicts; let Pydantic validate.
    return _coerce_to_directive_list(raw, notes)


def _coerce_to_directive_list(raw, notes: List[str]) -> LLMDirectiveList:
    """Validate the interpreter output through the strict Pydantic schema."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"interpreter returned invalid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise LLMError(f"interpreter returned non-list: {type(raw).__name__}")

    if len(raw) != len(notes):
        raise LLMError(
            f"interpreter returned {len(raw)} directives, expected {len(notes)}"
        )

    parsed: List[LLMDirective] = []
    for idx, entry in enumerate(raw):
        try:
            # Force note_index to match input order regardless of what the
            # backend returned.
            obj = dict(entry)
            obj["note_index"] = idx
            parsed.append(LLMDirective.model_validate(obj))
        except Exception as exc:
            raise LLMError(
                f"interpreter entry {idx} failed validation: {exc}"
            ) from exc
    return LLMDirectiveList(directives=parsed)


__all__ = ["interpret_notes", "LLMError"]
