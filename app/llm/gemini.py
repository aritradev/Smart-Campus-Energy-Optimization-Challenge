"""Google Gemini LLM backend using the official Interactions API.

Per the official Gemini API documentation (Sep 2026), the new endpoint is::

    POST https://generativelanguage.googleapis.com/v1beta/interactions
    Headers:
        x-goog-api-key: $GEMINI_API_KEY
        Content-Type:   application/json
    Body:
        {
          "model": "gemini-3.8-flash",
          "input": "...",
          "system_instruction": "...",
          "response_format": {"type": "array", "schema": {...}}
        }

Response::

    {
      "id": "v1_...",
      "status": "completed",
      "steps": [
        {"type": "thought", "signature": "..."},
        {"type": "model_output", "content": [{"type": "text", "text": "..."}]}
      ]
    }

Features:

* Strict **3.0-second** per-model timeout (``settings.llm_request_timeout_seconds``).
* Optimistic multi-model fallback over ``settings.gemini_model_fallback_list``.
* Process-local **LRU cache** keyed by SHA-256 of the note sequence to
  short-circuit duplicate queries and protect rate-limited (free-tier)
  API keys from quota exhaustion.
* JSON-mode structured output enforced via ``response_format`` with a
  JSON Schema mirroring the Pydantic directive models.
* On total outage raises :class:`LLMError`; orchestrator falls through
  to the deterministic interpreter.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings
from app.logging import get_logger
from app.llm import LLMError

logger = get_logger(__name__)


# Dedicated executor for Gemini SDK calls. Sized so that one slow model
# does NOT block the next model's call. We have up to ~7 models in the
# fallback list; 16 worker threads gives 2x headroom per model.
_GEMINI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="gemini"
)


# --------------------------------------------------------------------------- #
# In-memory LRU cache                                                          #
# --------------------------------------------------------------------------- #
# Many competition scenarios ask the same set of notes repeatedly. Without a
# cache, every retry against the live Gemini API burns quota. The cache key
# is a SHA-256 of the JSON-serialised note list so the (notes, scenario_id,
# model, prompt) tuple is implicitly preserved.
class _LRUCache:
    """Simple thread-safe LRU cache."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(0, int(capacity))
        self._data: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if self.capacity == 0:
            return None
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key: str, value: List[Dict[str, Any]]) -> None:
        if self.capacity == 0:
            return
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
                return
            self._data[key] = value
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# Module-level cache; populated lazily by GeminiInterpreter instances.
_NOTE_CACHE = _LRUCache(capacity=256)


def _cache_key(notes: List[str]) -> str:
    """Stable SHA-256 hash of the note sequence."""
    payload = json.dumps(notes, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clear_note_cache() -> None:
    """Wipe the in-memory cache (useful for tests)."""
    _NOTE_CACHE.clear()


INTERACTIONS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)

# Response JSON schema enforced via Gemini's structured output. Mirrors the
# Pydantic models in ``app/schemas/directives.py`` so the LLM is forced to
# produce exactly the directive shapes the rest of the pipeline expects.
RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "note_index": {"type": "integer", "minimum": 0},
            "applies": {"type": "boolean"},
            "directive_type": {
                "type": "string",
                "enum": [
                    "solar_reduction",
                    "minimum_battery_reserve",
                    "no_charge_window",
                    "no_discharge_window",
                    "max_grid_window",
                    "no_op",
                ],
            },
            "structured_adjustment": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "hours": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 23},
                                "minItems": 1,
                                "uniqueItems": True,
                            },
                            "factor": {"type": "number", "minimum": 0, "maximum": 1},
                            "minimum_energy_kwh": {"type": "number", "minimum": 0},
                            "max_grid_kwh": {"type": "number", "minimum": 0},
                        },
                        "required": ["hours"],
                        "additionalProperties": False,
                    },
                ],
            },
            "paraphrase": {"type": "string"},
        },
        "required": [
            "note_index",
            "applies",
            "directive_type",
            "structured_adjustment",
            "paraphrase",
        ],
        "additionalProperties": False,
    },
}


_SYSTEM_PROMPT = """You are GridWise, an expert directive interpreter for a campus energy optimization system.

You receive 1 to 3 free-form operator notes. Your ONLY job is to interpret each note as exactly ONE structured directive. Never combine multiple directives. Never invent directives. Never change scenario data.

## Six supported directive types

For each note, output exactly one of these `directive_type` values:

1. solar_reduction — solar availability is reduced by some percentage over some hours.
     structured_adjustment = {"hours": [int...], "factor": float in [0, 1]}
     The factor is the REMAINING usable solar fraction.
     "80% reduction" -> factor = 0.2
     "20% of normal" -> factor = 0.2
     "only 30% available" -> factor = 0.3

2. minimum_battery_reserve — battery must stay at or above a minimum kWh level.
     structured_adjustment = {"hours": [int...], "minimum_energy_kwh": float >= 0}

3. no_charge_window — battery must not charge during these hours.
     structured_adjustment = {"hours": [int...]}      (hours are 0..23)

4. no_discharge_window — battery must not discharge during these hours.
     structured_adjustment = {"hours": [int...]}      (hours are 0..23)

5. max_grid_window — grid import capped at max_grid_kwh during these hours.
     structured_adjustment = {"hours": [int...], "max_grid_kwh": float >= 0}

6. no_op — note is unrelated to energy. Use this for cafeteria, security, class schedule, events, etc.
     structured_adjustment = null, "applies": false

## Time conversion rules

- Hours must be integers 0..23.
- Ranges are START inclusive, END exclusive.
- "1 PM to 3 PM" -> [13, 14]
- "10:00 to 14:00" -> [10, 11, 12, 13]
- "from 2 PM through 4 PM" -> [14, 15]
- "midnight" -> [0]
- "noon" -> [12]
- "11 PM to 1 AM" -> [23, 0]
- "8h-10h" -> [8, 9]
- Sort ascending, dedupe.
- If no hours are mentioned and the directive type is per-hour, default to all 24 hours.

## Percentage semantics

- The `factor` in solar_reduction is the REMAINING usable fraction.
- "80% reduction" -> factor 0.2.
- "Reduce by 50%" -> factor 0.5.
- "Only 20% of normal" -> factor 0.2.

## Per-note rules

- Output one JSON object per operator note.
- Preserve order: output[i] corresponds to notes[i].
- Each object MUST have these fields:
    {"note_index": int, "applies": bool, "directive_type": str, "structured_adjustment": object|null, "paraphrase": str}
- If the note is unrelated to energy, set:
    {"note_index": i, "applies": false, "directive_type": "no_op", "structured_adjustment": null, "paraphrase": "..."}
- Active directives MUST have "applies": true and a non-null structured_adjustment matching the directive_type.

## Output format

Return ONLY a JSON ARRAY. No prose, no markdown, no comments.
"""


class GeminiInterpreter:
    """Gemini-backed LLM interpreter using the Interactions API.

    Implements:
      * Strict 2-second per-model timeout (``settings.llm_request_timeout_seconds``).
      * Optimistic multi-model fallback loop over ``settings.gemini_model_fallback_list``.
      * JSON-mode structured output enforced via ``response_format.json_schema``.
      * On total outage raises :class:`LLMError`; orchestrator falls through to deterministic.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is required for the Gemini backend")
        if not settings.gemini_model_fallback_list:
            raise LLMError(
                "gemini_model_fallback_list is empty; cannot run Gemini backend"
            )
        self.settings = settings
        self.api_key = settings.gemini_api_key
        self.models = list(settings.gemini_model_fallback_list)
        self.timeout_s = float(settings.llm_request_timeout_seconds)
        self.temperature = float(settings.llm_temperature)
        # Resize the global cache to honour the configured capacity. Setting
        # capacity=0 effectively disables caching.
        if _NOTE_CACHE.capacity != settings.gemini_cache_size:
            _NOTE_CACHE.__init__(capacity=settings.gemini_cache_size)

    # ------------------------------------------------------------------ #
    # Prompt construction                                                  #
    # ------------------------------------------------------------------ #
    def _build_prompts(self, notes: List[str]) -> tuple[str, str]:
        numbered = "\n".join(f"[{i}] {n}" for i, n in enumerate(notes))
        user_prompt = (
            "Interpret each of the following operator notes and return a JSON "
            "array of length exactly "
            f"{len(notes)}, one per note in the same order.\n\nNOTES:\n{numbered}"
        )
        return _SYSTEM_PROMPT, user_prompt

    # ------------------------------------------------------------------ #
    # Single-model synchronous HTTP call                                    #
    # ------------------------------------------------------------------ #
    def _call_model_sync(
        self, model_name: str, system_prompt: str, user_prompt: str
    ) -> str:
        """POST to the Interactions API; return the raw text response.

        Raises any underlying HTTP/JSON error. The orchestrator decides whether
        to retry on the next model.
        """
        body: Dict[str, Any] = {
            "model": model_name,
            "input": user_prompt,
            # Pin the conversation context to the canonical system prompt.
            "system_instruction": system_prompt,
            # Force structured JSON output to keep the pipeline deterministic.
            # The Interactions API accepts ``response_format.type`` of
            # ``"object"`` for a JSON object schema, or ``"array"`` for an
            # array schema. We return a top-level array of directives.
            "response_format": {
                "type": "array",
                "schema": RESPONSE_JSON_SCHEMA,
            },
        }
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout_s + 1.0) as client:
                resp = client.post(INTERACTIONS_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"gemini[{model_name}] HTTP error: {exc}"
            ) from exc

        if resp.status_code >= 400:
            # Capture the error body for diagnostics; truncate to 300 chars.
            snippet = (resp.text or "")[:300]
            raise LLMError(
                f"gemini[{model_name}] returned HTTP {resp.status_code}: {snippet}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"gemini[{model_name}] returned invalid JSON: {exc}: "
                f"{(resp.text or '')[:300]}"
            ) from exc

        # Extract text from the new Interactions response shape.
        text = _extract_text(data)
        if not text:
            raise LLMError(
                f"gemini[{model_name}] returned empty output "
                f"(keys: {list(data.keys())})"
            )
        return text

    async def _call_one_model(
        self, model_name: str, system_prompt: str, user_prompt: str
    ) -> str:
        """Run ``_call_model_sync`` with the strict per-model timeout."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    _GEMINI_EXECUTOR,
                    self._call_model_sync,
                    model_name,
                    system_prompt,
                    user_prompt,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise LLMError(
                f"gemini[{model_name}] exceeded {self.timeout_s:.2f}s timeout"
            ) from exc

    # ------------------------------------------------------------------ #
    # Optimistic multi-model fallback                                       #
    # ------------------------------------------------------------------ #
    async def _try_models(self, notes: List[str]) -> List[Dict[str, Any]]:
        # Cache check: short-circuit duplicate queries. The cache key is
        # just the note sequence — independent of model choice, scenario_id,
        # or system prompt. This is safe because all models share the same
        # output contract (LLMDirective schema).
        key = _cache_key(notes)
        cached = _NOTE_CACHE.get(key)
        if cached is not None:
            logger.info("gemini cache hit", extra={"note_count": len(notes)})
            return cached

        system_prompt, user_prompt = self._build_prompts(notes)
        last_error: Optional[Exception] = None

        for model_name in self.models:
            try:
                text = await self._call_one_model(
                    model_name, system_prompt, user_prompt
                )
            except LLMError as exc:
                logger.warning(
                    "gemini model failed or timed out; switching to next model",
                    extra={"model": model_name, "error": str(exc)},
                )
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 - any SDK / network error
                logger.warning(
                    "gemini model raised unexpected error; switching to next model",
                    extra={"model": model_name, "error": str(exc)},
                )
                last_error = exc
                continue

            # Strip Markdown code fences if Gemini returned any.
            text = _strip_code_fences(text)

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "gemini model returned non-JSON; switching to next model",
                    extra={
                        "model": model_name,
                        "error": str(exc),
                        "snippet": text[:200],
                    },
                )
                last_error = LLMError(f"gemini[{model_name}] bad JSON: {exc}")
                continue

            if isinstance(parsed, dict) and "directives" in parsed:
                directives = parsed["directives"]
            elif isinstance(parsed, list):
                directives = parsed
            else:
                logger.warning(
                    "gemini model returned unexpected JSON shape; switching",
                    extra={"model": model_name, "shape": type(parsed).__name__},
                )
                last_error = LLMError(
                    f"gemini[{model_name}] returned {type(parsed).__name__}, "
                    "expected list or {directives: [...]}"
                )
                continue

            if not isinstance(directives, list) or len(directives) != len(notes):
                logger.warning(
                    "gemini model returned wrong directive count; switching",
                    extra={
                        "model": model_name,
                        "expected": len(notes),
                        "got": (
                            len(directives) if isinstance(directives, list)
                            else None
                        ),
                    },
                )
                last_error = LLMError(
                    f"gemini[{model_name}] returned wrong directive count"
                )
                continue

            logger.info(
                "gemini model succeeded",
                extra={"model": model_name, "note_count": len(notes)},
            )
            # Cache successful result for future identical queries.
            _NOTE_CACHE.put(key, directives)
            return directives

        msg = "all gemini models failed"
        if last_error is not None:
            msg = f"{msg}: last error = {last_error}"
        raise LLMError(msg)

    # ------------------------------------------------------------------ #
    # Public sync facade                                                    #
    # ------------------------------------------------------------------ #
    def interpret(self, notes: List[str]) -> List[Dict[str, Any]]:
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False

        if loop_running:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(self._interpret_sync_via_new_loop, notes).result()
        return self._interpret_sync_via_new_loop(notes)

    def _interpret_sync_via_new_loop(self, notes: List[str]) -> List[Dict[str, Any]]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._try_models(notes))
        finally:
            loop.close()


# --------------------------------------------------------------------------- #
# Module-level helper matching the spec's exact function name                  #
# --------------------------------------------------------------------------- #
def interpret_notes_with_gemini(
    operator_notes: List[str],
    scenario_id: str,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Top-level convenience entry point.

    Constructs a :class:`GeminiInterpreter` from ``settings`` (or default
    ``Settings()`` if not supplied) and runs the multi-model fallback loop.

    Parameters
    ----------
    operator_notes:
        1–3 free-form operator notes.
    scenario_id:
        Echoed for logging only; the Gemini backend does not need it.
    settings:
        Optional :class:`Settings` instance (mostly useful for tests).

    Returns
    -------
    list of dict
        One dict per operator note, in input order, conforming to the
        :class:`LLMDirective` schema after downstream coercion.
    """
    from app.config import get_settings

    s = settings or get_settings()
    logger.info(
        "interpret_notes_with_gemini started",
        extra={"scenario_id": scenario_id, "note_count": len(operator_notes)},
    )
    interpreter = GeminiInterpreter(s)
    return interpreter.interpret(operator_notes)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _extract_text(data: Dict[str, Any]) -> str:
    """Extract the assistant text from a Gemini Interactions response.

    Supports both the new ``steps[].content[].text`` shape and the legacy
    ``candidates[].content.parts[].text`` shape for safety.
    """
    # New Interactions API shape
    steps = data.get("steps")
    if isinstance(steps, list):
        chunks: List[str] = []
        for step in steps:
            content = step.get("content") if isinstance(step, dict) else None
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        chunks.append(str(c.get("text", "")))
        if chunks:
            return "".join(chunks)

    # Convenience property exposed by SDKs
    if "output_text" in data and isinstance(data["output_text"], str):
        return data["output_text"]

    # Legacy / fallback shape
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        try:
            parts = candidates[0]["content"]["parts"]
            return "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        except (KeyError, TypeError, IndexError):
            pass

    return ""


def _strip_code_fences(text: str) -> str:
    """Remove Markdown ```json ... ``` fences if present."""
    t = text.strip()
    if t.startswith("```"):
        # Drop first line (```json or ```), then trailing fence.
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


__all__ = ["GeminiInterpreter", "interpret_notes_with_gemini", "INTERACTIONS_URL"]
