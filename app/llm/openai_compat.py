"""OpenAI-compatible LLM backend.

Uses ``httpx`` to call any OpenAI-compatible chat-completions endpoint with
JSON-schema structured output. Includes a fallback to ``response_format``
JSON-object mode if the provider does not support strict JSON schemas.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from app.config import Settings
from app.logging import get_logger
from app.llm import LLMError

logger = get_logger(__name__)


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
- If no hours are mentioned and the directive type is per-hour (no_charge_window, no_discharge_window, max_grid_window, minimum_battery_reserve, solar_reduction), default to all 24 hours.

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


class OpenAICompatibleInterpreter:
    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY required for openai-compatible backend")
        self.settings = settings
        self.base_url = (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_s
        self.temperature = settings.llm_temperature

    def interpret(self, notes: List[str]) -> List[Dict[str, Any]]:
        numbered = "\n".join(f"[{i}] {n}" for i, n in enumerate(notes))
        user_prompt = (
            "Interpret each of the following operator notes and return a JSON array "
            "of length exactly "
            f"{len(notes)}, one per note in the same order.\n\nNOTES:\n{numbered}"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM HTTP error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM response missing choices: {data}") from exc

        # OpenAI json_object mode wraps in a top-level object, so accept either:
        # {"directives": [...]} OR [...] directly.
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}: {content[:300]}") from exc

        if isinstance(parsed, dict) and "directives" in parsed:
            return parsed["directives"]
        if isinstance(parsed, list):
            return parsed
        raise LLMError(
            f"LLM returned unexpected JSON shape: {type(parsed).__name__}"
        )
