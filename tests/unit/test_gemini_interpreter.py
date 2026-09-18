"""Unit tests for the Gemini backend.

The Gemini interpreter now uses ``httpx`` to POST to the official
``/v1beta/interactions`` endpoint directly (no SDK dependency). These tests
mock ``httpx.Client.post`` so no network, no API key, and no SDK are needed.

Coverage:
  1. **Happy path** — first model returns valid JSON; no fallback.
  2. **Timeout on first model, success on second** — strict 2-second
     timeout enforced via ``asyncio.wait_for``; orchestrator switches models.
  3. **All models fail** — orchestrator raises
     ``LLMError("all gemini models failed")`` and ``interpret_notes``
     falls through to deterministic.
  4. **Malformed JSON on first model, success on second** — JSON validation
     happens before returning; bad JSON triggers fallback.
  5. **Wrong directive count** — guardrail rejects list-of-wrong-length.
  6. **Missing API key** — construction raises ``LLMError``.
  7. **Fallback list parsed from env** — comma-separated string is split.
  8. **End-to-end**: total Gemini failure falls back to deterministic.
  9. **Interactions API URL + headers** — verify the request shape.
 10. **Structured-output JSON schema** — verify response_format.json_schema.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.llm import LLMError, interpret_notes
from app.llm.gemini import INTERACTIONS_URL


# --------------------------------------------------------------------------- #
# Settings helper                                                             #
# --------------------------------------------------------------------------- #
def _settings(**overrides) -> Settings:
    base = dict(
        host="127.0.0.1",
        port=0,
        log_level="WARNING",
        llm_provider="gemini",
        llm_api_key=None,
        llm_model="gpt-4o-mini",
        llm_base_url=None,
        llm_timeout_s=20.0,
        llm_temperature=0.0,
        gemini_api_key="test-key",
        gemini_model_fallback_list=[
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
        ],
        llm_request_timeout_seconds=2.0,
        solver_time_limit_s=15.0,
        numerical_tolerance_kwh=0.01,
        numerical_tolerance_bdt=0.01,
    )
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _make_response(status_code: int, payload: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    if payload is None:
        r.json.side_effect = json.JSONDecodeError("err", "", 0)
        r.text = "not-json"
    else:
        r.json.return_value = payload
        r.text = json.dumps(payload)
    return r


def _interaction_response(text: str) -> Dict[str, Any]:
    """Build a Gemini Interactions API response containing ``text``."""
    return {
        "id": "v1_test",
        "status": "completed",
        "steps": [
            {"type": "thought", "signature": "..."},
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            },
        ],
    }


# --------------------------------------------------------------------------- #
# 1. Happy path                                                               #
# --------------------------------------------------------------------------- #
def test_first_model_success():
    from app.llm.gemini import GeminiInterpreter

    notes = ["Reduce solar by 80% from 10 AM to 2 PM."]
    body = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [10, 11, 12, 13], "factor": 0.2},
            "paraphrase": "Reduce solar by 80% from 10 AM to 2 PM.",
        }
    ]
    resp = _make_response(200, _interaction_response(json.dumps(body)))

    captured: Dict[str, Any] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return resp

    interp = GeminiInterpreter(_settings())
    with patch("httpx.Client.post", side_effect=fake_post):
        out = interp.interpret(notes)

    assert out == body
    assert captured["url"] == INTERACTIONS_URL
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    body_payload = captured["json"]
    assert body_payload["model"] == "gemini-3.8-flash"
    assert "response_format" in body_payload
    assert body_payload["response_format"]["type"] == "array"
    # NOTE: ``temperature`` is intentionally NOT sent — the Interactions
    # API rejects unknown parameters with HTTP 400.


# --------------------------------------------------------------------------- #
# 2. Timeout on first model, success on second                               #
# --------------------------------------------------------------------------- #
def test_timeout_triggers_fallback():
    """Force the first call to exceed the per-model timeout."""
    from app.llm.gemini import GeminiInterpreter

    notes = ["Cap grid at 5 kWh from 6 PM to 9 PM."]
    body = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [18, 19, 20], "max_grid_kwh": 5.0},
            "paraphrase": "Cap grid at 5 kWh from 6 PM to 9 PM.",
        }
    ]
    resp_slow = _make_response(200, _interaction_response("ignored"))
    resp_ok = _make_response(200, _interaction_response(json.dumps(body)))

    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Block longer than the per-model timeout. The asyncio.wait_for
            # wrapper inside the interpreter will give up and the next
            # model will be tried.
            import time as _t
            _t.sleep(0.5)
            return resp_slow
        return resp_ok

    interp = GeminiInterpreter(_settings(llm_request_timeout_seconds=0.1))
    with patch("httpx.Client.post", side_effect=fake_post):
        out = interp.interpret(notes)

    assert out == body
    # First model timed out, second succeeded.
    assert call_count["n"] == 2


# --------------------------------------------------------------------------- #
# 3. All models fail                                                          #
# --------------------------------------------------------------------------- #
def test_all_models_fail_raises_llm_error():
    from app.llm.gemini import GeminiInterpreter

    notes = ["Reduce solar by 80% from 10 AM to 2 PM."]

    def boom(*args, **kwargs):
        raise RuntimeError("simulated network outage")

    interp = GeminiInterpreter(
        _settings(
            llm_request_timeout_seconds=0.05,
            gemini_model_fallback_list=["a", "b", "c"],
        )
    )
    with patch("httpx.Client.post", side_effect=boom):
        with pytest.raises(LLMError) as excinfo:
            interp.interpret(notes)
    assert "all gemini models failed" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 4. Malformed JSON triggers fallback                                         #
# --------------------------------------------------------------------------- #
def test_malformed_json_triggers_fallback():
    from app.llm.gemini import GeminiInterpreter

    notes = ["Keep battery at min 20 kWh from 6 PM to 9 PM."]
    body = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {
                "hours": [18, 19, 20],
                "minimum_energy_kwh": 20.0,
            },
            "paraphrase": "Keep battery at min 20 kWh from 6 PM to 9 PM.",
        }
    ]

    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(200, _interaction_response("not json at all {oops"))
        return _make_response(200, _interaction_response(json.dumps(body)))

    interp = GeminiInterpreter(_settings())
    with patch("httpx.Client.post", side_effect=fake_post):
        out = interp.interpret(notes)
    assert out == body
    assert call_count["n"] == 2


# --------------------------------------------------------------------------- #
# 5. Wrong directive count triggers fallback                                  #
# --------------------------------------------------------------------------- #
def test_wrong_directive_count_triggers_fallback():
    from app.llm.gemini import GeminiInterpreter

    notes = ["A.", "B."]
    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Return only 1 directive when 2 are required.
            return _make_response(
                200,
                _interaction_response(
                    json.dumps(
                        [
                            {
                                "note_index": 0,
                                "applies": False,
                                "directive_type": "no_op",
                                "structured_adjustment": None,
                                "paraphrase": "A.",
                            }
                        ]
                    )
                ),
            )
        # Second attempt: correct length.
        return _make_response(
            200,
            _interaction_response(
                json.dumps(
                    [
                        {
                            "note_index": 0,
                            "applies": False,
                            "directive_type": "no_op",
                            "structured_adjustment": None,
                            "paraphrase": "A.",
                        },
                        {
                            "note_index": 1,
                            "applies": False,
                            "directive_type": "no_op",
                            "structured_adjustment": None,
                            "paraphrase": "B.",
                        },
                    ]
                )
            ),
        )

    interp = GeminiInterpreter(_settings())
    with patch("httpx.Client.post", side_effect=fake_post):
        out = interp.interpret(notes)
    assert len(out) == 2
    assert call_count["n"] == 2


# --------------------------------------------------------------------------- #
# 6. Missing API key                                                          #
# --------------------------------------------------------------------------- #
def test_missing_api_key_raises():
    from app.llm.gemini import GeminiInterpreter

    with pytest.raises(LLMError) as excinfo:
        GeminiInterpreter(_settings(gemini_api_key=None))
    assert "GEMINI_API_KEY" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 7. Env var fallback list parsing                                            #
# --------------------------------------------------------------------------- #
def test_fallback_list_parsed_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL_FALLBACK_LIST", "m1, m2 ,,m3")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    s = Settings()
    assert s.gemini_model_fallback_list == ["m1", "m2", "m3"]


# --------------------------------------------------------------------------- #
# 8. End-to-end: total Gemini failure falls back to deterministic            #
# --------------------------------------------------------------------------- #
def test_total_outage_falls_back_to_deterministic(monkeypatch):
    notes = [
        "Reduce solar by 80% from 10 AM to 2 PM.",
        "Cafeteria menu changed today.",
    ]
    s = _settings(
        llm_request_timeout_seconds=0.05,
        gemini_model_fallback_list=["only-model"],
    )
    assert s.llm_provider == "gemini"

    # Patch the underlying http post to always raise.
    with patch(
        "httpx.Client.post",
        side_effect=RuntimeError("simulated outage"),
    ):
        out = interpret_notes(notes, settings=s)

    assert len(out.directives) == 2
    assert out.directives[0].directive_type == "solar_reduction"
    assert out.directives[0].applies is True
    assert out.directives[1].directive_type == "no_op"
    assert out.directives[1].applies is False


# --------------------------------------------------------------------------- #
# 9. HTTP error from the API triggers fallback                                #
# --------------------------------------------------------------------------- #
def test_http_error_triggers_fallback():
    """A 401 / 403 / 404 from the API should be treated as a failure and
    the orchestrator should move to the next model."""
    from app.llm.gemini import GeminiInterpreter

    notes = ["Reduce solar by 80% from 10 AM to 2 PM."]
    body = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [10, 11, 12, 13], "factor": 0.2},
            "paraphrase": "ok",
        }
    ]

    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(401, {"error": "unauthorized"})
        return _make_response(200, _interaction_response(json.dumps(body)))

    interp = GeminiInterpreter(_settings())
    with patch("httpx.Client.post", side_effect=fake_post):
        out = interp.interpret(notes)
    assert out == body
    assert call_count["n"] == 2


# --------------------------------------------------------------------------- #
# 10. Strip Markdown code fences                                              #
# --------------------------------------------------------------------------- #
def test_markdown_code_fences_stripped():
    from app.llm.gemini import GeminiInterpreter

    notes = ["x"]
    body = [
        {"note_index": 0, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "paraphrase": "x"}
    ]
    fenced = "```json\n" + json.dumps(body) + "\n```"

    def fake_post(url, **kwargs):
        return _make_response(200, _interaction_response(fenced))

    interp = GeminiInterpreter(_settings())
    with patch("httpx.Client.post", side_effect=fake_post):
        out = interp.interpret(notes)
    assert out == body


# --------------------------------------------------------------------------- #
# 11. Legacy candidates shape still supported (forward compat)                #
# --------------------------------------------------------------------------- #
def test_legacy_candidates_shape_still_supported():
    from app.llm.gemini import GeminiInterpreter

    notes = ["x"]
    body = [{"note_index": 0, "applies": False, "directive_type": "no_op",
             "structured_adjustment": None, "paraphrase": "x"}]
    legacy = {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(body)}]}}
        ]
    }

    def fake_post(url, **kwargs):
        return _make_response(200, legacy)

    interp = GeminiInterpreter(_settings())
    with patch("httpx.Client.post", side_effect=fake_post):
        out = interp.interpret(notes)
    assert out == body
