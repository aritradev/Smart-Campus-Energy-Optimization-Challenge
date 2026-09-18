"""Phase 2 — directive interpreter tests.

Covers:
* canonical phrasings
* paraphrases / synonyms
* AM/PM and 24h time conversion
* percentage semantics (80% reduction -> factor 0.2)
* no_op detection (distractors)
* mixed 1..3 note scenarios
* robust against malformed inputs
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm import interpret_notes
from app.llm.deterministic import DeterministicInterpreter
from app.schemas.directives import LLMDirectiveList
from app.config import Settings
from dataclasses import replace


def _interpret(note: str | list[str]) -> LLMDirectiveList:
    notes = [note] if isinstance(note, str) else note
    return interpret_notes(notes, settings=Settings(llm_provider="deterministic"))


# ---------------------------------------------------------------------------
# solar_reduction
# ---------------------------------------------------------------------------


def test_solar_reduction_canonical():
    out = _interpret("Reduce solar availability by 80% from 10 AM to 2 PM.")
    assert out.directives[0].directive_type == "solar_reduction"
    assert out.directives[0].applies is True
    adj = out.directives[0].structured_adjustment
    assert adj.hours == [10, 11, 12, 13]
    assert adj.factor == pytest.approx(0.2)


def test_solar_reduction_paraphrase_same_meaning():
    a = _interpret("Solar production will be only 20% of normal between 10:00 and 14:00.").directives[0]
    b = _interpret("Reduce solar availability by 80% from 10 AM to 2 PM.").directives[0]
    assert a.directive_type == "solar_reduction"
    assert b.directive_type == "solar_reduction"
    assert a.structured_adjustment.hours == b.structured_adjustment.hours == [10, 11, 12, 13]
    assert a.structured_adjustment.factor == pytest.approx(0.2)
    assert b.structured_adjustment.factor == pytest.approx(0.2)


def test_solar_reduction_50_percent():
    out = _interpret("Cut solar output by 50% during hours 11 to 13.")
    d = out.directives[0]
    assert d.directive_type == "solar_reduction"
    assert d.structured_adjustment.hours == [11, 12]
    assert d.structured_adjustment.factor == pytest.approx(0.5)


def test_solar_reduction_clamps_factor():
    out = _interpret("Reduce solar by 120% from 10 to 11.")
    d = out.directives[0]
    assert d.directive_type == "solar_reduction"
    assert d.structured_adjustment.factor == 0.0


def test_solar_reduction_no_op_for_non_solar():
    out = _interpret("Reduce consumption by 80% from 10 AM to 2 PM.")
    # Not solar-related — should fall back (consumption not modelled) to no_op.
    assert out.directives[0].directive_type == "no_op"


# ---------------------------------------------------------------------------
# minimum_battery_reserve
# ---------------------------------------------------------------------------


def test_minimum_battery_reserve_canonical():
    out = _interpret("Keep the battery at a minimum of 20 kWh from 6 PM to 9 PM.")
    d = out.directives[0]
    assert d.directive_type == "minimum_battery_reserve"
    assert d.structured_adjustment.hours == [18, 19, 20]
    assert d.structured_adjustment.minimum_energy_kwh == 20.0


def test_minimum_battery_reserve_paraphrase():
    out = _interpret("Maintain a minimum battery reserve of 15 kWh during 14:00 to 16:00.")
    d = out.directives[0]
    assert d.directive_type == "minimum_battery_reserve"
    assert d.structured_adjustment.hours == [14, 15]
    assert d.structured_adjustment.minimum_energy_kwh == 15.0


# ---------------------------------------------------------------------------
# no_charge_window
# ---------------------------------------------------------------------------


def test_no_charge_window_canonical():
    out = _interpret("Do not charge the battery between 2 PM and 4 PM.")
    d = out.directives[0]
    assert d.directive_type == "no_charge_window"
    assert d.structured_adjustment.hours == [14, 15]


def test_no_charge_window_paraphrase_variants():
    samples = [
        # 14:00 through 15:00 -> [14] (start inclusive, end exclusive)
        ("Battery charging should be disabled from 14:00 through 15:00.", [14]),
        # 2-4 PM -> [14, 15] (both PM applied, start inclusive)
        ("Prevent the battery from charging during the 2-4 PM period.", [14, 15]),
        # 2 PM to 4 PM -> [14, 15]
        ("Stop charging the battery between 2 PM and 4 PM.", [14, 15]),
    ]
    for s, expected in samples:
        out = _interpret(s)
        d = out.directives[0]
        assert d.directive_type == "no_charge_window", s
        assert d.structured_adjustment.hours == expected, s


# ---------------------------------------------------------------------------
# no_discharge_window
# ---------------------------------------------------------------------------


def test_no_discharge_window_canonical():
    out = _interpret("Do not draw energy from the battery between 6 PM and 10 PM.")
    d = out.directives[0]
    assert d.directive_type == "no_discharge_window"
    assert d.structured_adjustment.hours == [18, 19, 20, 21]


def test_no_discharge_window_paraphrase():
    out = _interpret("Battery discharge should be disabled from 19:00 to 22:00.")
    d = out.directives[0]
    assert d.directive_type == "no_discharge_window"
    assert d.structured_adjustment.hours == [19, 20, 21]


# ---------------------------------------------------------------------------
# max_grid_window
# ---------------------------------------------------------------------------


def test_max_grid_window_canonical():
    out = _interpret("Cap grid usage at 5 kWh from 6 PM to 9 PM.")
    d = out.directives[0]
    assert d.directive_type == "max_grid_window"
    assert d.structured_adjustment.hours == [18, 19, 20]
    assert d.structured_adjustment.max_grid_kwh == 5.0


def test_max_grid_window_paraphrase():
    out = _interpret("Limit grid draw to 3 kWh during hours 17 to 20.")
    d = out.directives[0]
    assert d.directive_type == "max_grid_window"
    assert d.structured_adjustment.hours == [17, 18, 19]
    assert d.structured_adjustment.max_grid_kwh == 3.0


# ---------------------------------------------------------------------------
# no_op / distractors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("note", [
    "The cafeteria will serve a new menu tomorrow.",
    "Security patrol will run extra rounds tonight.",
    "Tomorrow's class schedule is postponed.",
    "The seminar room will be cleaned in the morning.",
    "Notice: parking lot B is closed today.",
    "Birthday celebration in the staff lounge this afternoon.",
])
def test_distractor_is_no_op(note):
    out = _interpret(note)
    d = out.directives[0]
    assert d.directive_type == "no_op"
    assert d.applies is False
    assert d.structured_adjustment is None


# ---------------------------------------------------------------------------
# Time-conversion edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("from 13:00 to 15:00", [13, 14]),
    ("from 1 PM to 3 PM", [13, 14]),
    ("between 1:00 PM and 3:00 PM", [13, 14]),
    ("during 22:00 through 02:00", [0, 1, 22, 23]),
    ("from 11 PM to 1 AM", [0, 23]),
])
def test_time_extraction(text, expected):
    out = _interpret(f"Do not charge the battery {text}.")
    d = out.directives[0]
    assert d.directive_type == "no_charge_window"
    assert d.structured_adjustment.hours == expected


# ---------------------------------------------------------------------------
# Mixed 1..3 note scenarios
# ---------------------------------------------------------------------------


def test_three_notes():
    notes = [
        "Reduce solar by 80% from 10 AM to 2 PM.",
        "Do not charge the battery between 14:00 and 16:00.",
        "Cafeteria will close early today.",
    ]
    out = interpret_notes(notes, settings=Settings(llm_provider="deterministic"))
    assert len(out.directives) == 3
    assert out.directives[0].directive_type == "solar_reduction"
    assert out.directives[1].directive_type == "no_charge_window"
    assert out.directives[2].directive_type == "no_op"
    assert [d.note_index for d in out.directives] == [0, 1, 2]


def test_single_distractor_note():
    notes = ["Staff meeting at 5 PM."]
    out = interpret_notes(notes, settings=Settings(llm_provider="deterministic"))
    assert len(out.directives) == 1
    assert out.directives[0].directive_type == "no_op"


# ---------------------------------------------------------------------------
# Robustness against malformed LLM-style output
# ---------------------------------------------------------------------------


def test_interpreter_handles_unicode_and_punctuation():
    notes = ["Solar — reduce by 50% from 10 to 12."]
    out = interpret_notes(notes, settings=Settings(llm_provider="deterministic"))
    d = out.directives[0]
    assert d.directive_type == "solar_reduction"
    assert d.structured_adjustment.factor == pytest.approx(0.5)


def test_interpreter_handles_garbage():
    notes = ["??????????"]
    out = interpret_notes(notes, settings=Settings(llm_provider="deterministic"))
    assert out.directives[0].directive_type == "no_op"


def test_interpreter_handles_multiple_time_mentions():
    notes = ["Do not discharge the battery between 6 PM and 9 PM, even though we usually discharge around 7 PM."]
    out = interpret_notes(notes, settings=Settings(llm_provider="deterministic"))
    d = out.directives[0]
    assert d.directive_type == "no_discharge_window"
    assert d.structured_adjustment.hours == [18, 19, 20]


def test_interpreter_does_not_crash_on_empty_string():
    # Empty strings should never reach here (Pydantic rejects them) but if
    # called directly they must not crash.
    di = DeterministicInterpreter()
    result = di.interpret([" "])
    # It still produces a (possibly no_op) directive.
    assert isinstance(result, list) and len(result) == 1
