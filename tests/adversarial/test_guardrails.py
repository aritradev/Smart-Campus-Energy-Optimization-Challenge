"""Adversarial tests for guardrails + directive compiler.

Target: every malformed directive must be rejected before it reaches the
solver, and overlapping directives must compose deterministically.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.directives import (
    LLMDirective,
    LLMDirectiveList,
    NoChargeWindowAdjustment,
    NoDischargeWindowAdjustment,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    SolarReductionAdjustment,
)
from app.validation.guardrails import (
    GuardrailError,
    safe_validate_directive_list,
    validate_directives_against_scenario,
)
from app.validation.compiler import compile_directives
from tests import default_battery, make_scenario


def _d(idx, applies, dtype, adj=None, paraphrase=""):
    return {
        "note_index": idx,
        "applies": applies,
        "directive_type": dtype,
        "structured_adjustment": adj,
        "paraphrase": paraphrase,
    }


# ---------------------------------------------------------------------------
# Pydantic-level guardrails
# ---------------------------------------------------------------------------


def test_duplicate_hours_rejected():
    with pytest.raises(ValidationError):
        SolarReductionAdjustment(hours=[5, 5], factor=0.2)


def test_unsorted_hours_rejected():
    with pytest.raises(ValidationError):
        SolarReductionAdjustment(hours=[10, 8], factor=0.2)


def test_negative_factor_rejected():
    with pytest.raises(ValidationError):
        SolarReductionAdjustment(hours=[10], factor=-0.1)


def test_factor_above_one_rejected():
    with pytest.raises(ValidationError):
        SolarReductionAdjustment(hours=[10], factor=1.1)


def test_hour_24_rejected():
    with pytest.raises(ValidationError):
        NoChargeWindowAdjustment(hours=[24])


def test_negative_hour_rejected():
    with pytest.raises(ValidationError):
        NoChargeWindowAdjustment(hours=[-1])


def test_negative_reserve_rejected():
    with pytest.raises(ValidationError):
        MinimumBatteryReserveAdjustment(hours=[5], minimum_energy_kwh=-1.0)


def test_negative_grid_cap_rejected():
    with pytest.raises(ValidationError):
        MaxGridWindowAdjustment(hours=[5], max_grid_kwh=-1.0)


def test_applies_false_with_no_op_enforced():
    d = LLMDirective.model_validate(_d(0, False, "no_op", None))
    assert d.directive_type == "no_op"


def test_active_directive_must_have_adjustment():
    with pytest.raises(ValidationError):
        LLMDirective.model_validate(_d(0, True, "no_charge_window", None))


def test_no_op_must_have_no_adjustment():
    with pytest.raises(ValidationError):
        LLMDirective.model_validate(_d(0, False, "no_op", {"hours": [5]}))


def test_wrong_adjustment_type_rejected():
    # directive_type=solar_reduction but adjustment is a NoChargeWindowAdjustment
    adj = NoChargeWindowAdjustment(hours=[5]).model_dump()
    with pytest.raises(ValidationError):
        LLMDirective.model_validate(_d(0, True, "solar_reduction", adj))


# ---------------------------------------------------------------------------
# Semantic guardrails
# ---------------------------------------------------------------------------


def test_semantic_rejects_reserve_above_capacity():
    scenario = make_scenario()
    big_battery_reserve = MinimumBatteryReserveAdjustment(
        hours=[10], minimum_energy_kwh=99999.0
    )
    raw = [_d(0, True, "minimum_battery_reserve", big_battery_reserve.model_dump())]
    with pytest.raises(GuardrailError):
        safe_validate_directive_list({"directives": raw}, scenario)


def test_semantic_rejects_wrong_note_index():
    scenario = make_scenario()
    raw = [_d(7, True, "no_charge_window", {"hours": [5]})]
    with pytest.raises(GuardrailError):
        safe_validate_directive_list({"directives": raw}, scenario)


def test_semantic_rejects_extra_directive():
    scenario = make_scenario()
    raw = [
        _d(0, True, "no_charge_window", {"hours": [5]}),
        _d(1, False, "no_op", None),
    ]
    with pytest.raises(GuardrailError):
        safe_validate_directive_list({"directives": raw}, scenario)


def test_semantic_rejects_unknown_directive_type():
    scenario = make_scenario()
    raw = [_d(0, True, "fake_directive", {"hours": [5]})]
    with pytest.raises(GuardrailError):
        safe_validate_directive_list({"directives": raw}, scenario)


def test_semantic_rejects_extra_field_in_adjustment():
    scenario = make_scenario()
    bad = {"hours": [5], "rogue": "field"}
    raw = [_d(0, True, "no_charge_window", bad)]
    with pytest.raises(GuardrailError):
        safe_validate_directive_list({"directives": raw}, scenario)


def test_semantic_rejects_null_adjustment_for_active():
    scenario = make_scenario()
    raw = [_d(0, True, "no_charge_window", None)]
    with pytest.raises(GuardrailError):
        safe_validate_directive_list({"directives": raw}, scenario)


def test_valid_directive_passes_guardrails():
    scenario = make_scenario()
    raw = [_d(0, True, "no_charge_window", {"hours": [5]})]
    out = safe_validate_directive_list({"directives": raw}, scenario)
    assert out.directives[0].directive_type == "no_charge_window"


# ---------------------------------------------------------------------------
# Directive compiler conflict resolution
# ---------------------------------------------------------------------------


def test_solar_reduction_overlap_takes_minimum_factor():
    scenario = make_scenario(solar=10.0, notes=["a", "b"])
    raw = [
        _d(0, True, "solar_reduction", {"hours": [10, 11], "factor": 0.5}),
        _d(1, True, "solar_reduction", {"hours": [11, 12], "factor": 0.2}),
    ]
    parsed = safe_validate_directive_list({"directives": raw}, scenario)
    compiled = compile_directives(scenario, parsed)
    assert compiled.effective.effective_solar[10] == pytest.approx(5.0)  # 0.5 * 10
    assert compiled.effective.effective_solar[11] == pytest.approx(2.0)  # min(0.5, 0.2) * 10
    assert compiled.effective.effective_solar[12] == pytest.approx(2.0)  # 0.2 * 10


def test_minimum_reserve_overlap_takes_maximum():
    scenario = make_scenario(notes=["a", "b"])
    raw = [
        _d(0, True, "minimum_battery_reserve", {"hours": [10, 11], "minimum_energy_kwh": 10.0}),
        _d(1, True, "minimum_battery_reserve", {"hours": [11, 12], "minimum_energy_kwh": 25.0}),
    ]
    parsed = safe_validate_directive_list({"directives": raw}, scenario)
    compiled = compile_directives(scenario, parsed)
    assert compiled.effective.effective_min_reserve[10] == 10.0
    assert compiled.effective.effective_min_reserve[11] == 25.0
    assert compiled.effective.effective_min_reserve[12] == 25.0


def test_max_grid_overlap_takes_tightest_cap():
    scenario = make_scenario(notes=["a", "b"])
    raw = [
        _d(0, True, "max_grid_window", {"hours": [10, 11], "max_grid_kwh": 5.0}),
        _d(1, True, "max_grid_window", {"hours": [11, 12], "max_grid_kwh": 2.0}),
    ]
    parsed = safe_validate_directive_list({"directives": raw}, scenario)
    compiled = compile_directives(scenario, parsed)
    assert compiled.windows.max_grid_kwh_per_hour[10] == 5.0
    assert compiled.windows.max_grid_kwh_per_hour[11] == 2.0
    assert compiled.windows.max_grid_kwh_per_hour[12] == 2.0


def test_no_charge_hours_unioned():
    scenario = make_scenario(notes=["a", "b"])
    raw = [
        _d(0, True, "no_charge_window", {"hours": [10, 11]}),
        _d(1, True, "no_charge_window", {"hours": [11, 12]}),
    ]
    parsed = safe_validate_directive_list({"directives": raw}, scenario)
    compiled = compile_directives(scenario, parsed)
    assert compiled.windows.no_charge_hours == [10, 11, 12]
