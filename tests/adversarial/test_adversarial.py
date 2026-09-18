"""Large adversarial test suite for the full pipeline.

Covers:
* boundary conditions (hour 0, hour 23, edge tariffs)
* constraint conflicts
* impossible scenarios
* large paraphrases
* every directive type at the API level
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.request import BatterySpec, EnergyHour, Scenario
from tests import default_battery, make_hour


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _scenario(
    scenario_id: str,
    notes,
    demands=None,
    solars=None,
    tariffs=None,
    battery=None,
):
    if demands is None:
        demands = [10.0] * 24
    if solars is None:
        solars = [3.0] * 24
    if tariffs is None:
        tariffs = [8.0] * 24
    return Scenario(
        scenario_id=scenario_id,
        operator_notes=notes,
        hourly_data=[
            make_hour(h, demand=d, solar=s, tariff=t)
            for h, (d, s, t) in enumerate(zip(demands, solars, tariffs))
        ],
        battery=battery or default_battery(),
    )


def _post(client, scenario):
    payload = {
        "scenario_id": scenario.scenario_id,
        "operator_notes": scenario.operator_notes,
        "hourly_data": [h.model_dump() for h in scenario.hourly_data],
        "battery": scenario.battery.model_dump(),
    }
    return client.post("/optimize-energy", json=payload)


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


def test_boundary_zero_demand(client):
    scenario = _scenario(
        "zero-demand",
        ["Cafeteria menu change."],
        demands=[0.0] * 24,
        solars=[0.0] * 24,
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    body = r.json()
    assert body["total_grid_kwh"] == pytest.approx(0.0, abs=0.01)
    assert body["total_cost_bdt"] == pytest.approx(0.0, abs=0.01)


def test_boundary_very_high_tariff(client):
    scenario = _scenario(
        "high-tariff",
        ["Do not charge between 14:00 and 15:00."],
        tariffs=[100.0] * 24,
    )
    r = _post(client, scenario)
    assert r.status_code == 200


def test_boundary_initial_equals_capacity(client):
    battery = BatterySpec(
        capacity_kwh=50.0,
        initial_energy_kwh=50.0,
        minimum_energy_kwh=0.0,
        max_charge_kwh_per_hour=10.0,
        max_discharge_kwh_per_hour=10.0,
    )
    scenario = _scenario(
        "init-full", ["ignore"], battery=battery,
        demands=[5.0] * 24, solars=[0.0] * 24,
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    body = r.json()
    # End-of-day neutrality must hold
    assert body["hourly_plan"][-1]["battery_energy_after_kwh"] == pytest.approx(50.0, abs=0.01)


def test_boundary_initial_equals_minimum(client):
    battery = BatterySpec(
        capacity_kwh=50.0,
        initial_energy_kwh=5.0,
        minimum_energy_kwh=5.0,
        max_charge_kwh_per_hour=10.0,
        max_discharge_kwh_per_hour=10.0,
    )
    scenario = _scenario(
        "init-min", ["ignore"], battery=battery,
        demands=[5.0] * 24, solars=[0.0] * 24,
    )
    r = _post(client, scenario)
    assert r.status_code == 200


def test_boundary_one_am_to_three_am(client):
    scenario = _scenario(
        "1am-3am",
        ["Do not charge between 1 AM and 3 AM."],
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    body = r.json()
    h = body["directive_interpretation"][0]
    assert h["directive_type"] == "no_charge_window"
    assert h["structured_adjustment"]["hours"] == [1, 2]


def test_boundary_eleven_pm_to_one_am(client):
    scenario = _scenario(
        "11pm-1am",
        ["Do not discharge between 11 PM and 1 AM."],
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    body = r.json()
    h = body["directive_interpretation"][0]
    assert h["directive_type"] == "no_discharge_window"
    assert h["structured_adjustment"]["hours"] == [0, 23]


def test_boundary_100_percent_reduction(client):
    """'Reduce by 100%' -> factor 0.0 (solar completely curtailed)."""
    scenario = _scenario(
        "100pct-reduction",
        ["Reduce solar availability by 100% from 12 PM to 1 PM."],
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    h = r.json()["directive_interpretation"][0]
    assert h["structured_adjustment"]["factor"] == 0.0


def test_boundary_zero_percent_reduction(client):
    """'Reduce by 0%' -> factor 1.0 (solar untouched)."""
    scenario = _scenario(
        "0pct-reduction",
        ["Reduce solar availability by 0% from 12 PM to 1 PM."],
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    h = r.json()["directive_interpretation"][0]
    assert h["structured_adjustment"]["factor"] == 1.0


# ---------------------------------------------------------------------------
# All directive types at the API level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("note,expected_type", [
    ("Reduce solar availability by 80% from 10 AM to 2 PM.", "solar_reduction"),
    ("Keep the battery at a minimum of 20 kWh from 6 PM to 9 PM.", "minimum_battery_reserve"),
    ("Do not charge the battery between 2 PM and 4 PM.", "no_charge_window"),
    ("Do not draw energy from the battery between 6 PM and 10 PM.", "no_discharge_window"),
    ("Cap grid usage at 5 kWh from 6 PM to 9 PM.", "max_grid_window"),
])
def test_all_directive_types(client, note, expected_type):
    scenario = _scenario(f"type-{expected_type}", [note])
    r = _post(client, scenario)
    assert r.status_code == 200
    h = r.json()["directive_interpretation"][0]
    assert h["directive_type"] == expected_type
    assert h["applies"] is True


@pytest.mark.parametrize("note", [
    "Cafeteria menu has changed today.",
    "Security patrol will run extra rounds tonight.",
    "Tomorrow's class schedule is postponed.",
    "Parking lot B is closed.",
])
def test_all_distractors(client, note):
    scenario = _scenario("distractor", [note])
    r = _post(client, scenario)
    assert r.status_code == 200
    h = r.json()["directive_interpretation"][0]
    assert h["directive_type"] == "no_op"
    assert h["applies"] is False
    assert h["structured_adjustment"] is None


# ---------------------------------------------------------------------------
# Constraint conflicts
# ---------------------------------------------------------------------------


def test_constraint_conflict_max_grid_and_no_discharge(client):
    """max_grid + no_discharge: feasible because solar can supply demand."""
    scenario = _scenario(
        "conflict-1",
        [
            "Cap grid usage at 2 kWh from 6 PM to 9 PM.",
            "Do not discharge the battery between 6 PM and 9 PM.",
        ],
        demands=[2.0] * 24,
        solars=[3.0] * 24,
        tariffs=[8.0 if h < 18 or h >= 21 else 15.0 for h in range(24)],
        battery=BatterySpec(
            capacity_kwh=80.0,
            initial_energy_kwh=40.0,
            minimum_energy_kwh=0.0,
            max_charge_kwh_per_hour=15.0,
            max_discharge_kwh_per_hour=15.0,
        ),
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    body = r.json()
    for e in body["hourly_plan"]:
        if 18 <= e["hour"] <= 20:
            assert e["grid_kwh"] <= 2.01
            assert e["battery_discharge_kwh"] == pytest.approx(0.0, abs=0.01)


def test_constraint_conflict_high_reserve_expensive_hours(client):
    """Battery must stay at 30 kWh during peak — feasible if pre-charged."""
    scenario = _scenario(
        "conflict-2",
        ["Keep the battery at a minimum of 30 kWh from 6 PM to 9 PM."],
        demands=[10.0] * 24,
        solars=[3.0] * 24,
        tariffs=[3.0 if h < 8 or h >= 22 else 12.0 for h in range(24)],
        battery=BatterySpec(
            capacity_kwh=50.0,
            initial_energy_kwh=30.0,
            minimum_energy_kwh=10.0,
            max_charge_kwh_per_hour=10.0,
            max_discharge_kwh_per_hour=10.0,
        ),
    )
    r = _post(client, scenario)
    assert r.status_code == 200
    body = r.json()
    for e in body["hourly_plan"]:
        if 18 <= e["hour"] <= 20:
            assert e["battery_energy_after_kwh"] >= 30.0 - 0.01


def test_impossible_max_grid_too_low(client):
    """Grid cap so low that demand cannot be met even with battery."""
    scenario = _scenario(
        "impossible-1",
        ["Cap grid usage at 1 kWh from 12 PM to 6 PM."],
        demands=[100.0] * 24,
        solars=[0.0] * 24,
        tariffs=[8.0] * 24,
        battery=BatterySpec(
            capacity_kwh=20.0,
            initial_energy_kwh=10.0,
            minimum_energy_kwh=0.0,
            max_charge_kwh_per_hour=10.0,
            max_discharge_kwh_per_hour=10.0,
        ),
    )
    r = _post(client, scenario)
    # 6 hours * 1 kWh = 6 kWh from grid. Battery can supply max 20. Demand 600. → infeasible.
    assert r.status_code in (422, 500)  # infeasibility surfaces as controlled error


# ---------------------------------------------------------------------------
# Final-validator robustness
# ---------------------------------------------------------------------------


def test_final_validator_rejects_negative_grid():
    """Inject a hand-crafted plan with negative grid → validator must reject."""
    from app.optimization.optimizer import OptimizationError
    from app.models.domain import PlanResult
    from app.validation.final_validator import (
        FinalValidatorError,
        validate_and_recalculate,
    )
    from app.models.domain import (
        CompiledScenario,
        EffectiveLimits,
        WindowConstraints,
        scenario_to_hourly_context,
    )
    from app.schemas.request import Scenario
    from app.config import Settings

    scenario = Scenario(
        scenario_id="validator",
        operator_notes=["ignore"],
        hourly_data=[make_hour(h, demand=10.0, solar=0.0, tariff=8.0) for h in range(24)],
        battery=default_battery(),
    )
    context = scenario_to_hourly_context(scenario)
    compiled = CompiledScenario(
        scenario_id="validator",
        context=context,
        battery=scenario.battery,
        effective=EffectiveLimits(
            effective_solar=list(context.solar),
            effective_min_reserve=[scenario.battery.minimum_energy_kwh] * 24,
        ),
        windows=WindowConstraints(),
    )
    bad_plan = PlanResult(
        grid_kwh=[-1.0] + [10.0] * 23,
        solar_used_kwh=[0.0] * 24,
        battery_charge_kwh=[0.0] * 24,
        battery_discharge_kwh=[0.0] * 24,
        battery_energy_after_kwh=[scenario.battery.initial_energy_kwh] * 24,
        objective_cost_bdt=240.0,
    )
    with pytest.raises(FinalValidatorError):
        validate_and_recalculate(compiled, bad_plan, settings=Settings())
