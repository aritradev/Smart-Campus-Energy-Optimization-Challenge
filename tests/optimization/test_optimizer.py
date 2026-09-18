"""Phase 4 — LP optimizer correctness tests.

Each scenario uses a hand-calculated expected result. Where brute-force is
possible (no battery), the test asserts the exact objective. Otherwise the
test asserts physical constraints and round-trip equalities.
"""
from __future__ import annotations

import math

import pytest

from app.config import Settings
from app.models.domain import (
    CompiledScenario,
    EffectiveLimits,
    WindowConstraints,
    scenario_to_hourly_context,
)
from app.optimization.optimizer import OptimizationError, optimize
from app.schemas.request import BatterySpec, Scenario
from tests import default_battery, make_24_hours, make_hour


def _compile(
    scenario: Scenario,
    effective_solar=None,
    effective_min_reserve=None,
    no_charge=None,
    no_discharge=None,
    max_grid=None,
):
    """Build a CompiledScenario from a scenario + directive overrides."""
    eff = EffectiveLimits(
        effective_solar=(
            effective_solar
            if effective_solar is not None
            else [h.solar_kwh for h in scenario.hourly_data]
        ),
        effective_min_reserve=(
            effective_min_reserve
            if effective_min_reserve is not None
            else [scenario.battery.minimum_energy_kwh] * 24
        ),
    )
    win = WindowConstraints(
        no_charge_hours=no_charge or [],
        no_discharge_hours=no_discharge or [],
        max_grid_kwh_per_hour=max_grid or {},
    )
    return CompiledScenario(
        scenario_id=scenario.scenario_id,
        context=scenario_to_hourly_context(scenario),
        battery=scenario.battery,
        effective=eff,
        windows=win,
    )


SETTINGS = Settings()


# ---------------------------------------------------------------------------
# Scenario A — no battery use
# ---------------------------------------------------------------------------


def test_scenario_a_no_battery_use():
    battery = BatterySpec(
        capacity_kwh=10.0,
        initial_energy_kwh=5.0,
        minimum_energy_kwh=0.0,
        max_charge_kwh_per_hour=10.0,
        max_discharge_kwh_per_hour=10.0,
    )
    scenario = Scenario(
        scenario_id="A",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=10.0, solar=0.0, tariff=8.0)
            for h in range(24)
        ],
        battery=battery,
    )
    compiled = _compile(scenario)
    result = optimize(compiled, settings=SETTINGS)
    expected = sum(10.0 * 8.0 for _ in range(24))
    # With constant tariff, cost equals demand * tariff * 24.
    assert result.objective_cost_bdt == pytest.approx(expected, abs=0.01)
    # Energy balance holds at every hour: grid = demand - solar + charge - discharge.
    for h in range(24):
        bal = result.grid_kwh[h] + result.solar_used_kwh[h] + result.battery_discharge_kwh[h]
        bal -= 10.0 + result.battery_charge_kwh[h]
        assert bal == pytest.approx(0.0, abs=0.01)
    # Total grid usage equals total demand (since solar=0 and battery net zero).
    assert sum(result.grid_kwh) == pytest.approx(240.0, abs=0.01)


# ---------------------------------------------------------------------------
# Scenario B — solar only (no deficit)
# ---------------------------------------------------------------------------


def test_scenario_b_solar_only():
    scenario = Scenario(
        scenario_id="B",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=5.0, solar=10.0, tariff=8.0) for h in range(24)
        ],
        battery=default_battery(initial=25.0, minimum=5.0, max_charge=15.0, max_discharge=15.0),
    )
    compiled = _compile(scenario)
    result = optimize(compiled, settings=SETTINGS)
    assert result.objective_cost_bdt == pytest.approx(0.0, abs=0.01)
    assert all(g == pytest.approx(0.0, abs=0.01) for g in result.grid_kwh)
    # solar_used ≤ solar_kwh (i.e. ≤ 10 every hour) — never exported.
    assert all(s <= 10.01 + 1e-6 for s in result.solar_used_kwh)
    # End-of-day neutrality
    assert result.battery_energy_after_kwh[23] == pytest.approx(25.0, abs=0.01)


# ---------------------------------------------------------------------------
# Scenario C — cheap night -> battery charge -> expensive day -> discharge
# ---------------------------------------------------------------------------


def test_scenario_c_cheap_peak_arbitrage():
    demands = [10.0] * 24
    solars = [0.0] * 24
    tariffs = [3.0 if h < 8 or h >= 22 else 12.0 for h in range(24)]
    battery = BatterySpec(
        capacity_kwh=30.0,
        initial_energy_kwh=10.0,
        minimum_energy_kwh=0.0,
        max_charge_kwh_per_hour=10.0,
        max_discharge_kwh_per_hour=10.0,
    )
    scenario = Scenario(
        scenario_id="C",
        operator_notes=["ignore"],
        hourly_data=[make_hour(h, demand=d, solar=s, tariff=t)
                     for h, (d, s, t) in enumerate(zip(demands, solars, tariffs))],
        battery=battery,
    )
    compiled = _compile(scenario)
    result = optimize(compiled, settings=SETTINGS)
    # Without battery, cost = 10 * 24 * avg tariff ≈ ...
    # With battery, expect cost to be lower than the no-battery cost.
    base_no_battery = sum(10.0 * t for t in tariffs)
    assert result.objective_cost_bdt < base_no_battery
    # End-of-day neutrality
    assert result.battery_energy_after_kwh[23] == pytest.approx(10.0, abs=1e-3)
    # Some charging in cheap hours, some discharging in expensive hours.
    cheap_charge = sum(result.battery_charge_kwh[h] for h in list(range(8)) + list(range(22, 24)))
    expensive_discharge = sum(
        result.battery_discharge_kwh[h] for h in range(8, 22)
    )
    assert cheap_charge > 0
    assert expensive_discharge > 0


# ---------------------------------------------------------------------------
# Scenario D — battery reserve constraint
# ---------------------------------------------------------------------------


def test_scenario_d_reserve_floor():
    scenario = Scenario(
        scenario_id="D",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=8.0, solar=0.0, tariff=8.0) for h in range(24)
        ],
        battery=default_battery(initial=25.0, minimum=0.0, max_charge=5.0, max_discharge=5.0),
    )
    compiled = _compile(scenario, effective_min_reserve=[20.0] * 24)
    result = optimize(compiled, settings=SETTINGS)
    assert all(
        e >= 20.0 - 0.01
        for e in result.battery_energy_after_kwh
    )
    # End-of-day neutrality
    assert result.battery_energy_after_kwh[23] == pytest.approx(25.0, abs=0.01)


# ---------------------------------------------------------------------------
# Scenario E — no charge window
# ---------------------------------------------------------------------------


def test_scenario_e_no_charge_window():
    scenario = Scenario(
        scenario_id="E",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=8.0, solar=0.0, tariff=8.0) for h in range(24)
        ],
        battery=default_battery(initial=25.0, minimum=0.0, max_charge=10.0, max_discharge=10.0),
    )
    no_charge = list(range(10, 14))
    compiled = _compile(scenario, no_charge=no_charge)
    result = optimize(compiled, settings=SETTINGS)
    for h in no_charge:
        assert result.battery_charge_kwh[h] == pytest.approx(0.0, abs=1e-6)
    assert result.battery_energy_after_kwh[23] == pytest.approx(25.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Scenario F — no discharge window
# ---------------------------------------------------------------------------


def test_scenario_f_no_discharge_window():
    scenario = Scenario(
        scenario_id="F",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=8.0, solar=0.0, tariff=8.0) for h in range(24)
        ],
        battery=default_battery(initial=25.0, minimum=0.0, max_charge=10.0, max_discharge=10.0),
    )
    no_discharge = list(range(18, 22))
    compiled = _compile(scenario, no_discharge=no_discharge)
    result = optimize(compiled, settings=SETTINGS)
    for h in no_discharge:
        assert result.battery_discharge_kwh[h] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scenario G — max grid constraint
# ---------------------------------------------------------------------------


def test_scenario_g_max_grid_constraint():
    """Tight grid cap forces battery use during peak hours."""
    demands = [20.0] * 24
    solars = [0.0] * 24
    tariffs = [3.0 if h < 8 or h >= 22 else 12.0 for h in range(24)]
    battery = BatterySpec(
        capacity_kwh=60.0,
        initial_energy_kwh=10.0,
        minimum_energy_kwh=0.0,
        max_charge_kwh_per_hour=10.0,
        max_discharge_kwh_per_hour=10.0,
    )
    scenario = Scenario(
        scenario_id="G",
        operator_notes=["ignore"],
        hourly_data=[make_hour(h, demand=d, solar=s, tariff=t)
                     for h, (d, s, t) in enumerate(zip(demands, solars, tariffs))],
        battery=battery,
    )
    max_grid = {h: 15.0 for h in range(12, 18)}
    compiled = _compile(scenario, max_grid=max_grid)
    result = optimize(compiled, settings=SETTINGS)
    for h in range(12, 18):
        assert result.grid_kwh[h] <= 15.01 + 1e-6
    # End-of-day neutrality
    assert result.battery_energy_after_kwh[23] == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Scenario H — solar reduction
# ---------------------------------------------------------------------------


def test_scenario_h_solar_reduction():
    scenario = Scenario(
        scenario_id="H",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=5.0, solar=10.0, tariff=8.0) for h in range(24)
        ],
        battery=default_battery(initial=0.0, minimum=0.0, max_charge=10.0, max_discharge=10.0),
    )
    eff_solar = [10.0 if h not in range(10, 14) else 2.0 for h in range(24)]
    compiled = _compile(scenario, effective_solar=eff_solar)
    result = optimize(compiled, settings=SETTINGS)
    for h in range(10, 14):
        assert result.solar_used_kwh[h] <= 2.01 + 1e-6


# ---------------------------------------------------------------------------
# Scenario I — multiple simultaneous directives
# ---------------------------------------------------------------------------


def test_scenario_i_multiple_directives():
    """All directives active simultaneously — must produce a feasible schedule."""
    demands = [10.0] * 24
    solars = [5.0] * 24
    tariffs = [3.0 if h < 8 or h >= 22 else 12.0 for h in range(24)]
    battery = BatterySpec(
        capacity_kwh=80.0,
        initial_energy_kwh=30.0,
        minimum_energy_kwh=0.0,
        max_charge_kwh_per_hour=15.0,
        max_discharge_kwh_per_hour=15.0,
    )
    scenario = Scenario(
        scenario_id="I",
        operator_notes=["a", "b", "c"],
        hourly_data=[make_hour(h, demand=d, solar=s, tariff=t)
                     for h, (d, s, t) in enumerate(zip(demands, solars, tariffs))],
        battery=battery,
    )
    eff_solar = [5.0 if h not in range(10, 14) else 1.0 for h in range(24)]
    eff_reserve = [scenario.battery.minimum_energy_kwh if h not in range(18, 22) else 40.0
                   for h in range(24)]
    compiled = _compile(
        scenario,
        effective_solar=eff_solar,
        effective_min_reserve=eff_reserve,
        no_charge=[10, 11, 12, 13],
        max_grid={h: 7.0 for h in range(8, 22)},
    )
    result = optimize(compiled, settings=SETTINGS)
    # End-of-day neutrality
    assert result.battery_energy_after_kwh[23] == pytest.approx(30.0, abs=0.01)
    # No charge during 10..13
    for h in [10, 11, 12, 13]:
        assert result.battery_charge_kwh[h] == pytest.approx(0.0, abs=0.01)
    # Reserve at 18..21
    for h in range(18, 22):
        assert result.battery_energy_after_kwh[h] >= 40.0 - 0.011
    # Solar limit
    for h in range(10, 14):
        assert result.solar_used_kwh[h] <= 1.01 + 1e-6
    # Max grid
    for h in range(8, 22):
        assert result.grid_kwh[h] <= 7.001 + 0.01


# ---------------------------------------------------------------------------
# Scenario J — end-of-day neutrality
# ---------------------------------------------------------------------------


def test_scenario_j_end_of_day_neutrality():
    battery = BatterySpec(
        capacity_kwh=50.0,
        initial_energy_kwh=25.0,
        minimum_energy_kwh=0.0,
        max_charge_kwh_per_hour=10.0,
        max_discharge_kwh_per_hour=10.0,
    )
    scenario = Scenario(
        scenario_id="J",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=12.0, solar=2.0, tariff=8.0) for h in range(24)
        ],
        battery=battery,
    )
    compiled = _compile(scenario)
    result = optimize(compiled, settings=SETTINGS)
    assert result.battery_energy_after_kwh[23] == pytest.approx(25.0, abs=0.01)


# ---------------------------------------------------------------------------
# Infeasibility handling
# ---------------------------------------------------------------------------


def test_infeasible_minimum_reserve():
    """A minimum reserve above capacity after rounding should be infeasible."""
    scenario = Scenario(
        scenario_id="K",
        operator_notes=["ignore"],
        hourly_data=[make_hour(h, demand=10.0, solar=0.0, tariff=8.0)
                     for h in range(24)],
        battery=default_battery(capacity=10.0, initial=10.0, minimum=10.0,
                                max_charge=10.0, max_discharge=10.0),
    )
    # Force impossible reserve
    compiled = _compile(scenario, effective_min_reserve=[20.0] * 24)
    with pytest.raises(OptimizationError):
        optimize(compiled, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Optimality: independent brute-force for a tiny scenario
# ---------------------------------------------------------------------------


def test_optimality_no_battery_brute_force():
    demands = [3.0, 5.0, 2.0, 7.0]
    solars = [0.0, 0.0, 0.0, 0.0]
    tariffs = [4.0, 5.0, 6.0, 7.0]
    battery = BatterySpec(
        capacity_kwh=1.0,  # tiny
        initial_energy_kwh=0.5,
        minimum_energy_kwh=0.0,
        max_charge_kwh_per_hour=0.001,  # effectively disable battery
        max_discharge_kwh_per_hour=0.001,
    )
    # Extend to 24 hours by padding with zeros
    full_demands = demands + [0.0] * 20
    full_solars = solars + [0.0] * 20
    full_tariffs = tariffs + [8.0] * 20
    scenario = Scenario(
        scenario_id="L",
        operator_notes=["ignore"],
        hourly_data=[make_hour(h, demand=d, solar=s, tariff=t)
                     for h, (d, s, t) in enumerate(zip(full_demands, full_solars, full_tariffs))],
        battery=battery,
    )
    compiled = _compile(scenario)
    result = optimize(compiled, settings=SETTINGS)
    expected = sum(d * t for d, t in zip(full_demands, full_tariffs))
    assert result.objective_cost_bdt == pytest.approx(expected, abs=0.01)
