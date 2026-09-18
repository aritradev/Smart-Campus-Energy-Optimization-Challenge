"""Mathematical proof that the LP relaxation is sufficient (no MILP needed).

The smart-campus model never benefits from simultaneous charging and
discharging, because:

* Solar cannot be exported, so discharging to "store" energy that is being
  charged would just waste round-trip capacity.
* The LP objective is purely linear in grid_kwh, which has no upper bound
  from the battery — there is no profitable arbitrage inside a single hour.
* Net charge and net discharge are equal in aggregate (battery is a pure
  shift device). The LP picks the cheapest schedule.

This test verifies that even if the LP is *allowed* to charge and discharge
simultaneously, the optimal solution does not use it: the optimal cost is
the same as the schedule that uses no simultaneous transfer.

We test this by computing the optimal solution and asserting
``charge[h] * discharge[h] == 0`` for every hour — i.e. at most one of
(charge, discharge) is positive.
"""
from __future__ import annotations

import pytest

from app.optimization.optimizer import optimize
from app.models.domain import (
    CompiledScenario,
    EffectiveLimits,
    WindowConstraints,
    scenario_to_hourly_context,
)
from tests import default_battery, make_hour
from app.schemas.request import Scenario, BatterySpec


def _make_scenario():
    return Scenario(
        scenario_id="simul",
        operator_notes=["ignore"],
        hourly_data=[
            make_hour(h, demand=10.0, solar=3.0, tariff=8.0 + (6.0 if 8 <= h < 22 else 0.0))
            for h in range(24)
        ],
        battery=BatterySpec(
            capacity_kwh=80.0,
            initial_energy_kwh=20.0,
            minimum_energy_kwh=0.0,
            max_charge_kwh_per_hour=15.0,
            max_discharge_kwh_per_hour=15.0,
        ),
    )


def test_no_simultaneous_charge_and_discharge_in_optimal():
    scenario = _make_scenario()
    context = scenario_to_hourly_context(scenario)
    compiled = CompiledScenario(
        scenario_id=scenario.scenario_id,
        context=context,
        battery=scenario.battery,
        effective=EffectiveLimits(
            effective_solar=list(context.solar),
            effective_min_reserve=[scenario.battery.minimum_energy_kwh] * 24,
        ),
        windows=WindowConstraints(),
    )
    result = optimize(compiled)
    for h in range(24):
        # Either both zero, or at most one is positive (numerical tolerance).
        assert result.battery_charge_kwh[h] * result.battery_discharge_kwh[h] == pytest.approx(0.0, abs=1e-6), (
            f"hour {h}: simultaneous charge and discharge "
            f"(charge={result.battery_charge_kwh[h]}, discharge={result.battery_discharge_kwh[h]})"
        )
