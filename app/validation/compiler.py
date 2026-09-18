"""Compile LLM directives into mathematical constraints for the optimizer.

The compiler is intentionally *deterministic*: it never re-interprets a
directive. Once a directive has been classified by the interpreter and
validated by the guardrails, the compiler simply emits per-hour limits.

Conflict resolution policy (deterministic):

* solar_reduction:        multiplicative -> MINIMUM factor (most reduction wins)
* minimum_battery_reserve: per-hour       -> MAXIMUM  reserve (most restrictive)
* max_grid_window:        per-hour       -> MINIMUM  cap (most restrictive)
* no_charge_window:       union          -> any hour in ANY directive = no-charge
* no_discharge_window:    union          -> any hour in ANY directive = no-discharge
"""
from __future__ import annotations

from typing import Dict, List

from app.models.domain import (
    CompiledScenario,
    EffectiveLimits,
    HourlyContext,
    WindowConstraints,
)
from app.schemas.directives import (
    LLMDirective,
    LLMDirectiveList,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    SolarReductionAdjustment,
)
from app.schemas.request import Scenario
from app.validation.guardrails import GuardrailError


def compile_directives(
    scenario: Scenario,
    directives: LLMDirectiveList,
) -> CompiledScenario:
    """Translate directives into the LP-ready CompiledScenario."""
    context = _build_context(scenario)
    effective = EffectiveLimits(
        effective_solar=list(context.solar),
        effective_min_reserve=[
            scenario.battery.minimum_energy_kwh for _ in range(24)
        ],
    )
    windows = WindowConstraints()

    for d in directives.directives:
        if not d.applies or d.directive_type == "no_op":
            continue
        if d.structured_adjustment is None:
            # Pydantic already enforces this; defensive.
            continue
        if d.directive_type == "solar_reduction":
            assert isinstance(d.structured_adjustment, SolarReductionAdjustment)
            for h in d.structured_adjustment.hours:
                effective.effective_solar[h] = min(
                    effective.effective_solar[h],
                    context.solar[h] * d.structured_adjustment.factor,
                )
        elif d.directive_type == "minimum_battery_reserve":
            assert isinstance(d.structured_adjustment, MinimumBatteryReserveAdjustment)
            for h in d.structured_adjustment.hours:
                effective.effective_min_reserve[h] = max(
                    effective.effective_min_reserve[h],
                    d.structured_adjustment.minimum_energy_kwh,
                )
        elif d.directive_type == "max_grid_window":
            assert isinstance(d.structured_adjustment, MaxGridWindowAdjustment)
            cap = d.structured_adjustment.max_grid_kwh
            for h in d.structured_adjustment.hours:
                prev = windows.max_grid_kwh_per_hour.get(h)
                windows.max_grid_kwh_per_hour[h] = cap if prev is None else min(prev, cap)
        elif d.directive_type == "no_charge_window":
            for h in d.structured_adjustment.hours:
                if h not in windows.no_charge_hours:
                    windows.no_charge_hours.append(h)
        elif d.directive_type == "no_discharge_window":
            for h in d.structured_adjustment.hours:
                if h not in windows.no_discharge_hours:
                    windows.no_discharge_hours.append(h)

    windows.no_charge_hours = sorted(set(windows.no_charge_hours))
    windows.no_discharge_hours = sorted(set(windows.no_discharge_hours))

    # Final battery-state checks
    for h, r in enumerate(effective.effective_min_reserve):
        if r > scenario.battery.capacity_kwh + 1e-9:
            raise GuardrailError(
                f"effective_min_reserve[{h}]={r} exceeds capacity "
                f"{scenario.battery.capacity_kwh}"
            )

    return CompiledScenario(
        scenario_id=scenario.scenario_id,
        context=context,
        battery=scenario.battery,
        effective=effective,
        windows=windows,
    )


def _build_context(scenario: Scenario) -> HourlyContext:
    return HourlyContext(
        demand=[float(h.demand_kwh) for h in scenario.hourly_data],
        solar=[float(h.solar_kwh) for h in scenario.hourly_data],
        tariff=[float(h.tariff_bdt_per_kwh) for h in scenario.hourly_data],
    )
