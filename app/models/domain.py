"""Internal domain models used by the optimizer and validator.

These are intentionally not part of the HTTP boundary. They are the *clean*
mathematical representation that the solver consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.schemas.request import BatterySpec, EnergyHour, Scenario


# ---------------------------------------------------------------------------
# Hour index 0..23
# ---------------------------------------------------------------------------

HOURS: List[int] = list(range(24))


@dataclass
class HourlyContext:
    demand: List[float]
    solar: List[float]
    tariff: List[float]

    def __post_init__(self) -> None:
        for name in ("demand", "solar", "tariff"):
            arr = getattr(self, name)
            assert len(arr) == 24, f"{name} must have exactly 24 entries"


@dataclass
class EffectiveLimits:
    """Per-hour effective limits derived from directives + base values."""
    effective_solar: List[float]              # solar available for use per hour
    effective_min_reserve: List[float]        # minimum battery energy per hour


@dataclass
class WindowConstraints:
    """Per-hour directive-imposed equality/upper-bound constraints."""
    no_charge_hours: List[int] = field(default_factory=list)
    no_discharge_hours: List[int] = field(default_factory=list)
    max_grid_kwh_per_hour: Dict[int, float] = field(default_factory=dict)
    # If a hour has no entry in max_grid_kwh_per_hour, no cap is applied.


@dataclass
class PlanResult:
    grid_kwh: List[float]
    solar_used_kwh: List[float]
    battery_charge_kwh: List[float]
    battery_discharge_kwh: List[float]
    battery_energy_after_kwh: List[float]
    objective_cost_bdt: float


@dataclass
class CompiledScenario:
    """The clean input passed to the LP solver."""
    scenario_id: str
    context: HourlyContext
    battery: BatterySpec
    effective: EffectiveLimits
    windows: WindowConstraints


def scenario_to_hourly_context(scenario: Scenario) -> HourlyContext:
    return HourlyContext(
        demand=[float(h.demand_kwh) for h in scenario.hourly_data],
        solar=[float(h.solar_kwh) for h in scenario.hourly_data],
        tariff=[float(h.tariff_bdt_per_kwh) for h in scenario.hourly_data],
    )
