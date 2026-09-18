"""Shared test fixtures."""
from __future__ import annotations

from typing import List

from app.schemas.request import BatterySpec, EnergyHour, Scenario


def make_hour(
    hour: int,
    demand: float = 10.0,
    solar: float = 5.0,
    tariff: float = 8.0,
) -> EnergyHour:
    return EnergyHour(
        hour=hour,
        demand_kwh=demand,
        solar_kwh=solar,
        tariff_bdt_per_kwh=tariff,
    )


def make_24_hours(
    demand: float | List[float] = 10.0,
    solar: float | List[float] = 5.0,
    tariff: float | List[float] = 8.0,
) -> List[EnergyHour]:
    if isinstance(demand, (int, float)):
        demand = [float(demand)] * 24
    if isinstance(solar, (int, float)):
        solar = [float(solar)] * 24
    if isinstance(tariff, (int, float)):
        tariff = [float(tariff)] * 24
    return [
        make_hour(h, demand=h_d, solar=h_s, tariff=h_t)
        for h, (h_d, h_s, h_t) in enumerate(zip(demand, solar, tariff))
    ]


def default_battery(
    capacity: float = 50.0,
    initial: float = 25.0,
    minimum: float = 5.0,
    max_charge: float = 10.0,
    max_discharge: float = 10.0,
) -> BatterySpec:
    return BatterySpec(
        capacity_kwh=capacity,
        initial_energy_kwh=initial,
        minimum_energy_kwh=minimum,
        max_charge_kwh_per_hour=max_charge,
        max_discharge_kwh_per_hour=max_discharge,
    )


def make_scenario(
    scenario_id: str = "test-001",
    notes: List[str] | None = None,
    demand: float | List[float] = 10.0,
    solar: float | List[float] = 5.0,
    tariff: float | List[float] = 8.0,
    battery: BatterySpec | None = None,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        operator_notes=notes if notes is not None else ["Do not charge between 14:00 and 15:00."],
        hourly_data=make_24_hours(demand=demand, solar=solar, tariff=tariff),
        battery=battery or default_battery(),
    )
