"""Unit tests for request schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.request import BatterySpec, EnergyHour, Scenario
from tests import default_battery, make_24_hours, make_scenario


def test_energy_hour_valid():
    h = EnergyHour(hour=5, demand_kwh=10.0, solar_kwh=2.0, tariff_bdt_per_kwh=8.0)
    assert h.hour == 5


def test_energy_hour_negative_demand_fails():
    with pytest.raises(ValidationError):
        EnergyHour(hour=5, demand_kwh=-1.0, solar_kwh=0.0, tariff_bdt_per_kwh=0.0)


def test_energy_hour_hour_out_of_range_fails():
    with pytest.raises(ValidationError):
        EnergyHour(hour=24, demand_kwh=1.0, solar_kwh=0.0, tariff_bdt_per_kwh=0.0)
    with pytest.raises(ValidationError):
        EnergyHour(hour=-1, demand_kwh=1.0, solar_kwh=0.0, tariff_bdt_per_kwh=0.0)


def test_battery_valid():
    b = default_battery()
    assert b.capacity_kwh == 50.0


def test_battery_initial_exceeds_capacity_fails():
    with pytest.raises(ValidationError):
        BatterySpec(
            capacity_kwh=10.0,
            initial_energy_kwh=20.0,
            minimum_energy_kwh=0.0,
            max_charge_kwh_per_hour=5.0,
            max_discharge_kwh_per_hour=5.0,
        )


def test_battery_minimum_exceeds_capacity_fails():
    with pytest.raises(ValidationError):
        BatterySpec(
            capacity_kwh=10.0,
            initial_energy_kwh=0.0,
            minimum_energy_kwh=20.0,
            max_charge_kwh_per_hour=5.0,
            max_discharge_kwh_per_hour=5.0,
        )


def test_battery_initial_below_minimum_fails():
    with pytest.raises(ValidationError):
        BatterySpec(
            capacity_kwh=10.0,
            initial_energy_kwh=0.0,
            minimum_energy_kwh=2.0,
            max_charge_kwh_per_hour=5.0,
            max_discharge_kwh_per_hour=5.0,
        )


def test_scenario_valid():
    s = make_scenario()
    assert s.scenario_id == "test-001"
    assert len(s.hourly_data) == 24


def test_scenario_requires_24_hours():
    with pytest.raises(ValidationError):
        Scenario(
            scenario_id="x",
            operator_notes=["a"],
            hourly_data=make_24_hours()[:12],
            battery=default_battery(),
        )


def test_scenario_rejects_too_many_notes():
    with pytest.raises(ValidationError):
        Scenario(
            scenario_id="x",
            operator_notes=["a", "b", "c", "d"],
            hourly_data=make_24_hours(),
            battery=default_battery(),
        )


def test_scenario_rejects_empty_note():
    with pytest.raises(ValidationError):
        Scenario(
            scenario_id="x",
            operator_notes=["   "],
            hourly_data=make_24_hours(),
            battery=default_battery(),
        )


def test_scenario_rejects_duplicate_hours():
    bad = make_24_hours()
    bad[5] = EnergyHour(hour=4, demand_kwh=10.0, solar_kwh=5.0, tariff_bdt_per_kwh=8.0)
    with pytest.raises(ValidationError):
        Scenario(
            scenario_id="x",
            operator_notes=["a"],
            hourly_data=bad,
            battery=default_battery(),
        )


def test_scenario_rejects_unsorted_hours():
    bad = make_24_hours()
    bad[0], bad[1] = bad[1], bad[0]
    with pytest.raises(ValidationError):
        Scenario(
            scenario_id="x",
            operator_notes=["a"],
            hourly_data=bad,
            battery=default_battery(),
        )


def test_scenario_rejects_extra_field():
    with pytest.raises(ValidationError):
        Scenario.model_validate({
            "scenario_id": "x",
            "operator_notes": ["a"],
            "hourly_data": [h.model_dump() for h in make_24_hours()],
            "battery": default_battery().model_dump(),
            "rogue": "field",
        })
