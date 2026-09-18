"""Request schema definitions for the GridWise LLM HTTP API.

The 24-hour dataset can be supplied under either of two field names for
backwards compatibility with the official competition test payloads:

* ``hourly_data``  — the canonical competition key (used by README §10
  and the existing e2e tests).
* ``hours``        — an alternate short alias accepted via Pydantic v2
  ``AliasChoices`` so the same parser works for either submission.

Both keys are functionally identical.
"""
from __future__ import annotations

from typing import Annotated, List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# EnergyHour
# ---------------------------------------------------------------------------

class EnergyHour(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23 inclusive")
    demand_kwh: float = Field(..., ge=0.0, description="Energy demand in kWh")
    solar_kwh: float = Field(..., ge=0.0, description="Available rooftop solar in kWh")
    tariff_bdt_per_kwh: float = Field(..., ge=0.0, description="Tariff in BDT per kWh")


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

class BatterySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity_kwh: float = Field(..., gt=0.0)
    initial_energy_kwh: float = Field(..., ge=0.0)
    minimum_energy_kwh: float = Field(..., ge=0.0)
    max_charge_kwh_per_hour: float = Field(..., gt=0.0)
    max_discharge_kwh_per_hour: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _validate(self) -> "BatterySpec":
        if self.initial_energy_kwh > self.capacity_kwh + 1e-9:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh + 1e-9:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh - 1e-9:
            raise ValueError("initial_energy_kwh cannot be less than minimum_energy_kwh")
        if self.max_charge_kwh_per_hour > self.capacity_kwh + 1e-9:
            raise ValueError("max_charge_kwh_per_hour cannot exceed capacity_kwh")
        if self.max_discharge_kwh_per_hour > self.capacity_kwh + 1e-9:
            raise ValueError("max_discharge_kwh_per_hour cannot exceed capacity_kwh")
        return self


# ---------------------------------------------------------------------------
# Scenario (full POST body)
# ---------------------------------------------------------------------------

class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    scenario_id: str = Field(..., min_length=1, max_length=128)
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hourly_data: List[EnergyHour] = Field(
        ...,
        min_length=24,
        max_length=24,
        validation_alias=AliasChoices("hourly_data", "hours"),
    )
    battery: BatterySpec

    @field_validator("operator_notes")
    @classmethod
    def _notes_nonempty(cls, v: List[str]) -> List[str]:
        for i, note in enumerate(v):
            if not isinstance(note, str):
                raise ValueError(f"note {i} must be a string")
            if not note.strip():
                raise ValueError(f"note {i} must not be empty or whitespace only")
        return v

    @field_validator("hourly_data")
    @classmethod
    def _hourly_data(cls, v: List[EnergyHour]) -> List[EnergyHour]:
        if len(v) != 24:
            raise ValueError("hourly_data must contain exactly 24 entries")
        hours = [h.hour for h in v]
        if sorted(hours) != list(range(24)):
            raise ValueError("hourly_data must cover every hour 0..23 exactly once")
        return v

    @model_validator(mode="after")
    def _sorted_check(self) -> "Scenario":
        # Pydantic field validator already verifies coverage; we still ensure
        # ascending order, which the problem statement explicitly requires.
        hours = [h.hour for h in self.hourly_data]
        if hours != sorted(hours):
            raise ValueError("hourly_data must be sorted ascending by hour")
        return self


# ---------------------------------------------------------------------------
# Top-level request wrapper
# ---------------------------------------------------------------------------

class OptimizeEnergyRequest(BaseModel):
    """Top-level request body for POST /optimize-energy.

    The competition API accepts the scenario fields at the top level (not nested),
    so we model it both ways:
      - The wrapper supports a flat body that contains scenario fields.

    The 24-hour dataset can be submitted as either ``hourly_data`` (the
    canonical competition key, used in README §10) or ``hours`` (an alias).
    Both keys are validated identically; no client migration is required.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    scenario_id: str = Field(..., min_length=1, max_length=128)
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hourly_data: List[EnergyHour] = Field(
        ...,
        min_length=24,
        max_length=24,
        validation_alias=AliasChoices("hourly_data", "hours"),
    )
    battery: BatterySpec

    def to_scenario(self) -> Scenario:
        return Scenario(
            scenario_id=self.scenario_id,
            operator_notes=self.operator_notes,
            hourly_data=self.hourly_data,
            battery=self.battery,
        )
