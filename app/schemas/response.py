"""Response schema definitions for the GridWise LLM HTTP API.

Per-hour plan entries expose both the raw charge/discharge kWh columns
(``battery_charge_kwh``, ``battery_discharge_kwh``) **and** the derived
single-column representation (``battery_action``, ``battery_kwh``) so the
caller never has to compute the action themselves.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


BatteryAction = Literal["charge", "discharge", "idle"]


# ---------------------------------------------------------------------------
# Per-hour plan entry
# ---------------------------------------------------------------------------

class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0.0)
    solar_used_kwh: float = Field(..., ge=0.0)
    battery_charge_kwh: float = Field(..., ge=0.0)
    battery_discharge_kwh: float = Field(..., ge=0.0)
    battery_energy_after_kwh: float = Field(..., ge=0.0)
    # Derived single-column action: ``charge`` | ``discharge`` | ``idle``.
    battery_action: BatteryAction
    # Magnitude in kWh (matches whichever of charge/discharge is non-zero).
    battery_kwh: float = Field(..., ge=0.0)
    demand_kwh: float = Field(..., ge=0.0)
    solar_available_kwh: float = Field(..., ge=0.0)
    tariff_bdt_per_kwh: float = Field(..., ge=0.0)
    cost_bdt: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Directive interpretation (echoed back to the client)
# ---------------------------------------------------------------------------

class StructuredAdjustmentEcho(BaseModel):
    """Generic echo of the structured_adjustment payload.

    Exactly one of these typed shapes matches the directive_type.
    """
    model_config = ConfigDict(extra="forbid")

    hours: Optional[List[int]] = None
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: str
    structured_adjustment: Optional[StructuredAdjustmentEcho] = None
    paraphrase: str = ""
    original_note: str = ""


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class OptimizeEnergyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry] = Field(..., min_length=24, max_length=24)
    total_grid_kwh: float = Field(..., ge=0.0)
    total_cost_bdt: float = Field(..., ge=0.0)
    peak_grid_kwh: float = Field(..., ge=0.0)
    plan_summary: str


__all__ = [
    "BatteryAction",
    "HourlyPlanEntry",
    "StructuredAdjustmentEcho",
    "DirectiveInterpretation",
    "OptimizeEnergyResponse",
]
