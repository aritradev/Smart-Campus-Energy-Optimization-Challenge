"""Strictly typed Pydantic models for the six supported directive types.

These are used as the *ground truth schema* the LLM must conform to.
Each active directive has its own typed payload model.
"""
from __future__ import annotations

from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HOUR_BOUNDS = (0, 23)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# ---------------------------------------------------------------------------
# Structured payloads (one per directive type)
# ---------------------------------------------------------------------------

class SolarReductionAdjustment(_StrictModel):
    hours: List[int] = Field(..., min_length=1)
    factor: float = Field(..., ge=0.0, le=1.0)

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: List[int]) -> List[int]:
        for h in v:
            if not isinstance(h, int) or isinstance(h, bool):
                raise ValueError("hour must be an integer")
            if h < HOUR_BOUNDS[0] or h > HOUR_BOUNDS[1]:
                raise ValueError(f"hour {h} outside [0,23]")
        if len(set(v)) != len(v):
            raise ValueError("duplicate hours")
        if v != sorted(v):
            raise ValueError("hours must be ascending")
        return v


class MinimumBatteryReserveAdjustment(_StrictModel):
    hours: List[int] = Field(..., min_length=1)
    minimum_energy_kwh: float = Field(..., ge=0.0)

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: List[int]) -> List[int]:
        for h in v:
            if not isinstance(h, int) or isinstance(h, bool):
                raise ValueError("hour must be an integer")
            if h < HOUR_BOUNDS[0] or h > HOUR_BOUNDS[1]:
                raise ValueError(f"hour {h} outside [0,23]")
        if len(set(v)) != len(v):
            raise ValueError("duplicate hours")
        if v != sorted(v):
            raise ValueError("hours must be ascending")
        return v


class NoChargeWindowAdjustment(_StrictModel):
    hours: List[int] = Field(..., min_length=1)

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: List[int]) -> List[int]:
        for h in v:
            if not isinstance(h, int) or isinstance(h, bool):
                raise ValueError("hour must be an integer")
            if h < HOUR_BOUNDS[0] or h > HOUR_BOUNDS[1]:
                raise ValueError(f"hour {h} outside [0,23]")
        if len(set(v)) != len(v):
            raise ValueError("duplicate hours")
        if v != sorted(v):
            raise ValueError("hours must be ascending")
        return v


class NoDischargeWindowAdjustment(_StrictModel):
    hours: List[int] = Field(..., min_length=1)

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: List[int]) -> List[int]:
        for h in v:
            if not isinstance(h, int) or isinstance(h, bool):
                raise ValueError("hour must be an integer")
            if h < HOUR_BOUNDS[0] or h > HOUR_BOUNDS[1]:
                raise ValueError(f"hour {h} outside [0,23]")
        if len(set(v)) != len(v):
            raise ValueError("duplicate hours")
        if v != sorted(v):
            raise ValueError("hours must be ascending")
        return v


class MaxGridWindowAdjustment(_StrictModel):
    hours: List[int] = Field(..., min_length=1)
    max_grid_kwh: float = Field(..., ge=0.0)

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: List[int]) -> List[int]:
        for h in v:
            if not isinstance(h, int) or isinstance(h, bool):
                raise ValueError("hour must be an integer")
            if h < HOUR_BOUNDS[0] or h > HOUR_BOUNDS[1]:
                raise ValueError(f"hour {h} outside [0,23]")
        if len(set(v)) != len(v):
            raise ValueError("duplicate hours")
        if v != sorted(v):
            raise ValueError("hours must be ascending")
        return v


# ---------------------------------------------------------------------------
# Single directive emitted by the LLM
# ---------------------------------------------------------------------------

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]


# Map directive_type to its adjustment model
_ADJUSTMENT_BY_TYPE = {
    "solar_reduction": SolarReductionAdjustment,
    "minimum_battery_reserve": MinimumBatteryReserveAdjustment,
    "no_charge_window": NoChargeWindowAdjustment,
    "no_discharge_window": NoDischargeWindowAdjustment,
    "max_grid_window": MaxGridWindowAdjustment,
}

StructuredAdjustmentModel = Union[
    SolarReductionAdjustment,
    MinimumBatteryReserveAdjustment,
    NoChargeWindowAdjustment,
    NoDischargeWindowAdjustment,
    MaxGridWindowAdjustment,
]


class LLMDirective(_StrictModel):
    """One directive returned by the LLM for one operator note."""

    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[StructuredAdjustmentModel] = None
    paraphrase: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_adjustment(cls, data: Any) -> Any:
        """Coerce ``structured_adjustment`` to the correct subtype based on
        ``directive_type``.

        Without this, Pydantic 2's smart-union matcher can confuse
        ``no_charge_window`` and ``no_discharge_window`` (identical shape).
        """
        if not isinstance(data, dict):
            return data
        dt = data.get("directive_type")
        adj = data.get("structured_adjustment")
        if isinstance(dt, str) and dt in _ADJUSTMENT_BY_TYPE and isinstance(adj, dict):
            model = _ADJUSTMENT_BY_TYPE[dt]
            data = dict(data)
            data["structured_adjustment"] = model.model_validate(adj)
        return data

    @model_validator(mode="after")
    def _validate_consistency(self) -> "LLMDirective":
        if self.applies:
            if self.directive_type == "no_op":
                raise ValueError("no_op must have applies=false")
            if self.structured_adjustment is None:
                raise ValueError("active directives must include structured_adjustment")
            # ensure structured_adjustment matches directive_type
            expected_cls = _ADJUSTMENT_BY_TYPE.get(self.directive_type)
            if expected_cls is None:
                raise ValueError(f"unknown directive_type {self.directive_type}")
            if not isinstance(self.structured_adjustment, expected_cls):
                raise ValueError(
                    f"structured_adjustment type {type(self.structured_adjustment).__name__}"
                    f" does not match directive_type {self.directive_type}"
                )
        else:
            if self.directive_type != "no_op":
                raise ValueError("inactive directives must have directive_type=no_op")
            if self.structured_adjustment is not None:
                raise ValueError("inactive directives must have structured_adjustment=null")
        return self


class LLMDirectiveList(_StrictModel):
    """List returned by the LLM, one entry per operator note."""

    directives: List[LLMDirective]

    @field_validator("directives")
    @classmethod
    def _non_empty(cls, v: List[LLMDirective]) -> List[LLMDirective]:
        if not v:
            raise ValueError("directives list must not be empty")
        return v
