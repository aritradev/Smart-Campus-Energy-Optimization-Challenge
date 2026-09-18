"""Top-level optimization service.

Pipeline:
    LLM interpretation
        →  Guardrails
            →  Directive compiler
                →  LP optimizer
                    →  Final independent validator
                        →  API response
"""
from __future__ import annotations

from typing import List

from fastapi import HTTPException

from app.config import Settings, get_settings
from app.llm import interpret_notes, LLMError
from app.logging import get_logger
from app.models.domain import scenario_to_hourly_context
from app.optimization.optimizer import optimize, OptimizationError
from app.schemas.directives import LLMDirectiveList
from app.schemas.request import Scenario
from app.schemas.response import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    OptimizeEnergyResponse,
    StructuredAdjustmentEcho,
)
from app.validation.guardrails import (
    GuardrailError,
    validate_directives_against_scenario,
)
from app.validation.compiler import compile_directives
from app.validation.final_validator import (
    FinalValidatorError,
    validate_and_recalculate,
)

logger = get_logger(__name__)


def optimize_scenario(
    scenario: Scenario,
    settings: Settings | None = None,
) -> OptimizeEnergyResponse:
    """Run the full pipeline and return a fully validated response."""
    settings = settings or get_settings()

    # 1. LLM interpretation (one call for all notes).
    try:
        directives = interpret_notes(
            scenario.operator_notes,
            settings=settings,
            battery_capacity_kwh=scenario.battery.capacity_kwh,
        )
    except LLMError as exc:
        logger.warning("LLM interpretation failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=422,
            detail=f"interpreter error: {exc}",
        ) from exc

    # 2. Guardrails (Pydantic + semantic).
    try:
        directives = validate_directives_against_scenario(directives, scenario)
    except GuardrailError as exc:
        logger.warning("guardrail rejection", extra={"error": str(exc)})
        raise HTTPException(
            status_code=422,
            detail=f"guardrail error: {exc}",
        ) from exc

    # 3. Directive compiler.
    compiled = compile_directives(scenario, directives)

    # 4. LP optimizer.
    try:
        plan = optimize(compiled, settings=settings)
    except OptimizationError as exc:
        logger.warning("LP optimization failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=422,
            detail=f"optimization error: {exc}",
        ) from exc

    # 5. Final independent validator.
    try:
        validated = validate_and_recalculate(compiled, plan, settings=settings)
    except FinalValidatorError as exc:
        logger.error(
            "final validator rejected plan",
            extra={"error": str(exc), "scenario_id": scenario.scenario_id},
        )
        raise HTTPException(
            status_code=500,
            detail=f"plan validation failed: {exc}",
        ) from exc

    # 6. Build response.
    interpretations = _build_interpretations(scenario.operator_notes, directives)
    hourly_plan = _build_hourly_plan(scenario, validated, compiled)

    summary = _build_summary(scenario, validated)

    return OptimizeEnergyResponse(
        scenario_id=scenario.scenario_id,
        directive_interpretation=interpretations,
        hourly_plan=hourly_plan,
        total_grid_kwh=_round(validated.total_grid_kwh),
        total_cost_bdt=_round(validated.total_cost_bdt),
        peak_grid_kwh=_round(validated.peak_grid_kwh),
        plan_summary=summary,
    )


def _build_interpretations(
    notes: List[str],
    directives: LLMDirectiveList,
) -> List[DirectiveInterpretation]:
    out: List[DirectiveInterpretation] = []
    for note, d in zip(notes, directives.directives):
        adj = d.structured_adjustment
        echo = None
        if adj is not None:
            echo = StructuredAdjustmentEcho(
                **adj.model_dump(),
            )
        out.append(
            DirectiveInterpretation(
                note_index=d.note_index,
                applies=d.applies,
                directive_type=d.directive_type,
                structured_adjustment=echo,
                paraphrase=d.paraphrase or note,
                original_note=note,
                warnings=list(d.warnings or []),
            )
        )
    return out


def _build_hourly_plan(scenario, validated, compiled):
    """Build the 24 ``HourlyPlanEntry`` rows.

    Every numeric field is rounded to 2 decimal places. The ``battery_action``
    and ``battery_kwh`` fields are derived: exactly one of (charge, discharge)
    is non-zero per hour by the LP model, so the action is whichever side
    has the larger magnitude (within a small tolerance for numerical noise).
    """
    out = []
    for h in range(24):
        charge = validated.battery_charge_kwh[h]
        discharge = validated.battery_discharge_kwh[h]
        tol = 1e-6
        if charge > discharge + tol:
            action = "charge"
            magnitude = charge
        elif discharge > charge + tol:
            action = "discharge"
            magnitude = discharge
        else:
            action = "idle"
            magnitude = max(charge, discharge)  # ~0.0 in practice
        out.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=_round(validated.grid_kwh[h]),
                solar_used_kwh=_round(validated.solar_used_kwh[h]),
                battery_charge_kwh=_round(charge),
                battery_discharge_kwh=_round(discharge),
                battery_energy_after_kwh=_round(validated.battery_energy_after_kwh[h]),
                battery_action=action,
                battery_kwh=_round(magnitude),
                demand_kwh=_round(validated.demand_kwh[h]),
                solar_available_kwh=_round(validated.solar_available_kwh[h]),
                tariff_bdt_per_kwh=_round(validated.tariff_bdt_per_kwh[h]),
                cost_bdt=_round(validated.cost_bdt_per_hour[h]),
            )
        )
    return out


def _build_summary(scenario, validated) -> str:
    return (
        f"{scenario.scenario_id}: "
        f"grid={validated.total_grid_kwh:.3f} kWh, "
        f"cost=BDT {validated.total_cost_bdt:.2f}, "
        f"peak={validated.peak_grid_kwh:.3f} kWh, "
        f"battery end={validated.battery_energy_after_kwh[23]:.3f} kWh"
    )


def _round(x: float, ndigits: int = 2) -> float:
    """Round a value while keeping numeric type for Pydantic compatibility.

    Default precision is **2 decimal places** for all response numerics, as
    required by the competition contract (≤ 0.01 kWh / BDT tolerance).
    """
    return float(round(float(x), ndigits))


# ---------------------------------------------------------------------------
# (Duplicate definitions removed; canonical versions are above.)
# ---------------------------------------------------------------------------
