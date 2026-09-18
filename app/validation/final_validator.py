"""Final independent validator.

Replays every constraint independently from the LP solution. Re-calculates
totals. Verifies battery neutrality, energy balance, solar limit, capacity,
reserve, charge/discharge rate, and directive enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.config import Settings, get_settings
from app.logging import get_logger
from app.models.domain import CompiledScenario, PlanResult

logger = get_logger(__name__)


class FinalValidatorError(Exception):
    """Plan failed final validation; do NOT return this plan to the client."""


@dataclass
class ValidatedPlan:
    grid_kwh: List[float]
    solar_used_kwh: List[float]
    battery_charge_kwh: List[float]
    battery_discharge_kwh: List[float]
    battery_energy_after_kwh: List[float]
    demand_kwh: List[float]
    solar_available_kwh: List[float]
    tariff_bdt_per_kwh: List[float]
    cost_bdt_per_hour: List[float]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float


def validate_and_recalculate(
    compiled: CompiledScenario,
    plan: PlanResult,
    settings: Settings | None = None,
) -> ValidatedPlan:
    """Replay constraints and re-compute totals.

    Raises ``FinalValidatorError`` if any constraint is violated beyond the
    configured numerical tolerance.
    """
    settings = settings or get_settings()
    tol_kwh = settings.numerical_tolerance_kwh
    tol_bdt = settings.numerical_tolerance_bdt
    ctx = compiled.context
    batt = compiled.battery

    grid = list(plan.grid_kwh)
    solar_used = list(plan.solar_used_kwh)
    charge = list(plan.battery_charge_kwh)
    discharge = list(plan.battery_discharge_kwh)
    energy_after = list(plan.battery_energy_after_kwh)

    if len(grid) != 24:
        raise FinalValidatorError(f"expected 24 hours of grid, got {len(grid)}")

    for h in range(24):
        # Non-negativity
        for name, arr in [
            ("grid", grid),
            ("solar_used", solar_used),
            ("charge", charge),
            ("discharge", discharge),
            ("energy_after", energy_after),
        ]:
            if arr[h] < -tol_kwh:
                raise FinalValidatorError(
                    f"hour {h}: {name}={arr[h]:.6f} is negative beyond tolerance"
                )

        # Solar limit
        if solar_used[h] > compiled.effective.effective_solar[h] + tol_kwh:
            raise FinalValidatorError(
                f"hour {h}: solar_used={solar_used[h]:.6f} exceeds "
                f"effective_solar={compiled.effective.effective_solar[h]:.6f}"
            )

        # Battery capacity
        if energy_after[h] > batt.capacity_kwh + tol_kwh:
            raise FinalValidatorError(
                f"hour {h}: energy_after={energy_after[h]:.6f} exceeds capacity={batt.capacity_kwh}"
            )

        # Reserve
        if energy_after[h] < compiled.effective.effective_min_reserve[h] - tol_kwh:
            raise FinalValidatorError(
                f"hour {h}: energy_after={energy_after[h]:.6f} below reserve="
                f"{compiled.effective.effective_min_reserve[h]:.6f}"
            )

        # Charge / discharge rates
        if charge[h] > batt.max_charge_kwh_per_hour + tol_kwh:
            raise FinalValidatorError(
                f"hour {h}: charge={charge[h]:.6f} exceeds max_charge_rate"
            )
        if discharge[h] > batt.max_discharge_kwh_per_hour + tol_kwh:
            raise FinalValidatorError(
                f"hour {h}: discharge={discharge[h]:.6f} exceeds max_discharge_rate"
            )

        # Energy balance: grid + solar_used + discharge = demand + charge
        lhs = grid[h] + solar_used[h] + discharge[h]
        rhs = ctx.demand[h] + charge[h]
        if abs(lhs - rhs) > tol_kwh:
            raise FinalValidatorError(
                f"hour {h}: energy balance violated, lhs={lhs:.6f}, rhs={rhs:.6f}, "
                f"diff={lhs - rhs:.6f}"
            )

        # Directive enforcement
        if h in compiled.windows.no_charge_hours:
            if charge[h] > tol_kwh:
                raise FinalValidatorError(
                    f"hour {h}: charge={charge[h]} but in no_charge_window"
                )
        if h in compiled.windows.no_discharge_hours:
            if discharge[h] > tol_kwh:
                raise FinalValidatorError(
                    f"hour {h}: discharge={discharge[h]} but in no_discharge_window"
                )
        if h in compiled.windows.max_grid_kwh_per_hour:
            cap = compiled.windows.max_grid_kwh_per_hour[h]
            if grid[h] > cap + tol_kwh:
                raise FinalValidatorError(
                    f"hour {h}: grid={grid[h]:.6f} exceeds cap={cap}"
                )

    # Battery state equations
    expected = batt.initial_energy_kwh + charge[0] - discharge[0]
    if abs(energy_after[0] - expected) > tol_kwh:
        raise FinalValidatorError(
            f"battery state init: expected {expected:.6f}, got {energy_after[0]:.6f}"
        )
    for h in range(1, 24):
        expected = energy_after[h - 1] + charge[h] - discharge[h]
        if abs(energy_after[h] - expected) > tol_kwh:
            raise FinalValidatorError(
                f"battery state[{h}]: expected {expected:.6f}, got {energy_after[h]:.6f}"
            )

    # End-of-day neutrality
    if abs(energy_after[23] - batt.initial_energy_kwh) > tol_kwh:
        raise FinalValidatorError(
            f"end-of-day neutrality violated: "
            f"energy_after[23]={energy_after[23]:.6f} != initial={batt.initial_energy_kwh}"
        )

    # Recalculate totals independently
    total_grid = sum(grid)
    cost_per_hour = [grid[h] * ctx.tariff[h] for h in range(24)]
    total_cost = sum(cost_per_hour)
    peak_grid = max(grid)

    # Compare against reported plan.objective_cost_bdt
    if abs(total_cost - plan.objective_cost_bdt) > tol_bdt:
        logger.warning(
            "validator objective mismatch (LP floating point)",
            extra={"recalculated": total_cost, "reported": plan.objective_cost_bdt},
        )

    return ValidatedPlan(
        grid_kwh=grid,
        solar_used_kwh=solar_used,
        battery_charge_kwh=charge,
        battery_discharge_kwh=discharge,
        battery_energy_after_kwh=energy_after,
        demand_kwh=list(ctx.demand),
        solar_available_kwh=compiled.effective.effective_solar,
        tariff_bdt_per_kwh=list(ctx.tariff),
        cost_bdt_per_hour=cost_per_hour,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
    )
