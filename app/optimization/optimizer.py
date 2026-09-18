"""Linear programming optimizer for the smart-campus energy problem."""
from __future__ import annotations

from typing import Optional

import pulp

from app.config import Settings, get_settings
from app.logging import get_logger
from app.models.domain import CompiledScenario, PlanResult
from app.optimization.constraints import build_constraints
from app.optimization.variables import HOURS, Variables


logger = get_logger(__name__)


class OptimizationError(Exception):
    """Raised when the LP fails to produce a feasible optimal solution."""


def optimize(
    compiled: CompiledScenario,
    settings: Optional[Settings] = None,
) -> PlanResult:
    """Solve the LP and return the plan.

    Raises ``OptimizationError`` on infeasibility, unboundedness, or solver
    numerical issues. Never returns a fabricated schedule.
    """
    settings = settings or get_settings()

    prob = pulp.LpProblem(
        f"gridwise_{compiled.scenario_id}",
        pulp.LpMinimize,
    )
    v = Variables(prob)

    # Objective: minimise grid cost
    prob += pulp.lpSum(
        v.grid[h] * compiled.context.tariff[h] for h in HOURS
    ), "total_cost"

    build_constraints(prob, v, compiled)

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=settings.solver_time_limit_s,
    )
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError(
            f"LP status: {pulp.LpStatus[status]} (solver returned {status})"
        )

    def val(x: pulp.LpVariable) -> float:
        return float(x.value() or 0.0)

    return PlanResult(
        grid_kwh=[val(v.grid[h]) for h in HOURS],
        solar_used_kwh=[val(v.solar_used[h]) for h in HOURS],
        battery_charge_kwh=[val(v.charge[h]) for h in HOURS],
        battery_discharge_kwh=[val(v.discharge[h]) for h in HOURS],
        battery_energy_after_kwh=[val(v.energy_after[h]) for h in HOURS],
        objective_cost_bdt=float(pulp.value(prob.objective) or 0.0),
    )
