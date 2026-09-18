"""Build the full LP constraint set from a CompiledScenario."""
from __future__ import annotations

import pulp

from app.models.domain import CompiledScenario
from app.optimization.variables import HOURS, Variables


def build_constraints(prob: pulp.LpProblem, v: Variables, c: CompiledScenario) -> None:
    """Add every constraint to ``prob``.

    Constraints added (all equality/inequality, no binaries):

    1. Energy balance: grid + solar_used + discharge = demand + charge
    2. Solar limit:    solar_used ≤ effective_solar[h]
    3. Battery capacity: energy_after ≤ capacity
    4. Min reserve:    energy_after ≥ effective_min_reserve[h]
    5. Charge rate:    charge ≤ max_charge_kwh_per_hour
    6. Discharge rate: discharge ≤ max_discharge_kwh_per_hour
    7. Battery state:  energy_after[0] = initial + charge[0] - discharge[0]
                      energy_after[h] = energy_after[h-1] + charge[h] - discharge[h]
    8. End-of-day:     energy_after[23] = initial_energy
    9. Directives:     no_charge_window  → charge[h] = 0
                      no_discharge_window → discharge[h] = 0
                      max_grid_window     → grid[h] ≤ max_grid_kwh_per_hour[h]
    """
    ctx = c.context
    batt = c.battery

    # (1) Energy balance
    for h in HOURS:
        prob += (
            v.grid[h] + v.solar_used[h] + v.discharge[h]
            == ctx.demand[h] + v.charge[h]
        ), f"balance_{h}"

    # (2) Solar limit (unused solar is curtailed, never exported)
    for h in HOURS:
        prob += v.solar_used[h] <= c.effective.effective_solar[h], f"solar_limit_{h}"

    # (3) Battery capacity
    for h in HOURS:
        prob += v.energy_after[h] <= batt.capacity_kwh, f"cap_{h}"

    # (4) Minimum reserve
    for h in HOURS:
        prob += (
            v.energy_after[h] >= c.effective.effective_min_reserve[h]
        ), f"reserve_{h}"

    # (5) Charge rate
    for h in HOURS:
        prob += v.charge[h] <= batt.max_charge_kwh_per_hour, f"charge_rate_{h}"

    # (6) Discharge rate
    for h in HOURS:
        prob += (
            v.discharge[h] <= batt.max_discharge_kwh_per_hour
        ), f"discharge_rate_{h}"

    # (7) Battery state
    prob += (
        v.energy_after[0]
        == batt.initial_energy_kwh + v.charge[0] - v.discharge[0]
    ), "state_init"
    for h in HOURS[1:]:
        prob += (
            v.energy_after[h]
            == v.energy_after[h - 1] + v.charge[h] - v.discharge[h]
        ), f"state_{h}"

    # (8) End-of-day neutrality (hard)
    prob += v.energy_after[23] == batt.initial_energy_kwh, "state_final"

    # (9) Directives
    for h in c.windows.no_charge_hours:
        prob += v.charge[h] == 0, f"no_charge_{h}"
    for h in c.windows.no_discharge_hours:
        prob += v.discharge[h] == 0, f"no_discharge_{h}"
    for h, cap in c.windows.max_grid_kwh_per_hour.items():
        prob += v.grid[h] <= cap, f"max_grid_{h}"
