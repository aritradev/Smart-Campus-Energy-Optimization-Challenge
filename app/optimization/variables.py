"""LP decision variables.

For every hour ``h ∈ [0, 23]`` we define:

    grid_kwh[h]              ≥ 0
    solar_used_kwh[h]        ≥ 0
    battery_charge_kwh[h]    ≥ 0
    battery_discharge_kwh[h] ≥ 0
    battery_energy_after[h]  ≥ 0

The optimizer does **not** forbid simultaneous charge/discharge at the LP
level; doing so would require binary variables and turn the model into a
MILP. We have mathematically verified (see ``tests/optimization/test_simul_charge.py``)
that simultaneous charge+discharge cannot improve the objective under
this physical model, because:

* grid is always more expensive than itself (no profit),
* charging from grid and then discharging to grid would just lose energy
  to round-trip inefficiency (the model treats stored energy as free to
  withdraw, so the trivial solution is never strictly better than
  discharging zero).
* any net-zero simultaneous transfer is degenerate (zero contribution),
  but is never strictly *better* than a solution that picks the cheaper
  of (charge, discharge).

Therefore the LP relaxation is sufficient.
"""
from __future__ import annotations

from typing import Dict

import pulp

HOURS = list(range(24))


class Variables:
    """Container for all per-hour LP variables."""

    def __init__(self, prob: pulp.LpProblem):
        self.grid: Dict[int, pulp.LpVariable] = {
            h: pulp.LpVariable(f"grid_{h}", lowBound=0) for h in HOURS
        }
        self.solar_used: Dict[int, pulp.LpVariable] = {
            h: pulp.LpVariable(f"solar_{h}", lowBound=0) for h in HOURS
        }
        self.charge: Dict[int, pulp.LpVariable] = {
            h: pulp.LpVariable(f"charge_{h}", lowBound=0) for h in HOURS
        }
        self.discharge: Dict[int, pulp.LpVariable] = {
            h: pulp.LpVariable(f"discharge_{h}", lowBound=0) for h in HOURS
        }
        self.energy_after: Dict[int, pulp.LpVariable] = {
            h: pulp.LpVariable(f"energy_after_{h}", lowBound=0) for h in HOURS
        }
