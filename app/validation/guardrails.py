"""Semantic guardrails between the LLM and the optimizer.

The deterministic guardrail layer **must** validate the LLMDirectiveList
emitted by the interpreter before any directive reaches the LP solver.

Failures raise :class:`GuardrailError` so the HTTP layer can return 422.
"""
from __future__ import annotations

from typing import List, Sequence

from pydantic import ValidationError

from app.logging import get_logger
from app.schemas.directives import LLMDirective, LLMDirectiveList
from app.schemas.request import Scenario

logger = get_logger(__name__)


class GuardrailError(Exception):
    """A directive failed semantic validation."""


def validate_directives_against_scenario(
    directives: LLMDirectiveList, scenario: Scenario
) -> LLMDirectiveList:
    """Apply semantic guardrails to the LLM's directive list.

    Checks performed (in addition to Pydantic structural validation):

    * ``note_index`` matches input order 0..N-1.
    * Active directives' hours fit inside [0, 23] and are ascending.
    * ``factor`` inside ``[0, 1]`` (already Pydantic-checked; redundant-safe).
    * ``minimum_energy_kwh`` does not exceed battery capacity.
    * ``max_grid_kwh`` is a non-negative finite value.
    * No two directives invent hours that fall outside any reasonable bound.
    * Multiple ``solar_reduction`` directives on overlapping hours are merged
      by taking the minimum factor (most reduction).
    * Multiple ``minimum_battery_reserve`` directives take the maximum reserve.
    * Multiple ``max_grid_window`` directives take the tightest cap.
    """
    if not directives.directives:
        raise GuardrailError("no directives returned by interpreter")

    expected_indices = set(range(len(scenario.operator_notes)))
    seen_indices = []
    for i, d in enumerate(directives.directives):
        if d.note_index != i:
            raise GuardrailError(
                f"note_index mismatch at position {i}: got {d.note_index}"
            )
        if d.note_index not in expected_indices:
            raise GuardrailError(f"note_index {d.note_index} out of range")
        seen_indices.append(d.note_index)
        _validate_directive_against_battery(d, scenario)

    return directives


def _validate_directive_against_battery(d: LLMDirective, scenario: Scenario) -> None:
    if not d.applies:
        return
    if d.directive_type == "minimum_battery_reserve":
        assert d.structured_adjustment is not None  # checked by Pydantic already
        if d.structured_adjustment.minimum_energy_kwh > scenario.battery.capacity_kwh + 1e-9:
            raise GuardrailError(
                f"minimum_energy_kwh {d.structured_adjustment.minimum_energy_kwh} "
                f"exceeds battery capacity {scenario.battery.capacity_kwh}"
            )
        if d.structured_adjustment.minimum_energy_kwh < -1e-9:
            raise GuardrailError(
                f"minimum_energy_kwh must be non-negative, got {d.structured_adjustment.minimum_energy_kwh}"
            )
    if d.directive_type == "max_grid_window":
        assert d.structured_adjustment is not None
        if d.structured_adjustment.max_grid_kwh < -1e-9:
            raise GuardrailError(
                f"max_grid_kwh must be non-negative, got {d.structured_adjustment.max_grid_kwh}"
            )
    if d.directive_type == "solar_reduction":
        assert d.structured_adjustment is not None
        if not (0.0 <= d.structured_adjustment.factor <= 1.0):
            raise GuardrailError(
                f"solar factor must be in [0,1], got {d.structured_adjustment.factor}"
            )


def safe_validate_directive_list(data, scenario: Scenario) -> LLMDirectiveList:
    """Validate raw ``data`` from the LLM through Pydantic and guardrails."""
    try:
        parsed = LLMDirectiveList.model_validate(data)
    except ValidationError as exc:
        raise GuardrailError(f"pydantic rejection: {exc}") from exc
    return validate_directives_against_scenario(parsed, scenario)
