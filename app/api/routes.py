"""HTTP API routes for GridWise LLM."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from app.api.dependencies import settings_dep
from app.config import Settings
from app.logging import get_logger
from app.schemas.request import OptimizeEnergyRequest
from app.schemas.response import OptimizeEnergyResponse
from app.services.optimization_service import optimize_scenario

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", summary="Liveness probe")
def health() -> Dict[str, Any]:
    """Cheap liveness probe. Never depends on the LLM or optimizer."""
    return {"status": "ok"}


@router.post(
    "/optimize-energy",
    response_model=OptimizeEnergyResponse,
    summary="Run end-to-end smart campus energy optimization",
)
def optimize_energy(
    request_body: OptimizeEnergyRequest,
    request: Request,
) -> OptimizeEnergyResponse:
    """Accept a scenario and return a fully validated 24-hour plan.

    Pipeline: schema -> LLM -> guardrails -> directive compiler -> LP solver
    -> final validator -> response.
    """
    settings: Settings = settings_dep()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()

    logger.info(
        "optimize-energy received",
        extra={
            "request_id": request_id,
            "scenario_id": request_body.scenario_id,
            "note_count": len(request_body.operator_notes),
        },
    )

    try:
        scenario = request_body.to_scenario()
        response = optimize_scenario(scenario)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "optimize-energy failed",
            extra={"request_id": request_id, "scenario_id": request_body.scenario_id},
        )
        raise HTTPException(
            status_code=500,
            detail=f"internal error: {type(exc).__name__}",
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "optimize-energy completed",
        extra={
            "request_id": request_id,
            "scenario_id": request_body.scenario_id,
            "duration_ms": round(elapsed_ms, 3),
            "total_grid_kwh": response.total_grid_kwh,
            "total_cost_bdt": response.total_cost_bdt,
        },
    )
    return response
