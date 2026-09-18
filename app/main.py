"""FastAPI application entry point."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import get_settings
from app.logging import get_logger, setup_logging


# Optional single-page frontend directory. When present, ``index.html`` is
# served on ``GET /`` and the directory is mounted at ``/static``.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger(__name__)

    app = FastAPI(
        title="GridWise LLM",
        version="1.0.0",
        description=(
            "Smart Campus Energy Optimization service for BUP CSE FEST 2026. "
            "Interprets natural-language operator notes via an LLM, compiles them "
            "into mathematical constraints, and solves a deterministic LP to "
            "produce the minimum-cost 24-hour grid schedule."
        ),
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", "")
        logger.warning(
            "request validation error",
            extra={"request_id": rid, "errors": exc.errors()},
        )
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "request_id": rid},
            headers={"x-request-id": rid},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        rid = getattr(request.state, "request_id", "")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": rid},
            headers={"x-request-id": rid},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "")
        logger.exception("unhandled exception", extra={"request_id": rid})
        return JSONResponse(
            status_code=500,
            content={"detail": f"internal error: {type(exc).__name__}", "request_id": rid},
            headers={"x-request-id": rid},
        )

    # ---- Optional single-page frontend -------------------------------------
    # If `frontend/` exists, mount the directory as static and serve
    # `index.html` on `/`. Otherwise expose a tiny JSON descriptor.
    if FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "index.html").is_file():
        @app.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

        app.mount(
            "/static",
            StaticFiles(directory=str(FRONTEND_DIR)),
            name="frontend-static",
        )
        logger.info("frontend mounted", extra={"dir": str(FRONTEND_DIR)})
    else:

        @app.get("/", include_in_schema=False)
        def root() -> dict[str, Any]:
            return {
                "service": "GridWise LLM",
                "version": "1.0.0",
                "endpoints": ["/health", "/optimize-energy", "/docs"],
            }

    app.include_router(router)
    return app


app = create_app()
