"""FastAPI application factory. Serves the API and (if built) the React frontend."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import ftad.modules  # noqa: F401  (registers domain hint/evidence providers)
from ftad import __version__
from ftad.api import routes_data, routes_inference, routes_ml, routes_reports
from ftad.api.state import AppState
from ftad.config import Settings, get_settings
from ftad.core.experiments.runner import package_versions

LOCAL_HOSTS = ["127.0.0.1", "localhost", "[::1]", "testserver"]


def create_app(settings: Settings | None = None, start_jobs: bool = True) -> FastAPI:
    settings = settings or get_settings()
    state = AppState(settings, start_jobs=start_jobs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if state.jobs:
            state.jobs.shutdown()

    app = FastAPI(title="FinTech Anomaly Detection (FTAD)", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.ftad = state
    # Local-only operation: reject requests whose Host header is not a loopback name (DNS-rebinding guard).
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS + os.environ.get("FTAD_EXTRA_HOSTS", "").split(","))
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/system")
    def system():
        return {"version": __version__, "packages": package_versions(), "cpu_count": os.cpu_count(),
                "max_upload_mb": settings.max_upload_mb, "workers_per_model": settings.max_workers_per_model,
                "data_dir": str(settings.data_dir), "export_dir": str(settings.export_dir) if settings.export_dir else None,
                "active_experiment": state.jobs.active() if state.jobs else None}

    app.include_router(routes_data.build_router(state))
    app.include_router(routes_ml.build_router(state))
    app.include_router(routes_inference.build_router(state))
    app.include_router(routes_reports.build_router(state))

    @app.exception_handler(FileNotFoundError)
    async def not_found(_: Request, exc: FileNotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    dist = settings.frontend_dist
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = (dist / path).resolve()
            if path and dist.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


def main() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("ftad.app:create_app", factory=True, host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
