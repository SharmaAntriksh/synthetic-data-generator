"""
web/api.py – FastAPI backend for the Synthetic Data Generator.

Exposes the pipeline via REST + SSE,
consumed by the React SPA served from web/frontend/.

Launch:  python -m uvicorn web.api:app --port 8502
   or:   .\\scripts\\run_web.ps1
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from web.routes.config_routes import router as config_router
from web.routes.models_routes import router as models_router
from web.routes.generation_routes import router as generation_router
from web.routes.presets_routes import router as presets_router

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Synthetic Data Generator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Include routers — backward-compatible paths (no prefix)
# ---------------------------------------------------------------------------

app.include_router(config_router)
app.include_router(models_router)
app.include_router(generation_router)
app.include_router(presets_router)

# ---------------------------------------------------------------------------
# Include routers — versioned paths (/v1/api/...)
# ---------------------------------------------------------------------------

app.include_router(config_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(generation_router, prefix="/v1")
app.include_router(presets_router, prefix="/v1")

# ---------------------------------------------------------------------------
# Favicon (suppress browser 404)
# ---------------------------------------------------------------------------

_FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="6" fill="#4f5bd5"/>'
    b'<text x="16" y="23" font-size="20" text-anchor="middle" fill="#fff" '
    b'font-family="system-ui">D</text></svg>'
)


@app.get("/favicon.ico")
def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = _FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Frontend not found. Place index.html in web/frontend/")
    return index.read_text(encoding="utf-8")


if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
