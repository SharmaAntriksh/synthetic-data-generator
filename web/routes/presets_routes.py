"""
web/routes/presets_routes.py -- /api/presets/* endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from web.shared_state import (
    PRESETS,
    apply_preset,
    build_presets_by_sales,
    _base_config,
)
import web.shared_state as _state

router = APIRouter(prefix="/api/presets", tags=["presets"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PresetApply(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("")
def get_presets():
    if build_presets_by_sales is None:
        return {}
    return build_presets_by_sales()


@router.post("/apply")
def apply_preset_route(body: PresetApply):
    if apply_preset is None:
        raise HTTPException(400, "Presets module not available")
    if body.name not in PRESETS:
        raise HTTPException(404, f"Unknown preset: {body.name}")
    with _state._cfg_lock:
        apply_preset(_state._cfg, _base_config, body.name)
    return {"ok": True, "applied": body.name}
