"""
web/routes/generation_routes.py -- /api/generate and /api/validate endpoints.
"""

from __future__ import annotations

import asyncio
import copy
import json as _json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from web.shared_state import (
    REPO_ROOT,
    _ANSI_RE,
    _g,
    _job_lock,
    cfg_to_dict,
)
import web.shared_state as _state

router = APIRouter(prefix="/api", tags=["generation"])

_VALID_ONLY = {"dimensions", "sales"}
_VALID_REGEN = {"all", "products", "customers", "stores", "employees", "dates",
                "geography", "currency", "exchange_rates", "promotions", "suppliers",
                "return_reasons", "superpowers", "lookups", "time"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    clean: bool = False
    only: Optional[str] = None
    regen_dimensions: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@router.get("/validate")
def validate_config():
    with _state._cfg_lock:
        cfg = copy.deepcopy(_state._cfg)
    errors: List[str] = []
    warnings: List[str] = []

    defaults = _g(cfg, "defaults", "dates", default={})
    start = str(getattr(defaults, "start", ""))
    end = str(getattr(defaults, "end", ""))
    if start and end and end < start:
        errors.append("End date must be after start date.")

    sales = _g(cfg, "sales", default={})
    rows = int(getattr(sales, "total_rows", 0))
    if rows <= 0:
        errors.append("Total rows must be greater than zero.")
    chunk = int(getattr(sales, "chunk_size", 0))
    if chunk > rows > 0:
        warnings.append("Chunk size exceeds total rows.")
    fmt = getattr(sales, "file_format", "parquet")
    if fmt == "csv" and rows > 5_000_000:
        warnings.append("Large CSV outputs can be slow. Consider parquet.")

    if getattr(sales, "skip_order_cols", False):
        warnings.append("Order columns skipped — Returns will not be generated.")

    customers = int(_g(cfg, "customers", "total_customers", default=0))
    products = int(_g(cfg, "products", "num_products", default=0))
    if customers > rows > 0:
        warnings.append(f"Customers ({customers:,}) exceed sales rows.")
    if products > rows > 0:
        warnings.append(f"Products ({products:,}) exceed sales rows.")

    pricing = _g(cfg, "products", "pricing", "base", default={})
    # pricing is a plain dict (products.pricing is Dict[str, Any])
    lo = float(pricing.get("min_unit_price", 10) if isinstance(pricing, dict) else getattr(pricing, "min_unit_price", 10))
    hi = float(pricing.get("max_unit_price", 5000) if isinstance(pricing, dict) else getattr(pricing, "max_unit_price", 5000))
    if hi <= lo:
        errors.append("Max unit price must exceed min unit price.")

    returns = _g(cfg, "returns", default={})
    if getattr(returns, "enabled", False):
        rr = float(getattr(returns, "return_rate", 0))
        if rr < 0 or rr > 1:
            errors.append("Return rate must be between 0 and 1.")

    geo = _g(cfg, "geography", "country_weights", default={})
    if geo:
        gs = sum(float(v) for v in geo.values())
        if abs(gs - 1.0) > 0.05:
            warnings.append(f"Geography weights sum to {gs*100:.0f}% (expected ~100%).")

    cust = _g(cfg, "customers", default={})
    rS = float(getattr(cust, "pct_india", 0)) + float(getattr(cust, "pct_us", 0)) + float(getattr(cust, "pct_eu", 0)) + float(getattr(cust, "pct_asia", 0))
    if rS <= 0:
        warnings.append("Customer regional percentages sum to 0.")

    ret = _g(cfg, "returns", default={})
    if getattr(ret, "enabled", False):
        mn = int(getattr(ret, "min_days_after_sale", 1))
        mx = int(getattr(ret, "max_days_after_sale", 60))
        if mx <= mn:
            warnings.append("Return max days should exceed min days.")

    return {"errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Generate (subprocess + SSE)
# ---------------------------------------------------------------------------

def _run_pipeline_thread(job: dict, cfg_snapshot: dict, models_snapshot: dict, req: GenerateRequest):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            models_path = Path(tmp) / "models.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_to_dict(cfg_snapshot), f, sort_keys=False)
            with open(models_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(models_snapshot, f, sort_keys=False)

            cmd = [
                sys.executable, "-u", str(REPO_ROOT / "main.py"),
                "--config", str(cfg_path),
                "--models-config", str(models_path),
            ]
            if req.clean:
                cmd.append("--clean")
            if req.only:
                cmd.extend(["--only", req.only])
            if req.regen_dimensions:
                cmd.extend(["--regen-dimensions", *req.regen_dimensions])

            job["command"] = " ".join(cmd)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, cwd=str(REPO_ROOT),
            )
            job["process"] = proc

            for line in proc.stdout:
                clean = _ANSI_RE.sub("", line.rstrip())
                job["logs"].append(clean)

            proc.wait()
            job["exit_code"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else "failed"
    except Exception as e:
        job["logs"].append(f"ERROR: {e}")
        job["status"] = "failed"
        job["exit_code"] = -1
    finally:
        job["ended"] = time.time()
        job["elapsed"] = job["ended"] - job["started"]


@router.post("/generate")
def start_generate(req: GenerateRequest):
    # Validate request before starting thread (HTTPException can't be returned from threads)
    if req.only and req.only not in _VALID_ONLY:
        raise HTTPException(400, f"Invalid 'only' value: {req.only}. Must be one of {_VALID_ONLY}")
    if req.regen_dimensions:
        invalid = set(req.regen_dimensions) - _VALID_REGEN
        if invalid:
            raise HTTPException(400, f"Invalid regen_dimensions: {invalid}. Must be from {_VALID_REGEN}")
    with _job_lock:
        if _state._current_job and _state._current_job["status"] == "running":
            raise HTTPException(409, "A pipeline is already running.")
        job = {
            "id": str(uuid.uuid4())[:8], "status": "running",
            "logs": [], "process": None,
            "exit_code": None, "started": time.time(),
            "ended": None, "elapsed": 0, "command": "",
        }
        _state._current_job = job
    with _state._cfg_lock:
        cfg_snapshot = copy.deepcopy(_state._cfg)
        models_snapshot = copy.deepcopy(_state._models_cfg)
    t = threading.Thread(target=_run_pipeline_thread, args=(job, cfg_snapshot, models_snapshot, req), daemon=True)
    t.start()
    return {"ok": True, "job_id": job["id"]}


@router.get("/generate/stream")
async def stream_logs():
    async def event_stream():
        idx = 0
        while True:
            with _job_lock:
                job = _state._current_job
            if job is None:
                yield f"data: {_json.dumps({'type': 'idle'})}\n\n"
                break
            logs = list(job["logs"])
            for line in logs[idx:]:
                yield f"data: {_json.dumps({'type': 'log', 'line': line})}\n\n"
            idx = len(logs)
            elapsed = time.time() - job["started"] if job["status"] == "running" else job.get("elapsed", 0)
            yield f"data: {_json.dumps({'type': 'status', 'status': job['status'], 'elapsed': round(elapsed, 1)})}\n\n"
            if job["status"] in ("done", "failed", "cancelled"):
                yield f"data: {_json.dumps({'type': 'end', 'status': job['status'], 'exit_code': job.get('exit_code'), 'elapsed': round(elapsed, 1)})}\n\n"
                break
            await asyncio.sleep(0.3)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/generate/cancel")
def cancel_generate():
    with _job_lock:
        if not _state._current_job or _state._current_job["status"] != "running":
            raise HTTPException(400, "No running job to cancel.")
        proc = _state._current_job.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            _state._current_job["status"] = "cancelled"
            _state._current_job["logs"].append("Pipeline cancelled by user.")
            return {"ok": True}
    raise HTTPException(400, "Could not cancel.")


@router.get("/generate/status")
def job_status():
    with _job_lock:
        job = _state._current_job
    if not job:
        return {"status": "idle"}
    elapsed = time.time() - job["started"] if job["status"] == "running" else job.get("elapsed", 0)
    return {
        "id": job["id"], "status": job["status"],
        "elapsed": round(elapsed, 1), "exit_code": job.get("exit_code"),
        "log_count": len(job["logs"]),
    }
