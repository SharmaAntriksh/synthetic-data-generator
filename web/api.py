"""
web/api.py – FastAPI backend for the Synthetic Data Generator.

Exposes the same pipeline as the Streamlit UI via REST + SSE,
consumed by the React SPA served from web/frontend/.

Launch:  python -m uvicorn web.api:app --port 8502
   or:   .\\scripts\\run_web.ps1
"""

from __future__ import annotations

import asyncio
import copy
import json as _json
import os
import re
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "main.py").exists():
            return d
        d = d.parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Config state
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_config_path = REPO_ROOT / "config.yaml"
_models_path = REPO_ROOT / "models.yaml"


def _load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _base_config() -> dict:
    return _load_yaml(_config_path)


_cfg: Dict[str, Any] = _base_config()
_models_cfg: Dict[str, Any] = _load_yaml(_models_path)

# ---------------------------------------------------------------------------
# Preset logic
# ---------------------------------------------------------------------------

try:
    from ui.presets import PRESETS, apply_preset, build_presets_by_sales
except ImportError:
    PRESETS = {}
    apply_preset = None
    build_presets_by_sales = None

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_job_lock = threading.Lock()
_current_job: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(d: dict, *keys, default=None):
    """Nested dict get."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur if cur is not None else default


def _promo_total(promos: dict) -> int:
    keys = ("num_seasonal", "num_clearance", "num_limited")
    if all(k in promos for k in keys):
        return sum(int(promos.get(k, 0) or 0) for k in keys)
    return int(promos.get("total_promotions", 0) or 0)


def _set_promotions_total(promos: dict, total: int):
    """Distribute total across buckets proportionally."""
    total = max(0, int(total))
    keys = ["num_seasonal", "num_clearance", "num_limited"]
    if all(k in promos for k in keys):
        cur = [int(promos.get(k, 0) or 0) for k in keys]
        s = sum(cur) or 3
        base = cur if sum(cur) > 0 else [1, 1, 1]
        scaled = [b * total / s for b in base]
        floors = [int(x) for x in scaled]
        remainder = total - sum(floors)
        fracs = sorted(range(3), key=lambda i: scaled[i] - floors[i], reverse=True)
        for i in range(remainder):
            floors[fracs[i % 3]] += 1
        for i, k in enumerate(keys):
            promos[k] = floors[i]
    else:
        promos["total_promotions"] = total


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Synthetic Data Generator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    values: Dict[str, Any]

class PresetApply(BaseModel):
    name: str

class GenerateRequest(BaseModel):
    clean: bool = False
    only: Optional[str] = None
    regen_dimensions: Optional[List[str]] = None

# ---------------------------------------------------------------------------
# GET /api/config — flat shape for the frontend
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    defaults = _g(_cfg, "defaults", "dates", default={})
    sales = _g(_cfg, "sales", default={})
    cust = _g(_cfg, "customers", default={})
    stores = _g(_cfg, "stores", default={})
    prods = _g(_cfg, "products", default={})
    promos = _g(_cfg, "promotions", default={})
    pricing = _g(_cfg, "products", "pricing", "base", default={})
    returns = _g(_cfg, "returns", default={})
    geo = _g(_cfg, "geography", default={})
    dates_cfg = _g(_cfg, "dates", default={})
    include = _g(dates_cfg, "include", default={})
    wc = _g(dates_cfg, "weekly_calendar", default={})
    lifecycle = _g(cust, "lifecycle", default={})
    prod_lifecycle = _g(prods, "lifecycle", default={})
    tuning = _g(_models_cfg, "tuning", default={})

    return {
        # Output
        "format": str(sales.get("file_format", "csv")),
        "salesOutput": str(sales.get("sales_output", "sales")),
        "skipOrderCols": bool(sales.get("skip_order_cols", False)),
        "compression": str(sales.get("compression", "snappy")),
        "rowGroupSize": int(sales.get("row_group_size", 2000000)),
        "mergeParquet": bool(sales.get("merge_parquet", True)),
        "partitionEnabled": bool(_g(sales, "partitioning", "enabled", default=True)),
        # Dates
        "startDate": str(defaults.get("start", "2023-01-01")),
        "endDate": str(defaults.get("end", "2026-12-31")),
        "fiscalMonthOffset": int(dates_cfg.get("fiscal_month_offset", dates_cfg.get("fiscal_start_month", 0)) or 0),
        "includeCalendar": True,
        "includeIso": bool(include.get("iso", False)),
        "includeFiscal": bool(include.get("fiscal", True)),
        "includeWeeklyFiscal": bool(include.get("weekly_fiscal", False)),
        "wfFirstDay": int(wc.get("first_day_of_week", 0)),
        "wfWeeklyType": str(wc.get("weekly_type", "Last")),
        "wfQuarterType": str(wc.get("quarter_week_type", "445")),
        # Volume
        "salesRows": int(sales.get("total_rows", 100000)),
        "chunkSize": int(sales.get("chunk_size", 200000)),
        "workers": int(sales.get("workers", 0)),
        # Dimensions
        "customers": int(cust.get("total_customers", 10000)),
        "stores": int(stores.get("num_stores", 100)),
        "products": int(prods.get("num_products", 5000)),
        "promotions": _promo_total(promos),
        # Customers detail
        "pctIndia": float(cust.get("pct_india", 10)),
        "pctUs": float(cust.get("pct_us", 51)),
        "pctEu": float(cust.get("pct_eu", 39)),
        "pctAsia": float(cust.get("pct_asia", 0)),
        "pctOrg": float(cust.get("pct_org", 10)),
        "customerActiveRatio": float(cust.get("active_ratio", 0.90)),
        "churnEnabled": bool(lifecycle.get("enable_churn", True)),
        "baseMonthlyChurn": float(lifecycle.get("base_monthly_churn", 0.52)),
        "minTenureMonths": int(lifecycle.get("min_tenure_months", 22)),
        "initialActiveCust": int(lifecycle.get("initial_active_customers", 2000)),
        "initialSpreadMonths": int(lifecycle.get("initial_spread_months", 36)),
        # Products detail
        "valueScale": float(pricing.get("value_scale", 1.0)),
        "minPrice": float(pricing.get("min_unit_price", 100)),
        "maxPrice": float(pricing.get("max_unit_price", 5000)),
        "productActiveRatio": float(prods.get("active_ratio", 0.94)),
        "lookbackYears": int(prod_lifecycle.get("lookback_years", 10)),
        "lookaheadYears": int(prod_lifecycle.get("lookahead_years", 2)),
        "preexistingShare": float(prod_lifecycle.get("preexisting_share", 0.70)),
        "discontinueRatio": float(prod_lifecycle.get("discontinue_ratio", 0.20)),
        # Geography
        "geoWeights": dict(geo.get("country_weights", {})),
        # Returns
        "returnsEnabled": bool(returns.get("enabled", True)),
        "returnRate": float(returns.get("return_rate", 0.03)),
        "returnMinDays": int(returns.get("min_days_after_sale", 1)),
        "returnMaxDays": int(returns.get("max_days_after_sale", 60)),
        # Tuning
        "tuningIntensity": float(tuning.get("acquisition_intensity", 0.55)),
        "tuningSmoothness": float(tuning.get("acquisition_smoothness", 0.90)),
        "tuningCycles": float(tuning.get("acquisition_cycles", 0.35)),
    }

# ---------------------------------------------------------------------------
# POST /api/config — apply partial updates from frontend
# ---------------------------------------------------------------------------

@app.post("/api/config")
def update_config(body: ConfigUpdate):
    v = body.values

    _cfg.setdefault("defaults", {}).setdefault("dates", {})
    _cfg.setdefault("sales", {}).setdefault("partitioning", {})
    _cfg.setdefault("customers", {}).setdefault("lifecycle", {})
    _cfg.setdefault("stores", {})
    _cfg.setdefault("products", {}).setdefault("pricing", {}).setdefault("base", {})
    _cfg["products"].setdefault("lifecycle", {})
    _cfg.setdefault("promotions", {})
    _cfg.setdefault("returns", {})
    _cfg.setdefault("geography", {}).setdefault("country_weights", {})
    _cfg.setdefault("dates", {}).setdefault("include", {})
    _cfg["dates"].setdefault("weekly_calendar", {})

    # Output
    if "format" in v: _cfg["sales"]["file_format"] = v["format"]
    if "salesOutput" in v: _cfg["sales"]["sales_output"] = v["salesOutput"]
    if "skipOrderCols" in v: _cfg["sales"]["skip_order_cols"] = bool(v["skipOrderCols"])
    if "compression" in v: _cfg["sales"]["compression"] = v["compression"]
    if "rowGroupSize" in v: _cfg["sales"]["row_group_size"] = int(v["rowGroupSize"])
    if "mergeParquet" in v: _cfg["sales"]["merge_parquet"] = bool(v["mergeParquet"])
    if "partitionEnabled" in v: _cfg["sales"]["partitioning"]["enabled"] = bool(v["partitionEnabled"])

    # Dates
    if "startDate" in v: _cfg["defaults"]["dates"]["start"] = v["startDate"]
    if "endDate" in v: _cfg["defaults"]["dates"]["end"] = v["endDate"]
    if "fiscalMonthOffset" in v: _cfg["dates"]["fiscal_month_offset"] = int(v["fiscalMonthOffset"])
    if "includeIso" in v: _cfg["dates"]["include"]["iso"] = bool(v["includeIso"])
    if "includeFiscal" in v: _cfg["dates"]["include"]["fiscal"] = bool(v["includeFiscal"])
    if "includeWeeklyFiscal" in v: _cfg["dates"]["include"]["weekly_fiscal"] = bool(v["includeWeeklyFiscal"])
    if "wfFirstDay" in v: _cfg["dates"]["weekly_calendar"]["first_day_of_week"] = int(v["wfFirstDay"])
    if "wfWeeklyType" in v: _cfg["dates"]["weekly_calendar"]["weekly_type"] = v["wfWeeklyType"]
    if "wfQuarterType" in v: _cfg["dates"]["weekly_calendar"]["quarter_week_type"] = v["wfQuarterType"]

    # Volume
    if "salesRows" in v: _cfg["sales"]["total_rows"] = int(v["salesRows"])
    if "chunkSize" in v: _cfg["sales"]["chunk_size"] = int(v["chunkSize"])
    if "workers" in v: _cfg["sales"]["workers"] = int(v["workers"])

    # Dimensions
    if "customers" in v: _cfg["customers"]["total_customers"] = int(v["customers"])
    if "stores" in v: _cfg["stores"]["num_stores"] = int(v["stores"])
    if "products" in v: _cfg["products"]["num_products"] = int(v["products"])
    if "promotions" in v: _set_promotions_total(_cfg["promotions"], int(v["promotions"]))

    # Customers detail
    if "pctIndia" in v: _cfg["customers"]["pct_india"] = float(v["pctIndia"])
    if "pctUs" in v: _cfg["customers"]["pct_us"] = float(v["pctUs"])
    if "pctEu" in v: _cfg["customers"]["pct_eu"] = float(v["pctEu"])
    if "pctAsia" in v: _cfg["customers"]["pct_asia"] = float(v["pctAsia"])
    if "pctOrg" in v: _cfg["customers"]["pct_org"] = float(v["pctOrg"])
    if "customerActiveRatio" in v: _cfg["customers"]["active_ratio"] = float(v["customerActiveRatio"])
    if "churnEnabled" in v: _cfg["customers"]["lifecycle"]["enable_churn"] = bool(v["churnEnabled"])
    if "baseMonthlyChurn" in v: _cfg["customers"]["lifecycle"]["base_monthly_churn"] = float(v["baseMonthlyChurn"])
    if "minTenureMonths" in v: _cfg["customers"]["lifecycle"]["min_tenure_months"] = int(v["minTenureMonths"])
    if "initialActiveCust" in v: _cfg["customers"]["lifecycle"]["initial_active_customers"] = int(v["initialActiveCust"])
    if "initialSpreadMonths" in v: _cfg["customers"]["lifecycle"]["initial_spread_months"] = int(v["initialSpreadMonths"])

    # Products detail
    if "valueScale" in v: _cfg["products"]["pricing"]["base"]["value_scale"] = float(v["valueScale"])
    if "minPrice" in v: _cfg["products"]["pricing"]["base"]["min_unit_price"] = float(v["minPrice"])
    if "maxPrice" in v: _cfg["products"]["pricing"]["base"]["max_unit_price"] = float(v["maxPrice"])
    if "productActiveRatio" in v: _cfg["products"]["active_ratio"] = float(v["productActiveRatio"])
    if "lookbackYears" in v: _cfg["products"]["lifecycle"]["lookback_years"] = int(v["lookbackYears"])
    if "lookaheadYears" in v: _cfg["products"]["lifecycle"]["lookahead_years"] = int(v["lookaheadYears"])
    if "preexistingShare" in v: _cfg["products"]["lifecycle"]["preexisting_share"] = float(v["preexistingShare"])
    if "discontinueRatio" in v: _cfg["products"]["lifecycle"]["discontinue_ratio"] = float(v["discontinueRatio"])

    # Geography
    if "geoWeights" in v and isinstance(v["geoWeights"], dict):
        _cfg["geography"]["country_weights"] = v["geoWeights"]

    # Returns
    if "returnsEnabled" in v: _cfg["returns"]["enabled"] = bool(v["returnsEnabled"])
    if "returnRate" in v: _cfg["returns"]["return_rate"] = float(v["returnRate"])
    if "returnMinDays" in v: _cfg["returns"]["min_days_after_sale"] = int(v["returnMinDays"])
    if "returnMaxDays" in v: _cfg["returns"]["max_days_after_sale"] = int(v["returnMaxDays"])

    # Models tuning
    _models_cfg.setdefault("tuning", {})
    if "tuningIntensity" in v: _models_cfg["tuning"]["acquisition_intensity"] = float(v["tuningIntensity"])
    if "tuningSmoothness" in v: _models_cfg["tuning"]["acquisition_smoothness"] = float(v["tuningSmoothness"])
    if "tuningCycles" in v: _models_cfg["tuning"]["acquisition_cycles"] = float(v["tuningCycles"])

    return {"ok": True}


@app.get("/api/config/download")
def download_config():
    return _cfg


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

@app.get("/api/presets")
def get_presets():
    if build_presets_by_sales is None:
        return {}
    return build_presets_by_sales()


@app.post("/api/presets/apply")
def apply_preset_route(body: PresetApply):
    if apply_preset is None:
        raise HTTPException(400, "Presets module not available")
    if body.name not in PRESETS:
        raise HTTPException(404, f"Unknown preset: {body.name}")
    apply_preset(_cfg, _base_config, body.name)
    return {"ok": True, "applied": body.name}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@app.get("/api/validate")
def validate_config():
    errors: List[str] = []
    warnings: List[str] = []

    defaults = _g(_cfg, "defaults", "dates", default={})
    start = str(defaults.get("start", ""))
    end = str(defaults.get("end", ""))
    if start and end and end < start:
        errors.append("End date must be after start date.")

    sales = _g(_cfg, "sales", default={})
    rows = int(sales.get("total_rows", 0))
    if rows <= 0:
        errors.append("Total rows must be greater than zero.")
    chunk = int(sales.get("chunk_size", 0))
    if chunk > rows > 0:
        warnings.append("Chunk size exceeds total rows.")
    fmt = sales.get("file_format", "csv")
    if fmt == "csv" and rows > 5_000_000:
        warnings.append("Large CSV outputs can be slow. Consider parquet.")

    customers = int(_g(_cfg, "customers", "total_customers", default=0))
    products = int(_g(_cfg, "products", "num_products", default=0))
    if customers > rows > 0:
        warnings.append(f"Customers ({customers:,}) exceed sales rows.")
    if products > rows > 0:
        warnings.append(f"Products ({products:,}) exceed sales rows.")

    pricing = _g(_cfg, "products", "pricing", "base", default={})
    lo = float(pricing.get("min_unit_price", 10))
    hi = float(pricing.get("max_unit_price", 5000))
    if hi <= lo:
        errors.append("Max unit price must exceed min unit price.")

    returns = _g(_cfg, "returns", default={})
    if returns.get("enabled"):
        rr = float(returns.get("return_rate", 0))
        if rr < 0 or rr > 1:
            errors.append("Return rate must be between 0 and 1.")

    geo = _g(_cfg, "geography", "country_weights", default={})
    if geo:
        gs = sum(float(v) for v in geo.values())
        if abs(gs - 1.0) > 0.05:
            warnings.append(f"Geography weights sum to {gs*100:.0f}% (expected ~100%).")

    cust = _g(_cfg, "customers", default={})
    rS = float(cust.get("pct_india", 0)) + float(cust.get("pct_us", 0)) + float(cust.get("pct_eu", 0)) + float(cust.get("pct_asia", 0))
    if rS <= 0:
        errors.append("Customer regional percentages must sum to > 0.")

    ret = _g(_cfg, "returns", default={})
    if ret.get("enabled"):
        mn = int(ret.get("min_days_after_sale", 1))
        mx = int(ret.get("max_days_after_sale", 60))
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
                yaml.safe_dump(cfg_snapshot, f, sort_keys=False)
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


@app.post("/api/generate")
def start_generate(req: GenerateRequest):
    global _current_job
    with _job_lock:
        if _current_job and _current_job["status"] == "running":
            raise HTTPException(409, "A pipeline is already running.")
        job = {
            "id": str(uuid.uuid4())[:8], "status": "running",
            "logs": deque(maxlen=5000), "process": None,
            "exit_code": None, "started": time.time(),
            "ended": None, "elapsed": 0, "command": "",
        }
        _current_job = job
    cfg_snapshot = copy.deepcopy(_cfg)
    models_snapshot = copy.deepcopy(_models_cfg)
    t = threading.Thread(target=_run_pipeline_thread, args=(job, cfg_snapshot, models_snapshot, req), daemon=True)
    t.start()
    return {"ok": True, "job_id": job["id"]}


@app.get("/api/generate/stream")
async def stream_logs():
    async def event_stream():
        idx = 0
        while True:
            job = _current_job
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


@app.post("/api/generate/cancel")
def cancel_generate():
    with _job_lock:
        if not _current_job or _current_job["status"] != "running":
            raise HTTPException(400, "No running job to cancel.")
        proc = _current_job.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            _current_job["status"] = "cancelled"
            _current_job["logs"].append("Pipeline cancelled by user.")
            return {"ok": True}
    raise HTTPException(400, "Could not cancel.")


@app.get("/api/generate/status")
def job_status():
    if not _current_job:
        return {"status": "idle"}
    elapsed = time.time() - _current_job["started"] if _current_job["status"] == "running" else _current_job.get("elapsed", 0)
    return {
        "id": _current_job["id"], "status": _current_job["status"],
        "elapsed": round(elapsed, 1), "exit_code": _current_job.get("exit_code"),
        "log_count": len(_current_job["logs"]),
    }


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
