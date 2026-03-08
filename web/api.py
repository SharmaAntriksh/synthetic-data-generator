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
from fastapi.responses import HTMLResponse, Response, StreamingResponse
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
# Config normalization (reuse pipeline's normalizers)
# ---------------------------------------------------------------------------

try:
    from src.engine.config.config import load_config as _load_pipeline_config
    _HAS_NORMALIZER = True
except ImportError:
    _HAS_NORMALIZER = False

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
    if _HAS_NORMALIZER:
        try:
            return _load_pipeline_config(str(_config_path))
        except Exception:
            pass
    return _load_yaml(_config_path)


_cfg: Dict[str, Any] = _base_config()
_cfg_disk_yaml: str = _config_path.read_text(encoding="utf-8") if _config_path.exists() else ""
_models_cfg: Dict[str, Any] = _load_yaml(_models_path)
_models_yaml_text: str = _models_path.read_text(encoding="utf-8") if _models_path.exists() else ""

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
    wf = _g(include, "weekly_fiscal", default={})
    if isinstance(wf, bool):
        wf = {"enabled": wf}

    return {
        # Output
        "format": str(sales.get("file_format", "parquet")),
        "salesOutput": str(sales.get("sales_output", "sales")),
        "skipOrderCols": bool(sales.get("skip_order_cols", False)),
        "compression": str(_g(sales, "compression", default="snappy")),
        "rowGroupSize": int(_g(sales, "row_group_size", default=2000000)),
        "mergeParquet": bool(sales.get("merge_parquet", True)),
        "partitionEnabled": bool(_g(sales, "partitioning", "enabled", default=True)),
        # Dates
        "startDate": str(defaults.get("start", "2023-01-01")),
        "endDate": str(defaults.get("end", "2026-12-31")),
        "fiscalMonthOffset": int(dates_cfg.get("fiscal_month_offset", dates_cfg.get("fiscal_start_month", 0)) or 0),
        "includeCalendar": True,
        "includeIso": bool(include.get("iso", False)),
        "includeFiscal": bool(include.get("fiscal", True)),
        "includeWeeklyFiscal": bool(wf.get("enabled", False)),
        "wfFirstDay": int(wf.get("first_day_of_week", 0)),
        "wfWeeklyType": str(wf.get("weekly_type", "Last")),
        "wfQuarterType": str(wf.get("quarter_week_type", "445")),
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
        "profile": str(cust.get("profile", "steady")),
        "firstYearPct": float(cust.get("first_year_pct", 0.27)),
        # Products detail
        "valueScale": float(pricing.get("value_scale", 1.0)),
        "minPrice": float(pricing.get("min_unit_price", 10)),
        "maxPrice": float(pricing.get("max_unit_price", 3000)),
        "productActiveRatio": float(prods.get("active_ratio", 0.94)),
        # Geography
        "geoWeights": dict(geo.get("country_weights", {})),
        # Returns
        "returnsEnabled": bool(returns.get("enabled", True)),
        "returnRate": float(returns.get("return_rate", 0.03)),
        "returnMinDays": int(returns.get("min_days_after_sale", 1)),
        "returnMaxDays": int(returns.get("max_days_after_sale", 60)),
        # Budget & Inventory
        "budgetEnabled": bool(_g(_cfg, "budget", "enabled", default=True)),
        "inventoryEnabled": bool(_g(_cfg, "inventory", "enabled", default=True)),
    }

# ---------------------------------------------------------------------------
# POST /api/config — apply partial updates from frontend
# ---------------------------------------------------------------------------

@app.post("/api/config")
def update_config(body: ConfigUpdate):
    v = body.values

    _cfg.setdefault("defaults", {}).setdefault("dates", {})
    _cfg.setdefault("sales", {}).setdefault("partitioning", {})
    _cfg.setdefault("customers", {})
    _cfg.setdefault("stores", {})
    _cfg.setdefault("products", {}).setdefault("pricing", {}).setdefault("base", {})
    _cfg.setdefault("promotions", {})
    _cfg.setdefault("returns", {})
    _cfg.setdefault("geography", {}).setdefault("country_weights", {})
    _cfg.setdefault("dates", {}).setdefault("include", {}).setdefault("weekly_fiscal", {})

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
    if "includeWeeklyFiscal" in v: _cfg["dates"]["include"]["weekly_fiscal"]["enabled"] = bool(v["includeWeeklyFiscal"])
    if "wfFirstDay" in v: _cfg["dates"]["include"]["weekly_fiscal"]["first_day_of_week"] = int(v["wfFirstDay"])
    if "wfWeeklyType" in v: _cfg["dates"]["include"]["weekly_fiscal"]["weekly_type"] = v["wfWeeklyType"]
    if "wfQuarterType" in v: _cfg["dates"]["include"]["weekly_fiscal"]["quarter_week_type"] = v["wfQuarterType"]

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
    if "profile" in v: _cfg["customers"]["profile"] = v["profile"]
    if "firstYearPct" in v: _cfg["customers"]["first_year_pct"] = float(v["firstYearPct"])

    # Products detail
    if "valueScale" in v: _cfg["products"]["pricing"]["base"]["value_scale"] = float(v["valueScale"])
    if "minPrice" in v: _cfg["products"]["pricing"]["base"]["min_unit_price"] = float(v["minPrice"])
    if "maxPrice" in v: _cfg["products"]["pricing"]["base"]["max_unit_price"] = float(v["maxPrice"])
    if "productActiveRatio" in v: _cfg["products"]["active_ratio"] = float(v["productActiveRatio"])

    # Geography
    if "geoWeights" in v and isinstance(v["geoWeights"], dict):
        _cfg["geography"]["country_weights"] = v["geoWeights"]

    # Returns
    if "returnsEnabled" in v: _cfg["returns"]["enabled"] = bool(v["returnsEnabled"])
    if "returnRate" in v: _cfg["returns"]["return_rate"] = float(v["returnRate"])
    if "returnMinDays" in v: _cfg["returns"]["min_days_after_sale"] = int(v["returnMinDays"])
    if "returnMaxDays" in v: _cfg["returns"]["max_days_after_sale"] = int(v["returnMaxDays"])

    # Budget & Inventory
    if "budgetEnabled" in v: _cfg.setdefault("budget", {})["enabled"] = bool(v["budgetEnabled"])
    if "inventoryEnabled" in v: _cfg.setdefault("inventory", {})["enabled"] = bool(v["inventoryEnabled"])

    return {"ok": True}


@app.get("/api/config/download")
def download_config():
    return _cfg


# ---------------------------------------------------------------------------
# Config YAML editor (in-memory only, never writes to disk)
# ---------------------------------------------------------------------------

@app.get("/api/config/yaml")
def get_config_yaml():
    """Return the current in-memory config serialized as YAML text."""
    text = yaml.safe_dump(_cfg, sort_keys=False, default_flow_style=False)
    return Response(content=text, media_type="text/plain")


@app.get("/api/config/yaml/disk")
def get_config_yaml_disk():
    """Return the original config.yaml from disk (for three-state tracking)."""
    return Response(content=_cfg_disk_yaml, media_type="text/plain")


class ConfigYamlUpdate(BaseModel):
    yaml_text: str


@app.post("/api/config/yaml")
def update_config_yaml(body: ConfigYamlUpdate):
    """Parse YAML, normalize, replace in-memory config. Original file untouched."""
    global _cfg
    text = body.yaml_text
    try:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise HTTPException(400, "Config YAML must be a mapping at the top level.")
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")

    # Normalize shorthand keys (scale → per-section, region_mix → pct_*, etc.)
    if _HAS_NORMALIZER:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as tmp:
                yaml.safe_dump(parsed, tmp, sort_keys=False)
                tmp_path = tmp.name
            _cfg = _load_pipeline_config(tmp_path)
            os.unlink(tmp_path)
        except Exception:
            _cfg = parsed
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        _cfg = parsed
    return {"ok": True}


@app.post("/api/config/yaml/reset")
def reset_config_yaml():
    """Reload config from disk, discarding in-memory edits."""
    global _cfg, _cfg_disk_yaml
    _cfg = _base_config()
    _cfg_disk_yaml = _config_path.read_text(encoding="utf-8") if _config_path.exists() else ""
    return {"ok": True}


# ---------------------------------------------------------------------------
# Models YAML editor (in-memory only, never writes to disk)
# ---------------------------------------------------------------------------

@app.get("/api/models")
def get_models():
    """Return the current models YAML as raw text for the editor."""
    return Response(content=_models_yaml_text, media_type="text/plain")


class ModelsUpdate(BaseModel):
    yaml_text: str


@app.post("/api/models")
def update_models(body: ModelsUpdate):
    """Parse and store updated models YAML in memory. Original file is untouched."""
    global _models_cfg, _models_yaml_text
    text = body.yaml_text
    try:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise HTTPException(400, "Models YAML must be a mapping at the top level.")
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")
    _models_cfg = parsed
    _models_yaml_text = text
    return {"ok": True}


@app.post("/api/models/reset")
def reset_models():
    """Reload models from disk, discarding in-memory edits."""
    global _models_cfg, _models_yaml_text
    _models_cfg = _load_yaml(_models_path)
    _models_yaml_text = _models_path.read_text(encoding="utf-8") if _models_path.exists() else ""
    return {"ok": True}


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
    fmt = sales.get("file_format", "parquet")
    if fmt == "csv" and rows > 5_000_000:
        warnings.append("Large CSV outputs can be slow. Consider parquet.")

    if sales.get("skip_order_cols"):
        warnings.append("Order columns skipped — Returns will not be generated.")

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
        warnings.append("Customer regional percentages sum to 0.")

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
