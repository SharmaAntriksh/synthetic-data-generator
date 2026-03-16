"""
web/routes/import_routes.py -- SQL Server import endpoints.

Runs the import script as a subprocess with SSE log streaming,
mirroring the generation_routes.py pattern.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from web.shared_state import REPO_ROOT, _ANSI_RE

router = APIRouter(prefix="/api", tags=["import"])

_DATASETS_DIR = REPO_ROOT / "generated_datasets"
_IMPORT_SCRIPT = REPO_ROOT / "scripts" / "sql" / "run_sql_server_import.py"

# ---------------------------------------------------------------------------
# Job state (single import at a time)
# ---------------------------------------------------------------------------

_import_lock = threading.Lock()
_current_import: dict | None = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ImportRequest(BaseModel):
    server: str
    database: str
    dataset: str                          # folder name in generated_datasets/
    trusted: bool = True
    user: Optional[str] = None
    password: Optional[str] = None
    apply_cci: bool = False
    odbc_driver: Optional[str] = None


# ---------------------------------------------------------------------------
# ODBC driver detection
# ---------------------------------------------------------------------------

@router.get("/import/drivers")
def list_odbc_drivers():
    """List available SQL Server ODBC drivers on this machine."""
    try:
        import pyodbc
        all_drivers = pyodbc.drivers()
        sql_drivers = [d for d in all_drivers if "sql server" in d.lower()]
        return {"available": True, "drivers": sql_drivers}
    except ImportError:
        return {"available": False, "drivers": [], "error": "pyodbc not installed"}
    except Exception as exc:
        return {"available": False, "drivers": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Import execution
# ---------------------------------------------------------------------------

def _run_import_thread(job: dict, req: ImportRequest):
    """Run the SQL Server import script as a subprocess."""
    global _current_import
    try:
        dataset_path = _DATASETS_DIR / req.dataset
        if not dataset_path.is_dir():
            job["logs"].append(f"ERROR: Dataset not found: {req.dataset}")
            job["status"] = "failed"
            job["exit_code"] = -1
            return

        cmd = [
            sys.executable, "-u", str(_IMPORT_SCRIPT),
            "--server", req.server,
            "--database", req.database,
            "--run-path", str(dataset_path),
        ]

        if req.trusted:
            cmd.append("--trusted")
        else:
            if req.user:
                cmd.extend(["--user", req.user])
            if req.password:
                cmd.extend(["--password", req.password])

        if req.apply_cci:
            cmd.append("--apply-cci")

        if req.odbc_driver:
            cmd.extend(["--odbc-driver", req.odbc_driver])

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


@router.post("/import/start")
def start_import(req: ImportRequest):
    """Start a SQL Server import job."""
    global _current_import

    # Validate inputs
    if not req.server or not req.server.strip():
        raise HTTPException(400, "Server name is required.")
    if not req.database or not req.database.strip():
        raise HTTPException(400, "Database name is required.")
    if not req.dataset or not req.dataset.strip():
        raise HTTPException(400, "Dataset must be selected.")
    if not req.trusted and not req.user:
        raise HTTPException(400, "Username is required when not using Windows Authentication.")

    # Validate dataset exists and has SQL scripts (CSV format only)
    dataset_path = _DATASETS_DIR / req.dataset
    if not dataset_path.is_dir():
        raise HTTPException(404, f"Dataset not found: {req.dataset}")
    sql_dir = dataset_path / "sql"
    if not sql_dir.is_dir():
        raise HTTPException(400, "Dataset has no SQL scripts. SQL import requires CSV format output.")

    with _import_lock:
        if _current_import and _current_import["status"] == "running":
            raise HTTPException(409, "An import is already running.")
        job = {
            "id": str(uuid.uuid4())[:8],
            "status": "running",
            "logs": [],
            "process": None,
            "exit_code": None,
            "started": time.time(),
            "ended": None,
            "elapsed": 0,
            "command": "",
            "dataset": req.dataset,
            "database": req.database,
            "server": req.server,
        }
        _current_import = job

    t = threading.Thread(target=_run_import_thread, args=(job, req), daemon=True)
    t.start()
    return {"ok": True, "job_id": job["id"]}


@router.get("/import/stream")
async def stream_import_logs():
    """SSE stream for import progress."""
    async def event_stream():
        idx = 0
        while True:
            with _import_lock:
                job = _current_import
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


@router.post("/import/cancel")
def cancel_import():
    """Cancel a running import."""
    global _current_import
    with _import_lock:
        if not _current_import or _current_import["status"] != "running":
            raise HTTPException(400, "No running import to cancel.")
        proc = _current_import.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            _current_import["status"] = "cancelled"
            _current_import["logs"].append("Import cancelled by user.")
            return {"ok": True}
    raise HTTPException(400, "Could not cancel.")


@router.get("/import/status")
def import_status():
    """Get current import job status."""
    with _import_lock:
        job = _current_import
    if not job:
        return {"status": "idle"}
    elapsed = time.time() - job["started"] if job["status"] == "running" else job.get("elapsed", 0)
    return {
        "id": job["id"],
        "status": job["status"],
        "elapsed": round(elapsed, 1),
        "exit_code": job.get("exit_code"),
        "log_count": len(job["logs"]),
        "dataset": job.get("dataset"),
        "database": job.get("database"),
        "server": job.get("server"),
    }
