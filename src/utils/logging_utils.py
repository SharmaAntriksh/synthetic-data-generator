from __future__ import annotations

import os
import sys
import time
import atexit
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# PROJECT ROOT + PATH SHORTENING
# ============================================================================
# Kept identical semantics: project root is the top-level folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def short_path(p: Any) -> Any:
    """
    If p is a path under PROJECT_ROOT, return a relative path string.
    Otherwise return the last component (name), matching previous behavior.
    """
    if not p:
        return p
    try:
        pth = Path(p)
    except Exception:
        return p

    try:
        return str(pth.relative_to(PROJECT_ROOT))
    except Exception:
        # Preserve previous fallback behavior: only last component
        return pth.name


def _shorten_path_in_msg(msg: Any) -> Any:
    """
    Backward-compatible behavior:
    - If message ends with a path after ': ', shorten that trailing path.
    - Otherwise return msg unchanged.
    """
    if not isinstance(msg, str):
        return msg

    head, sep, tail = msg.rpartition(": ")
    if sep and (("\\" in tail) or ("/" in tail)):
        shortened = short_path(tail)
        return f"{head}{sep}{shortened}"

    return msg


# ============================================================================
# CONFIG (kept names; logic improved)
# ============================================================================
ENABLE_COLORS = True          # Console colors (effective only if terminal supports)
ENABLE_FILE_LOG = False       # Save logs to file
LOG_FILE = "logs/generator.log"

COLORS = {
    "INFO":  "\033[94m",   # Blue
    "WORK":  "\033[93m",   # Yellow
    "DONE":  "\033[92m",   # Green
    "SKIP":  "\033[90m",   # Grey
    "WARN":  "\033[95m",   # Magenta
    "FAIL":  "\033[91m",   # Red
    "RESET": "\033[0m",
}

PIPELINE_START_TIME = time.time()


# ============================================================================
# INTERNAL STATE
# ============================================================================
_PRINT_LOCK = threading.Lock()

_LOG_DIR_READY = False
_LOG_FD: Optional[int] = None
_LOG_FILE_PATH: Optional[Path] = None


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _color_allowed() -> bool:
    """
    Colors are used if:
      - ENABLE_COLORS is True
      - and stdout is a TTY
      - and NO_COLOR isn't set
    """
    if not ENABLE_COLORS:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return _is_tty()


def configure_logging(
    *,
    enable_colors: Optional[bool] = None,
    enable_file_log: Optional[bool] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Optional runtime configuration without touching import sites.
    """
    global ENABLE_COLORS, ENABLE_FILE_LOG, LOG_FILE, _LOG_FILE_PATH, _LOG_DIR_READY, _LOG_FD

    if enable_colors is not None:
        ENABLE_COLORS = bool(enable_colors)

    if enable_file_log is not None:
        ENABLE_FILE_LOG = bool(enable_file_log)

    if log_file is not None:
        LOG_FILE = str(log_file)
        _LOG_FILE_PATH = None
        _LOG_DIR_READY = False
        if _LOG_FD is not None:
            try:
                os.close(_LOG_FD)
            except Exception:
                pass
            _LOG_FD = None


def fmt_sec(sec: float) -> str:
    """Return a clean human-readable time string."""
    if sec < 1:
        return f"{sec * 1000:.0f}ms"
    if sec < 60:
        return f"{sec:.1f}s"
    return str(timedelta(seconds=int(sec)))


def human_duration(seconds: float) -> str:
    """Backward-compatible alias."""
    return fmt_sec(seconds)


def pipeline_elapsed() -> str:
    """Elapsed time since module import (pipeline start)."""
    return fmt_sec(time.time() - PIPELINE_START_TIME)


def _format_level(level: str) -> str:
    if _color_allowed():
        color = COLORS.get(level, "")
        reset = COLORS["RESET"]
        return f"{color}{level:<5}{reset}"
    return f"{level:<5}"


def _line(level: str, msg: Any) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    return f"{ts} | {_format_level(level)} | {msg}"


def _ensure_log_file_open() -> None:
    """
    Lazily open a process-local FD in append mode.
    Keeps overhead low vs open/close on every log line.
    """
    global _LOG_DIR_READY, _LOG_FD, _LOG_FILE_PATH

    if not ENABLE_FILE_LOG:
        return

    if _LOG_FILE_PATH is None:
        _LOG_FILE_PATH = Path(LOG_FILE)

    if not _LOG_DIR_READY:
        # Avoid repeated mkdir calls
        parent = _LOG_FILE_PATH.parent
        if str(parent) and str(parent) != ".":
            parent.mkdir(parents=True, exist_ok=True)
        _LOG_DIR_READY = True

    if _LOG_FD is None:
        # O_APPEND ensures each write is appended (best-effort atomicity per process)
        _LOG_FD = os.open(str(_LOG_FILE_PATH), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)


@atexit.register
def _close_log_fd() -> None:
    global _LOG_FD
    if _LOG_FD is not None:
        try:
            os.close(_LOG_FD)
        except Exception:
            pass
        _LOG_FD = None


def _write_to_file(line: str) -> None:
    if not ENABLE_FILE_LOG:
        return
    _ensure_log_file_open()
    if _LOG_FD is None:
        return
    try:
        os.write(_LOG_FD, (line + "\n").encode("utf-8", errors="replace"))
    except Exception:
        # File logging should never crash the pipeline
        pass


def _flush(line: str) -> None:
    """
    Prints to stdout safely for multiprocessing (flush) and reduces interleaving
    within the same process via a lock. Multiprocess interleaving is still possible
    (expected), but each call flushes promptly.
    """
    with _PRINT_LOCK:
        print(line, flush=True)
        _write_to_file(line)


# ============================================================================
# BASIC LOG LEVEL FUNCTIONS (same names)
# ============================================================================
def info(msg: Any) -> None:
    msg = _shorten_path_in_msg(msg)
    _flush(_line("INFO", msg))


def warn(msg: Any) -> None:
    _flush(_line("WARN", msg))


def fail(msg: Any) -> None:
    _flush(_line("FAIL", msg))


def skip(msg: Any) -> None:
    _flush(_line("SKIP", msg))


def done(msg: Any) -> None:
    _flush(_line("DONE", msg))


# ============================================================================
# STAGE CONTEXT MANAGER (Auto-timed) – improved exception logging
# ============================================================================
@dataclass
class stage:
    """
    Usage:
        with stage("Generating Dates"):
            ...

    Behavior:
      - Logs INFO at enter
      - Logs DONE on successful exit with timing
      - Logs FAIL on exception with timing, then re-raises
    """
    msg: str
    start: Optional[float] = None

    def __enter__(self) -> "stage":
        self.start = time.time()
        info(self.msg)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = fmt_sec(time.time() - (self.start or time.time()))
        if exc_type is not None:
            # Include exception type + message, but keep it compact
            fail(f"{self.msg} failed in {elapsed}: {exc_type.__name__}: {exc}")
            return False  # re-raise
        done(f"{self.msg} completed in {elapsed}")
        return False


# ============================================================================
# ENHANCED WORK LOGGER (For Multiprocessing Workers) – same signature
# ============================================================================
def work(msg: str = "", *, outfile: Any = None, **_ignore: Any) -> None:
    """
    WORK logger (multiprocessing-safe).

    - Message is authoritative
    - If msg is empty and outfile is provided, log outfile name
    - Keeps previous visible format (including arrow) for compatibility
    """
    if msg:
        final = msg
    elif outfile:
        try:
            final = f"→ {Path(outfile).name}"
        except Exception:
            final = f"→ {outfile}"
    else:
        final = ""

    _flush(_line("WORK", final))
