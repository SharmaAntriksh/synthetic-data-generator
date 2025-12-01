import os
import time
from datetime import datetime, timedelta

from pathlib import Path

# Compute project root (top-level folder)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def short_path(p):
    if not p:
        return p
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except Exception:
        return p.name  # last component only


def _shorten_path_in_msg(msg: str) -> str:
    """
    If message ends with a path (after ': '), shorten that path preserving internal spaces.
    Fallback: return original msg unchanged.
    """
    if not isinstance(msg, str):
        return msg

    # If message contains a colon-space then the trailing part is likely a path in many logs:
    head, sep, tail = msg.rpartition(': ')
    if sep and (('\\' in tail) or ('/' in tail)):
        shortened = short_path(tail)
        return f"{head}{sep}{shortened}"

    # Otherwise leave unchanged
    return msg


# ============================================================================
# CONFIG
# ============================================================================
ENABLE_COLORS = True          # Colors in console
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

# Track entire pipeline start
PIPELINE_START_TIME = time.time()

# ============================================================================
# HELPERS
# ============================================================================

def fmt_sec(sec):
    """Return a clean human-readable time string."""
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    if sec < 60:
        return f"{sec:.1f}s"
    return str(timedelta(seconds=int(sec)))


def human_duration(seconds):
    """Backward-compatible for old code."""
    return fmt_sec(seconds)


def _line(level, msg):
    """ Standard log line formatter. """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if ENABLE_COLORS:
        color = COLORS.get(level, "")
        reset = COLORS["RESET"]
        level_str = f"{color}{level:<5}{reset}"
    else:
        level_str = f"{level:<5}"

    return f"{ts} | {level_str} | {msg}"


def _flush(line):
    """Prints to stdout safely for multiprocessing."""
    print(line, flush=True)

    if ENABLE_FILE_LOG:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================================
# BASIC LOG LEVEL FUNCTIONS
# ============================================================================
def info(msg):
    msg = _shorten_path_in_msg(msg)
    _flush(_line("INFO", msg))


def warn(msg):
    _flush(_line("WARN", msg))


def fail(msg):
    _flush(_line("FAIL", msg))


def skip(msg):
    _flush(_line("SKIP", msg))


def done(msg):
    _flush(_line("DONE", msg))


# ============================================================================
# STAGE CONTEXT MANAGER (Auto-timed)
# ============================================================================
class stage:
    """Clean stage logging without arrows or tick marks."""

    def __init__(self, msg):
        self.msg = msg
        self.start = None

    def __enter__(self):
        self.start = time.time()
        info(self.msg)   # No arrow
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = fmt_sec(time.time() - self.start)
        done(f"{self.msg} completed in {elapsed}")


# ============================================================================
# ENHANCED WORK LOGGER (For Multiprocessing Workers)
# ============================================================================
def work(msg="", *, chunk=None, total=None, outfile=None, **_ignore):
    """
    Simplified WORK logger.
    Only prints:
        - timestamp
        - WORK level
        - Chunk X/Y
        - Output path
    """

    parts = []

    # Always show chunk if provided
    if chunk is not None:
        if total is not None:
            parts.append(f"Chunk {chunk}/{total}")
        else:
            parts.append(f"Chunk {chunk}")

    # Auto shorten output paths
    if outfile:
        parts.append(f"→ {Path(outfile).name}")


    # Build final message
    final = " | ".join(parts) if parts else msg

    _flush(_line("WORK", final))
