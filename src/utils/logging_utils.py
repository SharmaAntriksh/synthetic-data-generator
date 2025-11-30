import os
import time
from datetime import datetime, timedelta

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

    # Show chunk progress if available
    if chunk is not None and total is not None:
        parts.append(f"Chunk {chunk}/{total}")

    # Show output file path if available
    if outfile:
        parts.append(f"→ {outfile}")

    # Build final message
    final = " | ".join(parts) if parts else msg

    _flush(_line("WORK", final))
