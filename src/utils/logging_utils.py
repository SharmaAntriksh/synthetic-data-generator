import os
from datetime import datetime
from contextlib import contextmanager

# ============================
# Configuration
# ============================

ENABLE_COLORS = True          # Set False if logging to parsers or CI
ENABLE_FILE_LOG = False       # Set True to log to a file
LOG_FILE = "logs/runtime.log" # File path if ENABLE_FILE_LOG = True


# ============================
# ANSI Colors
# ============================

COLORS = {
    "INFO": "\033[94m",   # Blue
    "SKIP": "\033[93m",   # Yellow
    "WORK": "\033[96m",   # Cyan
    "DONE": "\033[92m",   # Green
    "ERROR": "\033[91m",  # Red
    "RESET": "\033[0m",
}


# ============================
# Core Logging
# ============================

def log(level, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Apply color if enabled
    if ENABLE_COLORS:
        color = COLORS.get(level, "")
        reset = COLORS["RESET"]
        level_str = f"{color}{level:<5}{reset}"
    else:
        level_str = f"{level:<5}"

    line = f"{ts} | {level_str} | {msg}"

    # Print to console
    print(line, flush=True)

    # Optional logging to file
    if ENABLE_FILE_LOG:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {level:<5} | {msg}\n")


# ============================
# Helper Shortcuts
# ============================

def info(msg): log("INFO", msg)
def skip(msg): log("SKIP", msg)
def work(msg): log("WORK", msg)
def done(msg): log("DONE", msg)
def error(msg): log("ERROR", msg)


# ============================
# Stage Context Manager
# ============================

@contextmanager
def stage(label: str):
    """Prints start and end timestamps with duration."""
    info(f"{label}...")
    start = datetime.now()

    yield  # run the wrapped block

    duration = (datetime.now() - start).total_seconds()
    done(f"{label} ({duration:.2f}s)")
