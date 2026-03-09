# ui/helpers.py
#
# Shared lightweight helpers used across UI sections.
#
# Centralises the _as_int / _as_float coercion helpers that were previously
# copy-pasted into volume.py, dimensions.py, and pricing.py, and the
# project-root resolution that was duplicated in app.py and generate.py.

from __future__ import annotations

import sys
from pathlib import Path


# ------------------------------------------------------------------
# Type coercion (safe for Streamlit widget defaults)
# ------------------------------------------------------------------

def as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def as_float(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


# ------------------------------------------------------------------
# Project root resolution (used by app.py and generate.py)
# ------------------------------------------------------------------

def find_repo_root(*, anchor_file: str = "main.py", fallback_dir: str = "src") -> Path:
    """Walk upward from this file looking for anchor_file first, then
    fallback_dir.  Returns the first directory that matches."""
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for r in candidates[:6]:
        if (r / anchor_file).exists():
            return r
    for r in candidates[:6]:
        if (r / fallback_dir).exists():
            return r
    return here.parents[1]


def ensure_root_on_path(root: Path) -> None:
    """Insert root at the front of sys.path if not already present."""
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
