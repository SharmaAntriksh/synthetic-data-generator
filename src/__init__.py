from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synthetic-data-generator")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
