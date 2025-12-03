"""
Public API for versioning utilities.

This package exposes:
- should_regenerate: Check if a dimension must be regenerated.
- save_version: Write version metadata after generation.
- load_version: Load stored metadata.
- validate_all_dimensions: Ensure version files exist for dimensions.
"""

# Core version metadata store
from .version_store import (
    save_version,
    load_version,
    should_regenerate,
)

# Validation helpers
from .version_checker import (
    ensure_dimension_version_exists,
    validate_all_dimensions,
)

__all__ = [
    # store
    "save_version",
    "load_version",
    "should_regenerate",
    # validation
    "ensure_dimension_version_exists",
    "validate_all_dimensions",
]
