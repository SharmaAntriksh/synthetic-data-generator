from pathlib import Path
from src.utils.logging_utils import info
from versioning.version_store import save_version
from versioning.version_store import _version_file as version_file_path  # exact same path logic as versioning.py


def ensure_dimension_version_exists(name: str, parquet_path: Path, cfg_section):
    """
    If parquet exists but version metadata is missing,
    create the version file once.
    """
    vpath = version_file_path(name)

    if parquet_path.exists() and not vpath.exists():
        info(f"[versioning] Creating version metadata for '{name}'.")
        save_version(name, cfg_section, parquet_path)


def validate_all_dimensions(cfg: dict, parquet_dims: Path, dimension_names):
    """
    Ensures each dimension that has a parquet file
    also has a matching version metadata file.
    """
    for name in dimension_names:

        parquet_path = parquet_dims / f"{name}.parquet"

        # Config sections:
        if name in ["currency", "exchange_rates"]:
            cfg_section = cfg.get("exchange_rates", {})
        else:
            cfg_section = cfg.get(name, {})

        ensure_dimension_version_exists(name, parquet_path, cfg_section)
