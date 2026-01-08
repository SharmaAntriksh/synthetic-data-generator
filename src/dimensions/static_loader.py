import pandas as pd
from pathlib import Path

from src.utils import info, skip
from src.versioning import should_regenerate, save_version


def load_static_dimension(
    name: str,
    src_path: Path,
    output_path: Path,
):
    """
    Load a static reference dimension by copying it as-is.
    Returns: (DataFrame, regenerated: bool)
    """

    version_key = {
        "source": str(src_path),
    }

    if not should_regenerate(name, version_key, output_path):
        skip(f"{name} up-to-date; skipping")
        return pd.read_parquet(output_path), False

    info(f"Loading {name}")

    df = pd.read_parquet(src_path)
    df.to_parquet(output_path, index=False)

    save_version(name, version_key, output_path)
    return df, True
