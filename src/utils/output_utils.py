from __future__ import annotations

import csv
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from src.utils.logging_utils import stage, done, info

__all__ = [
    "write_parquet_with_date32",
    "format_number_short",
    "create_final_output_folder",
]

# ---------------------------------------------------------------------------
# Dimension filenames — single source of truth for exclusion logic.
# If a dimension's output filename changes, update here only.
# ---------------------------------------------------------------------------
_DIM_FILE_PLANS = "plans.parquet"
_DIM_FILE_SUBSCRIPTIONS_BRIDGE = "customer_subscriptions.parquet"
_DIM_FILE_WISHLISTS = "customer_wishlists.parquet"
_DIM_FILE_RETURN_REASON = "return_reason.parquet"


# ============================================================
# Parquet helpers (Power BI / Power Query friendliness)
# ============================================================

def _all_null_series(s: pd.Series) -> bool:
    try:
        return bool(s.isna().all())
    except (TypeError, ValueError):
        return False


def _object_series_looks_like_date(s: pd.Series) -> bool:
    """
    True if an object-dtype Series appears to hold Python date/datetime-like
    values, or is entirely null (common for optional date columns in small
    datasets).

    Samples up to 100 non-null rows rather than 25 to reduce false negatives
    on sparse columns.  Also accepts ISO-format date strings so that columns
    stored as plain strings (e.g. "2024-01-15") are correctly detected.
    """
    import datetime as _dt

    import numpy as _np

    try:
        if _all_null_series(s):
            return True

        nonnull = s.dropna()
        if nonnull.empty:
            return True

        # Sample more rows to reduce false negatives on sparse columns
        sample = nonnull.head(100).tolist()
        for v in sample:
            if isinstance(v, pd.Timestamp):
                return True
            if isinstance(v, (_dt.date, _dt.datetime, _np.datetime64)):
                return True
            # Accept ISO-format date strings ("2024-01-15" / "2024-01-15T00:00:00")
            if isinstance(v, str):
                try:
                    pd.Timestamp(v)
                    return True
                except (ValueError, TypeError):
                    pass
        return False
    except (TypeError, ValueError):
        return False


def _datetime_cols(df: pd.DataFrame) -> list[str]:
    """Return column names whose dtype is pandas datetime64 (with or without tz)."""
    cols: list[str] = []
    for c in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                cols.append(str(c))
        except (TypeError, ValueError):
            continue
    return cols


def _guess_date_cols(df: pd.DataFrame) -> list[str]:
    """
    Heuristic: return column names that should be written as Arrow date32.

    Rules (conservative — won't touch time strings like OpenTime/CloseTime etc.):
      1. datetime64 columns whose names contain a date-like token.
      2. object-dtype columns whose names contain "date" AND whose values look
         date-like (or are all-null), to prevent Arrow NullType on rewrite.

    Returns a de-duplicated, order-preserving list.
    """
    dt_cols: set[str] = set(_datetime_cols(df))
    out: list[str] = []

    dateish_tokens = ("date", "birth", "created", "updated", "effective", "expiry", "valid")

    # Rule 1 — proper datetime64 columns with date-like names
    for c in df.columns:
        if str(c) in dt_cols and any(tok in str(c).lower() for tok in dateish_tokens):
            out.append(str(c))

    # Rule 2 — object columns whose name contains "date"
    for c in df.columns:
        cs = str(c)
        if cs in dt_cols:
            continue
        if "date" not in cs.lower():
            continue
        try:
            if pd.api.types.is_object_dtype(df[c]) and _object_series_looks_like_date(df[c]):
                out.append(cs)
        except (TypeError, ValueError):
            continue

    # De-duplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def write_parquet_with_date32(
    df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    date_cols: Optional[Sequence[str]] = None,
    cast_all_datetime: bool = False,
    compression: str = "snappy",
    compression_level: Optional[int] = None,
    force_date32: bool = True,
) -> None:
    """
    Write a Parquet file with selected date-like columns stored as Arrow date32
    (Power BI / Power Query friendly — avoids the NullType crash).

    Behaviour:
      - If ``date_cols`` is given, those columns are cast (dtype not required).
      - If ``cast_all_datetime`` is True, all datetime64 columns are cast.
      - Otherwise, ``_guess_date_cols`` is used as the heuristic.

    Only the target columns are copied; the rest of the DataFrame is passed
    through by reference to keep peak memory low.
    """
    out_path = Path(out_path)

    dt_cols: set[str] = set(_datetime_cols(df))

    if date_cols is not None:
        target = [str(c) for c in date_cols if str(c) in df.columns]
    elif cast_all_datetime:
        target = list(dt_cols)
    else:
        target = _guess_date_cols(df)

    if not target:
        df.to_parquet(out_path, index=False)
        return

    # Surgical copy: only materialise new Series for target columns so we
    # avoid duplicating the full DataFrame in memory.
    overrides = {
        c: pd.to_datetime(df[c], errors="coerce").dt.normalize()
        for c in target
        if c in df.columns
    }
    df2 = df.assign(**overrides)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        # PyArrow unavailable — fall back to pandas writer
        if force_date32:
            date_overrides = {
                c: pd.to_datetime(df2[c], errors="coerce").dt.date
                for c in target
                if c in df2.columns
            }
            df2 = df2.assign(**date_overrides)
        df2.to_parquet(out_path, index=False)
        return

    table = pa.Table.from_pandas(df2, preserve_index=False)
    target_set = set(target)
    fields = [
        pa.field(f.name, pa.date32()) if f.name in target_set else f
        for f in table.schema
    ]
    table = table.cast(pa.schema(fields), safe=False)

    kwargs: dict = {"compression": str(compression)}
    if compression_level is not None:
        kwargs["compression_level"] = int(compression_level)

    pq.write_table(table, str(out_path), **kwargs)


# ============================================================
# Utility helpers
# ============================================================

def format_number_short(n: int) -> str:
    """Human-readable suffix for large numbers (rounded, not truncated)."""
    if n >= 1_000_000_000:
        return f"{round(n / 1_000_000_000)}B"
    if n >= 1_000_000:
        return f"{round(n / 1_000_000)}M"
    if n >= 1_000:
        return f"{round(n / 1_000)}K"
    return str(n)


def _copy_config_files_into_run_folder(
    final_folder: Path,
    config_yaml_path: Optional[Union[str, Path]] = None,
    model_yaml_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Copy config/model YAMLs into ``<final_folder>/config/`` for traceability.
    Logs a warning when an expected file is not found rather than silently skipping.
    """
    config_dir = final_folder / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Optional[Union[str, Path]], dest_name: str) -> None:
        if not src:
            return
        p = Path(str(src))
        if not p.exists():
            info(f"WARNING: config file not found, skipping copy: {p}")
            return
        shutil.copy2(p, config_dir / dest_name)

    _copy(config_yaml_path, "config.yaml")
    _copy(model_yaml_path, "models.yaml")


def _ensure_clean_dir(p: Path) -> None:
    """
    Remove and recreate a directory.

    Raises ``FileExistsError`` (rather than silently deleting) if the path
    exists AND already contains files — a common sign of a wrong path being
    passed in.  An *empty* existing directory is accepted and recreated
    without complaint.
    """
    if p.exists():
        existing_files = list(p.rglob("*"))
        non_empty = [f for f in existing_files if f.is_file()]
        if non_empty:
            raise FileExistsError(
                f"_ensure_clean_dir: refusing to delete non-empty directory '{p}' "
                f"({len(non_empty)} file(s) present). "
                "Pass an explicit empty or non-existent path to avoid data loss."
            )
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def _excluded_dim_files(cfg: dict) -> set[str]:
    """
    Return the set of dimension filenames that should be skipped during
    packaging, based on config flags.

    - ``enabled: false``        → exclude both the dim table and its bridge table
    - ``generate_bridge: false`` → exclude only the bridge table
    - returns effectively disabled → exclude ReturnReason
    """
    excluded: set[str] = set()

    sub_cfg = getattr(cfg, "subscriptions", None)
    if sub_cfg is not None:
        if not bool(getattr(sub_cfg, "enabled", True)):
            excluded.add(_DIM_FILE_PLANS)
            excluded.add(_DIM_FILE_SUBSCRIPTIONS_BRIDGE)
        elif not bool(getattr(sub_cfg, "generate_bridge", True)):
            excluded.add(_DIM_FILE_SUBSCRIPTIONS_BRIDGE)

    wl_cfg = getattr(cfg, "wishlists", None)
    if wl_cfg is not None:
        if not bool(getattr(wl_cfg, "enabled", True)):
            excluded.add(_DIM_FILE_WISHLISTS)

    returns_cfg = getattr(cfg, "returns", None)
    returns_on = bool(getattr(returns_cfg, "enabled", False)) if returns_cfg is not None else False
    if returns_on:
        sales_cfg = getattr(cfg, "sales", None) or {}
        skip_order = bool(getattr(sales_cfg, "skip_order_cols", False))
        sales_output = str(getattr(sales_cfg, "sales_output", "sales")).strip().lower()
        if skip_order and sales_output == "sales":
            returns_on = False
    if not returns_on:
        excluded.add(_DIM_FILE_RETURN_REASON)

    return excluded


# ============================================================
# Final output folder creation (dimensions only)
# ============================================================

def create_final_output_folder(
    final_folder_root: Path,
    parquet_dims: Path,
    fact_folder: Path,                        # retained for API compatibility
    sales_cfg: dict,
    file_format: str,
    sales_rows_expected: int = 0,             # deprecated — no longer used
    cfg: dict = {},
    config_yaml_path: Optional[Union[str, Path]] = None,
    model_yaml_path: Optional[Union[str, Path]] = None,
    package_facts: bool = False,              # deprecated — facts packaged externally
) -> Path:
    """
    Create the run output folder and package DIMENSIONS into it.

    Post-modularisation responsibilities:
      - Name and create the run folder hierarchy.
      - Copy ``config.yaml`` / ``models.yaml`` into ``<run>/config/`` for
        traceability.
      - Convert/copy DIMENSIONS from ``parquet_dims`` into the chosen format.

    Facts and SQL packaging are handled by ``src.engine.packaging.package_output()``.

    Deprecated parameters (accepted for backward compatibility only):
      - ``sales_rows_expected`` — no longer read; pass 0 or omit.
      - ``package_facts``       — no longer honoured; facts are always packaged
                                  externally.  Passing ``True`` emits a
                                  ``DeprecationWarning``.
      - ``fact_folder``         — accepted but ignored.
    """
    if sales_rows_expected != 0:
        warnings.warn(
            "sales_rows_expected is deprecated and no longer used by "
            "create_final_output_folder. Pass 0 or omit the argument.",
            DeprecationWarning,
            stacklevel=2,
        )
    if package_facts:
        warnings.warn(
            "package_facts=True is deprecated. Facts are now packaged exclusively "
            "by src.engine.packaging.package_output(). This flag has no effect.",
            DeprecationWarning,
            stacklevel=2,
        )

    with stage("Creating Final Output Folder"):
        ff = str(file_format).strip().lower()

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %I_%M_%S %p")  # windows-safe

        _cust = getattr(cfg, "customers", None) or {}
        customer_total = int(getattr(_cust, "total_customers", 0) or 0)
        sales_total = int(getattr(sales_cfg, "total_rows", 0) or 0)

        dataset_name = (
            f"{timestamp} Customers {format_number_short(customer_total)} "
            f"Sales {format_number_short(sales_total)} {ff.upper()}"
        )
        final_folder = Path(final_folder_root) / dataset_name

        _ensure_clean_dir(final_folder)

        dims_out = final_folder / "dimensions"
        dims_out.mkdir(parents=True, exist_ok=True)
        # NOTE: "facts/" is intentionally NOT created here; facts are packaged
        # externally and will create their own subdirectory as needed.

        _copy_config_files_into_run_folder(
            final_folder,
            config_yaml_path=config_yaml_path,
            model_yaml_path=model_yaml_path,
        )

        # --------------------------------------------------------
        # DIMENSIONS
        # --------------------------------------------------------
        packaging_cfg = cfg.packaging
        dim_parquet_compression: str = packaging_cfg.dim_parquet_compression
        dim_parquet_compression_level: Optional[int] = packaging_cfg.dim_parquet_compression_level
        dim_force_date32: bool = bool(packaging_cfg.dim_force_date32)

        parquet_dims = Path(parquet_dims)
        excluded_dims = _excluded_dim_files(cfg)

        if ff == "parquet":
            for f in parquet_dims.glob("*.parquet"):
                if f.name in excluded_dims:
                    continue
                df = pd.read_parquet(f)
                write_parquet_with_date32(
                    df,
                    dims_out / f.name,
                    cast_all_datetime=False,
                    compression=dim_parquet_compression,
                    compression_level=dim_parquet_compression_level,
                    force_date32=dim_force_date32,
                )

        elif ff == "csv":
            for f in parquet_dims.glob("*.parquet"):
                if f.name in excluded_dims:
                    continue
                # Nullable backend keeps Int columns as Int (avoids "10001.0")
                try:
                    df = pd.read_parquet(f, dtype_backend="numpy_nullable")
                except TypeError:
                    df = pd.read_parquet(f)

                # bool/boolean → 0/1 (Int8) for clean CSV output
                bool_cols = list(df.select_dtypes(include=["bool", "boolean"]).columns)
                if bool_cols:
                    df = df.assign(**{c: df[c].astype("Int8") for c in bool_cols})

                # integer-like floats → Int64 to avoid "10001.0" in CSV
                float_cols = list(df.select_dtypes(include=["float"]).columns)
                int_upgrades: dict = {}
                for c in float_cols:
                    s = pd.to_numeric(df[c], errors="coerce")
                    non_null = s.dropna()
                    if not non_null.empty and ((non_null % 1) == 0).all():
                        int_upgrades[c] = s.astype("Int64")
                if int_upgrades:
                    df = df.assign(**int_upgrades)

                df.to_csv(
                    dims_out / f"{f.stem}.csv",
                    index=False,
                    encoding="utf-8",
                    quoting=csv.QUOTE_MINIMAL,
                    na_rep="",
                )

        elif ff == "deltaparquet":
            try:
                from deltalake import write_deltalake
            except ImportError as e:
                raise RuntimeError(
                    "deltaparquet mode requested but 'deltalake' is not installed. "
                    "Run `pip install deltalake` or switch to parquet/csv."
                ) from e

            import pyarrow as pa

            for f in parquet_dims.glob("*.parquet"):
                if f.name in excluded_dims:
                    continue
                dim_name = f.stem
                delta_out = dims_out / dim_name
                delta_out.mkdir(parents=True, exist_ok=True)

                df = pd.read_parquet(f)

                # Fix: _guess_date_cols covers BOTH datetime64 AND object-dtype
                # date columns; the previous guard `c in dt_cols` incorrectly
                # skipped object-dtype cols, leaving NullType columns in the table.
                if dim_force_date32:
                    date_cols = _guess_date_cols(df)
                    if date_cols:
                        overrides = {
                            c: pd.to_datetime(df[c], errors="coerce").dt.normalize().dt.date
                            for c in date_cols
                            if c in df.columns
                        }
                        df = df.assign(**overrides)

                table = pa.Table.from_pandas(df, preserve_index=False)
                write_deltalake(str(delta_out), table, mode="overwrite")

        else:
            raise ValueError(
                f"Unknown file_format {file_format!r}. "
                "Expected one of: 'parquet', 'csv', 'deltaparquet'."
            )

        done(f"Created final folder: {final_folder.name}")
        return final_folder
