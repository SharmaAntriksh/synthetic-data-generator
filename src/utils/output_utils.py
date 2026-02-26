from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

from src.utils.logging_utils import stage, done, info


# ============================================================
# Parquet helpers (Power BI / Power Query friendliness)
# ============================================================
def _all_null_series(s) -> bool:
    try:
        return bool(getattr(s, "isna")().all())
    except Exception:
        return False


def _object_series_looks_like_date(s) -> bool:
    """
    True if object series appears to contain python date/datetime-like values,
    or is entirely null (common for optional date columns in small datasets).
    """
    import datetime as _dt
    import numpy as _np
    try:
        import pandas as _pd
    except Exception:
        _pd = None  # type: ignore

    try:
        if _all_null_series(s):
            return True

        nonnull = s.dropna()
        if nonnull.empty:
            return True

        sample = nonnull.head(25)
        for v in sample.tolist():
            if _pd is not None and isinstance(v, _pd.Timestamp):
                return True
            if isinstance(v, (_dt.date, _dt.datetime, _np.datetime64)):
                return True
        return False
    except Exception:
        return False
    

def _datetime_cols(df) -> list[str]:
    """Return columns that are pandas datetime64/tz-aware."""
    try:
        import pandas as pd
    except Exception:
        return []
    cols: list[str] = []
    for c in getattr(df, "columns", []):
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                cols.append(str(c))
        except Exception:
            continue
    return cols


def _guess_date_cols(df) -> list[str]:
    """
    Heuristic for date-like columns.

    Keeps prior conservatism (won't touch OpenMinute/CloseMinute etc):
      - include datetime64 columns whose names look date-like
      - additionally include *Date* columns that are object dtype but look date-like
        (or are all-null), to avoid Arrow NullType on rewrite.
    """
    try:
        import pandas as pd
    except Exception:
        return []

    dt_cols = set(_datetime_cols(df))
    out: list[str] = []

    # datetime columns: name-based filter
    dateish_tokens = ("date", "birth", "created", "updated", "effective", "expiry", "valid")
    for c in getattr(df, "columns", []):
        cl = str(c).lower()
        if c in dt_cols:
            if any(tok in cl for tok in dateish_tokens):
                out.append(str(c))

    # object columns: ONLY if name contains "date" (prevents OpenMinute/CloseMinute mistakes)
    for c in getattr(df, "columns", []):
        c = str(c)
        if c in dt_cols:
            continue
        if "date" not in c.lower():
            continue

        try:
            if pd.api.types.is_object_dtype(df[c]) and _object_series_looks_like_date(df[c]):
                out.append(c)
        except Exception:
            continue

    # de-dupe preserving order
    seen = set()
    deduped = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def write_parquet_with_date32(
    df,
    out_path: Union[str, Path],
    *,
    date_cols: Optional[Sequence[str]] = None,
    cast_all_datetime: bool = False,
    compression: str = "snappy",
    compression_level: Optional[int] = None,
    force_date32: bool = True,
) -> None:
    """
    Write Parquet with selected date-like columns stored as Arrow date32 (Power BI friendly).

    Fix: honor date_cols / guessed date cols even if pandas dtype is object/all-null,
    so we don't emit Arrow NullType columns (Power BI crash: dataType cannot be null).
    """
    import pandas as pd

    out_path = Path(out_path)

    dt_cols = set(_datetime_cols(df))

    if date_cols is not None:
        # IMPORTANT: don't require datetime dtype; we'll coerce below.
        target = [str(c) for c in date_cols if str(c) in df.columns]
    elif cast_all_datetime:
        target = [str(c) for c in dt_cols]
    else:
        target = _guess_date_cols(df)

    if not target:
        df.to_parquet(out_path, index=False)
        return

    df2 = df.copy()
    for c in target:
        if c not in df2.columns:
            continue
        # Coerce even object/all-null columns; normalize to midnight
        df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.normalize()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        if force_date32:
            for c in target:
                if c in df2.columns:
                    df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.date
        df2.to_parquet(out_path, index=False)
        return

    table = pa.Table.from_pandas(df2, preserve_index=False)

    target_set = set(target)
    fields = []
    for f in table.schema:
        fields.append(pa.field(f.name, pa.date32()) if f.name in target_set else f)

    table = table.cast(pa.schema(fields), safe=False)

    kwargs = {"compression": str(compression)}
    if compression_level is not None:
        kwargs["compression_level"] = int(compression_level)

    pq.write_table(table, str(out_path), **kwargs)


# ============================================================
# Final output folder creation (dimensions only)
# ============================================================

def format_number_short(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _copy_config_files_into_run_folder(
    final_folder: Path,
    config_yaml_path: Optional[Union[str, Path]] = None,
    model_yaml_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Copy config/model YAMLs into <final_folder>/config/ for traceability.
    """
    config_dir = final_folder / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Optional[Union[str, Path]], dest_name: str) -> None:
        if not src:
            return
        p = Path(str(src))
        if not p.exists():
            return
        shutil.copy2(p, config_dir / dest_name)

    _copy(config_yaml_path, "config.yaml")
    _copy(model_yaml_path, "models.yaml")

def _ensure_clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def create_final_output_folder(
    final_folder_root: Path,
    parquet_dims: Path,
    fact_folder: Path,
    sales_cfg: dict,
    file_format: str,
    sales_rows_expected: int,  # kept for signature compatibility
    cfg: dict,
    config_yaml_path: Optional[Union[str, Path]] = None,
    model_yaml_path: Optional[Union[str, Path]] = None,
    package_facts: bool = True,
) -> Path:
    """
    Create the final run folder and package DIMENSIONS into it.

    Post modularization:
      - Facts packaging is handled by src.engine.packaging.*
      - SQL script packaging is handled by src.engine.packaging.sql_scripts

    This function focuses on:
      - naming + creating the run folder structure
      - copying config/model YAML files into <final>/config/
      - converting/copying DIMENSIONS based on file_format

    Note: 'fact_folder' and 'package_facts' are retained for signature compatibility.
    """
    with stage("Creating Final Output Folder"):
        ff = str(file_format).strip().lower()
        # info(f"[DEBUG] create_final_output_folder file_format={file_format!r} ff={ff!r}")

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %I_%M_%S %p")  # windows-safe

        customer_total = int(cfg.get("customers", {}).get("total_customers", 0) or 0)
        sales_total = int(sales_cfg.get("total_rows") or sales_rows_expected or 0)

        dataset_name = (
            f"{timestamp} Customers {format_number_short(customer_total)} "
            f"Sales {format_number_short(sales_total)} {ff.upper()}"
        )
        final_folder = Path(final_folder_root) / dataset_name

        _ensure_clean_dir(final_folder)

        dims_out = final_folder / "dimensions"
        facts_out = final_folder / "facts"

        dims_out.mkdir(parents=True, exist_ok=True)
        facts_out.mkdir(parents=True, exist_ok=True)

        _copy_config_files_into_run_folder(
            final_folder,
            config_yaml_path=config_yaml_path,
            model_yaml_path=model_yaml_path,
        )

        # --------------------------------------------------------
        # DIMENSIONS
        # --------------------------------------------------------
        packaging_cfg = cfg.get("packaging", {}) if isinstance(cfg, dict) else {}
        dim_parquet_compression = packaging_cfg.get("dim_parquet_compression", "snappy")
        dim_parquet_compression_level = packaging_cfg.get("dim_parquet_compression_level", None)
        dim_force_date32 = bool(packaging_cfg.get("dim_force_date32", True))

        parquet_dims = Path(parquet_dims)

        if ff == "parquet":
            import pandas as pd

            for f in parquet_dims.glob("*.parquet"):
                df = pd.read_parquet(f)
                out_f = dims_out / f.name
                write_parquet_with_date32(
                    df,
                    out_f,
                    cast_all_datetime=False,
                    compression=dim_parquet_compression,
                    compression_level=dim_parquet_compression_level,
                    force_date32=dim_force_date32,
                )

        elif ff == "csv":
            import pandas as pd

            for f in parquet_dims.glob("*.parquet"):
                # Prefer nullable backend (also helps with your Int columns staying Int)
                try:
                    df = pd.read_parquet(f, dtype_backend="numpy_nullable")
                except TypeError:
                    df = pd.read_parquet(f)

                # 1) bool/boolean -> 0/1 (Int8)
                bool_cols = list(df.select_dtypes(include=["bool", "boolean"]).columns)
                for c in bool_cols:
                    df[c] = df[c].astype("Int8")  # writes 0/1 in CSV

                # 2) (optional but recommended) integer-like floats -> Int64 to avoid "10001.0"
                float_cols = list(df.select_dtypes(include=["float"]).columns)
                for c in float_cols:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if not s.dropna().empty and ((s.dropna() % 1) == 0).all():
                        df[c] = s.astype("Int64")

                out_csv = dims_out / f"{f.stem}.csv"
                df.to_csv(
                    out_csv,
                    index=False,
                    encoding="utf-8",
                    quoting=csv.QUOTE_MINIMAL,
                    na_rep="",
                )

        elif ff == "deltaparquet":
            try:
                from deltalake import write_deltalake
            except Exception as e:
                raise RuntimeError(
                    "deltaparquet mode requested, but deltalake is not available. "
                    "Install deltalake or switch to parquet/csv."
                ) from e

            import pandas as pd
            import pyarrow as pa

            for f in parquet_dims.glob("*.parquet"):
                dim_name = f.stem
                delta_out = dims_out / dim_name
                delta_out.mkdir(parents=True, exist_ok=True)

                df = pd.read_parquet(f)

                # For Power BI, keep date-like datetime columns as date32-ish
                if dim_force_date32:
                    dt_cols = _datetime_cols(df)
                    date_cols = _guess_date_cols(df)
                    for c in date_cols:
                        if c in dt_cols and c in df.columns:
                            df[c] = pd.to_datetime(df[c]).dt.normalize().dt.date

                table = pa.Table.from_pandas(df, preserve_index=False)
                write_deltalake(str(delta_out), table, mode="overwrite")

        else:
            raise ValueError(f"Unknown file_format: {file_format}")

        # Facts are packaged by engine/packaging/* now.
        if package_facts:
            info(
                "NOTE: create_final_output_folder no longer packages facts. "
                "Facts + SQL are packaged by src.engine.packaging.package_output()."
            )

        done(f"Created final folder: {final_folder.name}")
        return final_folder
