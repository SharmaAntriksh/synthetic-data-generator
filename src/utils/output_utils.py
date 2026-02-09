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
    Heuristic for date-like columns. Used only when date_cols is not provided.

    Conservative on purpose: only matches columns that both:
      - look date-like by name, AND
      - are datetime dtype.
    """
    dt_cols = set(_datetime_cols(df))
    if not dt_cols:
        return []

    dateish_tokens = (
        "date",
        "day",
        "month",
        "year",
        "start",
        "end",
        "open",
        "close",
        "birth",
        "created",
        "updated",
        "effective",
        "expiry",
        "valid",
    )

    out: list[str] = []
    for c in dt_cols:
        cl = c.lower()
        if any(tok in cl for tok in dateish_tokens):
            out.append(c)
    return out


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
    Write Parquet with selected datetime columns stored as Arrow date32 so Power Query imports them as Date.

    Selection:
      - If date_cols is provided: cast only those columns (if they are datetime dtype)
      - Else if cast_all_datetime=True: cast all datetime64 columns
      - Else: heuristic based on column names (see _guess_date_cols)

    Notes:
      - Casting timestamp -> date truncates time-of-day. We normalize selected cols to midnight before casting.
      - If pyarrow isn't available:
          - if force_date32=True: fallback converts selected cols to python date objects (object dtype) just for writing
          - else: plain df.to_parquet
    """
    import pandas as pd

    out_path = Path(out_path)

    dt_cols = _datetime_cols(df)
    if not dt_cols:
        df.to_parquet(out_path, index=False)
        return

    if date_cols is not None:
        target = [c for c in date_cols if c in df.columns and c in dt_cols]
    elif cast_all_datetime:
        target = list(dt_cols)
    else:
        target = _guess_date_cols(df)

    if not target:
        df.to_parquet(out_path, index=False)
        return

    df2 = df.copy()
    for c in target:
        # normalize to midnight to avoid "unexpected" day shifts after date cast
        df2[c] = pd.to_datetime(df2[c]).dt.normalize()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        if force_date32:
            # Fallback: write python date objects so parquet writer uses date logical type.
            for c in target:
                df2[c] = pd.to_datetime(df2[c]).dt.date
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

    # Backward-compat alias (some older scripts used model.yaml)
    if model_yaml_path:
        _copy(model_yaml_path, "model.yaml")


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
        sql_root = final_folder / "sql"

        dims_out.mkdir(parents=True, exist_ok=True)
        facts_out.mkdir(parents=True, exist_ok=True)
        sql_root.mkdir(parents=True, exist_ok=True)

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
                df = pd.read_parquet(f)
                out_csv = dims_out / f"{f.stem}.csv"
                df.to_csv(out_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

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

        done(f"Created final folder: {final_folder}")
        return final_folder
