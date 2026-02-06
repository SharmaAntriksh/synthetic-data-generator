from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from src.utils.logging_utils import stage, done, info

# ============================================================
# Constants / shared
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IGNORE_PATTERNS = shutil.ignore_patterns("_tmp_parts*", "tmp*", "*_tmp*")


# ============================================================
# Parquet helpers (Power Query Date typing)
# ============================================================

def _datetime_cols(df) -> list[str]:
    import pandas as pd
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _guess_date_cols(df) -> list[str]:
    """
    Heuristic: treat columns as "date columns" if:
      - dtype is datetime64
      - and name matches common date-like patterns
    This avoids accidentally truncating true timestamps.
    """
    import pandas as pd

    dt_cols = _datetime_cols(df)
    if not dt_cols:
        return []

    picks: list[str] = []
    for c in dt_cols:
        cl = c.lower()
        # common patterns in your project: Date, StartDate, EndDate, OpeningDate, ClosingDate, DOB, etc.
        if cl == "date" or cl.endswith("date") or "date" in cl or cl in {"dob", "birthdate"}:
            picks.append(c)
    return picks


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
      - Else: cast "date-like" datetime columns based on name heuristic

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

    # Nothing selected -> default parquet write
    if not target:
        df.to_parquet(out_path, index=False)
        return

    # Normalize selected cols (drop any time component deterministically)
    df2 = df.copy()
    for c in target:
        df2[c] = pd.to_datetime(df2[c]).dt.normalize()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        if force_date32:
            # Fallback: python date objects for selected cols
            for c in target:
                df2[c] = pd.to_datetime(df2[c]).dt.date
            df2.to_parquet(out_path, index=False)
        else:
            df2.to_parquet(out_path, index=False)
        return

    table = pa.Table.from_pandas(df2, preserve_index=False)

    target_set = set(target)
    fields = []
    for f in table.schema:
        if f.name in target_set:
            fields.append(pa.field(f.name, pa.date32()))
        else:
            fields.append(f)

    table = table.cast(pa.schema(fields), safe=False)

    kwargs = {"compression": str(compression)}
    if compression_level is not None:
        kwargs["compression_level"] = int(compression_level)

    pq.write_table(table, str(out_path), **kwargs)


# ============================================================
# SQL helper file copies
# ============================================================

def _copy_if_exists(src: Path, dst: Path, log_msg: str) -> None:
    if not src.exists():
        info(f"{src.name} not found; skipping")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    info(log_msg)


def copy_sql_constraints(sql_root: Path) -> None:
    constraints_file = _PROJECT_ROOT / "scripts" / "sql" / "bootstrap" / "create_constraints.sql"
    _copy_if_exists(
        constraints_file,
        sql_root / "schema" / "03_create_constraints.sql",
        "Included create_constraints.sql in final output",
    )


def copy_sql_views(sql_root: Path) -> None:
    views_file = _PROJECT_ROOT / "scripts" / "sql" / "views" / "create_views.sql"
    _copy_if_exists(
        views_file,
        sql_root / "schema" / "04_create_views.sql",
        "Included create_views.sql in final output",
    )


def copy_sql_indexes(sql_root: Path) -> None:
    cci_file = _PROJECT_ROOT / "scripts" / "sql" / "columnstore" / "create_drop_cci.sql"
    _copy_if_exists(
        cci_file,
        sql_root / "indexes" / "create_drop_cci.sql",
        "Included create_drop_cci.sql in SQL indexes",
    )


# ============================================================
# Misc helpers
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
    Copies the exact YAML files used for the run into:
      <final_folder>/config/config.yaml
      <final_folder>/config/model.yaml

    Non-fatal: missing paths are skipped.
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
    _copy(model_yaml_path, "model.yaml")


def _ensure_clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Final output packager
# ============================================================

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
) -> Path:
    """
    Packs cleaned dimension + fact data according to config rules.
    Also copies config/model YAML files into <final_folder>/config/ for traceability.

    Date logic:
      - When file_format == "parquet", dimensions are rewritten so date-like datetime
        columns are stored as Arrow date32 (Power Query imports as Date).
      - Facts are NOT rewritten (too large); they are copied as produced.
    """
    stage("Creating Final Output Folder")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %I_%M_%S %p")  # windows-safe (kept)

    customer_total = cfg["customers"]["total_customers"]
    sales_total = sales_cfg["total_rows"]

    cust_short = format_number_short(int(customer_total))
    sales_short = format_number_short(int(sales_total))

    ff = file_format.lower()
    fmt_label = {
        "deltaparquet": "DeltaParquet",
        "parquet": "Parquet",
        "csv": "CSV",
    }.get(ff, file_format)

    dataset_name = f"{timestamp} Customers {cust_short} Sales {sales_short} {fmt_label}"

    final_folder = final_folder_root / dataset_name
    dims_out = final_folder / "dimensions"
    facts_out = final_folder / "facts"

    _ensure_clean_dir(final_folder)
    dims_out.mkdir(parents=True, exist_ok=True)
    facts_out.mkdir(parents=True, exist_ok=True)

    _copy_config_files_into_run_folder(
        final_folder=final_folder,
        config_yaml_path=config_yaml_path,
        model_yaml_path=model_yaml_path,
    )

    # Optional packaging controls (safe defaults)
    packaging_cfg = cfg.get("packaging", {}) if isinstance(cfg, dict) else {}
    dim_parquet_compression = packaging_cfg.get("dim_parquet_compression", "snappy")
    dim_parquet_compression_level = packaging_cfg.get("dim_parquet_compression_level", None)
    dim_force_date32 = bool(packaging_cfg.get("dim_force_date32", True))

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------
    if ff == "parquet":
        # Rewrite dims to enforce Arrow date32 for date-like columns (Power Query friendly)
        import pandas as pd

        for f in parquet_dims.glob("*.parquet"):
            df = pd.read_parquet(f)
            out_f = dims_out / f.name
            write_parquet_with_date32(
                df,
                out_f,
                # conservative default: only date-like columns
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
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(
                out_csv,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_MINIMAL,
            )

    elif ff == "deltaparquet":
        # Keep behavior, but we can still enforce date32 by casting date-like timestamp fields in Arrow table.
        from deltalake import write_deltalake

        import pyarrow as pa
        import pyarrow.parquet as pq

        for f in parquet_dims.glob("*.parquet"):
            dim_name = f.stem
            delta_out = dims_out / dim_name
            delta_out.mkdir(parents=True, exist_ok=True)

            table = pq.read_table(f)

            # Cast date-like timestamp columns to date32 for better Power BI behavior
            schema = table.schema
            fields = []
            for field in schema:
                name = field.name
                nl = name.lower()
                # only cast if it's timestamp AND name looks date-like
                if pa.types.is_timestamp(field.type) and (nl == "date" or nl.endswith("date") or "date" in nl or nl in {"dob", "birthdate"}):
                    fields.append(pa.field(name, pa.date32()))
                else:
                    fields.append(field)

            try:
                table = table.cast(pa.schema(fields), safe=False)
            except Exception:
                # If cast fails, just keep original (non-fatal)
                pass

            write_deltalake(str(delta_out), table, mode="overwrite")

    else:
        raise ValueError(f"Unknown file_format: {file_format}")

    # --------------------------------------------------------
    # FACTS (sales)
    # --------------------------------------------------------
    sales_target = facts_out / "sales"
    if sales_target.exists():
        shutil.rmtree(sales_target, ignore_errors=True)

    # DeltaParquet
    if ff == "deltaparquet":
        delta_src: Optional[Path] = None

        cfg_delta = sales_cfg.get("delta_output_folder")
        if cfg_delta:
            d = Path(cfg_delta).expanduser().resolve()
            if d.exists():
                delta_src = d

        if delta_src is None:
            fb = fact_folder / "sales"
            if fb.exists():
                delta_src = fb

        if delta_src is None:
            raise RuntimeError("DeltaParquet output folder not found!")

        shutil.copytree(
            delta_src,
            sales_target,
            ignore=_IGNORE_PATTERNS,
        )

        done("Creating Final Output Folder")
        return final_folder

    # Parquet (partitioned folder copy)
    if ff == "parquet":
        partitioned_sales = fact_folder / "sales"
        if partitioned_sales.exists():
            shutil.copytree(
                partitioned_sales,
                sales_target,
                ignore=_IGNORE_PATTERNS,
            )
        else:
            info("No sales parquet folder found to copy; facts/sales will be empty.")

        done("Creating Final Output Folder")
        return final_folder

    # CSV (convert sales parquet partitions -> csv + include sql helpers)
    if ff == "csv":
        import pandas as pd

        partitioned_sales = fact_folder / "sales"

        sql_root = final_folder / "sql"
        sql_root.mkdir(parents=True, exist_ok=True)

        if partitioned_sales.exists():
            for file in partitioned_sales.rglob("*.parquet"):
                rel = file.relative_to(partitioned_sales)
                out_file = sales_target / rel.with_suffix(".csv")
                out_file.parent.mkdir(parents=True, exist_ok=True)

                df = pd.read_parquet(file)
                df.to_csv(
                    out_file,
                    index=False,
                    encoding="utf-8",
                    quoting=csv.QUOTE_MINIMAL,
                )
        else:
            info("No sales parquet folder found to convert; facts/sales will be empty.")

        copy_sql_indexes(sql_root)
        copy_sql_constraints(sql_root)
        copy_sql_views(sql_root)

        done("Creating Final Output Folder")
        return final_folder

    raise ValueError(f"Unknown file_format: {file_format}")
