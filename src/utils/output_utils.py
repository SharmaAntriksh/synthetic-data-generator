import shutil
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime
import csv
from typing import Optional, Union

from src.utils.logging_utils import stage, done, info


# ============================================================
# Helpers
# ============================================================
def copy_sql_constraints(sql_root: Path):
    project_root = Path(__file__).resolve().parents[2]
    constraints_file = project_root / "scripts" / "sql" / "bootstrap" / "create_constraints.sql"

    if not constraints_file.exists():
        info("No create_constraints.sql found; skipping SQL constraints copy")
        return

    schema_dir = sql_root / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(constraints_file, schema_dir / "03_create_constraints.sql")
    info("Included create_constraints.sql in final output")


def copy_sql_views(sql_root: Path):
    project_root = Path(__file__).resolve().parents[2]
    views_file = project_root / "scripts" / "sql" / "views" / "create_views.sql"

    if not views_file.exists():
        info("No create_views.sql found; skipping SQL views copy")
        return

    schema_dir = sql_root / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(views_file, schema_dir / "04_create_views.sql")
    info("Included create_views.sql in final output")


def copy_sql_indexes(sql_root: Path):
    project_root = Path(__file__).resolve().parents[2]
    cci_file = project_root / "scripts" / "sql" / "columnstore" / "create_drop_cci.sql"

    if not cci_file.exists():
        info("No create_drop_cci.sql found; skipping SQL indexes")
        return

    indexes_dir = sql_root / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cci_file, indexes_dir / "create_drop_cci.sql")
    info("Included create_drop_cci.sql in SQL indexes")


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
    """
    stage("Creating Final Output Folder")
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %I_%M_%S %p")  # windows-safe

    customer_total = cfg["customers"]["total_customers"]
    sales_total = sales_cfg["total_rows"]

    cust_short = format_number_short(customer_total)
    sales_short = format_number_short(sales_total)

    fmt_label = {
        "deltaparquet": "DeltaParquet",
        "parquet": "Parquet",
        "csv": "CSV",
    }.get(file_format.lower(), file_format)

    dataset_name = f"{timestamp} Customers {cust_short} Sales {sales_short} {fmt_label}"

    final_folder = final_folder_root / dataset_name
    dims_out = final_folder / "dimensions"
    facts_out = final_folder / "facts"

    if final_folder.exists():
        shutil.rmtree(final_folder, ignore_errors=True)

    final_folder.mkdir(parents=True, exist_ok=True)
    dims_out.mkdir(parents=True, exist_ok=True)
    facts_out.mkdir(parents=True, exist_ok=True)

    # NEW: copy yaml specs into <final_folder>/config/
    _copy_config_files_into_run_folder(
        final_folder=final_folder,
        config_yaml_path=config_yaml_path,
        model_yaml_path=model_yaml_path,
    )

    ff = file_format.lower()

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------
    if ff == "parquet":
        for f in parquet_dims.glob("*.parquet"):
            shutil.copy2(f, dims_out / f.name)

    elif ff == "csv":
        import pandas as pd

        for f in parquet_dims.glob("*.parquet"):
            df = pd.read_parquet(f)
            (dims_out / f"{f.stem}.csv").parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(
                dims_out / f"{f.stem}.csv",
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_MINIMAL,
            )

    elif ff == "deltaparquet":
        from deltalake import write_deltalake

        for f in parquet_dims.glob("*.parquet"):
            dim_name = f.stem
            delta_out = dims_out / dim_name
            delta_out.mkdir(parents=True, exist_ok=True)

            table = pq.read_table(f)
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
        delta_src = None

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
            ignore=shutil.ignore_patterns("_tmp_parts*", "tmp*", "*_tmp*"),
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
                ignore=shutil.ignore_patterns("_tmp_parts*", "tmp*", "*_tmp*"),
            )

        done("Creating Final Output Folder")
        return final_folder

    # CSV (convert sales parquet partitions -> csv + include sql helpers)
    if ff == "csv":
        import pandas as pd

        partitioned_sales = fact_folder / "sales"

        sql_root = final_folder / "sql"
        sql_root.mkdir(parents=True, exist_ok=True)

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

        copy_sql_indexes(sql_root)
        copy_sql_constraints(sql_root)
        copy_sql_views(sql_root)

        done("Creating Final Output Folder")
        return final_folder

    raise ValueError(f"Unknown file_format: {file_format}")
