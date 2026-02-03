import shutil
from pathlib import Path
from urllib.parse import unquote
from typing import Optional

from src.utils.output_utils import create_final_output_folder
from src.tools.sql.generate_bulk_insert_sql import generate_bulk_insert_script
from src.tools.sql.generate_create_table_scripts import generate_all_create_tables
from src.utils.logging_utils import stage, info, skip, done


def _get_first_existing_path(cfg: dict, keys: list[str]) -> Optional[Path]:
    project_root = Path(__file__).resolve().parents[2]  # repo root

    def _resolve_existing(v: str) -> Optional[Path]:
        p = Path(str(v)).expanduser()
        if p.is_absolute() and p.exists():
            return p

        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        repo_candidate = (project_root / p).resolve()
        if repo_candidate.exists():
            return repo_candidate

        return None

    for k in keys:
        v = cfg.get(k)
        if not v:
            continue
        resolved = _resolve_existing(v)
        if resolved:
            return resolved

    return None


def package_output(cfg, sales_cfg, parquet_dims: Path, fact_out: Path):
    """
    Handles:
    - Creating final packaged folder (dims + facts)
    - Copying Sales fact (Delta / Parquet / CSV)
    - Generating SQL scripts (CSV only)
    - Cleaning stale output

    Behavior is format-driven and deterministic.
    """

    file_format = sales_cfg["file_format"].lower()
    is_csv = file_format == "csv"

    # ============================================================
    # Normalize final output root ONCE
    # ============================================================
    final_root = Path(unquote(str(cfg["final_output_folder"]))).resolve()

    # ============================================================
    # Resolve config/model yaml paths (optional)
    # ============================================================
    # These keys are guesses to be resilient across different entrypoints.
    # Prefer storing the true file paths in cfg["config_yaml_path"] / cfg["model_yaml_path"] upstream.
    config_yaml_path = _get_first_existing_path(
        cfg,
        keys=[
            "config_yaml_path",
            "config_path",
            "config_file",
            "config_yaml",
            "config",
        ],
    )
    model_yaml_path = _get_first_existing_path(
        cfg,
        keys=[
            "model_yaml_path",
            "model_path",
            "model_file",
            "model_yaml",
            "model",
        ],
    )

    # ============================================================
    # Create final output folder
    # ============================================================
    with stage("Creating Final Output Folder"):
        final_folder = create_final_output_folder(
            final_folder_root=final_root,
            parquet_dims=parquet_dims,
            fact_folder=fact_out,
            sales_cfg=sales_cfg,
            file_format=file_format,
            sales_rows_expected=sales_cfg["total_rows"],
            cfg=cfg,
            # NEW: copy YAML specs into final_folder/config/
            config_yaml_path=config_yaml_path,
            model_yaml_path=model_yaml_path,
        )

        # ---------------------------------------------------------
        # HARD FIX: remove URL-encoded duplicate run folder (%20)
        # ---------------------------------------------------------
        parent = final_folder.parent
        real_name = final_folder.name

        for sibling in parent.iterdir():
            if (
                sibling.is_dir()
                and "%20" in sibling.name
                and unquote(sibling.name) == real_name
            ):
                shutil.rmtree(sibling)

        dims_out = final_folder / "dimensions"
        facts_out = final_folder / "facts"

        # ---------------------------------------------------------
        # Clean old packaged sales folder (idempotent)
        # ---------------------------------------------------------
        packaged_sales = facts_out / "sales"
        if packaged_sales.exists():
            shutil.rmtree(packaged_sales)

        # ---------------------------------------------------------
        # Determine destination sales folder
        # ---------------------------------------------------------
        if file_format == "deltaparquet":
            dst_sales = facts_out / "sales"
        else:
            dst_sales = facts_out

        dst_sales.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # PARQUET MODE — single file copy and exit early
        # ============================================================
        if file_format == "parquet":
            src_file = fact_out / "parquet" / "sales.parquet"
            dst_file = facts_out / "sales.parquet"

            if not src_file.exists():
                raise RuntimeError(f"Expected parquet file not found: {src_file}")

            if dst_file.exists():
                dst_file.unlink()

            shutil.copy2(src_file, dst_file)
            done("Sales fact copied (single parquet file).")

            # Parquet never generates SQL scripts
            return final_folder

        # ============================================================
        # Determine source sales folder
        # ============================================================
        if file_format == "deltaparquet":
            src_sales = fact_out / "sales"
        else:  # CSV
            src_sales = fact_out / "csv"

        if not src_sales.exists():
            raise RuntimeError(f"Expected sales output folder not found: {src_sales}")

        # ============================================================
        # CSV MODE — flat copy (schema already resolved upstream)
        # ============================================================
        if is_csv:
            csv_files = list(src_sales.glob("*.csv"))
            info(f"Copying {len(csv_files)} CSV sales files from: {src_sales}")

            for csv_file in csv_files:
                target = dst_sales / csv_file.name
                if target.exists():
                    raise RuntimeError(
                        f"Duplicate CSV filename detected during packaging: "
                        f"{csv_file.name}"
                    )
                shutil.copy2(csv_file, target)

            done("Sales fact copied (CSV flat).")

        # ============================================================
        # DELTA MODE — directory snapshot copy
        # ============================================================
        else:
            info(f"Copying sales fact from: {src_sales}")

            for item in src_sales.iterdir():
                if item.name == "_tmp_parts":
                    continue

                target = dst_sales / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)

            done("Sales fact copied (Delta snapshot).")

    # ============================================================
    # SQL SCRIPT GENERATION — CSV ONLY (correct & reachable)
    # ============================================================
    if is_csv:
        sql_root = final_folder / "sql"
        sql_root.mkdir(parents=True, exist_ok=True)

        with stage("Generating BULK INSERT Scripts"):
            dims_csv = sorted(dims_out.glob("*.csv"))
            facts_csv = sorted(facts_out.glob("*.csv"))

            if not dims_csv and not facts_csv:
                skip("No CSV files found — skipping BULK INSERT scripts.")
            else:
                generate_bulk_insert_script(
                    csv_folder=str(dims_out),
                    table_name=None,
                    output_sql_file=str(sql_root / "load" / "01_bulk_insert_dims.sql"),
                    mode="csv",
                )

                generate_bulk_insert_script(
                    csv_folder=str(facts_out),
                    table_name="Sales",
                    output_sql_file=str(sql_root / "load" / "02_bulk_insert_facts.sql"),
                    mode="legacy",
                    row_terminator="0x0a",
                )

        with stage("Generating CREATE TABLE Scripts"):
            generate_all_create_tables(
                dim_folder=dims_out,
                fact_folder=facts_out,
                output_folder=sql_root,
                cfg=cfg,
                skip_order_cols=sales_cfg.get("skip_order_cols", False),
            )

    else:
        info("Skipping SQL script generation for non-CSV format.")

    return final_folder
