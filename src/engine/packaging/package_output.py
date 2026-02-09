import shutil
from pathlib import Path
from urllib.parse import unquote

from src.utils.output_utils import create_final_output_folder
from src.utils.logging_utils import stage, info

from .paths import get_first_existing_path, tables_from_sales_cfg
from .parquet_packager import copy_parquet_facts
from .delta_packager import copy_delta_facts
from .csv_packager import copy_csv_facts
from .sql_scripts import write_bulk_insert_scripts, write_create_table_scripts


def package_output(cfg, sales_cfg, parquet_dims: Path, fact_out: Path):
    """
    Orchestrates packaging:
      - Creates final packaged folder (dims + config copied by output_utils)
      - Copies fact outputs (Sales / SalesOrderHeader / SalesOrderDetail)
      - Generates SQL scripts (CSV only)
    """
    file_format = str(sales_cfg["file_format"]).lower()
    is_csv = file_format == "csv"

    final_root = Path(unquote(str(cfg["final_output_folder"]))).resolve()

    config_yaml_path = get_first_existing_path(
        cfg,
        keys=["config_yaml_path", "config_path", "config_file", "config_yaml", "config"],
    )
    model_yaml_path = get_first_existing_path(
        cfg,
        keys=["model_yaml_path", "model_path", "model_file", "model_yaml", "model"],
    )

    with stage("Creating Final Output Folder"):
        final_folder = create_final_output_folder(
            final_folder_root=final_root,
            parquet_dims=parquet_dims,
            fact_folder=fact_out,
            sales_cfg=sales_cfg,
            file_format=file_format,
            sales_rows_expected=sales_cfg["total_rows"],
            cfg=cfg,
            config_yaml_path=config_yaml_path,
            model_yaml_path=model_yaml_path,
            package_facts=False,
        )

        # Remove URL-encoded duplicate run folder (%20)
        parent = final_folder.parent
        real_name = final_folder.name
        for sibling in parent.iterdir():
            if sibling.is_dir() and "%20" in sibling.name and unquote(sibling.name) == real_name:
                shutil.rmtree(sibling, ignore_errors=True)

        dims_out = final_folder / "dimensions"
        facts_out = final_folder / "facts"
        facts_out.mkdir(parents=True, exist_ok=True)

        tables = tables_from_sales_cfg(sales_cfg)

        if file_format == "parquet":
            copy_parquet_facts(fact_out=fact_out, facts_out=facts_out, sales_cfg=sales_cfg, tables=tables)
            return final_folder

        if file_format == "deltaparquet":
            copy_delta_facts(fact_out=fact_out, facts_out=facts_out, sales_cfg=sales_cfg, tables=tables)
            return final_folder

        if file_format != "csv":
            raise ValueError(f"Unsupported file_format in packaging: {file_format!r}")

        copy_csv_facts(fact_out=fact_out, facts_out=facts_out, tables=tables)

    # SQL SCRIPT GENERATION — CSV ONLY
    if is_csv:
        sql_root = final_folder / "sql"
        write_bulk_insert_scripts(dims_out=dims_out, facts_out=facts_out, sql_root=sql_root, sales_cfg=sales_cfg)
        write_create_table_scripts(dims_out=dims_out, facts_out=facts_out, sql_root=sql_root, cfg=cfg)
    else:
        info("Skipping SQL script generation for non-CSV format.")

    return final_folder
