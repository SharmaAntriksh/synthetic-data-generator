import shutil
from pathlib import Path
from typing import Optional

from src.utils.logging_utils import info, done
from src.facts.sales.output_paths import TABLE_SALES

from .paths import table_dir_name


def resolve_merged_parquet(fact_out: Path, sales_cfg: dict, table: str) -> Optional[Path]:
    """
    Locate merged parquet output in scratch fact_out.

    Layouts supported (scratch has evolved):
      - <fact_out>/parquet/<snake_table>/<snake_table>.parquet   (current order tables)
      - <fact_out>/<snake_table>/<snake_table>.parquet          (older)
      - Sales merged file may sit at <fact_out>/parquet/sales.parquet or <fact_out>/sales.parquet
    """
    parquet_root = fact_out / "parquet"
    tdir = table_dir_name(table)

    if table == TABLE_SALES:
        merged_name = str(sales_cfg.get("merged_file") or "sales.parquet")
        candidates = [
            parquet_root / merged_name,
            parquet_root / tdir / merged_name,
            fact_out / merged_name,
            fact_out / tdir / merged_name,
        ]
    else:
        merged_name = f"{table_dir_name(table)}.parquet"  # snake_case file
        candidates = [
            parquet_root / tdir / merged_name,
            parquet_root / table / merged_name,  # fallback if CamelCase dir
            parquet_root / merged_name,          # fallback if written to root
            fact_out / tdir / merged_name,
            fact_out / table / merged_name,
            fact_out / merged_name,
            # legacy fallback: older runs may still have CamelCase filename
            parquet_root / tdir / f"{table}.parquet",
            fact_out / tdir / f"{table}.parquet",
        ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def copy_parquet_facts(*, fact_out: Path, facts_out: Path, sales_cfg: dict, tables: list[str]) -> None:
    copied = 0
    missing: list[str] = []

    for t in tables:
        src = resolve_merged_parquet(fact_out, sales_cfg, t)
        if src is None:
            missing.append(t)
            info(f"No merged parquet found for {t}. Checked: {fact_out / 'parquet'} and {fact_out}.")
            continue

        # Keep Sales filename untouched; normalize others to <snake_table>.parquet
        dst_name = src.name if t == TABLE_SALES else f"{table_dir_name(t)}.parquet"
        dst = facts_out / dst_name

        if dst.exists():
            dst.unlink()

        shutil.copy2(src, dst)
        copied += 1

    if missing:
        raise RuntimeError(
            "No merged parquet found for table(s): "
            + ", ".join(missing)
            + f". Scratch roots checked: {fact_out} and {fact_out / 'parquet'}."
        )

    done(f"Copied {copied} parquet fact file(s).")
