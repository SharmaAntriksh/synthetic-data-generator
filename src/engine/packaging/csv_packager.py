import shutil
from pathlib import Path
from typing import Optional

from src.utils.logging_utils import info, done
from src.facts.sales.output_paths import TABLE_SALES

from .paths import table_dir_name


def resolve_csv_table_dir(fact_out: Path, table: str) -> Optional[Path]:
    """
    Locate CSV chunk folder for a table.
    Accepts:
      - <fact_out>/csv/<snake_table>/
      - <fact_out>/csv/<TableName>/
      - <fact_out>/csv/ (Sales may be rooted)
    """
    csv_root = fact_out / "csv"
    if not csv_root.exists():
        return None

    tdir = table_dir_name(table)

    if table == TABLE_SALES:
        candidates = [csv_root / tdir, csv_root / table, csv_root]
    else:
        candidates = [csv_root / tdir, csv_root / table, csv_root / table.lower()]

    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def copy_csv_facts(*, fact_out: Path, facts_out: Path, tables: list[str]) -> None:
    copied_files = 0
    missing_dirs: list[str] = []

    for t in tables:
        src_dir = resolve_csv_table_dir(fact_out, t)
        if src_dir is None:
            missing_dirs.append(t)
            info(f"No CSV folder found for {t}; expected under {fact_out / 'csv'}.")
            continue

        multi_table = len(tables) > 1
        if t == TABLE_SALES:
            dst_dir = (facts_out / table_dir_name(t)) if multi_table else facts_out
        else:
            dst_dir = facts_out / table_dir_name(t)

        dst_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(src_dir.glob("*.csv"))
        info(f"Copying {len(csv_files)} CSV file(s) for {t} from: {src_dir}")

        for f in csv_files:
            target = dst_dir / f.name
            if target.exists():
                raise RuntimeError(f"Duplicate CSV filename during packaging: {target}")
            shutil.copy2(f, target)
            copied_files += 1

    if missing_dirs:
        raise RuntimeError(
            "No CSV fact outputs found for table(s): "
            + ", ".join(missing_dirs)
            + f". Expected under: {fact_out / 'csv'}"
        )

    done(f"Copied {copied_files} CSV fact file(s).")
