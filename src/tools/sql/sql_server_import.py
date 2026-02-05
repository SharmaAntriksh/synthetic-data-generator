from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_GO_SPLIT_RE = re.compile(r"^\s*GO\s*$", flags=re.MULTILINE | re.IGNORECASE)


class SqlServerImportError(RuntimeError):
    """Raised when SQL Server import fails."""


# -------------------------
# SQL file execution helpers
# -------------------------
def _read_sql_text(sql_file: Path) -> str:
    try:
        try:
            return sql_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return sql_file.read_text(encoding="utf-16")
    except OSError as exc:
        raise SqlServerImportError(f"Failed reading SQL file '{sql_file.name}': {exc}") from exc


def execute_sql_batches(cursor, sql_file: Path) -> None:
    """
    Execute a .sql file split on line-only GO statements (case-insensitive).
    """
    sql_text = _read_sql_text(sql_file)
    batches = _GO_SPLIT_RE.split(sql_text)

    for idx, batch in enumerate(batches, start=1):
        batch = batch.strip()
        if not batch:
            continue
        try:
            cursor.execute(batch)
        except pyodbc.Error as exc:
            # Include driver payload for immediate diagnosis
            raise SqlServerImportError(
                f"Error executing batch {idx} in file '{sql_file.name}'. Details: {exc.args}"
            ) from exc


def execute_sql_files(cursor, files: Iterable[Path]) -> None:
    for f in files:
        execute_sql_batches(cursor, f)


def list_sql_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.sql"))


# -------------------------
# DB helpers
# -------------------------
def database_exists(cursor, database: str) -> bool:
    cursor.execute("SELECT 1 FROM sys.databases WHERE name = ?", database)
    return cursor.fetchone() is not None


def _quote_db_name(database: str) -> str:
    # Safe bracket quoting for SQL Server identifiers
    return f"[{database.replace(']', ']]')}]"


def create_database_if_not_exists(cursor, database: str) -> None:
    """
    Create the database if it does not already exist.
    Must be executed in autocommit mode.
    """
    try:
        cursor.execute(f"CREATE DATABASE {_quote_db_name(database)}")
    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed to create database '{database}'. Details: {exc.args}"
        ) from exc


# -------------------------
# Script discovery / ordering
# -------------------------
def _is_view_file(p: Path) -> bool:
    n = p.name.lower()
    return "view" in n or "views" in n


def _is_constraint_file(p: Path) -> bool:
    n = p.name.lower()
    return (
        "constraint" in n
        or n.startswith("fk")
        or "_fk" in n
        or "foreignkey" in n
        or "foreign_key" in n
        or n.startswith("pk")
        or "_pk" in n
        or "primarykey" in n
        or "primary_key" in n
    )


def _is_cci_file(p: Path) -> bool:
    n = p.name.lower()
    return "cci" in n or "columnstore" in n


def _collect_phase_scripts(sql_dir: Path) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """
    Collect scripts for phases:
      tables, views, constraints, cci_apply

    Supported layouts:

    A) Preferred layout:
      sql/schema/tables/*.sql
      sql/schema/views/*.sql
      sql/schema/constraints/*.sql
      sql/cci/*.sql                   (optional apply scripts)

    B) Alternate layout:
      sql/schema/*.sql                (flat)
      sql/views/*.sql (optional)
      sql/constraints/*.sql (optional)
      sql/cci/*.sql (optional)

    Notes:
    - Any file inferred as CCI (name contains 'cci'/'columnstore') is treated as OPTIONAL apply
      and is excluded from the always-run phases.
    """
    schema_dir = sql_dir / "schema"
    if not schema_dir.is_dir():
        return [], [], [], []

    # Preferred subfolders under schema/
    schema_tables_dir = schema_dir / "tables"
    schema_views_dir = schema_dir / "views"
    schema_constraints_dir = schema_dir / "constraints"

    # Optional top-level folders under sql/
    top_views_dir = sql_dir / "views"
    top_constraints_dir = sql_dir / "constraints"

    cci_dir = sql_dir / "cci"
    indexes_dir = sql_dir / "indexes"

    # 1) If schema subfolders exist, use them deterministically
    if schema_tables_dir.is_dir() or schema_views_dir.is_dir() or schema_constraints_dir.is_dir():
        tables = list_sql_files(schema_tables_dir)
        views = list_sql_files(schema_views_dir)
        constraints = list_sql_files(schema_constraints_dir)

        cci_apply = list_sql_files(cci_dir)
        # Allow CCI apply scripts in sql/indexes, but only if explicitly CCI/columnstore
        cci_apply += [p for p in list_sql_files(indexes_dir) if _is_cci_file(p)]

        return tables, views, constraints, cci_apply

    # 2) Otherwise: tables from schema/*.sql (minus inferred view/constraint/cci)
    schema_files = list_sql_files(schema_dir)

    # Views: prefer sql/views if present, else infer from schema
    views = list_sql_files(top_views_dir)
    if not views:
        views = [p for p in schema_files if _is_view_file(p)]

    # Constraints: prefer sql/constraints if present, else infer from schema
    constraints = list_sql_files(top_constraints_dir)
    if not constraints:
        constraints = [p for p in schema_files if _is_constraint_file(p)]

    # CCI apply: prefer sql/cci, plus CCI-named scripts in sql/indexes, plus inferred CCI from schema
    cci_apply = list_sql_files(cci_dir)
    cci_apply += [p for p in list_sql_files(indexes_dir) if _is_cci_file(p)]
    inferred_cci_from_schema = [p for p in schema_files if _is_cci_file(p)]

    # Tables: whatever remains in schema after excluding views/constraints/cci
    excluded = set(views) | set(constraints) | set(inferred_cci_from_schema)
    tables = [p for p in schema_files if p not in excluded]

    # Include inferred schema CCI as optional apply scripts (NOT always-run)
    cci_apply += inferred_cci_from_schema

    # Deduplicate while preserving order
    seen = set()
    cci_apply_unique: List[Path] = []
    for p in cci_apply:
        if p not in seen:
            seen.add(p)
            cci_apply_unique.append(p)

    return tables, views, constraints, cci_apply_unique


# -------------------------
# Main import
# -------------------------
def import_sql_server(
    *,
    server: str,
    database: str,
    run_dir: Path,
    connection_string: str,
    apply_cci: bool = False,
) -> None:
    """
    Flow:
      1) Create tables (always)
      2) Create views (always)
      3) Insert data (always)
      4) Apply constraints (always)
      5) Create CCI table type + stored proc (always)
      6) Optional: run CCI apply scripts if apply_cci=True

    Expected structure in run_dir (CSV mode):
      sql/schema/...
      sql/load/...
      (optional) sql/views/...
      (optional) sql/constraints/...
      (optional) sql/cci/...
      (optional) sql/indexes/...   (only CCI/columnstore-named files are considered, and only if apply_cci=True)
    """
    run_dir = Path(run_dir)
    sql_dir = run_dir / "sql"
    schema_dir = sql_dir / "schema"
    load_dir = sql_dir / "load"

    if not schema_dir.is_dir() or not load_dir.is_dir():
        raise SqlServerImportError(
            "SQL Server import is supported only for CSV runs. "
            "Expected 'sql/schema/' and 'sql/load/' folders in run directory."
        )

    # Bootstrap scripts (always executed each run)
    bootstrap_dir = PROJECT_ROOT / "scripts" / "sql" / "bootstrap"
    types_file = bootstrap_dir / "create_types.sql"
    procs_file = bootstrap_dir / "create_procs.sql"

    tables_files, view_files, constraint_files, cci_apply_files = _collect_phase_scripts(sql_dir)

    if not tables_files:
        raise SqlServerImportError(
            f"No table scripts found. Expected files under '{schema_dir}' "
            "or under 'sql/schema/tables/'."
        )

    # -------------------------
    # Step 1: ensure DB exists
    # -------------------------
    try:
        with pyodbc.connect(connection_string, autocommit=True) as conn:
            cursor = conn.cursor()
            cursor.timeout = 0
            if not database_exists(cursor, database):
                create_database_if_not_exists(cursor, database)
    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed connecting to SQL Server '{server}'. Details: {exc.args}"
        ) from exc

    db_conn_str = f"{connection_string};DATABASE={database}"

    # -------------------------
    # Step 2: core import (always)
    # -------------------------
    try:
        with pyodbc.connect(db_conn_str, autocommit=False) as conn:
            cursor = conn.cursor()
            cursor.timeout = 0

            # 2.1 Tables
            execute_sql_files(cursor, tables_files)

            # 2.2 Views
            if view_files:
                execute_sql_files(cursor, view_files)

            # 2.3 Insert data
            load_files = list_sql_files(load_dir)
            execute_sql_files(cursor, load_files)
            conn.commit()  # commit data before constraints

            # 2.4 Constraints
            if constraint_files:
                execute_sql_files(cursor, constraint_files)
            conn.commit()

        print(f"[INFO] Core import completed (tables/views/load/constraints) for '{database}'.")

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed importing SQL into database '{database}'. Details: {exc.args}"
        ) from exc

    # -------------------------
    # Step 3: CCI bootstrap (always)
    # -------------------------
    try:
        with pyodbc.connect(db_conn_str, autocommit=True) as conn:
            cursor = conn.cursor()
            cursor.timeout = 0

            if types_file.is_file():
                execute_sql_batches(cursor, types_file)
            else:
                print(f"[WARN] Missing bootstrap types file: {types_file}")

            if procs_file.is_file():
                execute_sql_batches(cursor, procs_file)
            else:
                print(f"[WARN] Missing bootstrap procs file: {procs_file}")

        print("[INFO] CCI bootstrap completed (TYPE + PROC).")

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed running CCI bootstrap in database '{database}'. Details: {exc.args}"
        ) from exc

    # -------------------------
    # Step 4: optional CCI apply
    # -------------------------
    if not apply_cci:
        print("[INFO] Skipping CCI apply scripts (apply_cci=False).")
        return

    if not cci_apply_files:
        print("[INFO] No CCI apply scripts found; nothing to do.")
        return

    try:
        with pyodbc.connect(db_conn_str, autocommit=True) as conn:
            cursor = conn.cursor()
            cursor.timeout = 0
            execute_sql_files(cursor, cci_apply_files)

        print("[INFO] CCI apply scripts executed.")

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed running CCI apply scripts in database '{database}'. Details: {exc.args}"
        ) from exc
