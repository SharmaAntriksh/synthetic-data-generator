from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    Execute a .sql file split on GO statements (case-insensitive, line-only GO).
    """
    import re

    sql_text = _read_sql_text(sql_file)

    batches = re.split(
        r"^\s*GO\s*$",
        sql_text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    for idx, batch in enumerate(batches, start=1):
        batch = batch.strip()
        if not batch:
            continue
        try:
            cursor.execute(batch)
        except pyodbc.Error as exc:
            # Bubble up the *real* SQL Server error text for fast debugging
            raise SqlServerImportError(
                f"Error executing batch {idx} in file '{sql_file.name}'. Details: {exc.args}"
            ) from exc


def execute_sql_files(cursor, files: Iterable[Path]) -> None:
    for f in files:
        execute_sql_batches(cursor, f)


def execute_sql_folder(cursor, folder: Path) -> List[Path]:
    """
    Execute all .sql files in a folder in sorted order.
    Returns the files executed (for logging).
    """
    sql_files = sorted(folder.glob("*.sql"))
    if not sql_files:
        return []
    execute_sql_files(cursor, sql_files)
    return sql_files


# -------------------------
# DB helpers
# -------------------------
def database_exists(cursor, database: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM sys.databases
        WHERE name = ?
        """,
        database,
    )
    return cursor.fetchone() is not None


def _quote_db_name(database: str) -> str:
    # Safe bracket quoting for SQL Server identifiers (handles closing bracket)
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

    Preferred layout (most deterministic):
      sql/schema/tables/*.sql
      sql/schema/views/*.sql
      sql/schema/constraints/*.sql
      sql/cci/*.sql                      (optional apply scripts)

    Back-compat layout:
      sql/schema/*.sql (flat)
        - views inferred by filename containing "view"/"views"
        - constraints inferred by filename containing "constraint"/fk/pk patterns
        - CCI apply inferred by filename containing "cci"/"columnstore" (treated as optional apply)
        - everything else treated as tables/other schema
    """
    schema_dir = sql_dir / "schema"

    # Structured folders (preferred)
    tables_dir = schema_dir / "tables"
    views_dir = schema_dir / "views"
    constraints_dir = schema_dir / "constraints"

    tables: List[Path] = []
    views: List[Path] = []
    constraints: List[Path] = []
    cci_apply: List[Path] = []

    if tables_dir.is_dir() or views_dir.is_dir() or constraints_dir.is_dir():
        if tables_dir.is_dir():
            tables = sorted(tables_dir.glob("*.sql"))
        if views_dir.is_dir():
            views = sorted(views_dir.glob("*.sql"))
        if constraints_dir.is_dir():
            constraints = sorted(constraints_dir.glob("*.sql"))
        # CCI apply scripts live outside schema/
        cci_dir = sql_dir / "cci"
        if cci_dir.is_dir():
            cci_apply = sorted(cci_dir.glob("*.sql"))
        return tables, views, constraints, cci_apply

    # Flat folder back-compat
    flat = sorted(schema_dir.glob("*.sql"))
    for f in flat:
        if _is_cci_file(f):
            cci_apply.append(f)
        elif _is_view_file(f):
            views.append(f)
        elif _is_constraint_file(f):
            constraints.append(f)
        else:
            tables.append(f)

    # Optional dedicated cci folder wins/extends
    cci_dir = sql_dir / "cci"
    if cci_dir.is_dir():
        cci_apply.extend(sorted(cci_dir.glob("*.sql")))

    return tables, views, constraints, cci_apply


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
      1) Create tables (and other base schema)
      2) Create views
      3) Insert data (BULK INSERT)
      4) Constraints
      5) Create CCI table type + stored proc (always)
      6) Optional: run CCI apply scripts (e.g., call proc) if apply_cci=True

    Expected structure in run_dir (CSV mode):
      sql/schema/...
      sql/load/...
      (optional) sql/cci/...
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

    # -------------------------
    # Step 1: ensure DB exists
    # -------------------------
    try:
        with pyodbc.connect(connection_string, autocommit=True) as conn:
            cursor = conn.cursor()
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

            # 2.1 Tables
            if tables_files:
                execute_sql_files(cursor, tables_files)
            else:
                # If schema exists but no *.sql matched (unexpected), fall back to folder run
                execute_sql_folder(cursor, schema_dir)

            # 2.2 Views
            if view_files:
                execute_sql_files(cursor, view_files)

            # 2.3 Insert data
            execute_sql_folder(cursor, load_dir)

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
            execute_sql_files(cursor, cci_apply_files)

        print("[INFO] CCI apply scripts executed.")

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed running CCI apply scripts in database '{database}'. Details: {exc.args}"
        ) from exc
