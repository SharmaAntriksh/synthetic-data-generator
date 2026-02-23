from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_GO_SPLIT_RE = re.compile(r"^\s*GO\s*$", flags=re.MULTILINE | re.IGNORECASE)


class SqlServerImportError(RuntimeError):
    """Raised when SQL Server import fails."""


# -------------------------
# Formatting helpers
# -------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(level: str, msg: str) -> None:
    # level: INFO/WARN/ERROR
    print(f"{_ts()} | {level:<5} | {msg}")


def _extract_tables_from_create_sql(sql_file: "Path") -> list[str]:
    """
    Extract table names from a CREATE TABLE script in execution order.
    Works with: CREATE TABLE dbo.Table, CREATE TABLE [dbo].[Table], CREATE TABLE [Table]
    """
    try:
        txt = sql_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for raw in txt.splitlines():
        line = re.sub(r"--.*$", "", raw).strip()
        if not line:
            continue
        if "CREATE" not in line.upper() or "TABLE" not in line.upper():
            continue

        m = re.search(
            r"CREATE\s+TABLE\s+"
            r"(?:(?:\[\s*(?P<s1>\w+)\s*\]|\b(?P<s2>\w+)\b)\s*\.\s*)?"
            r"(?:\[\s*(?P<t1>\w+)\s*\]|(?P<t2>\w+))",
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            continue

        table = (m.group("t1") or m.group("t2") or "").strip()
        if not table:
            continue
        if table.lower() in {"if", "exists"}:
            continue

        if table not in seen:
            seen.add(table)
            out.append(table)

    return out


def _find_table_schema(cursor: "pyodbc.Cursor", table_name: str) -> str:
    cursor.execute(
        "SELECT s.name "
        "FROM sys.tables t "
        "JOIN sys.schemas s ON s.schema_id = t.schema_id "
        "WHERE t.name = ? "
        "ORDER BY CASE WHEN s.name = 'dbo' THEN 0 ELSE 1 END, s.name;",
        table_name,
    )
    row = cursor.fetchone()
    if not row:
        raise SqlServerImportError(f"Table not found in database: {table_name}")
    return str(row[0])


def _fast_rowcount(cursor: "pyodbc.Cursor", schema: str, table: str) -> int:
    cursor.execute(
        "SELECT COALESCE(SUM(row_count),0) "
        "FROM sys.dm_db_partition_stats "
        "WHERE object_id = OBJECT_ID(?) AND index_id IN (0,1);",
        f"{schema}.{table}",
    )
    return int(cursor.fetchone()[0])


def _print_table_counts(cursor: "pyodbc.Cursor", *, tables: list[str], title: str) -> None:
    _log("INFO", title)
    for t in tables:
        try:
            schema = _find_table_schema(cursor, t)
            n = _fast_rowcount(cursor, schema, t)
            print(f"  - {schema}.{t}: {n:,}")
        except Exception as exc:
            print(f"  - {t}: [SKIP] {exc}")


def _cci_count(cursor: "pyodbc.Cursor", schema: str, table: str) -> int:
    cursor.execute(
        "SELECT COUNT(*) "
        "FROM sys.indexes i "
        "JOIN sys.tables t ON t.object_id = i.object_id "
        "JOIN sys.schemas s ON s.schema_id = t.schema_id "
        "WHERE s.name = ? AND t.name = ? AND i.type_desc = 'CLUSTERED COLUMNSTORE';",
        schema,
        table,
    )
    return int(cursor.fetchone()[0])


def _print_cci_summary(cursor: "pyodbc.Cursor", *, tables: list[str]) -> None:
    cursor.execute("SELECT COUNT(*) FROM sys.indexes WHERE type_desc = 'CLUSTERED COLUMNSTORE';")
    total_ccis = int(cursor.fetchone()[0])

    _log("INFO", f"CCI summary: total CCIs in DB = {total_ccis}")
    for t in tables:
        try:
            schema = _find_table_schema(cursor, t)
            c = _cci_count(cursor, schema, t)
            print(f"  - {schema}.{t}: CCI={c}")
        except Exception as exc:
            print(f"  - {t}: [SKIP] {exc}")


def _short_path(p: Path, *, base: Path | None = None) -> str:
    """
    Prefer printing paths relative to `base` (if possible), otherwise just the filename.
    """
    try:
        if base is not None:
            return p.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        pass
    return p.name


# -------------------------
# SQL file execution helpers
# -------------------------
def _read_sql_text(sql_file: Path) -> str:
    raw = sql_file.read_bytes()

    # BOM detection
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        # Heuristic: NULs early usually mean UTF-16
        if b"\x00" in raw[:4096]:
            try:
                text = raw.decode("utf-16")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-16")

    # Strip NULs that break identifiers (dbo -> d\0b\0o\0)
    if "\x00" in text:
        text = text.replace("\x00", "")

    return text


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


def _try_disable_query_timeout(conn) -> None:
    """
    pyodbc timeout support differs by build/driver.
    Prefer connection-level timeout; do nothing if unsupported.
    """
    try:
        # 0 commonly means "no timeout" for ODBC query timeout.
        conn.timeout = 0
    except Exception:
        pass


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

    schema_tables_dir = schema_dir / "tables"
    schema_views_dir = schema_dir / "views"
    schema_constraints_dir = schema_dir / "constraints"

    top_views_dir = sql_dir / "views"
    top_constraints_dir = sql_dir / "constraints"

    cci_dir = sql_dir / "cci"
    indexes_dir = sql_dir / "indexes"

    # Structured schema folders (preferred)
    if schema_tables_dir.is_dir() or schema_views_dir.is_dir() or schema_constraints_dir.is_dir():
        tables = list_sql_files(schema_tables_dir)
        views = list_sql_files(schema_views_dir)
        constraints = list_sql_files(schema_constraints_dir)

        cci_apply = list_sql_files(cci_dir)
        cci_apply += [p for p in list_sql_files(indexes_dir) if _is_cci_file(p)]
        return tables, views, constraints, cci_apply

    # Flat schema folder with inference
    schema_files = list_sql_files(schema_dir)

    views = list_sql_files(top_views_dir) or [p for p in schema_files if _is_view_file(p)]
    constraints = list_sql_files(top_constraints_dir) or [p for p in schema_files if _is_constraint_file(p)]

    cci_apply = list_sql_files(cci_dir)
    cci_apply += [p for p in list_sql_files(indexes_dir) if _is_cci_file(p)]
    inferred_cci_from_schema = [p for p in schema_files if _is_cci_file(p)]

    excluded = set(views) | set(constraints) | set(inferred_cci_from_schema)
    tables = [p for p in schema_files if p not in excluded]

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
# Budget cache refresh (optional)
# -------------------------
def _maybe_refresh_budget_cache(db_conn_str: str, *, target: str) -> None:
    t = (target or "FX").strip().upper()
    if t in {"", "NONE"}:
        _log("INFO", "Budget cache target is NONE; skipping refresh.")
        return
    if t not in {"FX", "LOCAL", "BOTH"}:
        raise SqlServerImportError(f"Invalid budget_cache_target={target!r}. Use FX, LOCAL, BOTH, or NONE.")

    try:
        with pyodbc.connect(db_conn_str, autocommit=True) as conn:
            _try_disable_query_timeout(conn)
            cursor = conn.cursor()

            cursor.execute("SELECT 1 WHERE OBJECT_ID(N'dbo.sp_RefreshBudgetCache', N'P') IS NOT NULL;")
            if cursor.fetchone() is None:
                _log("WARN", "dbo.sp_RefreshBudgetCache not found; skipping budget cache refresh.")
                return

            _log("INFO", f"Executing dbo.sp_RefreshBudgetCache (@Target='{t}') ...")
            try:
                cursor.execute(
                    "EXEC dbo.sp_RefreshBudgetCache @RebuildIfSchemaChanged = 1, @Target = ?;",
                    (t,),
                )
            except pyodbc.Error as exc:
                # Backward-compatible fallback if someone has an older proc without @Target
                msg = " ".join(str(x) for x in (exc.args or ()))
                if "too many arguments" in msg.lower():
                    cursor.execute("EXEC dbo.sp_RefreshBudgetCache @RebuildIfSchemaChanged = 1;")
                else:
                    raise

            _log("INFO", "Budget cache refresh completed.")

    except pyodbc.Error as exc:
        raise SqlServerImportError(f"Failed refreshing budget cache. Details: {exc.args}") from exc


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
    refresh_budget_cache: bool = False,
    budget_cache_target: str = "FX",
) -> None:
    run_dir = Path(run_dir)
    sql_dir = run_dir / "sql"
    schema_dir = sql_dir / "schema"
    load_dir = sql_dir / "load"

    if not schema_dir.is_dir() or not load_dir.is_dir():
        raise SqlServerImportError(
            "SQL Server import is supported only for CSV runs. "
            "Expected 'sql/schema/' and 'sql/load/' folders in run directory."
        )

    bootstrap_dir = PROJECT_ROOT / "scripts" / "sql" / "bootstrap"
    types_file = bootstrap_dir / "create_types.sql"
    procs_file = bootstrap_dir / "create_procs.sql"

    tables_files, view_files, constraint_files, cci_apply_files = _collect_phase_scripts(sql_dir)

    if not tables_files:
        raise SqlServerImportError(
            f"No table scripts found. Expected files under '{schema_dir}' "
            "or under 'sql/schema/tables/'."
        )

    # Step 1: ensure DB exists
    try:
        with pyodbc.connect(connection_string, autocommit=True) as conn:
            _try_disable_query_timeout(conn)
            cursor = conn.cursor()

            if database_exists(cursor, database):
                raise SqlServerImportError(
                    f"Database '{database}' already exists. "
                    "Import aborted to avoid partial drops / FK failures. "
                    "Use a new database name or drop the database first."
                )

            create_database_if_not_exists(cursor, database)

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed connecting to SQL Server '{server}'. Details: {exc.args}"
        ) from exc

    db_conn_str = f"{connection_string};DATABASE={database}"

    # Step 2: core import (tables/constraints/views/load)
    try:
        with pyodbc.connect(db_conn_str, autocommit=False) as conn:
            _try_disable_query_timeout(conn)
            cursor = conn.cursor()

            _log("INFO", "Executing schema + load scripts:")

            # 2.1 Tables
            print("  Tables:")
            for f in tables_files:
                print(f"    - {_short_path(f, base=run_dir)}")
            execute_sql_files(cursor, tables_files)

            # 2.2 Constraints
            if constraint_files:
                print("  Constraints:")
                for f in constraint_files:
                    print(f"    - {_short_path(f, base=run_dir)}")
                execute_sql_files(cursor, constraint_files)

            # 2.3 Views
            if view_files:
                print("  Views:")
                for f in view_files:
                    print(f"    - {_short_path(f, base=run_dir)}")
                execute_sql_files(cursor, view_files)

            # 2.4 Load (dims then facts)
            load_files = list_sql_files(load_dir)

            dims_sql = next((p for p in load_files if p.name.lower() == "01_bulk_insert_dims.sql"), None)
            facts_sql = next((p for p in load_files if p.name.lower() == "02_bulk_insert_facts.sql"), None)

            ordered_load: List[Path] = []
            if dims_sql is not None:
                ordered_load.append(dims_sql)
            if facts_sql is not None:
                ordered_load.append(facts_sql)

            if not ordered_load:
                ordered_load = load_files
            else:
                seen = set(ordered_load)
                extras = [p for p in load_files if p not in seen]
                if extras:
                    _log("WARN", "Extra load scripts present; skipping by default:")
                    for f in extras:
                        print(f"    - {_short_path(f, base=run_dir)}")

            print("  Load:")
            for f in ordered_load:
                print(f"    - {_short_path(f, base=run_dir)}")
            execute_sql_files(cursor, ordered_load)

            # 2.5 Row count verification
            try:
                dim_create = next((p for p in tables_files if p.name.lower().endswith("create_dimensions.sql")), None)
                fact_create = next((p for p in tables_files if p.name.lower().endswith("create_facts.sql")), None)

                dim_tables = _extract_tables_from_create_sql(dim_create) if dim_create else []
                fact_tables = _extract_tables_from_create_sql(fact_create) if fact_create else []

                _print_table_counts(cursor, tables=dim_tables, title="Loaded dimension row counts:")
                _print_table_counts(cursor, tables=fact_tables, title="Loaded fact row counts:")
            except Exception as _exc:
                _log("WARN", f"Row count verification skipped: {_exc}")

            conn.commit()

        _log("INFO", f"Core import completed (tables/constraints/views/load) for '{database}'.")

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed importing SQL into database '{database}'. Details: {exc.args}"
        ) from exc

    # Step 2.6: optional budget cache refresh (materialize cache tables)
    if refresh_budget_cache:
        _maybe_refresh_budget_cache(db_conn_str, target=budget_cache_target)

    # Step 3: optional CCI (types/procs + apply)
    if not apply_cci:
        _log("INFO", "Skipping CCI bootstrap/apply (apply_cci=False).")
        return

    # 3.1 Bootstrap (TYPE + PROC)
    try:
        with pyodbc.connect(db_conn_str, autocommit=True) as conn:
            _try_disable_query_timeout(conn)
            cursor = conn.cursor()

            if types_file.is_file():
                execute_sql_batches(cursor, types_file)
            else:
                _log("WARN", f"Missing bootstrap types file: {_short_path(types_file, base=PROJECT_ROOT)}")

            if procs_file.is_file():
                execute_sql_batches(cursor, procs_file)
            else:
                _log("WARN", f"Missing bootstrap procs file: {_short_path(procs_file, base=PROJECT_ROOT)}")

        _log("INFO", "CCI bootstrap completed (TYPE + PROC).")

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed running CCI bootstrap in database '{database}'. Details: {exc.args}"
        ) from exc

    # 3.2 Apply scripts (from run output)
    _log("INFO", "CCI apply scripts discovered:")
    for f in cci_apply_files:
        print(f"    - {_short_path(f, base=run_dir)}")

    if not cci_apply_files:
        _log("INFO", "No CCI apply scripts found; nothing to do.")
        return

    try:
        with pyodbc.connect(db_conn_str, autocommit=True) as conn:
            _try_disable_query_timeout(conn)
            cursor = conn.cursor()

            execute_sql_files(cursor, cci_apply_files)

            # Verification: total CCIs + per-fact CCIs (derived from create_facts.sql)
            fact_create = next((p for p in tables_files if p.name.lower().endswith("create_facts.sql")), None)
            fact_tables = _extract_tables_from_create_sql(fact_create) if fact_create else []
            _print_cci_summary(cursor, tables=fact_tables)

            cursor.execute("SELECT COUNT(*) FROM sys.indexes WHERE type_desc = 'CLUSTERED COLUMNSTORE';")
            total_ccis = int(cursor.fetchone()[0])

        if total_ccis == 0:
            raise SqlServerImportError(
                "CCI apply completed without errors, but no CLUSTERED COLUMNSTORE index exists in the database. "
                "Likely the script is a no-op (wrong @Action, empty table list, or wrong table names)."
            )

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed running CCI apply scripts in database '{database}'. Details: {exc.args}"
        ) from exc