from pathlib import Path
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class SqlServerImportError(RuntimeError):
    """Raised when SQL Server import fails."""
    pass


def execute_sql_folder(cursor, folder: Path):
    """
    Execute all .sql files in a folder in sorted order.
    """
    sql_files = sorted(folder.glob("*.sql"))
    if not sql_files:
        return

    for sql_file in sql_files:
        execute_sql_batches(cursor, sql_file)


def database_exists(cursor, database: str) -> bool:
    """
    Check whether a database already exists.
    """
    cursor.execute(
        """
        SELECT 1
        FROM sys.databases
        WHERE name = ?
        """,
        database,
    )
    return cursor.fetchone() is not None


def create_database_if_not_exists(cursor, database: str):
    """
    Create the database if it does not already exist.
    Must be executed in autocommit mode.
    """
    try:
        cursor.execute(f"CREATE DATABASE [{database}]")
    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed to create database '{database}'"
        ) from exc
    

def execute_sql_batches(cursor, sql_file: Path):
    import re

    try:
        try:
            sql_text = sql_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            sql_text = sql_file.read_text(encoding="utf-16")

        batches = re.split(
            r'^\s*GO\s*$',
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
                raise SqlServerImportError(
                    f"Error executing batch {idx} in file '{sql_file.name}'"
                ) from exc

    except OSError as exc:
        raise SqlServerImportError(
            f"Failed reading SQL file '{sql_file.name}'"
        ) from exc


def execute_sql_single_batch(cursor, sql_file: Path):
    try:
        try:
            sql_text = sql_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            sql_text = sql_file.read_text(encoding="utf-16")

        sql_text = sql_text.strip()
        if not sql_text:
            return

        cursor.execute(sql_text)

    except pyodbc.Error as exc:
        print("\n=== SQL SERVER ERROR (bootstrap) ===")
        for arg in exc.args:
            print(arg)
        print("==================================\n")
        raise SqlServerImportError(
            f"Error executing SQL file '{sql_file.name}'"
        ) from exc



def import_sql_server(
    *,
    server: str,
    database: str,
    run_dir: Path,
    connection_string: str,
):
    """
    Create database (if needed) and execute generated SQL scripts.

    Behavior:
    - If database already exists → reuse database and import
    - If database does not exist → create and import

    Expected structure in run_dir (CSV mode only):
        schema/   → tables, constraints, views
        load/     → bulk insert scripts
        indexes/  → optional performance indexes

    Optional (executed last if present):
      - create_views.sql
    """
    run_dir = Path(run_dir)

    sql_dir     = run_dir / "sql"
    schema_dir  = sql_dir / "schema"
    load_dir    = sql_dir / "load"
    indexes_dir = sql_dir / "indexes"

    # CSV-only guard
    if not schema_dir.is_dir() or not load_dir.is_dir():
        raise SqlServerImportError(
            "SQL Server import is supported only for CSV runs. "
            "Expected 'schema/' and 'load/' folders in run directory."
        )

    bootstrap_dir = PROJECT_ROOT / "scripts" / "sql" / "bootstrap"

    types_file = bootstrap_dir / "create_types.sql"
    procs_file = bootstrap_dir / "create_procs.sql"

    # ------------------------------------------------------------
    # Step 1: connect to server context and check DB existence
    # ------------------------------------------------------------
    try:
        with pyodbc.connect(connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            db_exists = database_exists(cursor, database)

            if not db_exists:
                create_database_if_not_exists(cursor, database)

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed connecting to SQL Server '{server}'"
        ) from exc
    
    # ------------------------------------------------------------
    # Step 2: connect to database and execute scripts
    # (tables, data, views only — transactional)
    # ------------------------------------------------------------
    db_conn_str = f"{connection_string};DATABASE={database}"

    try:
        with pyodbc.connect(db_conn_str, autocommit=False) as conn:
            cursor = conn.cursor()

            # ------------------------------------------------------------
            # Step 2a: schema (tables, constraints, views) — transactional
            # ------------------------------------------------------------
            execute_sql_folder(cursor, schema_dir)

            # ------------------------------------------------------------
            # Step 2b: data load — transactional
            # ------------------------------------------------------------
            execute_sql_folder(cursor, load_dir)

            conn.commit()

            print(
                f"[INFO] SQL import completed successfully for database '{database}'."
            )

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed importing SQL into database '{database}'"
        ) from exc

    # ------------------------------------------------------------
    # Step 3: columnstore bootstrap (TYPE + PROC only)
    # Must run in autocommit mode
    # ------------------------------------------------------------
    with pyodbc.connect(db_conn_str, autocommit=True) as conn:
        cursor = conn.cursor()

        if types_file.is_file():
            execute_sql_single_batch(cursor, types_file)

        if procs_file.is_file():
            execute_sql_single_batch(cursor, procs_file)

    print("[INFO] Columnstore bootstrap completed (TYPE + PROC).")

