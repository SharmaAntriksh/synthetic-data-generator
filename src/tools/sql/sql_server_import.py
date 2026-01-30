from pathlib import Path
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class SqlServerImportError(RuntimeError):
    """Raised when SQL Server import fails."""
    pass


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

    Expected files in run_dir (executed in order):
      1. create_dimensions.sql
      2. create_facts.sql
      3. bulk_insert_dims.sql
      4. bulk_insert_facts.sql

    Optional (executed last if present):
      - create_views.sql
    """
    run_dir = Path(run_dir)

    sql_sequence = [
        run_dir / "create_dimensions.sql",
        run_dir / "create_facts.sql",
        run_dir / "bulk_insert_dims.sql",
        run_dir / "bulk_insert_facts.sql",
    ]

    for sql_file in sql_sequence:
        if not sql_file.is_file():
            raise SqlServerImportError(
                f"Missing required SQL file: {sql_file.name} in {run_dir}"
            )

    views_file = run_dir / "create_views.sql"

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

            # Core schema + data
            for sql_file in sql_sequence:
                execute_sql_batches(cursor, sql_file)

            # Optional views (must run last)
            if views_file.is_file():
                execute_sql_batches(cursor, views_file)

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

