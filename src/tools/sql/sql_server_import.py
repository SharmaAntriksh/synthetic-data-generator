from pathlib import Path
import pyodbc


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
        cursor.execute(
            f"""
            CREATE DATABASE [{database}]
            """
        )
    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed to create database '{database}'"
        ) from exc


def execute_sql_file(cursor, sql_file: Path):
    """
    Execute a SQL Server script, splitting batches on GO.

    Rules:
    - GO must appear alone on a line (ignoring whitespace)
    - Case-insensitive
    - Empty batches are ignored
    """
    if not sql_file.exists():
        raise SqlServerImportError(f"SQL file not found: {sql_file}")

    # Handle UTF-8 and UTF-16 (common from SSMS)
    try:
        sql_text = sql_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        sql_text = sql_file.read_text(encoding="utf-16")

    if not sql_text.strip():
        return

    batches: list[str] = []
    current_batch: list[str] = []

    for line in sql_text.splitlines():
        if line.strip().upper() == "GO":
            if current_batch:
                batches.append("\n".join(current_batch).strip())
                current_batch.clear()
        else:
            current_batch.append(line)

    if current_batch:
        batches.append("\n".join(current_batch).strip())

    for idx, batch in enumerate(batches, start=1):
        if not batch:
            continue
        try:
            cursor.execute(batch)
        except pyodbc.Error as exc:
            raise SqlServerImportError(
                f"Error executing batch {idx} in file '{sql_file.name}'"
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
    - If database already exists → exit with message, no execution
    - If database does not exist → create and import

    Expected files in run_dir (executed in order):
      1. create_dimensions.sql
      2. create_facts.sql
      3. bulk_insert_dims.sql
      4. bulk_insert_facts.sql
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

    # Step 1: connect to server context and check DB existence
    try:
        with pyodbc.connect(connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            if database_exists(cursor, database):
                print(
                    f"[INFO] Database '{database}' already exists. "
                    "Skipping SQL import."
                )
                return

            create_database_if_not_exists(cursor, database)

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed connecting to SQL Server '{server}'"
        ) from exc

    # Step 2: connect to database and execute scripts
    db_conn_str = f"{connection_string};DATABASE={database}"

    try:
        with pyodbc.connect(db_conn_str, autocommit=False) as conn:
            cursor = conn.cursor()

            for sql_file in sql_sequence:
                execute_sql_file(cursor, sql_file)

            conn.commit()

            print(
                f"[INFO] SQL import completed successfully for database '{database}'."
            )

    except pyodbc.Error as exc:
        raise SqlServerImportError(
            f"Failed importing SQL into database '{database}'"
        ) from exc
