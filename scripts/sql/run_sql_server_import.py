import argparse
import sys
from pathlib import Path

from src.tools.sql.sql_server_import import (
    import_sql_server,
    SqlServerImportError,
)


def build_connection_string(args) -> str:
    """
    Build a SQL Server ODBC connection string from CLI arguments.
    """
    driver = args.odbc_driver or "ODBC Driver 17 for SQL Server"

    if args.trusted:
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={args.server};"
            "Trusted_Connection=yes;"
        )

    if not args.user or not args.password:
        raise ValueError(
            "Username and password must be provided when not using --trusted"
        )

    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={args.server};"
        f"UID={args.user};PWD={args.password};"
    )


def _resolve_run_dir(run_path: str) -> Path:
    run_dir = Path(run_path).expanduser().resolve()

    sql_dir = run_dir / "sql"
    schema_dir = sql_dir / "schema"
    load_dir = sql_dir / "load"

    if not run_dir.exists():
        raise ValueError(f"Run path does not exist: {run_dir}")
    if not sql_dir.is_dir():
        raise ValueError(f"Missing folder: {sql_dir}")
    if not schema_dir.is_dir():
        raise ValueError(f"Missing folder (CSV runs only): {schema_dir}")
    if not load_dir.is_dir():
        raise ValueError(f"Missing folder (CSV runs only): {load_dir}")

    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create SQL Server database and import generated SQL scripts"
    )

    parser.add_argument(
        "--server",
        required=True,
        help="SQL Server hostname or instance name",
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Target database name",
    )
    parser.add_argument(
        "--run-path",
        required=True,
        help="Path to generator run output folder containing SQL scripts",
    )

    parser.add_argument("--user", help="SQL Server username")
    parser.add_argument("--password", help="SQL Server password")

    parser.add_argument(
        "--trusted",
        action="store_true",
        help="Use Windows Integrated Authentication",
    )

    parser.add_argument(
        "--apply-cci",
        action="store_true",
        help="Optionally execute CCI/columnstore apply scripts (if present). "
             "All other steps run every time.",
    )

    parser.add_argument(
        "--odbc-driver",
        default=None,
        help="Override ODBC driver name (default: ODBC Driver 17 for SQL Server).",
    )

    args = parser.parse_args()

    try:
        run_dir = _resolve_run_dir(args.run_path)
        connection_string = build_connection_string(args)

        import_sql_server(
            server=args.server,
            database=args.database,
            run_dir=run_dir,
            connection_string=connection_string,
            apply_cci=args.apply_cci,
        )

    except (SqlServerImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
