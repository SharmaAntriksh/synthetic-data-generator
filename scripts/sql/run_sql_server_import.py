import argparse
import sys
from pathlib import Path

# Ensure repo root is importable so "import src...." works when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/sql/ -> scripts -> repo root
sys.path.insert(0, str(REPO_ROOT))

from src.tools.sql.sql_server_import import (
    import_sql_server,
    SqlServerImportError,
)


def build_connection_string(args) -> str:
    """
    Build a SQL Server ODBC connection string from CLI arguments.
    """
    driver = args.odbc_driver or "ODBC Driver 17 for SQL Server"

    # Sanitize driver name to prevent connection string injection
    if ";" in driver or "{" in driver or "}" in driver:
        raise ValueError(f"Invalid ODBC driver name (contains illegal characters): {driver}")

    if args.trusted:
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={args.server};"
            "Trusted_Connection=yes;"
        )

    if not args.user or not args.user.strip():
        raise ValueError("Username must be a non-empty string when not using --trusted")
    password = args.password
    if getattr(args, "password_env", False):
        import os
        password = os.environ.get("SYNDATA_DB_PASSWORD", password)
    if not password:
        raise ValueError("Password must be provided when not using --trusted")

    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={args.server};"
        f"UID={args.user};PWD={password};"
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
        "--password-env",
        action="store_true",
        help="Read password from SYNDATA_DB_PASSWORD environment variable",
    )

    parser.add_argument(
        "--trusted",
        action="store_true",
        help="Use Windows Integrated Authentication",
    )

    parser.add_argument(
        "--apply-cci",
        action="store_true",
        help=(
            "Optionally execute CCI/columnstore apply scripts (if present). "
            "All other steps run every time."
        ),
    )

    parser.add_argument(
        "--drop-pk",
        action="store_true",
        help=(
            "Drop primary key and foreign key constraints after import. "
            "Reduces database size significantly for analytics workloads."
        ),
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Run data verification after import. "
            "Executes verify.RunAll scorecard and reports PASS/FAIL results."
        ),
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
            apply_cci=bool(args.apply_cci),
            drop_pk=bool(args.drop_pk),
            verify=bool(args.verify),
        )

    except ValueError as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2
    except SqlServerImportError as exc:
        print(f"IMPORT ERROR: {exc}", file=sys.stderr)
        return 1
    except ConnectionError as exc:
        print(f"CONNECTION ERROR: {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())