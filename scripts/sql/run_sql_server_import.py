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
    if args.trusted:
        return (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={args.server};"
            "Trusted_Connection=yes;"
        )

    if not args.user or not args.password:
        raise ValueError(
            "Username and password must be provided "
            "when not using --trusted"
        )

    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={args.server};"
        f"UID={args.user};PWD={args.password};"
    )


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

    args = parser.parse_args()

    try:
        connection_string = build_connection_string(args)

        import_sql_server(
            server=args.server,
            database=args.database,
            run_dir=Path(args.run_path),
            connection_string=connection_string,
        )

    except (SqlServerImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
