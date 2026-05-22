"""SQL Server (T-SQL) dialect renderer.

Renders byte-identically to the previous string-based helpers — any change
to spelling, spacing, or ordering here will diverge from existing baseline
CREATE TABLE scripts and break downstream BULK INSERT scripts.
"""
from __future__ import annotations

from .base import ColumnSpec, Dialect, SqlType

_MISSING = object()

_SIMPLE: dict[SqlType, str] = {
    SqlType.INT: "INT",
    SqlType.BIGINT: "BIGINT",
    SqlType.SMALLINT: "SMALLINT",
    SqlType.TINYINT: "TINYINT",
    SqlType.BIT: "BIT",
    SqlType.FLOAT: "FLOAT",
    SqlType.DATE: "DATE",
    SqlType.DATETIME: "DATETIME",
}


class SqlServerDialect(Dialect):
    name = "sqlserver"
    batch_separator = "GO"

    def render_type(self, spec: ColumnSpec) -> str:
        suffix = "NULL" if spec.nullable else "NOT NULL"
        return f"{self._render_base(spec)} {suffix}"

    def quote_ident(self, name: str) -> str:
        raw = str(name).strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        return f"[{raw.replace(']', ']]')}]"

    def drop_table_if_exists(self, schema: str, table: str) -> str:
        fq = f"{self.quote_ident(schema)}.{self.quote_ident(table)}"
        literal = fq.replace("'", "''")
        return (
            f"IF OBJECT_ID(N'{literal}', N'U') IS NOT NULL\n"
            f"    DROP TABLE {fq};"
        )

    def _render_base(self, spec: ColumnSpec) -> str:
        t = spec.sql_type
        simple = _SIMPLE.get(t, _MISSING)
        if simple is not _MISSING:
            if spec.args:
                raise ValueError(f"{t.name} takes no args, got {spec.args!r}")
            return simple

        if t is SqlType.VARCHAR:
            (n,) = spec.args
            return f"VARCHAR({n})"
        if t is SqlType.CHAR:
            (n,) = spec.args
            return f"CHAR({n})"
        if t is SqlType.DECIMAL:
            p, s = spec.args
            return f"DECIMAL({p}, {s})"
        if t is SqlType.DATETIME2:
            (p,) = spec.args
            return f"DATETIME2({p})"
        if t is SqlType.TIME:
            (p,) = spec.args
            return f"TIME({p})"

        raise ValueError(f"Unhandled SqlType: {t!r}")


DEFAULT_DIALECT: Dialect = SqlServerDialect()
