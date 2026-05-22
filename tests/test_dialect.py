"""Unit tests for the SQL dialect layer (Phase 1).

Asserts that every SqlType renders to the exact SQL Server string the
previous string-based helpers produced, for both NULL and NOT NULL.
"""
from __future__ import annotations

import pytest

from src.tools.sql.dialect import ColumnSpec, DEFAULT_DIALECT, SqlServerDialect, SqlType
from src.utils.static_schemas import (
    BIT,
    BIGINT,
    CHAR,
    DATE,
    DATETIME,
    DATETIME2,
    DECIMAL,
    FLOAT,
    INT,
    SMALLINT,
    TIME,
    TINYINT,
    VARCHAR,
    STATIC_SCHEMAS,
)


class TestSqlServerRendering:
    """Every SqlType -> expected T-SQL fragment."""

    def setup_method(self) -> None:
        self.dialect = SqlServerDialect()

    @pytest.mark.parametrize(
        "spec,expected",
        [
            (ColumnSpec(SqlType.INT, nullable=False), "INT NOT NULL"),
            (ColumnSpec(SqlType.INT, nullable=True), "INT NULL"),
            (ColumnSpec(SqlType.BIGINT, nullable=False), "BIGINT NOT NULL"),
            (ColumnSpec(SqlType.BIGINT, nullable=True), "BIGINT NULL"),
            (ColumnSpec(SqlType.SMALLINT, nullable=False), "SMALLINT NOT NULL"),
            (ColumnSpec(SqlType.TINYINT, nullable=False), "TINYINT NOT NULL"),
            (ColumnSpec(SqlType.BIT, nullable=False), "BIT NOT NULL"),
            (ColumnSpec(SqlType.FLOAT, nullable=False), "FLOAT NOT NULL"),
            (ColumnSpec(SqlType.DATE, nullable=False), "DATE NOT NULL"),
            (ColumnSpec(SqlType.DATE, nullable=True), "DATE NULL"),
            (ColumnSpec(SqlType.DATETIME, nullable=False), "DATETIME NOT NULL"),
            (ColumnSpec(SqlType.DATETIME2, nullable=False, args=(7,)), "DATETIME2(7) NOT NULL"),
            (ColumnSpec(SqlType.TIME, nullable=False, args=(0,)), "TIME(0) NOT NULL"),
            (ColumnSpec(SqlType.VARCHAR, nullable=False, args=(100,)), "VARCHAR(100) NOT NULL"),
            (ColumnSpec(SqlType.VARCHAR, nullable=True, args=(50,)), "VARCHAR(50) NULL"),
            (ColumnSpec(SqlType.VARCHAR, nullable=True, args=("MAX",)), "VARCHAR(MAX) NULL"),
            (ColumnSpec(SqlType.CHAR, nullable=False, args=(1,)), "CHAR(1) NOT NULL"),
            (ColumnSpec(SqlType.DECIMAL, nullable=False, args=(8, 2)), "DECIMAL(8, 2) NOT NULL"),
            (ColumnSpec(SqlType.DECIMAL, nullable=True, args=(10, 6)), "DECIMAL(10, 6) NULL"),
        ],
    )
    def test_render_type(self, spec: ColumnSpec, expected: str) -> None:
        assert self.dialect.render_type(spec) == expected

    def test_default_dialect_is_sqlserver(self) -> None:
        assert isinstance(DEFAULT_DIALECT, SqlServerDialect)
        assert DEFAULT_DIALECT.name == "sqlserver"

    def test_simple_type_rejects_args(self) -> None:
        with pytest.raises(ValueError, match="takes no args"):
            self.dialect.render_type(ColumnSpec(SqlType.INT, args=(10,)))


class TestStaticSchemaHelpers:
    """Type helper functions in static_schemas should produce ColumnSpecs that
    render to the exact strings the old string-based helpers produced."""

    @pytest.mark.parametrize(
        "spec,expected",
        [
            (INT(), "INT NOT NULL"),
            (INT(not_null=False), "INT NULL"),
            (BIGINT(), "BIGINT NOT NULL"),
            (SMALLINT(not_null=False), "SMALLINT NULL"),
            (TINYINT(), "TINYINT NOT NULL"),
            (BIT(), "BIT NOT NULL"),
            (FLOAT(not_null=False), "FLOAT NULL"),
            (DATE(), "DATE NOT NULL"),
            (DATETIME(), "DATETIME NOT NULL"),
            (DATETIME2(7), "DATETIME2(7) NOT NULL"),
            (TIME(0), "TIME(0) NOT NULL"),
            # VARCHAR / CHAR default to NULL (not_null=False) — matches legacy
            (VARCHAR(100), "VARCHAR(100) NULL"),
            (VARCHAR(50, not_null=True), "VARCHAR(50) NOT NULL"),
            (VARCHAR("MAX"), "VARCHAR(MAX) NULL"),
            (CHAR(1), "CHAR(1) NULL"),
            (CHAR(1, not_null=True), "CHAR(1) NOT NULL"),
            (DECIMAL(8, 2), "DECIMAL(8, 2) NOT NULL"),
            (DECIMAL(10, 6, not_null=False), "DECIMAL(10, 6) NULL"),
        ],
    )
    def test_helper_renders_to_expected_sqlserver_string(
        self, spec: ColumnSpec, expected: str
    ) -> None:
        assert DEFAULT_DIALECT.render_type(spec) == expected

    def test_every_static_schema_column_is_renderable(self) -> None:
        """Round-trip safety: every column in STATIC_SCHEMAS must render."""
        for table, schema in STATIC_SCHEMAS.items():
            for col, spec in schema:
                rendered = DEFAULT_DIALECT.render_type(spec)
                assert rendered, f"{table}.{col} produced empty SQL"
                assert rendered.endswith("NULL"), (
                    f"{table}.{col} rendered as {rendered!r} — missing nullability"
                )
