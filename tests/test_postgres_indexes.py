"""Tests for the Postgres btree-index composer (Step 5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.engine.packaging.sql_scripts import compose_postgres_indexes_sql


@pytest.fixture
def run_layout(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "sql").mkdir(parents=True)
    return run


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestComposerOutput:
    def test_writes_under_postgres_indexes(self, run_layout: Path) -> None:
        compose_postgres_indexes_sql(sql_root=run_layout / "sql")
        out = run_layout / "postgres" / "indexes" / "01_create_btree_indexes.sql"
        assert out.is_file()

    def test_does_not_touch_sql_root(self, run_layout: Path) -> None:
        compose_postgres_indexes_sql(sql_root=run_layout / "sql")
        assert not (run_layout / "sql" / "indexes").exists(), \
            "Postgres indexes must not leak into the SQL Server sql/ tree"


class TestIndexFileContent:
    def _file(self) -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        return (
            repo_root / "scripts" / "sql" / "postgres" / "indexes" / "01_create_btree_indexes.sql"
        )

    @staticmethod
    def _strip_comments(text: str) -> str:
        import re
        text = re.sub(r"/\*[\s\S]*?\*/", "", text)
        text = re.sub(r"--[^\n]*", "", text)
        return text

    def test_file_exists(self) -> None:
        assert self._file().is_file()

    def test_uses_create_index_if_not_exists(self) -> None:
        text = self._strip_comments(_read(self._file()))
        assert "CREATE INDEX IF NOT EXISTS" in text
        # No CREATE UNIQUE INDEX — FK source columns are not unique.
        assert "CREATE UNIQUE INDEX" not in text

    def test_no_sql_server_idioms(self) -> None:
        text = self._strip_comments(_read(self._file()))
        assert "[dbo]" not in text
        assert "OBJECT_ID(" not in text
        assert "NONCLUSTERED" not in text
        assert "\nGO\n" not in text

    def test_covers_key_fact_fk_columns(self) -> None:
        """Spot-check: the load-bearing FK columns on Sales must be indexed."""
        text = _read(self._file())
        for ix in (
            "IX_Sales_CustomerKey",
            "IX_Sales_ProductKey",
            "IX_Sales_StoreKey",
            "IX_Sales_OrderDate",
            "IX_SalesOrderHeader_CustomerKey",
            "IX_SalesOrderDetail_ProductKey",
            "IX_SalesReturn_ReturnDate",
            "IX_InventorySnapshot_WarehouseKey",
        ):
            assert ix in text, f"Missing expected index: {ix}"

    def test_does_not_duplicate_pk_leading_columns(self) -> None:
        """ProductKey leads InventorySnapshot's composite PK — don't add a redundant standalone btree."""
        text = _read(self._file())
        assert "IX_InventorySnapshot_ProductKey" not in text, (
            "ProductKey already leads the InventorySnapshot composite PK; "
            "a standalone btree on it would be redundant."
        )

    def test_every_create_index_is_guarded(self) -> None:
        """No raw CREATE INDEX outside a DO $$ block (everything must be table/column-guarded)."""
        text = self._strip_comments(_read(self._file()))
        # Crude check: every CREATE INDEX line should appear after a DO $$ within the file.
        # Counts: number of DO blocks vs number of CREATE INDEX statements should leave no
        # CREATE INDEX statements unaccounted for outside DO blocks.
        do_count = text.count("DO $$")
        # A DO block can hold multiple CREATE INDEX lines (the Sales block does).
        assert do_count > 0, "expected DO $$ guards"
        # No CREATE INDEX should appear at file top level (lines starting with CREATE INDEX
        # would be at column 0; inside DO blocks the file indents them).
        for line in text.splitlines():
            if line.startswith("CREATE INDEX"):
                raise AssertionError(
                    f"Unguarded CREATE INDEX at file top level: {line!r}"
                )
