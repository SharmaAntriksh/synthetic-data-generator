"""Tests for the Postgres views composer + admin script copier (Steps 3+4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.engine.config.config_schema import AppConfig
from src.engine.packaging.sql_scripts import (
    _resolve_postgres_view_schema,
    _rewrite_postgres_view_schema,
    compose_postgres_views_sql,
    copy_postgres_admin_sql,
)


@pytest.fixture
def run_layout(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "sql").mkdir(parents=True)
    return run


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestResolvePostgresViewSchema:
    def test_none_cfg_defaults_to_public(self) -> None:
        assert _resolve_postgres_view_schema(None) == "public"

    def test_dbo_remaps_to_public(self) -> None:
        cfg = AppConfig.model_validate({"defaults": {"view_schema": "dbo"}})
        assert _resolve_postgres_view_schema(cfg) == "public"

    def test_explicit_public_kept(self) -> None:
        cfg = AppConfig.model_validate({"defaults": {"view_schema": "public"}})
        assert _resolve_postgres_view_schema(cfg) == "public"

    def test_custom_kept_verbatim(self) -> None:
        cfg = AppConfig.model_validate({"defaults": {"view_schema": "bi"}})
        assert _resolve_postgres_view_schema(cfg) == "bi"


class TestRewritePostgresViewSchema:
    def test_default_returns_input_unchanged(self) -> None:
        sql = 'CREATE OR REPLACE VIEW "public"."vw_Sales" AS SELECT * FROM "public"."Sales"'
        assert _rewrite_postgres_view_schema(sql, "public") == sql

    def test_custom_drops_vw_prefix_and_reschemas(self) -> None:
        sql = 'CREATE OR REPLACE VIEW "public"."vw_Sales" AS SELECT * FROM "public"."Sales"'
        out = _rewrite_postgres_view_schema(sql, "bi")
        # View target rewritten, source table reference untouched.
        assert '"bi"."Sales"' in out
        assert '"public"."Sales"' in out  # source table still public
        assert '"public"."vw_' not in out  # all view targets relocated


class TestComposePostgresViewsOutput:
    def test_writes_under_postgres_schema(self, run_layout: Path) -> None:
        compose_postgres_views_sql(sql_root=run_layout / "sql", cfg=None)
        out = run_layout / "postgres" / "schema" / "04_create_views.sql"
        assert out.is_file()

    def test_default_schema_no_create_schema_preamble(self, run_layout: Path) -> None:
        compose_postgres_views_sql(sql_root=run_layout / "sql", cfg=None)
        out = _read(run_layout / "postgres" / "schema" / "04_create_views.sql")
        assert "CREATE SCHEMA IF NOT EXISTS" not in out
        assert '"public"."vw_Sales"' in out  # vw_ prefix retained

    def test_custom_schema_emits_create_schema(self, run_layout: Path) -> None:
        cfg = AppConfig.model_validate({"defaults": {"view_schema": "bi"}})
        compose_postgres_views_sql(sql_root=run_layout / "sql", cfg=cfg)
        out = _read(run_layout / "postgres" / "schema" / "04_create_views.sql")
        assert 'CREATE SCHEMA IF NOT EXISTS "bi";' in out
        assert '"bi"."Sales"' in out  # vw_ stripped, schema rewritten
        assert '"public"."vw_' not in out
        # Source table refs still under public.
        assert '"public"."Sales"' in out

    def test_includes_both_view_files(self, run_layout: Path) -> None:
        compose_postgres_views_sql(sql_root=run_layout / "sql", cfg=None)
        out = _read(run_layout / "postgres" / "schema" / "04_create_views.sql")
        assert "00_model_views.sql" in out
        assert "10_budget_views.sql" in out

    def test_dbo_cfg_treated_as_public(self, run_layout: Path) -> None:
        """A SQL Server-style view_schema=dbo should produce default (public) output."""
        cfg = AppConfig.model_validate({"defaults": {"view_schema": "dbo"}})
        compose_postgres_views_sql(sql_root=run_layout / "sql", cfg=cfg)
        out = _read(run_layout / "postgres" / "schema" / "04_create_views.sql")
        assert "CREATE SCHEMA IF NOT EXISTS" not in out
        assert '"public"."vw_Sales"' in out


class TestPostgresViewFilesContent:
    def _all_files(self) -> list[Path]:
        repo_root = Path(__file__).resolve().parents[1]
        return sorted((repo_root / "scripts" / "sql" / "postgres" / "views").glob("*.sql"))

    @staticmethod
    def _strip_comments(text: str) -> str:
        import re
        text = re.sub(r"/\*[\s\S]*?\*/", "", text)
        text = re.sub(r"--[^\n]*", "", text)
        return text

    def test_both_files_present(self) -> None:
        names = {p.name for p in self._all_files()}
        assert names == {"00_model_views.sql", "10_budget_views.sql"}

    def test_no_sql_server_idioms(self) -> None:
        for p in self._all_files():
            text = self._strip_comments(p.read_text(encoding="utf-8"))
            assert "OBJECT_ID(" not in text, f"{p.name}: OBJECT_ID() leaked"
            assert "EXEC(" not in text, f"{p.name}: T-SQL EXEC() leaked"
            assert "sp_executesql" not in text, f"{p.name}: sp_executesql leaked"
            assert "[dbo]" not in text, f"{p.name}: [dbo] bracket leaked"
            assert "\nGO\n" not in text, f"{p.name}: GO separator leaked"
            assert "SET NOCOUNT" not in text, f"{p.name}: SET NOCOUNT leaked"
            assert " MONEY" not in text.upper().replace("NUMERIC", "X"), (
                f"{p.name}: MONEY type leaked (should be NUMERIC(19,4))"
            )

    def test_uses_create_or_replace(self) -> None:
        for p in self._all_files():
            text = self._strip_comments(p.read_text(encoding="utf-8"))
            assert "CREATE OR REPLACE VIEW" in text, f"{p.name}: missing CREATE OR REPLACE VIEW"
            assert "CREATE OR ALTER VIEW" not in text, f"{p.name}: T-SQL CREATE OR ALTER leaked"


class TestCopyPostgresAdmin:
    def test_copies_pk_proc(self, run_layout: Path) -> None:
        copy_postgres_admin_sql(sql_root=run_layout / "sql")
        out = run_layout / "postgres" / "admin" / "create_pk_proc.sql"
        assert out.is_file()
        text = _read(out)
        assert "CREATE OR REPLACE PROCEDURE admin.manage_primary_keys" in text
        assert "DROP" in text
        assert "RESTORE" in text
        # FKs are also managed so PK drops aren't blocked by FK dependencies.
        assert "foreign key" in text.lower()
        assert "contype IN ('p', 'f')" in text

    def test_does_not_touch_schema_folder(self, run_layout: Path) -> None:
        copy_postgres_admin_sql(sql_root=run_layout / "sql")
        # admin file lands under postgres/admin/, NOT postgres/schema/
        assert not (run_layout / "postgres" / "schema" / "create_pk_proc.sql").exists()


class TestPostgresAdminProcContent:
    def _proc_file(self) -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        return (
            repo_root / "scripts" / "sql" / "postgres" / "admin" / "create_pk_proc.sql"
        )

    def test_creates_admin_schema(self) -> None:
        text = _read(self._proc_file())
        assert "CREATE SCHEMA IF NOT EXISTS admin" in text

    def test_creates_backup_table(self) -> None:
        text = _read(self._proc_file())
        assert "CREATE TABLE IF NOT EXISTS admin._pk_backup" in text

    def test_rejects_unknown_action(self) -> None:
        text = _read(self._proc_file())
        assert "Unknown action" in text
        assert "RAISE EXCEPTION" in text

    def test_excludes_system_schemas(self) -> None:
        text = _read(self._proc_file())
        for s in ("pg_catalog", "information_schema", "admin"):
            assert s in text, f"Expected system-schema exclusion to mention {s}"
