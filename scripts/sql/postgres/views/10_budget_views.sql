/*
  10_budget_views.sql – Pass-through views for the Budget fact tables.
  Hand-translated from scripts/sql/views/10_budget_views.sql.

  Views land in "public" by default; the composer rewrites them when
  cfg.defaults.view_schema is set to a non-default value.
*/

DO $$
BEGIN
    IF to_regclass('"public"."BudgetYearly"') IS NOT NULL THEN
        EXECUTE 'CREATE OR REPLACE VIEW "public"."vw_BudgetYearly" AS SELECT * FROM "public"."BudgetYearly"';
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('"public"."BudgetMonthly"') IS NOT NULL THEN
        EXECUTE 'CREATE OR REPLACE VIEW "public"."vw_BudgetMonthly" AS SELECT * FROM "public"."BudgetMonthly"';
    END IF;
END $$;
