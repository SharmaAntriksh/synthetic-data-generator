-----------------------------------------------------------------------
-- FACT: Budget tables (PK + FOREIGN KEYS WITH CHECK)
-- Aligned to src/utils/static_schemas.py
--
-- Tables:
--   BudgetYearly          - annual grain (Country, Category, BudgetYear, Scenario)
--   BudgetMonthly         - monthly grain (Country, Category, BudgetYear, BudgetMonthStart, Scenario)
-----------------------------------------------------------------------

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- BudgetYearly: composite PK (annual grain)
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.BudgetYearly', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_BudgetYearly'
      AND parent_object_id = OBJECT_ID(N'dbo.BudgetYearly')
)
BEGIN
    ALTER TABLE dbo.BudgetYearly
    ADD CONSTRAINT PK_BudgetYearly
        PRIMARY KEY NONCLUSTERED ([Country], [Category], [BudgetYear], [Scenario]);
END;

-----------------------------------------------------------------------
-- BudgetMonthly: composite PK (monthly grain)
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.BudgetMonthly', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_BudgetMonthly'
      AND parent_object_id = OBJECT_ID(N'dbo.BudgetMonthly')
)
BEGIN
    ALTER TABLE dbo.BudgetMonthly
    ADD CONSTRAINT PK_BudgetMonthly
        PRIMARY KEY NONCLUSTERED ([Country], [Category], [BudgetYear], [BudgetMonthStart], [Scenario]);
END;
