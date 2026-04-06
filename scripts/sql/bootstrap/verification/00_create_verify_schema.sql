SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

-- ============================================================================
-- VERIFICATION SCHEMA — bootstrap
--
-- Creates the [verify] schema and the RunAll dispatcher procedure.
-- Individual verification procs are created by subsequent scripts
-- (10_*, 20_*, etc.) and are auto-discovered by RunAll.
--
-- Result set contract (all procs must return these 5 columns):
--   Category     VARCHAR(50)   — Referential | SCD2 | Distribution | Temporal | ...
--   [Check]      VARCHAR(200)  — short name
--   Description  VARCHAR(500)  — what it means
--   Result       VARCHAR(10)   — PASS | FAIL | INFO
--   ActualValue  VARCHAR(100)  — the measured value (even on PASS)
--
-- Adding a new suite:
--   1. Create a new .sql file in this folder (e.g. 50_my_check.sql)
--   2. Define a proc in the [verify] schema returning the 5 columns above
--   3. RunAll will auto-discover it — no changes needed here
-- ============================================================================

IF SCHEMA_ID('verify') IS NULL
    EXEC('CREATE SCHEMA [verify] AUTHORIZATION [dbo];');
GO


CREATE OR ALTER PROCEDURE [verify].[RunAll]
    @Suite VARCHAR(100) = NULL,
    @Help  BIT = 0
AS
BEGIN
    SET NOCOUNT ON;

    IF @Help = 1
    BEGIN
        SELECT
            p.name           AS Suite,
            'EXEC verify.' + QUOTENAME(p.name) + ';' AS [Usage]
        FROM sys.procedures p
        JOIN sys.schemas s ON s.schema_id = p.schema_id
        WHERE s.name = 'verify'
          AND p.name <> 'RunAll'
        ORDER BY p.name;
        RETURN;
    END

    CREATE TABLE #VR (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(500) NOT NULL
    );

    CREATE TABLE #Final (
        Suite       VARCHAR(100) NOT NULL,
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(500) NOT NULL
    );

    DECLARE @proc_name SYSNAME;
    DECLARE @sql NVARCHAR(500);

    DECLARE proc_cursor CURSOR LOCAL FAST_FORWARD FOR
        SELECT p.name
        FROM sys.procedures p
        JOIN sys.schemas s ON s.schema_id = p.schema_id
        WHERE s.name = 'verify'
          AND p.name <> 'RunAll'
          AND (@Suite IS NULL OR p.name = @Suite)
        ORDER BY p.name;

    OPEN proc_cursor;
    FETCH NEXT FROM proc_cursor INTO @proc_name;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        BEGIN TRY
            TRUNCATE TABLE #VR;

            SET @sql = N'INSERT INTO #VR (Category, [Check], Description, Result, ActualValue) '
                     + N'EXEC verify.' + QUOTENAME(@proc_name);
            EXEC sp_executesql @sql;

            INSERT INTO #Final (Suite, Category, [Check], Description, Result, ActualValue)
            SELECT @proc_name, Category, [Check], Description, Result, ActualValue
            FROM #VR;
        END TRY
        BEGIN CATCH
            INSERT INTO #Final (Suite, Category, [Check], Description, Result, ActualValue)
            VALUES (@proc_name, 'Error', 'EXECUTION ERROR', ERROR_MESSAGE(), 'FAIL', '-');
        END CATCH

        FETCH NEXT FROM proc_cursor INTO @proc_name;
    END

    CLOSE proc_cursor;
    DEALLOCATE proc_cursor;

    -- Detail rows: FAILs first, then by suite/category
    SELECT Suite, Category, [Check], Description, Result, ActualValue
    FROM #Final
    ORDER BY
        CASE Result WHEN 'FAIL' THEN 0 WHEN 'INFO' THEN 2 ELSE 1 END,
        Suite, Category, [Check];

    -- Summary
    SELECT
        COUNT(*)                                              AS TotalChecks,
        SUM(CASE WHEN Result = 'PASS' THEN 1 ELSE 0 END)     AS Passed,
        SUM(CASE WHEN Result = 'FAIL' THEN 1 ELSE 0 END)     AS Failed,
        SUM(CASE WHEN Result = 'INFO' THEN 1 ELSE 0 END)     AS Info,
        CASE
            WHEN SUM(CASE WHEN Result = 'FAIL' THEN 1 ELSE 0 END) = 0
            THEN 'ALL PASSED'
            ELSE CAST(SUM(CASE WHEN Result = 'FAIL' THEN 1 ELSE 0 END) AS VARCHAR)
                 + ' FAILED'
        END                                                    AS Verdict
    FROM #Final;

    DROP TABLE #VR;
    DROP TABLE #Final;
END;
GO
