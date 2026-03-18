/*
Create CLUSTERED COLUMNSTORE indexes on all user tables.

This script is packaged into the generated output folder and executed
by the automated SQL Server import pipeline.  It is CREATE-only and
has no configurable options — the stored procedure
[admin].[ManageColumnstoreIndexes] (installed automatically) provides
interactive CREATE/DROP/REBUILD with table-level targeting.

Compatibility: SQL Server 2016+ (no STRING_AGG, no CONCAT).

Batch-separated (GO) so each phase is independently visible to pyodbc
and errors are never deferred across result sets.
Uses #temp tables (not @table variables) so state survives across GO.
*/

-- ===================================================================
-- BATCH 1: Resolve targets and pre-flight checks
-- ===================================================================
SET NOCOUNT ON;
SET XACT_ABORT ON;

IF OBJECT_ID('tempdb..#CCI_Targets') IS NOT NULL DROP TABLE #CCI_Targets;
CREATE TABLE #CCI_Targets (
    schema_name sysname NOT NULL,
    table_name  sysname NOT NULL,
    object_id   int     NOT NULL
);

INSERT INTO #CCI_Targets(schema_name, table_name, object_id)
SELECT s.name, t.name, t.object_id
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE t.is_ms_shipped = 0
  AND s.name NOT IN ('sys', 'INFORMATION_SCHEMA');

IF NOT EXISTS (SELECT 1 FROM #CCI_Targets)
    THROW 51000, 'No user tables found in the database.', 1;

DECLARE @target_count int = (SELECT COUNT(*) FROM #CCI_Targets);
PRINT '-- Resolved ' + CAST(@target_count AS varchar(10)) + ' target table(s)';

-- Block if any target has a CLUSTERED rowstore index (e.g. clustered PK).
IF EXISTS (
    SELECT 1
    FROM #CCI_Targets tt
    JOIN sys.indexes i ON i.object_id = tt.object_id
    WHERE i.type = 1   -- CLUSTERED (rowstore)
)
BEGIN
    DECLARE @blocked nvarchar(2048) = N'';
    DECLARE @s sysname, @t sysname;

    DECLARE cur_block CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
    SELECT tt.schema_name, tt.table_name
    FROM #CCI_Targets tt
    JOIN sys.indexes i ON i.object_id = tt.object_id
    WHERE i.type = 1
    ORDER BY tt.schema_name, tt.table_name;

    OPEN cur_block;
    FETCH NEXT FROM cur_block INTO @s, @t;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        IF LEN(@blocked) > 0 SET @blocked = @blocked + N', ';
        SET @blocked = @blocked + QUOTENAME(@s) + N'.' + QUOTENAME(@t);
        FETCH NEXT FROM cur_block INTO @s, @t;
    END

    CLOSE cur_block;
    DEALLOCATE cur_block;

    DECLARE @msg nvarchar(2048) =
        N'Blocked: clustered rowstore index exists on: ' + @blocked +
        N'. Convert PKs to NONCLUSTERED or drop clustered indexes before applying CCI.';

    THROW 51001, @msg, 1;
END;

PRINT '-- Pre-flight checks passed';
GO

-- ===================================================================
-- BATCH 2: Create CCIs
-- ===================================================================
SET NOCOUNT ON;
SET XACT_ABORT ON;

DECLARE @schema   sysname,
        @table    sysname,
        @obj      nvarchar(517),
        @sql      nvarchar(max);

IF OBJECT_ID('tempdb..#CCI_Counts') IS NOT NULL DROP TABLE #CCI_Counts;
CREATE TABLE #CCI_Counts (created int, skipped int);
INSERT INTO #CCI_Counts VALUES (0, 0);

DECLARE cur CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
SELECT schema_name, table_name
FROM #CCI_Targets
ORDER BY schema_name, table_name;

OPEN cur;
FETCH NEXT FROM cur INTO @schema, @table;

WHILE @@FETCH_STATUS = 0
BEGIN
    SET @obj = QUOTENAME(@schema) + N'.' + QUOTENAME(@table);

    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE object_id = OBJECT_ID(@obj)
          AND type = 5 -- CLUSTERED COLUMNSTORE
    )
    BEGIN
        SET @sql = N'CREATE CLUSTERED COLUMNSTORE INDEX [CCI] ON ' + @obj + N';';
        EXEC sys.sp_executesql @sql;
        UPDATE #CCI_Counts SET created = created + 1;
        PRINT '  [CREATE] ' + @obj;
    END
    ELSE
    BEGIN
        UPDATE #CCI_Counts SET skipped = skipped + 1;
        PRINT '  [SKIP]   ' + @obj + ' (CCI already exists)';
    END

    FETCH NEXT FROM cur INTO @schema, @table;
END

CLOSE cur;
DEALLOCATE cur;
GO

-- ===================================================================
-- BATCH 3: Verification
-- ===================================================================
SET NOCOUNT ON;

DECLARE @cci_count int =
(
    SELECT COUNT(*)
    FROM sys.indexes i
    WHERE i.type = 5
      AND i.object_id IN (SELECT object_id FROM #CCI_Targets)
);

DECLARE @target_count3 int = (SELECT COUNT(*) FROM #CCI_Targets);
DECLARE @created3 int, @skipped3 int;
SELECT @created3 = created, @skipped3 = skipped FROM #CCI_Counts;

IF @cci_count = 0
BEGIN
    DECLARE @warn_msg nvarchar(256) =
        N'CCI apply completed but 0 of ' + CAST(@target_count3 AS nvarchar(10)) + N' target tables have CCIs.';
    THROW 51002, @warn_msg, 1;
END

PRINT '';
PRINT '-- Complete: '
    + CAST(@created3 AS varchar(10)) + ' created, '
    + CAST(@skipped3 AS varchar(10)) + ' skipped (already existed), '
    + CAST(@cci_count AS varchar(10)) + '/' + CAST(@target_count3 AS varchar(10)) + ' tables now have CCI';

-- Cleanup
DROP TABLE IF EXISTS #CCI_Targets;
DROP TABLE IF EXISTS #CCI_Counts;
GO
