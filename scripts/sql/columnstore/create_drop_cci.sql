/*
Apply / drop CLUSTERED COLUMNSTORE indexes across user tables.
- No CONCAT()
- No STRING_AGG()
- Schema-agnostic (works even if tables are not dbo)

Set:
  @Action = 'CREATE' or 'DROP'
  @Mode   = 'ALL' or 'SALES_ONLY'
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

DECLARE @Action varchar(10) = 'CREATE';     -- 'CREATE' | 'DROP'
DECLARE @Mode   varchar(20) = 'ALL';        -- 'ALL' | 'SALES_ONLY'

IF @Action NOT IN ('CREATE', 'DROP')
BEGIN
    THROW 51000, 'Invalid @Action. Use CREATE or DROP.', 1;
END;

IF @Mode NOT IN ('ALL', 'SALES_ONLY')
BEGIN
    THROW 51001, 'Invalid @Mode. Use ALL or SALES_ONLY.', 1;
END;

DECLARE @Targets TABLE (
    schema_name sysname NOT NULL,
    table_name  sysname NOT NULL,
    object_id   int     NOT NULL
);

INSERT INTO @Targets(schema_name, table_name, object_id)
SELECT s.name, t.name, t.object_id
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE t.is_ms_shipped = 0
  AND s.name NOT IN ('sys', 'INFORMATION_SCHEMA')
  AND (@Mode = 'ALL' OR t.name = 'Sales');

IF NOT EXISTS (SELECT 1 FROM @Targets)
BEGIN
    THROW 51002, 'No target tables resolved for CCI apply.', 1;
END;

-- If any target table has a CLUSTERED rowstore index, CCI creation is blocked.
-- (We fail fast with a clear error.)
IF @Action = 'CREATE'
AND EXISTS (
    SELECT 1
    FROM @Targets tt
    JOIN sys.indexes i ON i.object_id = tt.object_id
    WHERE i.type = 1   -- CLUSTERED (rowstore)
)
BEGIN
    -- Build a readable list without STRING_AGG/CONCAT
    DECLARE @blocked nvarchar(2048) = N'';
    DECLARE @s sysname, @t sysname;

    DECLARE cur_block CURSOR FAST_FORWARD FOR
    SELECT tt.schema_name, tt.table_name
    FROM @Targets tt
    JOIN sys.indexes i ON i.object_id = tt.object_id
    WHERE i.type = 1
    ORDER BY tt.schema_name, tt.table_name;

    OPEN cur_block;
    FETCH NEXT FROM cur_block INTO @s, @t;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        IF LEN(@blocked) > 0 SET @blocked = @blocked + N', ';
        SET @blocked = @blocked + @s + N'.' + @t;
        FETCH NEXT FROM cur_block INTO @s, @t;
    END

    CLOSE cur_block;
    DEALLOCATE cur_block;

    DECLARE @msg nvarchar(2048) =
        N'Blocked: clustered rowstore index exists on: ' + @blocked +
        N'. Convert PKs to NONCLUSTERED or drop clustered indexes before applying CCI.';

    THROW 51003, @msg, 1;
END;

DECLARE @schema sysname, @table sysname, @obj nvarchar(517), @sql nvarchar(max), @ix sysname;
DECLARE cur CURSOR FAST_FORWARD FOR
SELECT schema_name, table_name
FROM @Targets
ORDER BY schema_name, table_name;

OPEN cur;
FETCH NEXT FROM cur INTO @schema, @table;

WHILE @@FETCH_STATUS = 0
BEGIN
    SET @obj = QUOTENAME(@schema) + N'.' + QUOTENAME(@table);

    IF @Action = 'CREATE'
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM sys.indexes
            WHERE object_id = OBJECT_ID(@obj)
              AND type = 5 -- CLUSTERED COLUMNSTORE
        )
        BEGIN
            -- Use a consistent per-table name. Index names are scoped per table.
            SET @sql = N'CREATE CLUSTERED COLUMNSTORE INDEX [CCI] ON ' + @obj + N';';
            EXEC sys.sp_executesql @sql;
        END
    END
    ELSE
    BEGIN
        SELECT TOP (1) @ix = name
        FROM sys.indexes
        WHERE object_id = OBJECT_ID(@obj)
          AND type = 5;

        IF @ix IS NOT NULL
        BEGIN
            SET @sql = N'DROP INDEX ' + QUOTENAME(@ix) + N' ON ' + @obj + N';';
            EXEC sys.sp_executesql @sql;
        END
    END

    FETCH NEXT FROM cur INTO @schema, @table;
END

CLOSE cur;
DEALLOCATE cur;

-- Verification: ensure something happened (prevents silent no-op)
DECLARE @cci_count int =
(
    SELECT COUNT(*)
    FROM sys.indexes i
    WHERE i.type = 5
      AND i.object_id IN (SELECT object_id FROM @Targets)
);

IF @Action = 'CREATE' AND @cci_count = 0
BEGIN
    THROW 51004, 'CCI apply completed but created 0 CCIs (unexpected).', 1;
END;

SELECT @cci_count AS cci_tables_with_columnstore;
