/*
DBA utility procedure: CREATE, DROP, or REBUILD clustered columnstore
indexes on user tables in the current database.

Targets all user tables by default, or specific tables via a
comma-delimited list.

Compatibility: SQL Server 2016+ (uses STRING_SPLIT).

Install into the target database after import:
    sqlcmd -S server -d MyDB -i create_cci_proc.sql

Usage:
    EXEC [admin].[ManageColumnstoreIndexes] @Help = 1;
*/

IF SCHEMA_ID('admin') IS NULL
    EXEC('CREATE SCHEMA [admin] AUTHORIZATION [dbo];');
GO

CREATE OR ALTER PROCEDURE [admin].[ManageColumnstoreIndexes]
    @Action  varchar(10)   = 'CREATE',   -- 'CREATE' | 'DROP' | 'REBUILD'
    @Tables  nvarchar(max) = NULL,       -- comma-delimited table names; NULL = all user tables
    @Help    bit           = 0
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    -----------------------------------------------------------------
    -- Help / usage
    -----------------------------------------------------------------
    IF @Help = 1
    BEGIN
        PRINT N'=================================================================';
        PRINT N'  [admin].[ManageColumnstoreIndexes]';
        PRINT N'  Create, drop, or rebuild CLUSTERED COLUMNSTORE indexes.';
        PRINT N'  Schema-agnostic (works even if tables are not in dbo).';
        PRINT N'=================================================================';
        PRINT N'';
        PRINT N'PARAMETERS:';
        PRINT N'  @Action  varchar(10)   = ''CREATE''  (CREATE | DROP | REBUILD)';
        PRINT N'  @Tables  nvarchar(max) = NULL       comma-delimited table names';
        PRINT N'                                       NULL = all user tables';
        PRINT N'  @Help    bit           = 0';
        PRINT N'';
        PRINT N'EXAMPLES:';
        PRINT N'';
        PRINT N'  -- 1. Create CCI on ALL user tables';
        PRINT N'  EXEC [admin].[ManageColumnstoreIndexes];';
        PRINT N'';
        PRINT N'  -- 2. Create CCI on specific tables';
        PRINT N'  EXEC [admin].[ManageColumnstoreIndexes]';
        PRINT N'      @Tables = N''Sales, SalesReturn, Complaints'';';
        PRINT N'';
        PRINT N'  -- 3. Drop CCI from specific tables';
        PRINT N'  EXEC [admin].[ManageColumnstoreIndexes]';
        PRINT N'      @Action = ''DROP'', @Tables = N''Sales, SalesReturn'';';
        PRINT N'';
        PRINT N'  -- 4. Drop ALL CCIs';
        PRINT N'  EXEC [admin].[ManageColumnstoreIndexes] @Action = ''DROP'';';
        PRINT N'';
        PRINT N'  -- 5. Rebuild CCIs on all tables (recompress + clean deleted rows)';
        PRINT N'  EXEC [admin].[ManageColumnstoreIndexes] @Action = ''REBUILD'';';
        PRINT N'';
        PRINT N'  -- 6. Rebuild CCI on specific tables';
        PRINT N'  EXEC [admin].[ManageColumnstoreIndexes]';
        PRINT N'      @Action = ''REBUILD'', @Tables = N''Sales'';';
        PRINT N'=================================================================';
        RETURN;
    END;

    -----------------------------------------------------------------
    -- Parameter validation
    -----------------------------------------------------------------
    IF @Action NOT IN ('CREATE', 'DROP', 'REBUILD')
        THROW 51000, 'Invalid @Action. Use CREATE, DROP, or REBUILD.', 1;

    -----------------------------------------------------------------
    -- Resolve target tables
    -----------------------------------------------------------------
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
      AND (
          @Tables IS NULL
          OR t.name IN (
              SELECT LTRIM(RTRIM(value)) FROM STRING_SPLIT(@Tables, ',')
          )
      );

    IF NOT EXISTS (SELECT 1 FROM @Targets)
        THROW 51001, 'No target tables resolved. Check @Tables and available tables.', 1;

    DECLARE @target_count int = (SELECT COUNT(*) FROM @Targets);

    PRINT CONCAT(N'[OK] Action=', @Action,
                 N' | Targets=', @target_count,
                 CASE WHEN @Tables IS NOT NULL
                      THEN N' | Tables=' + @Tables
                      ELSE N' | Tables=ALL'
                 END);

    -----------------------------------------------------------------
    -- Pre-flight: block CREATE if clustered rowstore index exists
    -----------------------------------------------------------------
    IF @Action = 'CREATE'
    BEGIN
        DECLARE @blocked nvarchar(max);

        SELECT @blocked = STRING_AGG(
            QUOTENAME(tt.schema_name) + N'.' + QUOTENAME(tt.table_name), N', '
        )
        FROM @Targets tt
        JOIN sys.indexes i ON i.object_id = tt.object_id
        WHERE i.type = 1;   -- CLUSTERED (rowstore)

        IF @blocked IS NOT NULL
        BEGIN
            DECLARE @block_msg nvarchar(2048) =
                N'Blocked: clustered rowstore index exists on: ' + @blocked +
                N'. Convert PKs to NONCLUSTERED or drop clustered indexes before applying CCI.';
            THROW 51002, @block_msg, 1;
        END;
    END;

    -----------------------------------------------------------------
    -- Apply / drop CCIs
    -----------------------------------------------------------------
    DECLARE @schema sysname, @table sysname, @obj nvarchar(517),
            @sql nvarchar(max), @ix sysname;
    DECLARE @created int = 0, @dropped int = 0, @rebuilt int = 0, @skipped int = 0;

    DECLARE cur CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
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
                SELECT 1 FROM sys.indexes
                WHERE object_id = OBJECT_ID(@obj) AND type = 5
            )
            BEGIN
                SET @sql = N'CREATE CLUSTERED COLUMNSTORE INDEX [CCI] ON ' + @obj + N';';
                EXEC sys.sp_executesql @sql;
                SET @created += 1;
                PRINT CONCAT(N'  [CREATE] ', @obj);
            END
            ELSE
            BEGIN
                SET @skipped += 1;
                PRINT CONCAT(N'  [SKIP]   ', @obj, N' (CCI already exists)');
            END;
        END
        ELSE IF @Action = 'DROP'
        BEGIN
            SET @ix = NULL;

            SELECT TOP (1) @ix = name
            FROM sys.indexes
            WHERE object_id = OBJECT_ID(@obj) AND type = 5;

            IF @ix IS NOT NULL
            BEGIN
                SET @sql = N'DROP INDEX ' + QUOTENAME(@ix) + N' ON ' + @obj + N';';
                EXEC sys.sp_executesql @sql;
                SET @dropped += 1;
                PRINT CONCAT(N'  [DROP]   ', QUOTENAME(@ix), N' ON ', @obj);
            END
            ELSE
            BEGIN
                SET @skipped += 1;
                PRINT CONCAT(N'  [SKIP]   ', @obj, N' (no CCI found)');
            END;
        END
        ELSE -- REBUILD
        BEGIN
            SET @ix = NULL;

            SELECT TOP (1) @ix = name
            FROM sys.indexes
            WHERE object_id = OBJECT_ID(@obj) AND type = 5;

            IF @ix IS NOT NULL
            BEGIN
                SET @sql = N'ALTER INDEX ' + QUOTENAME(@ix) + N' ON ' + @obj + N' REBUILD;';
                EXEC sys.sp_executesql @sql;
                SET @rebuilt += 1;
                PRINT CONCAT(N'  [REBUILD] ', QUOTENAME(@ix), N' ON ', @obj);
            END
            ELSE
            BEGIN
                SET @skipped += 1;
                PRINT CONCAT(N'  [SKIP]    ', @obj, N' (no CCI to rebuild)');
            END;
        END;

        FETCH NEXT FROM cur INTO @schema, @table;
    END;

    CLOSE cur;
    DEALLOCATE cur;

    -----------------------------------------------------------------
    -- Verification
    -----------------------------------------------------------------
    DECLARE @cci_count int = (
        SELECT COUNT(*)
        FROM sys.indexes i
        WHERE i.type = 5
          AND i.object_id IN (SELECT object_id FROM @Targets)
    );

    IF @Action = 'CREATE' AND @cci_count = 0
    BEGIN
        DECLARE @warn_msg nvarchar(256) =
            N'CCI apply completed but 0 of ' + CAST(@target_count AS nvarchar(10)) + N' target tables have CCIs.';
        THROW 51003, @warn_msg, 1;
    END;

    IF @Action = 'DROP' AND @cci_count > 0
    BEGIN
        DECLARE @drop_warn nvarchar(256) =
            N'DROP completed but ' + CAST(@cci_count AS nvarchar(10))
            + N' CCI(s) still remain on target tables.';
        THROW 51004, @drop_warn, 1;
    END;

    IF @Action = 'REBUILD' AND @rebuilt = 0 AND @skipped = @target_count
    BEGIN
        DECLARE @rebuild_warn nvarchar(256) =
            N'REBUILD skipped all ' + CAST(@target_count AS nvarchar(10))
            + N' target tables (none have CCIs).';
        THROW 51005, @rebuild_warn, 1;
    END;

    PRINT N'';
    IF @Action = 'CREATE'
        PRINT CONCAT(N'[OK] ', @created, N' created, ', @skipped, N' skipped, ',
                     @cci_count, N'/', @target_count, N' tables now have CCI');
    ELSE IF @Action = 'DROP'
        PRINT CONCAT(N'[OK] ', @dropped, N' dropped, ', @skipped, N' skipped');
    ELSE
        PRINT CONCAT(N'[OK] ', @rebuilt, N' rebuilt, ', @skipped, N' skipped');
END;
GO
