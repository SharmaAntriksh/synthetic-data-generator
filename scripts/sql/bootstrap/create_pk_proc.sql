/*
DBA utility procedure: DROP or RESTORE primary key and foreign key
constraints on user tables in the current database.

DROP removes all FK constraints first, then all PK constraints,
and reports the space freed per table.

RESTORE re-creates PKs and FKs from the constraint definitions
stored in [admin].[_PK_Backup] during the last DROP.

Targets all user tables by default, or specific tables via a
comma-delimited list.

Compatibility: SQL Server 2016+ (uses STRING_SPLIT, STRING_AGG).

Install into the target database after import:
    sqlcmd -S server -d MyDB -i create_pk_proc.sql

Usage:
    EXEC [admin].[ManagePrimaryKeys] @Help = 1;
*/

IF SCHEMA_ID('admin') IS NULL
    EXEC('CREATE SCHEMA [admin] AUTHORIZATION [dbo];');
GO

-- Backup table for FK/PK definitions (survives across calls)
IF OBJECT_ID('admin._PK_Backup') IS NULL
BEGIN
    CREATE TABLE [admin].[_PK_Backup] (
        id              int IDENTITY(1,1) PRIMARY KEY,
        constraint_type varchar(2)    NOT NULL,  -- 'FK' or 'PK'
        schema_name     sysname       NOT NULL,
        table_name      sysname       NOT NULL,
        constraint_name sysname       NOT NULL,
        definition_sql  nvarchar(max) NOT NULL,
        dropped_at      datetime2     NOT NULL DEFAULT SYSDATETIME()
    );
END;
GO

CREATE OR ALTER PROCEDURE [admin].[ManagePrimaryKeys]
    @Action  varchar(10)   = 'DROP',      -- 'DROP' | 'RESTORE' | 'STATUS'
    @Tables  nvarchar(max) = NULL,        -- comma-delimited table names; NULL = all user tables
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
        PRINT N'  [admin].[ManagePrimaryKeys]';
        PRINT N'  Drop or restore PRIMARY KEY and FOREIGN KEY constraints.';
        PRINT N'  Schema-agnostic (works even if tables are not in dbo).';
        PRINT N'=================================================================';
        PRINT N'';
        PRINT N'PARAMETERS:';
        PRINT N'  @Action  varchar(10)   = ''DROP''  (DROP | RESTORE | STATUS)';
        PRINT N'  @Tables  nvarchar(max) = NULL     comma-delimited table names';
        PRINT N'                                     NULL = all user tables';
        PRINT N'  @Help    bit           = 0';
        PRINT N'';
        PRINT N'EXAMPLES:';
        PRINT N'';
        PRINT N'  -- 1. Drop all PKs and FKs (saves definitions for restore)';
        PRINT N'  EXEC [admin].[ManagePrimaryKeys];';
        PRINT N'';
        PRINT N'  -- 2. Drop PKs on specific tables only';
        PRINT N'  EXEC [admin].[ManagePrimaryKeys]';
        PRINT N'      @Tables = N''Sales, InventorySnapshot'';';
        PRINT N'';
        PRINT N'  -- 3. Restore previously dropped PKs and FKs';
        PRINT N'  EXEC [admin].[ManagePrimaryKeys] @Action = ''RESTORE'';';
        PRINT N'';
        PRINT N'  -- 4. Check current PK/FK status and index sizes';
        PRINT N'  EXEC [admin].[ManagePrimaryKeys] @Action = ''STATUS'';';
        PRINT N'=================================================================';
        RETURN;
    END;

    -----------------------------------------------------------------
    -- Parameter validation
    -----------------------------------------------------------------
    IF @Action NOT IN ('DROP', 'RESTORE', 'STATUS')
        THROW 51000, 'Invalid @Action. Use DROP, RESTORE, or STATUS.', 1;

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
      AND s.name NOT IN ('sys', 'INFORMATION_SCHEMA', 'admin')
      AND (
          @Tables IS NULL
          OR t.name IN (
              SELECT LTRIM(RTRIM(value)) FROM STRING_SPLIT(@Tables, ',')
          )
      );

    DECLARE @target_count int = (SELECT COUNT(*) FROM @Targets);

    -----------------------------------------------------------------
    -- STATUS: show current PK/FK state with sizes
    -----------------------------------------------------------------
    IF @Action = 'STATUS'
    BEGIN
        PRINT CONCAT(N'[STATUS] ', @target_count, N' target table(s)');
        PRINT N'';

        -- PK status
        DECLARE @pk_count int = 0;
        SELECT @pk_count = COUNT(*)
        FROM @Targets tt
        JOIN sys.key_constraints kc ON kc.parent_object_id = tt.object_id
        WHERE kc.type = 'PK';

        PRINT CONCAT(N'  Primary keys: ', @pk_count);

        -- Show PK sizes
        IF @pk_count > 0
        BEGIN
            DECLARE @s_schema sysname, @s_table sysname, @s_pk sysname;
            DECLARE @s_mb decimal(10,1);

            DECLARE cur_status CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
                SELECT tt.schema_name, tt.table_name, kc.name,
                       CAST(SUM(ps.used_page_count) * 8.0 / 1024 AS decimal(10,1))
                FROM @Targets tt
                JOIN sys.key_constraints kc ON kc.parent_object_id = tt.object_id AND kc.type = 'PK'
                JOIN sys.indexes i ON i.object_id = kc.parent_object_id AND i.name = kc.name
                JOIN sys.dm_db_partition_stats ps ON ps.object_id = i.object_id AND ps.index_id = i.index_id
                GROUP BY tt.schema_name, tt.table_name, kc.name
                ORDER BY SUM(ps.used_page_count) DESC;

            OPEN cur_status;
            FETCH NEXT FROM cur_status INTO @s_schema, @s_table, @s_pk, @s_mb;
            WHILE @@FETCH_STATUS = 0
            BEGIN
                PRINT CONCAT(N'    ', QUOTENAME(@s_schema), N'.', QUOTENAME(@s_table),
                             N'  ', @s_pk, N'  ', @s_mb, N' MB');
                FETCH NEXT FROM cur_status INTO @s_schema, @s_table, @s_pk, @s_mb;
            END;
            CLOSE cur_status;
            DEALLOCATE cur_status;
        END;

        -- FK status
        DECLARE @fk_count int = 0;
        SELECT @fk_count = COUNT(*)
        FROM sys.foreign_keys fk
        JOIN @Targets tt ON tt.object_id = fk.parent_object_id;

        PRINT CONCAT(N'  Foreign keys: ', @fk_count);

        -- Backup status
        DECLARE @backup_count int = 0;
        SELECT @backup_count = COUNT(*) FROM [admin].[_PK_Backup];
        IF @backup_count > 0
            PRINT CONCAT(N'  Backed-up constraints: ', @backup_count, N' (available for RESTORE)');

        RETURN;
    END;

    -----------------------------------------------------------------
    -- DROP: save definitions, then drop FKs and PKs
    -----------------------------------------------------------------
    IF @Action = 'DROP'
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM @Targets tt
            JOIN sys.key_constraints kc ON kc.parent_object_id = tt.object_id
            WHERE kc.type = 'PK'
        )
        BEGIN
            PRINT N'[OK] No primary keys found on target tables. Nothing to drop.';
            RETURN;
        END;

        PRINT CONCAT(N'[OK] Action=DROP | Targets=', @target_count,
                     CASE WHEN @Tables IS NOT NULL
                          THEN N' | Tables=' + @Tables
                          ELSE N' | Tables=ALL'
                     END);

        -- Clear old backup for target tables before saving new definitions
        DELETE b FROM [admin].[_PK_Backup] b
        WHERE EXISTS (
            SELECT 1 FROM @Targets tt
            WHERE tt.schema_name = b.schema_name AND tt.table_name = b.table_name
        );

        -- Save FK definitions
        DECLARE @fk_schema sysname, @fk_table sysname, @fk_name sysname;
        DECLARE @fk_ref_schema sysname, @fk_ref_table sysname;
        DECLARE @fk_object_id int;
        DECLARE @fk_cols nvarchar(max), @fk_ref_cols nvarchar(max);
        DECLARE @fk_sql nvarchar(max);
        DECLARE @fk_on_delete nvarchar(60), @fk_on_update nvarchar(60);

        DECLARE cur_fk_save CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
            SELECT
                s.name, t.name, fk.name, fk.object_id,
                rs.name, rt.name,
                fk.delete_referential_action_desc,
                fk.update_referential_action_desc
            FROM sys.foreign_keys fk
            JOIN sys.tables t ON t.object_id = fk.parent_object_id
            JOIN sys.schemas s ON s.schema_id = t.schema_id
            JOIN sys.tables rt ON rt.object_id = fk.referenced_object_id
            JOIN sys.schemas rs ON rs.schema_id = rt.schema_id
            WHERE EXISTS (SELECT 1 FROM @Targets tt WHERE tt.object_id = fk.parent_object_id);

        OPEN cur_fk_save;
        FETCH NEXT FROM cur_fk_save INTO @fk_schema, @fk_table, @fk_name, @fk_object_id,
                                         @fk_ref_schema, @fk_ref_table,
                                         @fk_on_delete, @fk_on_update;

        WHILE @@FETCH_STATUS = 0
        BEGIN
            -- Build column lists using fk.object_id directly (OBJECT_ID on constraint names is unreliable)
            SELECT @fk_cols = STRING_AGG(QUOTENAME(COL_NAME(fkc.parent_object_id, fkc.parent_column_id)), N', ')
                              WITHIN GROUP (ORDER BY fkc.constraint_column_id),
                   @fk_ref_cols = STRING_AGG(QUOTENAME(COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id)), N', ')
                                  WITHIN GROUP (ORDER BY fkc.constraint_column_id)
            FROM sys.foreign_key_columns fkc
            WHERE fkc.constraint_object_id = @fk_object_id;

            SET @fk_sql = N'ALTER TABLE ' + QUOTENAME(@fk_schema) + N'.' + QUOTENAME(@fk_table) +
                           N' ADD CONSTRAINT ' + QUOTENAME(@fk_name) +
                           N' FOREIGN KEY (' + @fk_cols + N') REFERENCES ' +
                           QUOTENAME(@fk_ref_schema) + N'.' + QUOTENAME(@fk_ref_table) +
                           N' (' + @fk_ref_cols + N')';

            -- Preserve ON DELETE / ON UPDATE referential actions
            IF @fk_on_delete <> 'NO_ACTION'
                SET @fk_sql += N' ON DELETE ' + REPLACE(@fk_on_delete, N'_', N' ');
            IF @fk_on_update <> 'NO_ACTION'
                SET @fk_sql += N' ON UPDATE ' + REPLACE(@fk_on_update, N'_', N' ');

            SET @fk_sql += N';';

            INSERT INTO [admin].[_PK_Backup] (constraint_type, schema_name, table_name, constraint_name, definition_sql)
            VALUES ('FK', @fk_schema, @fk_table, @fk_name, @fk_sql);

            FETCH NEXT FROM cur_fk_save INTO @fk_schema, @fk_table, @fk_name, @fk_object_id,
                                             @fk_ref_schema, @fk_ref_table,
                                             @fk_on_delete, @fk_on_update;
        END;
        CLOSE cur_fk_save;
        DEALLOCATE cur_fk_save;

        -- Save PK definitions (preserving CLUSTERED vs NONCLUSTERED type)
        DECLARE @pk_schema sysname, @pk_table sysname, @pk_name sysname;
        DECLARE @pk_cols_str nvarchar(max), @pk_sql nvarchar(max);
        DECLARE @pk_type tinyint;  -- 1 = CLUSTERED, 2 = NONCLUSTERED

        DECLARE cur_pk_save CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
            SELECT tt.schema_name, tt.table_name, kc.name, i.type
            FROM @Targets tt
            JOIN sys.key_constraints kc ON kc.parent_object_id = tt.object_id AND kc.type = 'PK'
            JOIN sys.indexes i ON i.object_id = kc.parent_object_id AND i.name = kc.name;

        OPEN cur_pk_save;
        FETCH NEXT FROM cur_pk_save INTO @pk_schema, @pk_table, @pk_name, @pk_type;

        WHILE @@FETCH_STATUS = 0
        BEGIN
            SELECT @pk_cols_str = STRING_AGG(QUOTENAME(COL_NAME(ic.object_id, ic.column_id)), N', ')
                                  WITHIN GROUP (ORDER BY ic.key_ordinal)
            FROM sys.index_columns ic
            JOIN sys.indexes i ON i.object_id = ic.object_id AND i.index_id = ic.index_id
            WHERE i.object_id = OBJECT_ID(QUOTENAME(@pk_schema) + N'.' + QUOTENAME(@pk_table))
              AND i.name = @pk_name
              AND ic.is_included_column = 0;

            SET @pk_sql = N'ALTER TABLE ' + QUOTENAME(@pk_schema) + N'.' + QUOTENAME(@pk_table) +
                           N' ADD CONSTRAINT ' + QUOTENAME(@pk_name) +
                           N' PRIMARY KEY ' +
                           CASE WHEN @pk_type = 1 THEN N'CLUSTERED' ELSE N'NONCLUSTERED' END +
                           N' (' + @pk_cols_str + N');';

            INSERT INTO [admin].[_PK_Backup] (constraint_type, schema_name, table_name, constraint_name, definition_sql)
            VALUES ('PK', @pk_schema, @pk_table, @pk_name, @pk_sql);

            FETCH NEXT FROM cur_pk_save INTO @pk_schema, @pk_table, @pk_name, @pk_type;
        END;
        CLOSE cur_pk_save;
        DEALLOCATE cur_pk_save;

        -- Drop FKs first
        DECLARE @sql nvarchar(max);
        DECLARE @drop_tbl nvarchar(517), @drop_name sysname;
        DECLARE @fk_dropped int = 0;

        DECLARE cur_fk_drop CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
            SELECT QUOTENAME(s.name) + N'.' + QUOTENAME(t.name), fk.name
            FROM sys.foreign_keys fk
            JOIN sys.tables t ON t.object_id = fk.parent_object_id
            JOIN sys.schemas s ON s.schema_id = t.schema_id
            WHERE EXISTS (SELECT 1 FROM @Targets tt WHERE tt.object_id = fk.parent_object_id);

        OPEN cur_fk_drop;
        FETCH NEXT FROM cur_fk_drop INTO @drop_tbl, @drop_name;
        WHILE @@FETCH_STATUS = 0
        BEGIN
            SET @sql = N'ALTER TABLE ' + @drop_tbl + N' DROP CONSTRAINT ' + QUOTENAME(@drop_name) + N';';
            EXEC sys.sp_executesql @sql;
            SET @fk_dropped += 1;
            FETCH NEXT FROM cur_fk_drop INTO @drop_tbl, @drop_name;
        END;
        CLOSE cur_fk_drop;
        DEALLOCATE cur_fk_drop;

        IF @fk_dropped > 0
            PRINT CONCAT(N'  [DROP] ', @fk_dropped, N' foreign key constraint(s)');

        -- Measure PK sizes, then drop
        DECLARE @pk_dropped int = 0;
        DECLARE @total_freed decimal(10,1) = 0;
        DECLARE @d_schema sysname, @d_table sysname, @d_pk sysname;
        DECLARE @d_mb decimal(10,1);

        DECLARE cur_pk_drop CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
            SELECT tt.schema_name, tt.table_name, kc.name,
                   CAST(SUM(ps.used_page_count) * 8.0 / 1024 AS decimal(10,1))
            FROM @Targets tt
            JOIN sys.key_constraints kc ON kc.parent_object_id = tt.object_id AND kc.type = 'PK'
            JOIN sys.indexes i ON i.object_id = kc.parent_object_id AND i.name = kc.name
            JOIN sys.dm_db_partition_stats ps ON ps.object_id = i.object_id AND ps.index_id = i.index_id
            GROUP BY tt.schema_name, tt.table_name, kc.name
            ORDER BY SUM(ps.used_page_count) DESC;

        OPEN cur_pk_drop;
        FETCH NEXT FROM cur_pk_drop INTO @d_schema, @d_table, @d_pk, @d_mb;
        WHILE @@FETCH_STATUS = 0
        BEGIN
            SET @sql = N'ALTER TABLE ' + QUOTENAME(@d_schema) + N'.' + QUOTENAME(@d_table) +
                       N' DROP CONSTRAINT ' + QUOTENAME(@d_pk) + N';';
            EXEC sys.sp_executesql @sql;
            SET @pk_dropped += 1;
            SET @total_freed += @d_mb;
            IF @d_mb >= 1.0
                PRINT CONCAT(N'  [DROP] ', QUOTENAME(@d_schema), N'.', QUOTENAME(@d_table),
                             N'  ', @d_pk, N'  freed ', @d_mb, N' MB');
            FETCH NEXT FROM cur_pk_drop INTO @d_schema, @d_table, @d_pk, @d_mb;
        END;
        CLOSE cur_pk_drop;
        DEALLOCATE cur_pk_drop;

        PRINT N'';
        PRINT CONCAT(N'[OK] Dropped ', @pk_dropped, N' PK(s), ', @fk_dropped, N' FK(s) — freed ',
                     @total_freed, N' MB');
        PRINT N'     Definitions saved to [admin].[_PK_Backup] for RESTORE.';
        RETURN;
    END;

    -----------------------------------------------------------------
    -- RESTORE: re-create PKs and FKs from backup
    -----------------------------------------------------------------
    IF @Action = 'RESTORE'
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM [admin].[_PK_Backup])
        BEGIN
            PRINT N'[OK] No backed-up constraints found. Nothing to restore.';
            RETURN;
        END;

        DECLARE @restored int = 0, @restore_errors int = 0;
        DECLARE @r_type varchar(2), @r_sql nvarchar(max), @r_name sysname, @r_id int;

        -- Restore PKs first (FKs reference them)
        DECLARE cur_restore CURSOR LOCAL FAST_FORWARD READ_ONLY FOR
            SELECT id, constraint_type, constraint_name, definition_sql
            FROM [admin].[_PK_Backup]
            WHERE (
                @Tables IS NULL
                OR table_name IN (SELECT LTRIM(RTRIM(value)) FROM STRING_SPLIT(@Tables, ','))
            )
            ORDER BY
                CASE constraint_type WHEN 'PK' THEN 1 WHEN 'FK' THEN 2 END,
                id;

        OPEN cur_restore;
        FETCH NEXT FROM cur_restore INTO @r_id, @r_type, @r_name, @r_sql;
        WHILE @@FETCH_STATUS = 0
        BEGIN
            BEGIN TRY
                EXEC sys.sp_executesql @r_sql;
                SET @restored += 1;
                PRINT CONCAT(N'  [RESTORE] ', @r_type, N': ', @r_name);
                DELETE FROM [admin].[_PK_Backup] WHERE id = @r_id;
            END TRY
            BEGIN CATCH
                SET @restore_errors += 1;
                PRINT CONCAT(N'  [FAIL]    ', @r_type, N': ', @r_name,
                             N' — ', ERROR_MESSAGE());
            END CATCH;
            FETCH NEXT FROM cur_restore INTO @r_id, @r_type, @r_name, @r_sql;
        END;
        CLOSE cur_restore;
        DEALLOCATE cur_restore;

        PRINT N'';
        PRINT CONCAT(N'[OK] Restored ', @restored, N' constraint(s)',
                     CASE WHEN @restore_errors > 0
                          THEN CONCAT(N', ', @restore_errors, N' failed')
                          ELSE N'' END);
        RETURN;
    END;
END;
GO
