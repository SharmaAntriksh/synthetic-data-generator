CREATE OR ALTER PROCEDURE dbo.ManageClusteredColumnstoreIndexes
    @Action  varchar(10),
    @Tables  dbo.TableNameList READONLY
AS
BEGIN
    SET NOCOUNT ON;

    IF @Action NOT IN ('CREATE', 'DROP')
    BEGIN
        THROW 50000, 'Invalid @Action. Use CREATE or DROP.', 1;
        RETURN;
    END;

    IF NOT EXISTS (SELECT 1 FROM @Tables)
    BEGIN
        THROW 50001, 'No tables provided.', 1;
        RETURN;
    END;

    -- Guard: if any target table has a CLUSTERED PRIMARY KEY, CCI will clash.
    IF EXISTS (
        SELECT 1
        FROM @Tables t
        WHERE EXISTS (
            SELECT 1
            FROM sys.indexes i
            WHERE i.object_id = OBJECT_ID('dbo.' + t.TableName)
              AND i.type_desc = 'CLUSTERED'
              AND i.is_primary_key = 1
        )
    )
    BEGIN
        THROW 50002, 'One or more target tables have a CLUSTERED PRIMARY KEY. Make PK NONCLUSTERED before applying CCI.', 1;
        RETURN;
    END;

    DECLARE @sql nvarchar(max);

    IF @Action = 'CREATE'
    BEGIN
        SELECT @sql = STRING_AGG(
        N'
IF OBJECT_ID(''dbo.' + TableName + ''') IS NULL
BEGIN
    THROW 50003, ''Table not found: dbo.' + TableName + ''', 1;
END;

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = ''CCI_' + TableName + '''
      AND object_id = OBJECT_ID(''dbo.' + TableName + ''')
)
BEGIN
    CREATE CLUSTERED COLUMNSTORE INDEX ' + QUOTENAME('CCI_' + TableName) + N'
    ON dbo.' + QUOTENAME(TableName) + N';
END;',
        CHAR(10))
        FROM @Tables;
    END
    ELSE
    BEGIN
        SELECT @sql = STRING_AGG(
        N'
IF OBJECT_ID(''dbo.' + TableName + ''') IS NULL
BEGIN
    THROW 50003, ''Table not found: dbo.' + TableName + ''', 1;
END;

IF EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = ''CCI_' + TableName + '''
      AND object_id = OBJECT_ID(''dbo.' + TableName + ''')
)
BEGIN
    DROP INDEX ' + QUOTENAME('CCI_' + TableName) + N'
    ON dbo.' + QUOTENAME(TableName) + N';
END;',
        CHAR(10))
        FROM @Tables;
    END;

    EXEC sys.sp_executesql @sql;
END;
