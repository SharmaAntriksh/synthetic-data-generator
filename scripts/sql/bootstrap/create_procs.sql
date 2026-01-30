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

    DECLARE @sql nvarchar(max);

    IF @Action = 'CREATE'
    BEGIN
        SELECT @sql = STRING_AGG(
        N'
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = ''CCI_' + TableName + '''
      AND object_id = OBJECT_ID(''dbo.' + TableName + ''')
)
BEGIN
    CREATE CLUSTERED COLUMNSTORE INDEX CCI_' + TableName + '
    ON dbo.' + QUOTENAME(TableName) + ';
END;',
        CHAR(10))
        FROM @Tables;
    END
    ELSE
    BEGIN
        SELECT @sql = STRING_AGG(
        N'
IF EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = ''CCI_' + TableName + '''
      AND object_id = OBJECT_ID(''dbo.' + TableName + ''')
)
BEGIN
    DROP INDEX CCI_' + TableName + '
    ON dbo.' + QUOTENAME(TableName) + ';
END;',
        CHAR(10))
        FROM @Tables;
    END;

    EXEC sys.sp_executesql @sql;
END;
