IF NOT EXISTS (
    SELECT 1
    FROM sys.types
    WHERE is_table_type = 1
      AND name = 'TableNameList'
      AND schema_id = SCHEMA_ID('dbo')
)
BEGIN
    CREATE TYPE dbo.TableNameList AS TABLE
    (
        TableName sysname NOT NULL
    );
END;
