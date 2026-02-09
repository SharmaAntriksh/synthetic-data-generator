-----------------------------------------------------------------------
-- FACT RELATION: SalesOrderDetail -> SalesOrderHeader (WITH CHECK)
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_SalesOrderHeader'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_SalesOrderHeader
        FOREIGN KEY ([SalesOrderNumber])
        REFERENCES dbo.SalesOrderHeader ([SalesOrderNumber]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_SalesOrderHeader;
END;
