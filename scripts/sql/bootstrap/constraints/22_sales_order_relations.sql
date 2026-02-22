-----------------------------------------------------------------------
-- FACT RELATIONS (WITH CHECK)
--  - SalesOrderDetail -> SalesOrderHeader
--  - SalesReturn      -> SalesOrderDetail
-----------------------------------------------------------------------

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- SalesOrderDetail -> SalesOrderHeader
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
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

-----------------------------------------------------------------------
-- SalesReturn -> SalesOrderDetail (natural key)
-- Requires PK/unique key on SalesOrderDetail(SalesOrderNumber, SalesOrderLineNumber)
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'SalesOrderNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'SalesOrderLineNumber') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesReturn_SalesOrderDetail'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesReturn')
)
BEGIN
    ALTER TABLE dbo.SalesReturn WITH CHECK
    ADD CONSTRAINT FK_SalesReturn_SalesOrderDetail
        FOREIGN KEY ([SalesOrderNumber], [SalesOrderLineNumber])
        REFERENCES dbo.SalesOrderDetail ([SalesOrderNumber], [SalesOrderLineNumber]);

    ALTER TABLE dbo.SalesReturn CHECK CONSTRAINT FK_SalesReturn_SalesOrderDetail;
END;
