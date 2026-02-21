-----------------------------------------------------------------------
-- FACT RELATIONS (WITH CHECK)
--  - SalesOrderDetail -> SalesOrderHeader
--  - SalesReturn      -> SalesOrderDetail
-----------------------------------------------------------------------

-----------------------------------------------------------------------
-- SalesOrderDetail -> SalesOrderHeader
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'SalesOrderNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'SalesOrderNumber') IS NOT NULL
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
-- SalesReturn -> SalesOrderDetail (order line grain)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'SalesOrderNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'SalesOrderLineNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'SalesOrderNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'SalesOrderLineNumber') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesReturn_SalesOrderDetail'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesReturn')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns pc
    JOIN sys.columns rc
      ON rc.object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
     AND rc.name = N'SalesOrderNumber'
    WHERE pc.object_id = OBJECT_ID(N'dbo.SalesReturn')
      AND pc.name = N'SalesOrderNumber'
      AND pc.system_type_id = rc.system_type_id
      AND pc.user_type_id = rc.user_type_id
      AND pc.max_length = rc.max_length
      AND pc.precision = rc.precision
      AND pc.scale = rc.scale
)
AND EXISTS (
    SELECT 1
    FROM sys.columns pc
    JOIN sys.columns rc
      ON rc.object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
     AND rc.name = N'SalesOrderLineNumber'
    WHERE pc.object_id = OBJECT_ID(N'dbo.SalesReturn')
      AND pc.name = N'SalesOrderLineNumber'
      AND pc.system_type_id = rc.system_type_id
      AND pc.user_type_id = rc.user_type_id
      AND pc.max_length = rc.max_length
      AND pc.precision = rc.precision
      AND pc.scale = rc.scale
)
BEGIN
    ALTER TABLE dbo.SalesReturn WITH CHECK
    ADD CONSTRAINT FK_SalesReturn_SalesOrderDetail
        FOREIGN KEY ([SalesOrderNumber], [SalesOrderLineNumber])
        REFERENCES dbo.SalesOrderDetail ([SalesOrderNumber], [SalesOrderLineNumber]);

    ALTER TABLE dbo.SalesReturn CHECK CONSTRAINT FK_SalesReturn_SalesOrderDetail;
END;
