-----------------------------------------------------------------------
-- FACT: SalesOrderHeader (CANDIDATE KEY + FOREIGN KEYS + CHECKS)
-- Columns (per static schema):
--   SalesOrderNumber, CustomerKey, OrderDate, TimeKey, SalesChannelKey, IsOrderDelayed
-----------------------------------------------------------------------

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- Primary / candidate key
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_SalesOrderHeader'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader
    ADD CONSTRAINT PK_SalesOrderHeader PRIMARY KEY NONCLUSTERED ([SalesOrderNumber]);
END;

-----------------------------------------------------------------------
-- Foreign keys
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Customers;
END;

IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Dates_OrderDate'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Dates_OrderDate
        FOREIGN KEY ([OrderDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Dates_OrderDate;
END;

IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Time', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'TimeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Time'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Time
        FOREIGN KEY ([TimeKey])
        REFERENCES dbo.Time ([TimeKey]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Time;
END;

IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.SalesChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'SalesChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_SalesChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_SalesChannels
        FOREIGN KEY ([SalesChannelKey])
        REFERENCES dbo.SalesChannels ([SalesChannelKey]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_SalesChannels;
END;

-----------------------------------------------------------------------
-- Checks
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'IsOrderDelayed') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_SalesOrderHeader_IsOrderDelayed'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader
    ADD CONSTRAINT CK_SalesOrderHeader_IsOrderDelayed
        CHECK ([IsOrderDelayed] IN (0, 1));
END;
