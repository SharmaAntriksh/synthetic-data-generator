-----------------------------------------------------------------------
-- FACT: SalesOrderHeader (CANDIDATE KEY + FOREIGN KEYS + CHECKS)
-- Columns (per static schema):
--   SalesOrderNumber, CustomerKey, OrderDate, TimeKey, SalesChannelKey, IsOrderDelayed
-----------------------------------------------------------------------

-----------------------------------------------------------------------
-- CANDIDATE KEY (required for SalesOrderDetail -> SalesOrderHeader FK)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'SalesOrderNumber') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'UQ_SalesOrderHeader_SalesOrderNumber'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader
    ADD CONSTRAINT UQ_SalesOrderHeader_SalesOrderNumber
        UNIQUE NONCLUSTERED ([SalesOrderNumber]);
END;

-----------------------------------------------------------------------
-- FOREIGN KEYS (WITH CHECK)
-----------------------------------------------------------------------

-- SalesOrderHeader -> Customers
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'CustomerKey') IS NOT NULL
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

-- SalesOrderHeader -> Dates (OrderDate)
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'OrderDate') IS NOT NULL
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

-- SalesOrderHeader -> Time
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

-- SalesOrderHeader -> SalesChannels
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
-- CHECK CONSTRAINTS
-----------------------------------------------------------------------

-- IsOrderDelayed is a bit/flag represented as INT in the fact schema (0/1).
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'IsOrderDelayed') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_SalesOrderHeader_IsOrderDelayed_01'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT CK_SalesOrderHeader_IsOrderDelayed_01
        CHECK ([IsOrderDelayed] IN (0, 1));
END;
