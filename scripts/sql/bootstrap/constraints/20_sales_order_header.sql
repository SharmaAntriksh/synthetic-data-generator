-----------------------------------------------------------------------
-- FACT: SalesOrderHeader (CANDIDATE KEY + FOREIGN KEYS + CHECKS)
-- Columns (expected):
--   SalesOrderNumber, CustomerKey, StoreKey, SalesPersonEmployeeKey,
--   OrderDate, TimeKey, SalesChannelKey, IsOrderDelayed
-----------------------------------------------------------------------

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- CLEANUP: remove legacy/incorrect constraint if it exists
-- (DeliveryDate is NOT a header column in the current schema)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Dates_DeliveryDate'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader DROP CONSTRAINT FK_SalesOrderHeader_Dates_DeliveryDate;
END;

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

-- SalesOrderHeader -> Stores
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'StoreKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Stores'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Stores
        FOREIGN KEY ([StoreKey])
        REFERENCES dbo.Stores ([StoreKey]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Stores;
END;

-- SalesOrderHeader -> Employees (SalesPersonEmployeeKey) [guarded for type compatibility]
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'SalesPersonEmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Employees_SalesPersonEmployeeKey'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns pc
    JOIN sys.columns rc
      ON rc.object_id = OBJECT_ID(N'dbo.Employees')
     AND rc.name = N'EmployeeKey'
    WHERE pc.object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
      AND pc.name = N'SalesPersonEmployeeKey'
      AND pc.system_type_id = rc.system_type_id
      AND pc.user_type_id = rc.user_type_id
      AND pc.max_length = rc.max_length
      AND pc.precision = rc.precision
      AND pc.scale = rc.scale
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Employees_SalesPersonEmployeeKey
        FOREIGN KEY ([SalesPersonEmployeeKey])
        REFERENCES dbo.Employees ([EmployeeKey]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Employees_SalesPersonEmployeeKey;
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
