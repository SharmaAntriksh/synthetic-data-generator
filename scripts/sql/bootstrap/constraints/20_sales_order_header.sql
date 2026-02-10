-----------------------------------------------------------------------
-- FACT: SalesOrderHeader (CANDIDATE KEYS + FOREIGN KEYS WITH CHECK)
--
-- Columns (per static schema):
--   SalesOrderNumber, CustomerKey, StoreKey, PromotionKey, CurrencyKey,
--   OrderDate, DueDate
-----------------------------------------------------------------------

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
    ALTER TABLE dbo.SalesOrderHeader
    DROP CONSTRAINT FK_SalesOrderHeader_Dates_DeliveryDate;
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
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'CustomerKey') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
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
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'StoreKey') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Stores', N'StoreKey') IS NOT NULL
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

-- SalesOrderHeader -> Promotions
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'PromotionKey') IS NOT NULL
AND OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Promotions', N'PromotionKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Promotions'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Promotions
        FOREIGN KEY ([PromotionKey])
        REFERENCES dbo.Promotions ([PromotionKey]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Promotions;
END;

-- SalesOrderHeader -> Currency
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'CurrencyKey') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Currency'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Currency
        FOREIGN KEY ([CurrencyKey])
        REFERENCES dbo.Currency ([CurrencyKey]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Currency;
END;

-- SalesOrderHeader -> Dates (OrderDate)
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'OrderDate') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
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

-- SalesOrderHeader -> Dates (DueDate)
IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderHeader', N'DueDate') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Dates_DueDate'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Dates_DueDate
        FOREIGN KEY ([DueDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Dates_DueDate;
END;
