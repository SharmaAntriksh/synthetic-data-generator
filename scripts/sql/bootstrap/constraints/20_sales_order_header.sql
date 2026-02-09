-----------------------------------------------------------------------
-- FACT: SalesOrderHeader (FOREIGN KEYS WITH CHECK)
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

IF OBJECT_ID(N'dbo.SalesOrderHeader', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderHeader_Dates_DeliveryDate'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderHeader')
)
BEGIN
    ALTER TABLE dbo.SalesOrderHeader WITH CHECK
    ADD CONSTRAINT FK_SalesOrderHeader_Dates_DeliveryDate
        FOREIGN KEY ([DeliveryDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.SalesOrderHeader CHECK CONSTRAINT FK_SalesOrderHeader_Dates_DeliveryDate;
END;
