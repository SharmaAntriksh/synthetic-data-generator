-----------------------------------------------------------------------
-- FACT: SalesOrderDetail (CANDIDATE KEY + FOREIGN KEYS WITH CHECK)
-- Columns (per static schema):
--   SalesOrderNumber, SalesOrderLineNumber, ProductKey, StoreKey,
--   SalesPersonEmployeeKey, PromotionKey, CurrencyKey,
--   DueDate, DeliveryDate, Quantity, NetPrice, UnitCost, UnitPrice,
--   DiscountAmount, DeliveryStatus
-----------------------------------------------------------------------

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- Candidate key: (SalesOrderNumber, SalesOrderLineNumber)
-- Needed for joining / FK from SalesReturn and to protect grain.
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_SalesOrderDetail'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail
    ADD CONSTRAINT PK_SalesOrderDetail
        PRIMARY KEY NONCLUSTERED ([SalesOrderNumber], [SalesOrderLineNumber]);
END;

-----------------------------------------------------------------------
-- Foreign keys (dimension links)
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Products'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Products
        FOREIGN KEY ([ProductKey])
        REFERENCES dbo.Products ([ProductKey]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Products;
END;

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Stores'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Stores
        FOREIGN KEY ([StoreKey])
        REFERENCES dbo.Stores ([StoreKey]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Stores;
END;

-- Employees (type-mismatch guard: SalesPersonEmployeeKey is often INT while Employees.EmployeeKey may be BIGINT)
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'SalesPersonEmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Employees'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns c1
    JOIN sys.columns c2 ON 1=1
    WHERE c1.object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
      AND c1.name = N'SalesPersonEmployeeKey'
      AND c2.object_id = OBJECT_ID(N'dbo.Employees')
      AND c2.name = N'EmployeeKey'
      AND c1.user_type_id = c2.user_type_id
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Employees
        FOREIGN KEY ([SalesPersonEmployeeKey])
        REFERENCES dbo.Employees ([EmployeeKey]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Employees;
END;

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Promotions'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Promotions
        FOREIGN KEY ([PromotionKey])
        REFERENCES dbo.Promotions ([PromotionKey]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Promotions;
END;

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Currency'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Currency
        FOREIGN KEY ([CurrencyKey])
        REFERENCES dbo.Currency ([CurrencyKey]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Currency;
END;

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Dates_DueDate'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Dates_DueDate
        FOREIGN KEY ([DueDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Dates_DueDate;
END;

IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Dates_DeliveryDate'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Dates_DeliveryDate
        FOREIGN KEY ([DeliveryDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Dates_DeliveryDate;
END;
