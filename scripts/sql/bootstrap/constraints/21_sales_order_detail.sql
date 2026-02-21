-----------------------------------------------------------------------
-- FACT: SalesOrderDetail (CANDIDATE KEY + FOREIGN KEYS WITH CHECK)
-- Columns (per static schema):
--   SalesOrderNumber, SalesOrderLineNumber, ProductKey, StoreKey,
--   SalesPersonEmployeeKey, PromotionKey, CurrencyKey,
--   DueDate, DeliveryDate, Quantity, NetPrice, UnitCost, UnitPrice,
--   DiscountAmount, DeliveryStatus
-----------------------------------------------------------------------

-----------------------------------------------------------------------
-- CANDIDATE KEY (supports SalesReturn -> SalesOrderDetail FK)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'SalesOrderNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'SalesOrderLineNumber') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'UQ_SalesOrderDetail_OrderLine'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail
    ADD CONSTRAINT UQ_SalesOrderDetail_OrderLine
        UNIQUE NONCLUSTERED ([SalesOrderNumber], [SalesOrderLineNumber]);
END;

-----------------------------------------------------------------------
-- FOREIGN KEYS (WITH CHECK)
-----------------------------------------------------------------------

-- SalesOrderDetail -> Products
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'ProductKey') IS NOT NULL
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

-- SalesOrderDetail -> Stores
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'StoreKey') IS NOT NULL
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

-- SalesOrderDetail -> Promotions
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'PromotionKey') IS NOT NULL
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

-- SalesOrderDetail -> Currency
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'CurrencyKey') IS NOT NULL
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

-- SalesOrderDetail -> Employees (SalesPersonEmployeeKey) [guarded for type compatibility]
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'SalesPersonEmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesOrderDetail_Employees_SalesPersonEmployeeKey'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns pc
    JOIN sys.columns rc
      ON rc.object_id = OBJECT_ID(N'dbo.Employees')
     AND rc.name = N'EmployeeKey'
    WHERE pc.object_id = OBJECT_ID(N'dbo.SalesOrderDetail')
      AND pc.name = N'SalesPersonEmployeeKey'
      AND pc.system_type_id = rc.system_type_id
      AND pc.user_type_id = rc.user_type_id
      AND pc.max_length = rc.max_length
      AND pc.precision = rc.precision
      AND pc.scale = rc.scale
)
BEGIN
    ALTER TABLE dbo.SalesOrderDetail WITH CHECK
    ADD CONSTRAINT FK_SalesOrderDetail_Employees_SalesPersonEmployeeKey
        FOREIGN KEY ([SalesPersonEmployeeKey])
        REFERENCES dbo.Employees ([EmployeeKey]);

    ALTER TABLE dbo.SalesOrderDetail CHECK CONSTRAINT FK_SalesOrderDetail_Employees_SalesPersonEmployeeKey;
END;

-- SalesOrderDetail -> Dates (DueDate)
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'DueDate') IS NOT NULL
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

-- SalesOrderDetail -> Dates (DeliveryDate)
IF OBJECT_ID(N'dbo.SalesOrderDetail', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesOrderDetail', N'DeliveryDate') IS NOT NULL
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
