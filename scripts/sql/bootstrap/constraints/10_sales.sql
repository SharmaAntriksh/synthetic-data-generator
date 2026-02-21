-----------------------------------------------------------------------
-- FACT: Sales + SalesReturn (FOREIGN KEYS / PK) WITH CHECK
-- Aligned to src/utils/static_schemas.py
-----------------------------------------------------------------------

-----------------------------------------------------------------------
-- Sales: dimension links
-----------------------------------------------------------------------

-- Sales -> Customers
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'CustomerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Customers;
END;

-- Sales -> Products
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'ProductKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Products'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Products
        FOREIGN KEY ([ProductKey])
        REFERENCES dbo.Products ([ProductKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Products;
END;

-- Sales -> Stores
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'StoreKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Stores'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Stores
        FOREIGN KEY ([StoreKey])
        REFERENCES dbo.Stores ([StoreKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Stores;
END;

-- Sales -> Promotions
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'PromotionKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Promotions'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Promotions
        FOREIGN KEY ([PromotionKey])
        REFERENCES dbo.Promotions ([PromotionKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Promotions;
END;

-- Sales -> Currency
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'CurrencyKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Currency'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Currency
        FOREIGN KEY ([CurrencyKey])
        REFERENCES dbo.Currency ([CurrencyKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Currency;
END;

-- Sales -> SalesChannels
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.SalesChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'SalesChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_SalesChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_SalesChannels
        FOREIGN KEY ([SalesChannelKey])
        REFERENCES dbo.SalesChannels ([SalesChannelKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_SalesChannels;
END;

-- Sales -> Time
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Time', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'TimeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Time'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Time
        FOREIGN KEY ([TimeKey])
        REFERENCES dbo.Time ([TimeKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Time;
END;

-- Sales -> Employees (SalesPersonEmployeeKey) [guarded for type compatibility]
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'SalesPersonEmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Employees_SalesPersonEmployeeKey'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns pc
    JOIN sys.columns rc
      ON rc.object_id = OBJECT_ID(N'dbo.Employees')
     AND rc.name = N'EmployeeKey'
    WHERE pc.object_id = OBJECT_ID(N'dbo.Sales')
      AND pc.name = N'SalesPersonEmployeeKey'
      AND pc.system_type_id = rc.system_type_id
      AND pc.user_type_id = rc.user_type_id
      AND pc.max_length = rc.max_length
      AND pc.precision = rc.precision
      AND pc.scale = rc.scale
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Employees_SalesPersonEmployeeKey
        FOREIGN KEY ([SalesPersonEmployeeKey])
        REFERENCES dbo.Employees ([EmployeeKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Employees_SalesPersonEmployeeKey;
END;

-- Sales -> Dates (OrderDate)
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'OrderDate') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Dates'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Dates
        FOREIGN KEY ([OrderDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Dates;
END;

-- Sales -> Dates (DueDate)
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'DueDate') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Dates_DueDate'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Dates_DueDate
        FOREIGN KEY ([DueDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Dates_DueDate;
END;

-- Sales -> Dates (DeliveryDate)
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'DeliveryDate') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Dates_DeliveryDate'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Dates_DeliveryDate
        FOREIGN KEY ([DeliveryDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Dates_DeliveryDate;
END;

-----------------------------------------------------------------------
-- SalesReturn: PK + dimension links (order-line relation lives in 22_*.sql)
-----------------------------------------------------------------------

-- PK: (SalesOrderNumber, SalesOrderLineNumber, ReturnDate, ReturnReasonKey)
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'SalesOrderNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'SalesOrderLineNumber') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'ReturnDate') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'ReturnReasonKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_SalesReturn'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesReturn')
)
BEGIN
    ALTER TABLE dbo.SalesReturn
    ADD CONSTRAINT PK_SalesReturn
        PRIMARY KEY NONCLUSTERED ([SalesOrderNumber], [SalesOrderLineNumber], [ReturnDate], [ReturnReasonKey]);
END;

-- SalesReturn -> Dates (ReturnDate)
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'ReturnDate') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_SalesReturn_Dates_ReturnDate'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesReturn')
)
BEGIN
    ALTER TABLE dbo.SalesReturn WITH CHECK
    ADD CONSTRAINT FK_SalesReturn_Dates_ReturnDate
        FOREIGN KEY ([ReturnDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.SalesReturn CHECK CONSTRAINT FK_SalesReturn_Dates_ReturnDate;
END;

-- SalesReturn -> ReturnReason (guarded for type compatibility)
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.ReturnReason', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'ReturnReasonKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_SalesReturn_ReturnReason'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesReturn')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns pc
    JOIN sys.columns rc
      ON rc.object_id = OBJECT_ID(N'dbo.ReturnReason')
     AND rc.name = N'ReturnReasonKey'
    WHERE pc.object_id = OBJECT_ID(N'dbo.SalesReturn')
      AND pc.name = N'ReturnReasonKey'
      AND pc.system_type_id = rc.system_type_id
      AND pc.user_type_id = rc.user_type_id
      AND pc.max_length = rc.max_length
      AND pc.precision = rc.precision
      AND pc.scale = rc.scale
)
BEGIN
    ALTER TABLE dbo.SalesReturn WITH CHECK
    ADD CONSTRAINT FK_SalesReturn_ReturnReason
        FOREIGN KEY ([ReturnReasonKey])
        REFERENCES dbo.ReturnReason ([ReturnReasonKey]);

    ALTER TABLE dbo.SalesReturn CHECK CONSTRAINT FK_SalesReturn_ReturnReason;
END;
