-----------------------------------------------------------------------
-- FACT: Sales + SalesReturn (PK + FOREIGN KEYS WITH CHECK)
-- Aligned to src/utils/static_schemas.py
--
-- SalesReturn change (Option B):
--   - Add ReturnEventKey BIGINT to SalesReturn
--   - PK_SalesReturn becomes (ReturnEventKey)
--   - Natural key gets a non-unique index for joins:
--       (SalesOrderNumber, SalesOrderLineNumber, ReturnDate, ReturnReasonKey)
-----------------------------------------------------------------------

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- Sales: candidate key (order line grain)
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Sales'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales
    ADD CONSTRAINT PK_Sales
        PRIMARY KEY NONCLUSTERED ([SalesOrderNumber], [SalesOrderLineNumber]);
END;

-----------------------------------------------------------------------
-- Sales: dimension links
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

-- Employees (type-mismatch guard)
IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'SalesPersonEmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Employees'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns c1
    JOIN sys.columns c2 ON 1=1
    WHERE c1.object_id = OBJECT_ID(N'dbo.Sales')
      AND c1.name = N'SalesPersonEmployeeKey'
      AND c2.object_id = OBJECT_ID(N'dbo.Employees')
      AND c2.name = N'EmployeeKey'
      AND c1.user_type_id = c2.user_type_id
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Employees
        FOREIGN KEY ([SalesPersonEmployeeKey])
        REFERENCES dbo.Employees ([EmployeeKey]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Employees;
END;

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.SalesChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'SalesChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Time', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Sales', N'TimeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Sales_Dates_OrderDate'
      AND parent_object_id = OBJECT_ID(N'dbo.Sales')
)
BEGIN
    ALTER TABLE dbo.Sales WITH CHECK
    ADD CONSTRAINT FK_Sales_Dates_OrderDate
        FOREIGN KEY ([OrderDate])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.Sales CHECK CONSTRAINT FK_Sales_Dates_OrderDate;
END;

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

IF OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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
-- SalesReturn: PK + dimension links
-----------------------------------------------------------------------

-- PK: ReturnEventKey (only when column exists in the table)
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'ReturnEventKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_SalesReturn'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesReturn')
)
BEGIN
    ALTER TABLE dbo.SalesReturn
    ADD CONSTRAINT PK_SalesReturn PRIMARY KEY NONCLUSTERED ([ReturnEventKey]);
END;

-- Natural key access path (non-unique)
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_SalesReturn_NaturalKey'
      AND object_id = OBJECT_ID(N'dbo.SalesReturn')
)
BEGIN
    CREATE INDEX IX_SalesReturn_NaturalKey
    ON dbo.SalesReturn ([SalesOrderNumber], [SalesOrderLineNumber], [ReturnDate], [ReturnReasonKey]);
END;

-- ReturnDate -> Dates(Date)
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'ReturnDate') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
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

-- ReturnReasonKey -> ReturnReason(ReturnReasonKey)
-- Guarded due to potential INT/BIGINT mismatch between fact/dim.
IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.ReturnReason', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesReturn', N'ReturnReasonKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_SalesReturn_ReturnReason'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesReturn')
)
AND EXISTS (
    SELECT 1
    FROM sys.columns c1
    JOIN sys.columns c2 ON 1=1
    WHERE c1.object_id = OBJECT_ID(N'dbo.SalesReturn')
      AND c1.name = N'ReturnReasonKey'
      AND c2.object_id = OBJECT_ID(N'dbo.ReturnReason')
      AND c2.name = N'ReturnReasonKey'
      AND c1.user_type_id = c2.user_type_id
)
BEGIN
    ALTER TABLE dbo.SalesReturn WITH CHECK
    ADD CONSTRAINT FK_SalesReturn_ReturnReason
        FOREIGN KEY ([ReturnReasonKey])
        REFERENCES dbo.ReturnReason ([ReturnReasonKey]);

    ALTER TABLE dbo.SalesReturn CHECK CONSTRAINT FK_SalesReturn_ReturnReason;
END;
