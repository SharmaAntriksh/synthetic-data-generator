SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- PRIMARY KEYS
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers
    ADD CONSTRAINT PK_Customers PRIMARY KEY NONCLUSTERED ([CustomerKey]);
END;

IF OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Products'
      AND parent_object_id = OBJECT_ID(N'dbo.Products')
)
BEGIN
    ALTER TABLE dbo.Products
    ADD CONSTRAINT PK_Products PRIMARY KEY NONCLUSTERED ([ProductKey]);
END;

IF OBJECT_ID(N'dbo.ProductSubcategory', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_ProductSubcategory'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductSubcategory')
)
BEGIN
    ALTER TABLE dbo.ProductSubcategory
    ADD CONSTRAINT PK_ProductSubcategory PRIMARY KEY NONCLUSTERED ([SubcategoryKey]);
END;

IF OBJECT_ID(N'dbo.ProductCategory', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_ProductCategory'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductCategory')
)
BEGIN
    ALTER TABLE dbo.ProductCategory
    ADD CONSTRAINT PK_ProductCategory PRIMARY KEY NONCLUSTERED ([CategoryKey]);
END;

IF OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Geography'
      AND parent_object_id = OBJECT_ID(N'dbo.Geography')
)
BEGIN
    ALTER TABLE dbo.Geography
    ADD CONSTRAINT PK_Geography PRIMARY KEY NONCLUSTERED ([GeographyKey]);
END;

IF OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Currency'
      AND parent_object_id = OBJECT_ID(N'dbo.Currency')
)
BEGIN
    ALTER TABLE dbo.Currency
    ADD CONSTRAINT PK_Currency PRIMARY KEY NONCLUSTERED ([CurrencyKey]);
END;

IF OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Dates'
      AND parent_object_id = OBJECT_ID(N'dbo.Dates')
)
BEGIN
    ALTER TABLE dbo.Dates
    ADD CONSTRAINT PK_Dates PRIMARY KEY NONCLUSTERED ([Date]);
END;

IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_ExchangeRates'
      AND parent_object_id = OBJECT_ID(N'dbo.ExchangeRates')
)
BEGIN
    ALTER TABLE dbo.ExchangeRates
    ADD CONSTRAINT PK_ExchangeRates PRIMARY KEY NONCLUSTERED ([Date], [FromCurrency], [ToCurrency]);
END;

IF OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Stores', N'StoreKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Stores'
      AND parent_object_id = OBJECT_ID(N'dbo.Stores')
)
BEGIN
    ALTER TABLE dbo.Stores
    ADD CONSTRAINT PK_Stores PRIMARY KEY NONCLUSTERED ([StoreKey]);
END;

IF OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Promotions', N'PromotionKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Promotions'
      AND parent_object_id = OBJECT_ID(N'dbo.Promotions')
)
BEGIN
    ALTER TABLE dbo.Promotions
    ADD CONSTRAINT PK_Promotions PRIMARY KEY NONCLUSTERED ([PromotionKey]);
END;

-----------------------------------------------------------------------
-- UNIQUE CONSTRAINTS
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'UQ_Currency_ToCurrency'
      AND parent_object_id = OBJECT_ID(N'dbo.Currency')
)
BEGIN
    ALTER TABLE dbo.Currency
    ADD CONSTRAINT UQ_Currency_ToCurrency UNIQUE NONCLUSTERED ([ToCurrency]);
END;

-----------------------------------------------------------------------
-- FOREIGN KEYS (WITH CHECK)
-----------------------------------------------------------------------

-- Product hierarchy
IF OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Products_ProductSubcategory'
      AND parent_object_id = OBJECT_ID(N'dbo.Products')
)
BEGIN
    ALTER TABLE dbo.Products WITH CHECK
    ADD CONSTRAINT FK_Products_ProductSubcategory
        FOREIGN KEY ([SubcategoryKey])
        REFERENCES dbo.ProductSubcategory ([SubcategoryKey]);

    ALTER TABLE dbo.Products CHECK CONSTRAINT FK_Products_ProductSubcategory;
END;

IF OBJECT_ID(N'dbo.ProductSubcategory', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_ProductSubcategory_ProductCategory'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductSubcategory')
)
BEGIN
    ALTER TABLE dbo.ProductSubcategory WITH CHECK
    ADD CONSTRAINT FK_ProductSubcategory_ProductCategory
        FOREIGN KEY ([CategoryKey])
        REFERENCES dbo.ProductCategory ([CategoryKey]);

    ALTER TABLE dbo.ProductSubcategory CHECK CONSTRAINT FK_ProductSubcategory_ProductCategory;
END;

-- Customer geography
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Customers_Geography'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers WITH CHECK
    ADD CONSTRAINT FK_Customers_Geography
        FOREIGN KEY ([GeographyKey])
        REFERENCES dbo.Geography ([GeographyKey]);

    ALTER TABLE dbo.Customers CHECK CONSTRAINT FK_Customers_Geography;
END;

-- ExchangeRates links
IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_ExchangeRates_FromCurrency'
      AND parent_object_id = OBJECT_ID(N'dbo.ExchangeRates')
)
BEGIN
    ALTER TABLE dbo.ExchangeRates WITH CHECK
    ADD CONSTRAINT FK_ExchangeRates_FromCurrency
        FOREIGN KEY ([FromCurrency])
        REFERENCES dbo.Currency ([ToCurrency]);

    ALTER TABLE dbo.ExchangeRates CHECK CONSTRAINT FK_ExchangeRates_FromCurrency;
END;

IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_ExchangeRates_ToCurrency'
      AND parent_object_id = OBJECT_ID(N'dbo.ExchangeRates')
)
BEGIN
    ALTER TABLE dbo.ExchangeRates WITH CHECK
    ADD CONSTRAINT FK_ExchangeRates_ToCurrency
        FOREIGN KEY ([ToCurrency])
        REFERENCES dbo.Currency ([ToCurrency]);

    ALTER TABLE dbo.ExchangeRates CHECK CONSTRAINT FK_ExchangeRates_ToCurrency;
END;

-----------------------------------------------------------------------
-- EMPLOYEES: ensure candidate key on EmployeeKey (required for FKs)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Employees', N'EmployeeKey') IS NOT NULL
BEGIN
    -- If Employees has no PK, add one on EmployeeKey.
    IF NOT EXISTS (
        SELECT 1
        FROM sys.key_constraints
        WHERE parent_object_id = OBJECT_ID(N'dbo.Employees')
          AND [type] = N'PK'
    )
    BEGIN
        ALTER TABLE dbo.Employees
        ADD CONSTRAINT PK_Employees PRIMARY KEY NONCLUSTERED ([EmployeeKey]);
    END;

    -- If a different PK exists, ensure EmployeeKey is still a candidate key for FKs.
    IF EXISTS (
        SELECT 1
        FROM sys.key_constraints
        WHERE parent_object_id = OBJECT_ID(N'dbo.Employees')
          AND [type] = N'PK'
          AND name <> N'PK_Employees'
    )
    AND NOT EXISTS (
        SELECT 1
        FROM sys.key_constraints
        WHERE parent_object_id = OBJECT_ID(N'dbo.Employees')
          AND name = N'UX_Employees_EmployeeKey'
    )
    BEGIN
        ALTER TABLE dbo.Employees
        ADD CONSTRAINT UX_Employees_EmployeeKey UNIQUE ([EmployeeKey]);
    END;
END;

-----------------------------------------------------------------------
-- Ensure candidate key for FK targets: dbo.Employees(EmployeeKey)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Employees', N'EmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.indexes i
    JOIN sys.index_columns ic
      ON i.object_id = ic.object_id AND i.index_id = ic.index_id
    JOIN sys.columns c
      ON c.object_id = ic.object_id AND c.column_id = ic.column_id
    WHERE i.object_id = OBJECT_ID(N'dbo.Employees')
      AND i.is_unique = 1
      AND ic.is_included_column = 0
    GROUP BY i.index_id
    HAVING COUNT(*) = 1
       AND MAX(CASE WHEN c.name = N'EmployeeKey' AND ic.key_ordinal = 1 THEN 1 ELSE 0 END) = 1
)
BEGIN
    CREATE UNIQUE INDEX [UX_Employees_EmployeeKey]
    ON dbo.Employees([EmployeeKey]);
END;
GO

-----------------------------------------------------------------------
-- Ensure candidate key for FK targets: dbo.SalesChannels(SalesChannelKey)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.SalesChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesChannels', N'SalesChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.indexes i
    JOIN sys.index_columns ic
      ON i.object_id = ic.object_id AND i.index_id = ic.index_id
    JOIN sys.columns c
      ON c.object_id = ic.object_id AND c.column_id = ic.column_id
    WHERE i.object_id = OBJECT_ID(N'dbo.SalesChannels')
      AND i.is_unique = 1
      AND ic.is_included_column = 0
    GROUP BY i.index_id
    HAVING COUNT(*) = 1
       AND MAX(CASE WHEN c.name = N'SalesChannelKey' AND ic.key_ordinal = 1 THEN 1 ELSE 0 END) = 1
)
BEGIN
    CREATE UNIQUE INDEX [UX_SalesChannels_SalesChannelKey]
    ON dbo.SalesChannels([SalesChannelKey]);
END;
GO

-----------------------------------------------------------------------
-- Ensure candidate key for FK targets: dbo.Time(TimeKey)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.Time', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Time', N'TimeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.indexes i
    JOIN sys.index_columns ic
      ON i.object_id = ic.object_id AND i.index_id = ic.index_id
    JOIN sys.columns c
      ON c.object_id = ic.object_id AND c.column_id = ic.column_id
    WHERE i.object_id = OBJECT_ID(N'dbo.Time')
      AND i.is_unique = 1
      AND ic.is_included_column = 0
    GROUP BY i.index_id
    HAVING COUNT(*) = 1
       AND MAX(CASE WHEN c.name = N'TimeKey' AND ic.key_ordinal = 1 THEN 1 ELSE 0 END) = 1
)
BEGIN
    CREATE UNIQUE INDEX [UX_Time_TimeKey]
    ON dbo.Time([TimeKey]);
END;
GO

-----------------------------------------------------------------------
-- Ensure candidate key for FK targets: dbo.ReturnReason(ReturnReasonKey)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.ReturnReason', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ReturnReason', N'ReturnReasonKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.indexes i
    JOIN sys.index_columns ic
      ON i.object_id = ic.object_id AND i.index_id = ic.index_id
    JOIN sys.columns c
      ON c.object_id = ic.object_id AND c.column_id = ic.column_id
    WHERE i.object_id = OBJECT_ID(N'dbo.ReturnReason')
      AND i.is_unique = 1
      AND ic.is_included_column = 0
    GROUP BY i.index_id
    HAVING COUNT(*) = 1
       AND MAX(CASE WHEN c.name = N'ReturnReasonKey' AND ic.key_ordinal = 1 THEN 1 ELSE 0 END) = 1
)
BEGIN
    CREATE UNIQUE INDEX [UX_ReturnReason_ReturnReasonKey]
    ON dbo.ReturnReason([ReturnReasonKey]);
END;
GO

-----------------------------------------------------------------------
-- Ensure candidate key for FK targets: dbo.Currency(ToCurrency)
-----------------------------------------------------------------------
IF OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Currency', N'ToCurrency') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.indexes i
    JOIN sys.index_columns ic
      ON i.object_id = ic.object_id AND i.index_id = ic.index_id
    JOIN sys.columns c
      ON c.object_id = ic.object_id AND c.column_id = ic.column_id
    WHERE i.object_id = OBJECT_ID(N'dbo.Currency')
      AND i.is_unique = 1
      AND ic.is_included_column = 0
    GROUP BY i.index_id
    HAVING COUNT(*) = 1
       AND MAX(CASE WHEN c.name = N'ToCurrency' AND ic.key_ordinal = 1 THEN 1 ELSE 0 END) = 1
)
BEGIN
    CREATE UNIQUE INDEX [UX_Currency_ToCurrency]
    ON dbo.Currency([ToCurrency]);
END;
GO
