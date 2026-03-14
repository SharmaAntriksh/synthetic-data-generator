/*
  00_dimensions.sql – Dimension table constraints (PKs, UKs, FKs).
  Idempotent: every statement is guarded; safe to re-run.

  Design notes
  ────────────
  • All primary keys are NONCLUSTERED because a clustered columnstore index
    (CCI) is applied separately via create_drop_cci.sql.  Tables remain heaps
    until CCI creation; NONCLUSTERED PKs coexist with the CCI for point
    lookups and FK enforcement.

  • Candidate-key checks use a robust sys.indexes / sys.index_columns pattern
    that detects ANY unique index on the target column(s), regardless of name.
    This avoids silent re-creation if an index is renamed out-of-band.

  • COL_LENGTH guards are applied uniformly to tables whose schema may vary
    depending on config (Stores, Promotions, Employees, SalesChannels, Time,
    ReturnReason).  Core tables with guaranteed schemas omit the guard for
    clarity.

  • Each logical section is separated by GO so a failure in one batch does
    not roll back unrelated constraints.  Individual statements are
    independently guarded, so partial execution leaves the database in a
    consistent state.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- 1. PRIMARY KEYS
-----------------------------------------------------------------------

-- Customers
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

-- Customers — SCD2 constraints
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Customers', N'IsCurrent') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_Customers_IsCurrent'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers WITH CHECK
    ADD CONSTRAINT CK_Customers_IsCurrent
        CHECK ([IsCurrent] IN (0, 1));
    ALTER TABLE dbo.Customers CHECK CONSTRAINT CK_Customers_IsCurrent;
END;

IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Customers', N'VersionNumber') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_Customers_VersionNumber'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers WITH CHECK
    ADD CONSTRAINT CK_Customers_VersionNumber
        CHECK ([VersionNumber] >= 1);
    ALTER TABLE dbo.Customers CHECK CONSTRAINT CK_Customers_VersionNumber;
END;


-- CustomerProfile (1:1 with Customers)
IF OBJECT_ID(N'dbo.CustomerProfile', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_CustomerProfile'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerProfile')
)
BEGIN
    ALTER TABLE dbo.CustomerProfile
    ADD CONSTRAINT PK_CustomerProfile PRIMARY KEY NONCLUSTERED ([CustomerKey]);
END;

-- OrganizationProfile (1:1 with org-type Customers)
IF OBJECT_ID(N'dbo.OrganizationProfile', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_OrganizationProfile'
      AND parent_object_id = OBJECT_ID(N'dbo.OrganizationProfile')
)
BEGIN
    ALTER TABLE dbo.OrganizationProfile
    ADD CONSTRAINT PK_OrganizationProfile PRIMARY KEY NONCLUSTERED ([CustomerKey]);
END;

-- Products
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

-- ProductProfile (1:1 with Products)
IF OBJECT_ID(N'dbo.ProductProfile', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_ProductProfile'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductProfile')
)
BEGIN
    ALTER TABLE dbo.ProductProfile
    ADD CONSTRAINT PK_ProductProfile PRIMARY KEY NONCLUSTERED ([ProductKey]);
END;

-- ProductSubcategory
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

-- ProductCategory
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

-- Geography
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

-- Currency
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

-- Dates
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

-- ExchangeRates (composite PK)
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

-- Stores (schema may vary)
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

-- Promotions (schema may vary)
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

-- Employees (schema may vary; BIGINT key)
IF OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Employees', N'EmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE parent_object_id = OBJECT_ID(N'dbo.Employees')
      AND [type] = N'PK'
)
BEGIN
    ALTER TABLE dbo.Employees
    ADD CONSTRAINT PK_Employees PRIMARY KEY NONCLUSTERED ([EmployeeKey]);
END;

-- Suppliers (schema may vary)
IF OBJECT_ID(N'dbo.Suppliers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Suppliers', N'SupplierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Suppliers'
      AND parent_object_id = OBJECT_ID(N'dbo.Suppliers')
)
BEGIN
    ALTER TABLE dbo.Suppliers
    ADD CONSTRAINT PK_Suppliers PRIMARY KEY NONCLUSTERED ([SupplierKey]);
END;

-- SalesChannels (schema may vary)
IF OBJECT_ID(N'dbo.SalesChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesChannels', N'SalesChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_SalesChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesChannels')
)
BEGIN
    ALTER TABLE dbo.SalesChannels
    ADD CONSTRAINT PK_SalesChannels PRIMARY KEY NONCLUSTERED ([SalesChannelKey]);
END;

-- Time (schema may vary)
IF OBJECT_ID(N'dbo.Time', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Time', N'TimeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Time'
      AND parent_object_id = OBJECT_ID(N'dbo.Time')
)
BEGIN
    ALTER TABLE dbo.Time
    ADD CONSTRAINT PK_Time PRIMARY KEY NONCLUSTERED ([TimeKey]);
END;

-- ReturnReason (schema may vary)
IF OBJECT_ID(N'dbo.ReturnReason', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ReturnReason', N'ReturnReasonKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_ReturnReason'
      AND parent_object_id = OBJECT_ID(N'dbo.ReturnReason')
)
BEGIN
    ALTER TABLE dbo.ReturnReason
    ADD CONSTRAINT PK_ReturnReason PRIMARY KEY NONCLUSTERED ([ReturnReasonKey]);
END;
GO

-----------------------------------------------------------------------
-- 2. CANDIDATE KEYS FOR FK TARGETS
--
-- These ensure a unique index exists on columns referenced by foreign
-- keys in downstream scripts (10_sales.sql etc.), even when the PK
-- uses a different column.  The sys.indexes check detects any existing
-- unique index on the target column regardless of naming.
-----------------------------------------------------------------------

-- Employees(EmployeeKey) – needed when PK is on a different column
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

-- Currency(ToCurrency) – referenced by ExchangeRates FKs
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

-- SalesChannels(SalesChannelKey)
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

-- Time(TimeKey)
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

-- ReturnReason(ReturnReasonKey)
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
-- 3. FOREIGN KEYS (WITH CHECK)
-----------------------------------------------------------------------
SET NOCOUNT ON;
SET XACT_ABORT ON;

-- Product hierarchy: Products -> ProductSubcategory
IF OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.ProductSubcategory', N'U') IS NOT NULL
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

-- Product hierarchy: ProductSubcategory -> ProductCategory
IF OBJECT_ID(N'dbo.ProductSubcategory', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.ProductCategory', N'U') IS NOT NULL
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

-- ProductProfile -> Products (1:1)
IF OBJECT_ID(N'dbo.ProductProfile', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_ProductProfile_Products'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductProfile')
)
BEGIN
    ALTER TABLE dbo.ProductProfile WITH CHECK
    ADD CONSTRAINT FK_ProductProfile_Products
        FOREIGN KEY ([ProductKey])
        REFERENCES dbo.Products ([ProductKey]);

    ALTER TABLE dbo.ProductProfile CHECK CONSTRAINT FK_ProductProfile_Products;
END;

-- ProductProfile -> Suppliers (SupplierKey lives on ProductProfile)
IF OBJECT_ID(N'dbo.ProductProfile', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Suppliers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ProductProfile', N'SupplierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_ProductProfile_Suppliers'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductProfile')
)
BEGIN
    ALTER TABLE dbo.ProductProfile WITH CHECK
    ADD CONSTRAINT FK_ProductProfile_Suppliers
        FOREIGN KEY ([SupplierKey])
        REFERENCES dbo.Suppliers ([SupplierKey]);

    ALTER TABLE dbo.ProductProfile CHECK CONSTRAINT FK_ProductProfile_Suppliers;
END;

-- Customers -> Geography
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
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

-- CustomerProfile -> Customers (1:1)
IF OBJECT_ID(N'dbo.CustomerProfile', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_CustomerProfile_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerProfile')
)
BEGIN
    ALTER TABLE dbo.CustomerProfile WITH CHECK
    ADD CONSTRAINT FK_CustomerProfile_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.CustomerProfile CHECK CONSTRAINT FK_CustomerProfile_Customers;
END;

-- OrganizationProfile -> Customers (1:1, org-type rows only)
IF OBJECT_ID(N'dbo.OrganizationProfile', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_OrganizationProfile_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.OrganizationProfile')
)
BEGIN
    ALTER TABLE dbo.OrganizationProfile WITH CHECK
    ADD CONSTRAINT FK_OrganizationProfile_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.OrganizationProfile CHECK CONSTRAINT FK_OrganizationProfile_Customers;
END;

-- Stores -> Geography
IF OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Stores', N'GeographyKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Stores_Geography'
      AND parent_object_id = OBJECT_ID(N'dbo.Stores')
)
BEGIN
    ALTER TABLE dbo.Stores WITH CHECK
    ADD CONSTRAINT FK_Stores_Geography
        FOREIGN KEY ([GeographyKey])
        REFERENCES dbo.Geography ([GeographyKey]);

    ALTER TABLE dbo.Stores CHECK CONSTRAINT FK_Stores_Geography;
END;

-- ExchangeRates -> Currency (FromCurrency)
IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
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

-- ExchangeRates -> Currency (ToCurrency)
IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
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

-- ExchangeRates -> Dates (Date)
IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_ExchangeRates_Dates'
      AND parent_object_id = OBJECT_ID(N'dbo.ExchangeRates')
)
BEGIN
    ALTER TABLE dbo.ExchangeRates WITH CHECK
    ADD CONSTRAINT FK_ExchangeRates_Dates
        FOREIGN KEY ([Date])
        REFERENCES dbo.Dates ([Date]);

    ALTER TABLE dbo.ExchangeRates CHECK CONSTRAINT FK_ExchangeRates_Dates;
END;
GO

-----------------------------------------------------------------------
-- 4. LOYALTY TIERS
-----------------------------------------------------------------------

-- LoyaltyTiers: PK
IF OBJECT_ID(N'dbo.LoyaltyTiers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.LoyaltyTiers', N'LoyaltyTierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_LoyaltyTiers'
      AND parent_object_id = OBJECT_ID(N'dbo.LoyaltyTiers')
)
BEGIN
    ALTER TABLE dbo.LoyaltyTiers
    ADD CONSTRAINT PK_LoyaltyTiers PRIMARY KEY NONCLUSTERED ([LoyaltyTierKey]);
END;
GO

-- Customers -> LoyaltyTiers FK
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.LoyaltyTiers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Customers', N'LoyaltyTierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Customers_LoyaltyTiers'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers WITH CHECK
    ADD CONSTRAINT FK_Customers_LoyaltyTiers
        FOREIGN KEY ([LoyaltyTierKey])
        REFERENCES dbo.LoyaltyTiers ([LoyaltyTierKey]);

    ALTER TABLE dbo.Customers CHECK CONSTRAINT FK_Customers_LoyaltyTiers;
END;
GO

-----------------------------------------------------------------------
-- 5. CUSTOMER ACQUISITION CHANNELS
-----------------------------------------------------------------------

-- CustomerAcquisitionChannels: PK
IF OBJECT_ID(N'dbo.CustomerAcquisitionChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerAcquisitionChannels', N'CustomerAcquisitionChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_CustomerAcquisitionChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerAcquisitionChannels')
)
BEGIN
    ALTER TABLE dbo.CustomerAcquisitionChannels
    ADD CONSTRAINT PK_CustomerAcquisitionChannels
        PRIMARY KEY NONCLUSTERED ([CustomerAcquisitionChannelKey]);
END;
GO

-- Customers -> CustomerAcquisitionChannels FK
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.CustomerAcquisitionChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Customers', N'CustomerAcquisitionChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Customers_AcquisitionChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers WITH CHECK
    ADD CONSTRAINT FK_Customers_AcquisitionChannels
        FOREIGN KEY ([CustomerAcquisitionChannelKey])
        REFERENCES dbo.CustomerAcquisitionChannels ([CustomerAcquisitionChannelKey]);

    ALTER TABLE dbo.Customers CHECK CONSTRAINT FK_Customers_AcquisitionChannels;
END;
GO

-----------------------------------------------------------------------
-- 6. PLANS + CUSTOMER SUBSCRIPTIONS BRIDGE
-----------------------------------------------------------------------

-- Plans: PK
IF OBJECT_ID(N'dbo.Plans', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Plans', N'PlanKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Plans'
      AND parent_object_id = OBJECT_ID(N'dbo.Plans')
)
BEGIN
    ALTER TABLE dbo.Plans
    ADD CONSTRAINT PK_Plans PRIMARY KEY NONCLUSTERED ([PlanKey]);
END;
GO

-- CustomerSubscriptions: PK on SubscriptionKey
IF OBJECT_ID(N'dbo.CustomerSubscriptions', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSubscriptions', N'SubscriptionKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_CustomerSubscriptions'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSubscriptions')
)
BEGIN
    ALTER TABLE dbo.CustomerSubscriptions
    ADD CONSTRAINT PK_CustomerSubscriptions
        PRIMARY KEY NONCLUSTERED ([SubscriptionKey]);
END;
GO

-- CustomerSubscriptions -> Customers FK
IF OBJECT_ID(N'dbo.CustomerSubscriptions', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSubscriptions', N'CustomerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_CustomerSubscriptions_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSubscriptions')
)
BEGIN
    ALTER TABLE dbo.CustomerSubscriptions WITH CHECK
    ADD CONSTRAINT FK_CustomerSubscriptions_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.CustomerSubscriptions CHECK CONSTRAINT FK_CustomerSubscriptions_Customers;
END;
GO

-- CustomerSubscriptions -> Plans FK
IF OBJECT_ID(N'dbo.CustomerSubscriptions', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Plans', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSubscriptions', N'PlanKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_CustomerSubscriptions_Plans'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSubscriptions')
)
BEGIN
    ALTER TABLE dbo.CustomerSubscriptions WITH CHECK
    ADD CONSTRAINT FK_CustomerSubscriptions_Plans
        FOREIGN KEY ([PlanKey])
        REFERENCES dbo.Plans ([PlanKey]);

    ALTER TABLE dbo.CustomerSubscriptions CHECK CONSTRAINT FK_CustomerSubscriptions_Plans;
END;
GO

-- Plans: CHECK constraints
IF OBJECT_ID(N'dbo.Plans', N'U') IS NOT NULL
BEGIN
    IF COL_LENGTH(N'dbo.Plans', N'BillingCycle') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_Plans_BillingCycle'
          AND parent_object_id = OBJECT_ID(N'dbo.Plans')
    )
        ALTER TABLE dbo.Plans
        ADD CONSTRAINT CK_Plans_BillingCycle
            CHECK ([BillingCycle] IN ('Monthly', 'Quarterly', 'Half-Yearly', 'Annual'));

    IF COL_LENGTH(N'dbo.Plans', N'Discount') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_Plans_Discount'
          AND parent_object_id = OBJECT_ID(N'dbo.Plans')
    )
        ALTER TABLE dbo.Plans
        ADD CONSTRAINT CK_Plans_Discount
            CHECK ([Discount] >= 0 AND [Discount] < 1);

    IF COL_LENGTH(N'dbo.Plans', N'MonthlyPrice') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_Plans_MonthlyPrice'
          AND parent_object_id = OBJECT_ID(N'dbo.Plans')
    )
        ALTER TABLE dbo.Plans
        ADD CONSTRAINT CK_Plans_MonthlyPrice
            CHECK ([MonthlyPrice] >= 0);
END;
GO

-- CustomerSubscriptions: CHECK constraints
IF OBJECT_ID(N'dbo.CustomerSubscriptions', N'U') IS NOT NULL
BEGIN
    IF COL_LENGTH(N'dbo.CustomerSubscriptions', N'Status') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_CustomerSubscriptions_Status'
          AND parent_object_id = OBJECT_ID(N'dbo.CustomerSubscriptions')
    )
        ALTER TABLE dbo.CustomerSubscriptions
        ADD CONSTRAINT CK_CustomerSubscriptions_Status
            CHECK ([Status] IN ('Active', 'Cancelled', 'Expired'));

    IF COL_LENGTH(N'dbo.CustomerSubscriptions', N'AutoRenew') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_CustomerSubscriptions_AutoRenew'
          AND parent_object_id = OBJECT_ID(N'dbo.CustomerSubscriptions')
    )
        ALTER TABLE dbo.CustomerSubscriptions
        ADD CONSTRAINT CK_CustomerSubscriptions_AutoRenew
            CHECK ([AutoRenew] IN (0, 1));

    IF COL_LENGTH(N'dbo.CustomerSubscriptions', N'MonthlyPrice') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_CustomerSubscriptions_MonthlyPrice'
          AND parent_object_id = OBJECT_ID(N'dbo.CustomerSubscriptions')
    )
        ALTER TABLE dbo.CustomerSubscriptions
        ADD CONSTRAINT CK_CustomerSubscriptions_MonthlyPrice
            CHECK ([MonthlyPrice] >= 0);

    IF COL_LENGTH(N'dbo.CustomerSubscriptions', N'LoyaltyDiscount') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_CustomerSubscriptions_LoyaltyDiscount'
          AND parent_object_id = OBJECT_ID(N'dbo.CustomerSubscriptions')
    )
        ALTER TABLE dbo.CustomerSubscriptions
        ADD CONSTRAINT CK_CustomerSubscriptions_LoyaltyDiscount
            CHECK ([LoyaltyDiscount] >= 0 AND [LoyaltyDiscount] <= 1);
END;
GO

-----------------------------------------------------------------------
-- 7. CUSTOMER WISHLISTS BRIDGE
-----------------------------------------------------------------------

-- CustomerWishlists: PK
IF OBJECT_ID(N'dbo.CustomerWishlists', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerWishlists', N'WishlistKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_CustomerWishlists'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerWishlists')
)
BEGIN
    ALTER TABLE dbo.CustomerWishlists
    ADD CONSTRAINT PK_CustomerWishlists PRIMARY KEY NONCLUSTERED ([WishlistKey]);
END;
GO

-- CustomerWishlists -> Customers FK
IF OBJECT_ID(N'dbo.CustomerWishlists', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerWishlists', N'CustomerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_CustomerWishlists_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerWishlists')
)
BEGIN
    ALTER TABLE dbo.CustomerWishlists WITH CHECK
    ADD CONSTRAINT FK_CustomerWishlists_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.CustomerWishlists CHECK CONSTRAINT FK_CustomerWishlists_Customers;
END;
GO

-- CustomerWishlists -> Products FK
IF OBJECT_ID(N'dbo.CustomerWishlists', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerWishlists', N'ProductKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_CustomerWishlists_Products'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerWishlists')
)
BEGIN
    ALTER TABLE dbo.CustomerWishlists WITH CHECK
    ADD CONSTRAINT FK_CustomerWishlists_Products
        FOREIGN KEY ([ProductKey])
        REFERENCES dbo.Products ([ProductKey]);

    ALTER TABLE dbo.CustomerWishlists CHECK CONSTRAINT FK_CustomerWishlists_Products;
END;
GO

-- CustomerWishlists: CHECK constraints
IF OBJECT_ID(N'dbo.CustomerWishlists', N'U') IS NOT NULL
BEGIN
    IF COL_LENGTH(N'dbo.CustomerWishlists', N'Quantity') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_CustomerWishlists_Quantity'
          AND parent_object_id = OBJECT_ID(N'dbo.CustomerWishlists')
    )
        ALTER TABLE dbo.CustomerWishlists
        ADD CONSTRAINT CK_CustomerWishlists_Quantity
            CHECK ([Quantity] >= 1);

    IF COL_LENGTH(N'dbo.CustomerWishlists', N'Priority') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_CustomerWishlists_Priority'
          AND parent_object_id = OBJECT_ID(N'dbo.CustomerWishlists')
    )
        ALTER TABLE dbo.CustomerWishlists
        ADD CONSTRAINT CK_CustomerWishlists_Priority
            CHECK ([Priority] IN (N'High', N'Medium', N'Low'));

    IF COL_LENGTH(N'dbo.CustomerWishlists', N'NetPrice') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_CustomerWishlists_NetPrice'
          AND parent_object_id = OBJECT_ID(N'dbo.CustomerWishlists')
    )
        ALTER TABLE dbo.CustomerWishlists
        ADD CONSTRAINT CK_CustomerWishlists_NetPrice
            CHECK ([NetPrice] >= 0);
END;
GO

------------------------------------------------------------------------
-- 8. Complaints (optional)
------------------------------------------------------------------------

-- Complaints: PK
IF OBJECT_ID(N'dbo.Complaints', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Complaints', N'ComplaintKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_Complaints'
      AND parent_object_id = OBJECT_ID(N'dbo.Complaints')
)
BEGIN
    ALTER TABLE dbo.Complaints
    ADD CONSTRAINT PK_Complaints PRIMARY KEY NONCLUSTERED ([ComplaintKey]);
END;
GO

-- Complaints -> Customers FK
IF OBJECT_ID(N'dbo.Complaints', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Complaints', N'CustomerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_Complaints_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.Complaints')
)
BEGIN
    ALTER TABLE dbo.Complaints WITH CHECK
    ADD CONSTRAINT FK_Complaints_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.Complaints CHECK CONSTRAINT FK_Complaints_Customers;
END;
GO

-- Complaints: CHECK constraints
IF OBJECT_ID(N'dbo.Complaints', N'U') IS NOT NULL
BEGIN
    IF COL_LENGTH(N'dbo.Complaints', N'Severity') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_Complaints_Severity'
          AND parent_object_id = OBJECT_ID(N'dbo.Complaints')
    )
        ALTER TABLE dbo.Complaints
        ADD CONSTRAINT CK_Complaints_Severity
            CHECK ([Severity] IN (N'Low', N'Medium', N'High', N'Critical'));

    IF COL_LENGTH(N'dbo.Complaints', N'Status') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_Complaints_Status'
          AND parent_object_id = OBJECT_ID(N'dbo.Complaints')
    )
        ALTER TABLE dbo.Complaints
        ADD CONSTRAINT CK_Complaints_Status
            CHECK ([Status] IN (N'Open', N'Resolved', N'Escalated', N'Closed'));

    IF COL_LENGTH(N'dbo.Complaints', N'Channel') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_Complaints_Channel'
          AND parent_object_id = OBJECT_ID(N'dbo.Complaints')
    )
        ALTER TABLE dbo.Complaints
        ADD CONSTRAINT CK_Complaints_Channel
            CHECK ([Channel] IN (N'Email', N'Phone', N'In-Store', N'Website', N'Chat'));

    IF COL_LENGTH(N'dbo.Complaints', N'ResponseDays') IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM sys.check_constraints
        WHERE name = N'CK_Complaints_ResponseDays'
          AND parent_object_id = OBJECT_ID(N'dbo.Complaints')
    )
        ALTER TABLE dbo.Complaints
        ADD CONSTRAINT CK_Complaints_ResponseDays
            CHECK ([ResponseDays] >= 0);
END;
GO
