SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- DIMENSIONS + LOOKUPS: CONSTRAINTS (PK/UQ/FK)
-- Aligned to src/utils/static_schemas.py
-----------------------------------------------------------------------

-----------------------------------------------------------------------
-- PRIMARY KEYS
-----------------------------------------------------------------------

-- Core dimensions
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers
    ADD CONSTRAINT PK_Customers PRIMARY KEY NONCLUSTERED ([CustomerKey]);
END;

IF OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Products'
      AND parent_object_id = OBJECT_ID(N'dbo.Products')
)
BEGIN
    ALTER TABLE dbo.Products
    ADD CONSTRAINT PK_Products PRIMARY KEY NONCLUSTERED ([ProductKey]);
END;

IF OBJECT_ID(N'dbo.ProductSubcategory', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_ProductSubcategory'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductSubcategory')
)
BEGIN
    ALTER TABLE dbo.ProductSubcategory
    ADD CONSTRAINT PK_ProductSubcategory PRIMARY KEY NONCLUSTERED ([SubcategoryKey]);
END;

IF OBJECT_ID(N'dbo.ProductCategory', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_ProductCategory'
      AND parent_object_id = OBJECT_ID(N'dbo.ProductCategory')
)
BEGIN
    ALTER TABLE dbo.ProductCategory
    ADD CONSTRAINT PK_ProductCategory PRIMARY KEY NONCLUSTERED ([CategoryKey]);
END;

IF OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Geography'
      AND parent_object_id = OBJECT_ID(N'dbo.Geography')
)
BEGIN
    ALTER TABLE dbo.Geography
    ADD CONSTRAINT PK_Geography PRIMARY KEY NONCLUSTERED ([GeographyKey]);
END;

IF OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Currency'
      AND parent_object_id = OBJECT_ID(N'dbo.Currency')
)
BEGIN
    ALTER TABLE dbo.Currency
    ADD CONSTRAINT PK_Currency PRIMARY KEY NONCLUSTERED ([CurrencyKey]);
END;

IF OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Dates'
      AND parent_object_id = OBJECT_ID(N'dbo.Dates')
)
BEGIN
    ALTER TABLE dbo.Dates
    -- Dates is keyed by [Date] (DATE) per current SQL model
    ADD CONSTRAINT PK_Dates PRIMARY KEY NONCLUSTERED ([Date]);
END;

IF OBJECT_ID(N'dbo.Time', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Time', N'TimeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Time'
      AND parent_object_id = OBJECT_ID(N'dbo.Time')
)
BEGIN
    ALTER TABLE dbo.Time
    ADD CONSTRAINT PK_Time PRIMARY KEY NONCLUSTERED ([TimeKey]);
END;

IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
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
    SELECT 1 FROM sys.key_constraints
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
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Promotions'
      AND parent_object_id = OBJECT_ID(N'dbo.Promotions')
)
BEGIN
    ALTER TABLE dbo.Promotions
    ADD CONSTRAINT PK_Promotions PRIMARY KEY NONCLUSTERED ([PromotionKey]);
END;

-- New lookup / auxiliary dimensions
IF OBJECT_ID(N'dbo.Suppliers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Suppliers', N'SupplierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Suppliers'
      AND parent_object_id = OBJECT_ID(N'dbo.Suppliers')
)
BEGIN
    ALTER TABLE dbo.Suppliers
    ADD CONSTRAINT PK_Suppliers PRIMARY KEY NONCLUSTERED ([SupplierKey]);
END;

IF OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Employees', N'EmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Employees'
      AND parent_object_id = OBJECT_ID(N'dbo.Employees')
)
BEGIN
    ALTER TABLE dbo.Employees
    ADD CONSTRAINT PK_Employees PRIMARY KEY NONCLUSTERED ([EmployeeKey]);
END;

IF OBJECT_ID(N'dbo.EmployeeStoreAssignments', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.EmployeeStoreAssignments', N'EmployeeKey') IS NOT NULL
AND COL_LENGTH(N'dbo.EmployeeStoreAssignments', N'StoreKey') IS NOT NULL
AND COL_LENGTH(N'dbo.EmployeeStoreAssignments', N'AssignmentSequence') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_EmployeeStoreAssignments'
      AND parent_object_id = OBJECT_ID(N'dbo.EmployeeStoreAssignments')
)
BEGIN
    -- Natural key: (EmployeeKey, StoreKey, AssignmentSequence)
    ALTER TABLE dbo.EmployeeStoreAssignments
    ADD CONSTRAINT PK_EmployeeStoreAssignments
        PRIMARY KEY NONCLUSTERED ([EmployeeKey], [StoreKey], [AssignmentSequence]);
END;

IF OBJECT_ID(N'dbo.LoyaltyTiers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.LoyaltyTiers', N'LoyaltyTierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_LoyaltyTiers'
      AND parent_object_id = OBJECT_ID(N'dbo.LoyaltyTiers')
)
BEGIN
    ALTER TABLE dbo.LoyaltyTiers
    ADD CONSTRAINT PK_LoyaltyTiers PRIMARY KEY NONCLUSTERED ([LoyaltyTierKey]);
END;

IF OBJECT_ID(N'dbo.SalesChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.SalesChannels', N'SalesChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_SalesChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.SalesChannels')
)
BEGIN
    ALTER TABLE dbo.SalesChannels
    ADD CONSTRAINT PK_SalesChannels PRIMARY KEY NONCLUSTERED ([SalesChannelKey]);
END;

IF OBJECT_ID(N'dbo.CustomerAcquisitionChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerAcquisitionChannels', N'CustomerAcquisitionChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_CustomerAcquisitionChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerAcquisitionChannels')
)
BEGIN
    ALTER TABLE dbo.CustomerAcquisitionChannels
    ADD CONSTRAINT PK_CustomerAcquisitionChannels
        PRIMARY KEY NONCLUSTERED ([CustomerAcquisitionChannelKey]);
END;

IF OBJECT_ID(N'dbo.CustomerSegment', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSegment', N'SegmentKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_CustomerSegment'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSegment')
)
BEGIN
    ALTER TABLE dbo.CustomerSegment
    ADD CONSTRAINT PK_CustomerSegment PRIMARY KEY NONCLUSTERED ([SegmentKey]);
END;

IF OBJECT_ID(N'dbo.CustomerSegmentMembership', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSegmentMembership', N'CustomerKey') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSegmentMembership', N'SegmentKey') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSegmentMembership', N'ValidFromDate') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_CustomerSegmentMembership'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSegmentMembership')
)
BEGIN
    ALTER TABLE dbo.CustomerSegmentMembership
    ADD CONSTRAINT PK_CustomerSegmentMembership
        PRIMARY KEY NONCLUSTERED ([CustomerKey], [SegmentKey], [ValidFromDate]);
END;

IF OBJECT_ID(N'dbo.Superpowers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Superpowers', N'SuperpowerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_Superpowers'
      AND parent_object_id = OBJECT_ID(N'dbo.Superpowers')
)
BEGIN
    ALTER TABLE dbo.Superpowers
    ADD CONSTRAINT PK_Superpowers PRIMARY KEY NONCLUSTERED ([SuperpowerKey]);
END;

IF OBJECT_ID(N'dbo.CustomerSuperpowers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSuperpowers', N'CustomerKey') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSuperpowers', N'SuperpowerKey') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSuperpowers', N'ValidFromDate') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_CustomerSuperpowers'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSuperpowers')
)
BEGIN
    ALTER TABLE dbo.CustomerSuperpowers
    ADD CONSTRAINT PK_CustomerSuperpowers
        PRIMARY KEY NONCLUSTERED ([CustomerKey], [SuperpowerKey], [ValidFromDate]);
END;

IF OBJECT_ID(N'dbo.ReturnReason', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ReturnReason', N'ReturnReasonKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
    WHERE name = N'PK_ReturnReason'
      AND parent_object_id = OBJECT_ID(N'dbo.ReturnReason')
)
BEGIN
    ALTER TABLE dbo.ReturnReason
    ADD CONSTRAINT PK_ReturnReason PRIMARY KEY NONCLUSTERED ([ReturnReasonKey]);
END;

-----------------------------------------------------------------------
-- UNIQUE CONSTRAINTS
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.key_constraints
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
AND OBJECT_ID(N'dbo.ProductSubcategory', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Products', N'SubcategoryKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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
AND OBJECT_ID(N'dbo.ProductCategory', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ProductSubcategory', N'CategoryKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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

-- Products -> Suppliers
IF OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Suppliers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Products', N'SupplierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Products_Suppliers'
      AND parent_object_id = OBJECT_ID(N'dbo.Products')
)
BEGIN
    ALTER TABLE dbo.Products WITH CHECK
    ADD CONSTRAINT FK_Products_Suppliers
        FOREIGN KEY ([SupplierKey])
        REFERENCES dbo.Suppliers ([SupplierKey]);

    ALTER TABLE dbo.Products CHECK CONSTRAINT FK_Products_Suppliers;
END;

-- Products self-reference (BaseProductKey -> ProductKey)
IF OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Products', N'BaseProductKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Products_BaseProduct'
      AND parent_object_id = OBJECT_ID(N'dbo.Products')
)
BEGIN
    ALTER TABLE dbo.Products WITH CHECK
    ADD CONSTRAINT FK_Products_BaseProduct
        FOREIGN KEY ([BaseProductKey])
        REFERENCES dbo.Products ([ProductKey]);

    ALTER TABLE dbo.Products CHECK CONSTRAINT FK_Products_BaseProduct;
END;

-- Customer geography
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Customers', N'GeographyKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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

-- Customers -> LoyaltyTiers
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.LoyaltyTiers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Customers', N'LoyaltyTierKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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

-- Customers -> CustomerAcquisitionChannels
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.CustomerAcquisitionChannels', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Customers', N'CustomerAcquisitionChannelKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Customers_CustomerAcquisitionChannels'
      AND parent_object_id = OBJECT_ID(N'dbo.Customers')
)
BEGIN
    ALTER TABLE dbo.Customers WITH CHECK
    ADD CONSTRAINT FK_Customers_CustomerAcquisitionChannels
        FOREIGN KEY ([CustomerAcquisitionChannelKey])
        REFERENCES dbo.CustomerAcquisitionChannels ([CustomerAcquisitionChannelKey]);

    ALTER TABLE dbo.Customers CHECK CONSTRAINT FK_Customers_CustomerAcquisitionChannels;
END;

-- Stores -> Geography
IF OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Stores', N'GeographyKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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

-- Employees self hierarchy
IF OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.Employees', N'ParentEmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_Employees_ParentEmployee'
      AND parent_object_id = OBJECT_ID(N'dbo.Employees')
)
BEGIN
    ALTER TABLE dbo.Employees WITH CHECK
    ADD CONSTRAINT FK_Employees_ParentEmployee
        FOREIGN KEY ([ParentEmployeeKey])
        REFERENCES dbo.Employees ([EmployeeKey]);

    ALTER TABLE dbo.Employees CHECK CONSTRAINT FK_Employees_ParentEmployee;
END;

-- EmployeeStoreAssignments links
IF OBJECT_ID(N'dbo.EmployeeStoreAssignments', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.EmployeeStoreAssignments', N'EmployeeKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_EmployeeStoreAssignments_Employees'
      AND parent_object_id = OBJECT_ID(N'dbo.EmployeeStoreAssignments')
)
BEGIN
    ALTER TABLE dbo.EmployeeStoreAssignments WITH CHECK
    ADD CONSTRAINT FK_EmployeeStoreAssignments_Employees
        FOREIGN KEY ([EmployeeKey])
        REFERENCES dbo.Employees ([EmployeeKey]);

    ALTER TABLE dbo.EmployeeStoreAssignments CHECK CONSTRAINT FK_EmployeeStoreAssignments_Employees;
END;

IF OBJECT_ID(N'dbo.EmployeeStoreAssignments', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.EmployeeStoreAssignments', N'StoreKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_EmployeeStoreAssignments_Stores'
      AND parent_object_id = OBJECT_ID(N'dbo.EmployeeStoreAssignments')
)
BEGIN
    ALTER TABLE dbo.EmployeeStoreAssignments WITH CHECK
    ADD CONSTRAINT FK_EmployeeStoreAssignments_Stores
        FOREIGN KEY ([StoreKey])
        REFERENCES dbo.Stores ([StoreKey]);

    ALTER TABLE dbo.EmployeeStoreAssignments CHECK CONSTRAINT FK_EmployeeStoreAssignments_Stores;
END;

-- Customer segment membership links
IF OBJECT_ID(N'dbo.CustomerSegmentMembership', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSegmentMembership', N'CustomerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_CustomerSegmentMembership_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSegmentMembership')
)
BEGIN
    ALTER TABLE dbo.CustomerSegmentMembership WITH CHECK
    ADD CONSTRAINT FK_CustomerSegmentMembership_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.CustomerSegmentMembership CHECK CONSTRAINT FK_CustomerSegmentMembership_Customers;
END;

IF OBJECT_ID(N'dbo.CustomerSegmentMembership', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.CustomerSegment', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSegmentMembership', N'SegmentKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_CustomerSegmentMembership_CustomerSegment'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSegmentMembership')
)
BEGIN
    ALTER TABLE dbo.CustomerSegmentMembership WITH CHECK
    ADD CONSTRAINT FK_CustomerSegmentMembership_CustomerSegment
        FOREIGN KEY ([SegmentKey])
        REFERENCES dbo.CustomerSegment ([SegmentKey]);

    ALTER TABLE dbo.CustomerSegmentMembership CHECK CONSTRAINT FK_CustomerSegmentMembership_CustomerSegment;
END;

-- Customer superpowers links
IF OBJECT_ID(N'dbo.CustomerSuperpowers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSuperpowers', N'CustomerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_CustomerSuperpowers_Customers'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSuperpowers')
)
BEGIN
    ALTER TABLE dbo.CustomerSuperpowers WITH CHECK
    ADD CONSTRAINT FK_CustomerSuperpowers_Customers
        FOREIGN KEY ([CustomerKey])
        REFERENCES dbo.Customers ([CustomerKey]);

    ALTER TABLE dbo.CustomerSuperpowers CHECK CONSTRAINT FK_CustomerSuperpowers_Customers;
END;

IF OBJECT_ID(N'dbo.CustomerSuperpowers', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Superpowers', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.CustomerSuperpowers', N'SuperpowerKey') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = N'FK_CustomerSuperpowers_Superpowers'
      AND parent_object_id = OBJECT_ID(N'dbo.CustomerSuperpowers')
)
BEGIN
    ALTER TABLE dbo.CustomerSuperpowers WITH CHECK
    ADD CONSTRAINT FK_CustomerSuperpowers_Superpowers
        FOREIGN KEY ([SuperpowerKey])
        REFERENCES dbo.Superpowers ([SuperpowerKey]);

    ALTER TABLE dbo.CustomerSuperpowers CHECK CONSTRAINT FK_CustomerSuperpowers_Superpowers;
END;

-- ExchangeRates links
IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ExchangeRates', N'FromCurrency') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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
AND OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ExchangeRates', N'ToCurrency') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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

-- ExchangeRates -> Dates
IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
AND COL_LENGTH(N'dbo.ExchangeRates', N'Date') IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
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
