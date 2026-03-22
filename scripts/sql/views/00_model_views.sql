SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

-----------------------------------------------------------------------
-- DIMENSION VIEWS
--
-- Every view is guarded with IF OBJECT_ID so the script is resilient
-- to partial deployments (e.g. a minimal config that omits LoyaltyTiers,
-- Subscriptions, etc.).
--
-- All guards use EXEC() to work around the CREATE VIEW must-be-first-
-- statement-in-batch rule.
-----------------------------------------------------------------------

-- Currency
IF OBJECT_ID(N'dbo.Currency', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Currency] AS SELECT * FROM [dbo].[Currency];');
GO

-- Customers
IF OBJECT_ID(N'dbo.Customers', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Customers] AS SELECT * FROM [dbo].[Customers];');
GO

-- CustomerProfile
IF OBJECT_ID(N'dbo.CustomerProfile', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_CustomerProfile] AS SELECT * FROM [dbo].[CustomerProfile];');
GO

-- OrganizationProfile
IF OBJECT_ID(N'dbo.OrganizationProfile', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_OrganizationProfile] AS SELECT * FROM [dbo].[OrganizationProfile];');
GO

-- CustomerAcquisitionChannels
IF OBJECT_ID(N'dbo.CustomerAcquisitionChannels', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_CustomerAcquisitionChannels] AS SELECT * FROM [dbo].[CustomerAcquisitionChannels];');
GO

-- CustomerWishlists (optional)
IF OBJECT_ID(N'dbo.CustomerWishlists', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_CustomerWishlists] AS SELECT * FROM [dbo].[CustomerWishlists];');
GO

-- Complaints (optional)
IF OBJECT_ID(N'dbo.Complaints', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Complaints] AS SELECT * FROM [dbo].[Complaints];');
GO

-- CustomerSubscriptions (optional)
IF OBJECT_ID(N'dbo.CustomerSubscriptions', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_CustomerSubscriptions] AS SELECT * FROM [dbo].[CustomerSubscriptions];');
GO

-- Plans (optional)
IF OBJECT_ID(N'dbo.Plans', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Plans] AS SELECT * FROM [dbo].[Plans];');
GO

-- Dates
IF OBJECT_ID(N'dbo.Dates', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Dates] AS SELECT * FROM [dbo].[Dates];');
GO

-- Time
IF OBJECT_ID(N'dbo.Time', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Time] AS SELECT * FROM [dbo].[Time];');
GO

-- Employees
IF OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Employees] AS SELECT * FROM [dbo].[Employees];');
GO

-- EmployeeStoreAssignments
IF OBJECT_ID(N'dbo.EmployeeStoreAssignments', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_EmployeeStoreAssignments] AS SELECT * FROM [dbo].[EmployeeStoreAssignments];');
GO

-- ExchangeRates
IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_ExchangeRates] AS SELECT * FROM [dbo].[ExchangeRates];');
GO

-- Geography
IF OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Geography] AS SELECT * FROM [dbo].[Geography];');
GO

-- LoyaltyTiers (optional)
IF OBJECT_ID(N'dbo.LoyaltyTiers', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_LoyaltyTiers] AS SELECT * FROM [dbo].[LoyaltyTiers];');
GO

-- ProductCategory
IF OBJECT_ID(N'dbo.ProductCategory', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_ProductCategory] AS SELECT * FROM [dbo].[ProductCategory];');
GO

-- Products
IF OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Products] AS SELECT * FROM [dbo].[Products];');
GO

-- ProductProfile
IF OBJECT_ID(N'dbo.ProductProfile', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_ProductProfile] AS SELECT * FROM [dbo].[ProductProfile];');
GO

-- ProductSubcategory
IF OBJECT_ID(N'dbo.ProductSubcategory', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_ProductSubcategory] AS SELECT * FROM [dbo].[ProductSubcategory];');
GO

-- Promotions
IF OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Promotions] AS SELECT * FROM [dbo].[Promotions];');
GO

-- ReturnReason
IF OBJECT_ID(N'dbo.ReturnReason', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_ReturnReason] AS SELECT * FROM [dbo].[ReturnReason];');
GO

-- SalesChannels
IF OBJECT_ID(N'dbo.SalesChannels', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_SalesChannels] AS SELECT * FROM [dbo].[SalesChannels];');
GO

-- Stores
IF OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Stores] AS SELECT * FROM [dbo].[Stores];');
GO

-- Suppliers
IF OBJECT_ID(N'dbo.Suppliers', N'U') IS NOT NULL
    EXEC('CREATE OR ALTER VIEW [dbo].[vw_Suppliers] AS SELECT * FROM [dbo].[Suppliers];');
GO

-----------------------------------------------------------------------
-- FACT VIEWS (conditional)
--   - sales_output: sales       -> vw_Sales from Sales
--   - sales_output: sales_order -> vw_SalesOrderHeader/Detail + vw_Sales join
--   - sales_output: both        -> vw_Sales from Sales + vw_SalesOrderHeader/Detail
--
-- FIX: DECIMAL(19,4) replaces MONEY to avoid silent precision loss
--      during intermediate arithmetic (MONEY uses integer division).
-- FIX: Schema two-part name now uses QUOTENAME consistently
--      (was unquoted for COL_LENGTH, broke on special-char schemas).
-----------------------------------------------------------------------

DECLARE @SalesSchema  sysname = NULL;
DECLARE @HdrSchema    sysname = NULL;
DECLARE @DtlSchema    sysname = NULL;
DECLARE @ReturnSchema sysname = NULL;
DECLARE @InvSchema    sysname = NULL;

SELECT TOP (1) @SalesSchema = s.name
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE t.name = N'Sales';

SELECT TOP (1) @HdrSchema = s.name
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE t.name = N'SalesOrderHeader';

SELECT TOP (1) @DtlSchema = s.name
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE t.name = N'SalesOrderDetail';

SELECT TOP (1) @ReturnSchema = s.name
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE t.name = N'SalesReturn';

SELECT TOP (1) @InvSchema = s.name
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE t.name = N'InventorySnapshot';

-----------------------------------------------------------------------
-- vw_SalesOrderHeader (if present)
-----------------------------------------------------------------------
IF @HdrSchema IS NOT NULL
BEGIN
    DECLARE @HdrFrom nvarchar(300) =
        QUOTENAME(@HdrSchema) + N'.' + QUOTENAME(N'SalesOrderHeader');

    DECLARE @sql_hdr nvarchar(max) = N'
CREATE OR ALTER VIEW [dbo].[vw_SalesOrderHeader]
AS
SELECT
    SalesOrderNumber,
    CustomerKey,
    StoreKey,
    EmployeeKey,
    PromotionKey,
    CurrencyKey,
    SalesChannelKey,
    OrderDate,
    TimeKey,
    IsOrderDelayed
FROM ' + @HdrFrom + N';';

    EXEC sys.sp_executesql @sql_hdr;
END;

-----------------------------------------------------------------------
-- vw_SalesOrderDetail (if present)
-----------------------------------------------------------------------
IF @DtlSchema IS NOT NULL
BEGIN
    DECLARE @DtlFrom nvarchar(300) =
        QUOTENAME(@DtlSchema) + N'.' + QUOTENAME(N'SalesOrderDetail');

    DECLARE @sql_dtl nvarchar(max) = N'
CREATE OR ALTER VIEW [dbo].[vw_SalesOrderDetail]
AS
SELECT
    SalesOrderNumber,
    SalesOrderLineNumber,
    ProductKey,
    DueDate,
    DeliveryDate,
    Quantity,
    CAST(NetPrice       AS decimal(19,4)) AS NetPrice,
    CAST(UnitCost       AS decimal(19,4)) AS UnitCost,
    CAST(UnitPrice      AS decimal(19,4)) AS UnitPrice,
    CAST(DiscountAmount AS decimal(19,4)) AS DiscountAmount,
    DeliveryStatus
FROM ' + @DtlFrom + N';';

    EXEC sys.sp_executesql @sql_dtl;
END;

-----------------------------------------------------------------------
-- vw_SalesReturn (if present)
-----------------------------------------------------------------------
IF @ReturnSchema IS NOT NULL
BEGIN
    DECLARE @ReturnFrom nvarchar(300) =
        QUOTENAME(@ReturnSchema) + N'.' + QUOTENAME(N'SalesReturn');

    DECLARE @sql_ret nvarchar(max) = N'
CREATE OR ALTER VIEW [dbo].[vw_SalesReturn]
AS
SELECT
    ReturnEventKey,
    SalesOrderNumber,
    SalesOrderLineNumber,
    ReturnDate,
    ReturnReasonKey,
    ReturnQuantity,
    CAST(ReturnNetPrice AS decimal(19,4)) AS ReturnNetPrice
FROM ' + @ReturnFrom + N';';

    EXEC sys.sp_executesql @sql_ret;
END;

-----------------------------------------------------------------------
-- vw_InventorySnapshot (if present)
-----------------------------------------------------------------------
IF @InvSchema IS NOT NULL
BEGIN
    DECLARE @InvFrom nvarchar(300) =
        QUOTENAME(@InvSchema) + N'.' + QUOTENAME(N'InventorySnapshot');

    DECLARE @sql_inv nvarchar(max) = N'
CREATE OR ALTER VIEW [dbo].[vw_InventorySnapshot]
AS
SELECT *
FROM ' + @InvFrom + N';';

    EXEC sys.sp_executesql @sql_inv;
END;

-----------------------------------------------------------------------
-- vw_Sales (Sales table if present; otherwise join Header+Detail)
-----------------------------------------------------------------------
IF @SalesSchema IS NOT NULL
BEGIN
    -- FIX: both COL_LENGTH lookup and FROM clause use QUOTENAME consistently
    DECLARE @SalesQualified nvarchar(300) =
        QUOTENAME(@SalesSchema) + N'.' + QUOTENAME(N'Sales');

    DECLARE @selectPrefix nvarchar(max) = N'';

    -- Only include order cols if they exist in the Sales table
    IF COL_LENGTH(@SalesQualified, 'SalesOrderNumber') IS NOT NULL
       AND COL_LENGTH(@SalesQualified, 'SalesOrderLineNumber') IS NOT NULL
    BEGIN
        SET @selectPrefix = N'
        SalesOrderNumber,
        SalesOrderLineNumber,';
    END

    DECLARE @sql_sales nvarchar(max) = N'
CREATE OR ALTER VIEW [dbo].[vw_Sales]
AS
SELECT'
+ @selectPrefix + N'
        CustomerKey,
        ProductKey,
        StoreKey,
        EmployeeKey,
        PromotionKey,
        CurrencyKey,
        SalesChannelKey,
        TimeKey,
        OrderDate,
        DueDate,
        DeliveryDate,
        Quantity,
        CAST(NetPrice         AS decimal(19,4)) AS NetPrice,
        CAST(UnitCost         AS decimal(19,4)) AS UnitCost,
        CAST(UnitPrice        AS decimal(19,4)) AS UnitPrice,
        CAST(DiscountAmount   AS decimal(19,4)) AS DiscountAmount,
        DeliveryStatus,
        IsOrderDelayed
FROM ' + @SalesQualified + N';';

    EXEC sys.sp_executesql @sql_sales;
END
ELSE IF @HdrSchema IS NOT NULL AND @DtlSchema IS NOT NULL
BEGIN
    DECLARE @HdrFrom2 nvarchar(300) =
        QUOTENAME(@HdrSchema) + N'.' + QUOTENAME(N'SalesOrderHeader');

    DECLARE @DtlFrom2 nvarchar(300) =
        QUOTENAME(@DtlSchema) + N'.' + QUOTENAME(N'SalesOrderDetail');

    DECLARE @sql_sales_join nvarchar(max) = N'
CREATE OR ALTER VIEW [dbo].[vw_Sales]
AS
SELECT
    d.SalesOrderNumber,
    d.SalesOrderLineNumber,
    h.CustomerKey,
    d.ProductKey,
    h.StoreKey,
    h.EmployeeKey,
    h.PromotionKey,
    h.CurrencyKey,
    h.SalesChannelKey,
    h.TimeKey,
    h.OrderDate,
    d.DueDate,
    d.DeliveryDate,
    d.Quantity,
    CAST(d.NetPrice         AS decimal(19,4)) AS NetPrice,
    CAST(d.UnitCost         AS decimal(19,4)) AS UnitCost,
    CAST(d.UnitPrice        AS decimal(19,4)) AS UnitPrice,
    CAST(d.DiscountAmount   AS decimal(19,4)) AS DiscountAmount,
    d.DeliveryStatus,
    h.IsOrderDelayed
FROM ' + @DtlFrom2 + N' AS d
JOIN ' + @HdrFrom2 + N' AS h
  ON h.SalesOrderNumber = d.SalesOrderNumber;';

    EXEC sys.sp_executesql @sql_sales_join;
END
ELSE
BEGIN
    THROW 50020, 'No fact tables found for vw_Sales. Expected Sales OR (SalesOrderHeader + SalesOrderDetail).', 1;
END;
GO
