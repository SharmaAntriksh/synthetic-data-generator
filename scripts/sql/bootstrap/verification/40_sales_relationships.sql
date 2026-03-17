SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[SalesRelationships]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.Sales', N'U') IS NULL
       OR OBJECT_ID(N'dbo.Stores', N'U') IS NULL
       OR OBJECT_ID(N'dbo.SalesChannels', N'U') IS NULL
        RETURN;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- ================================================================
    -- CORRELATIONS
    -- ================================================================

    -- Store-Channel correlation
    DECLARE @phys_pct DECIMAL(5,1);
    SELECT @phys_pct = ISNULL(
        SUM(CASE WHEN sc.IsPhysical = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
    FROM dbo.Sales f
    JOIN dbo.Stores s         ON s.StoreKey = f.StoreKey
    JOIN dbo.SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
    WHERE s.StoreType NOT IN ('Online', 'Fulfillment');
    INSERT INTO #R VALUES ('Correlation', 'Physical stores use physical channels >50%',
        'Physical stores should skew toward physical channels',
        CASE WHEN @phys_pct > 50 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@phys_pct AS VARCHAR) + '%');

    -- Customer-Store geo
    IF OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
    BEGIN
        DECLARE @geo_pct DECIMAL(5,1);
        SELECT @geo_pct = ISNULL(
            SUM(CASE WHEN cg.Country = sg.Country THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
        FROM dbo.Sales f
        JOIN dbo.Customers c  ON c.CustomerKey = f.CustomerKey AND c.IsCurrent = 1
        JOIN dbo.Stores s     ON s.StoreKey = f.StoreKey
        JOIN dbo.Geography cg ON cg.GeographyKey = c.GeographyKey
        JOIN dbo.Geography sg ON sg.GeographyKey = s.GeographyKey;
        INSERT INTO #R VALUES ('Correlation', 'Same-country sales >60%',
            'Customers should mostly buy from stores in their own country',
            CASE WHEN @geo_pct > 60 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@geo_pct AS VARCHAR) + '%');
    END

    -- Channel-Delivery
    DECLARE @avg_due DECIMAL(5,1);
    SELECT @avg_due = ISNULL(AVG(CAST(DATEDIFF(DAY, f.OrderDate, f.DueDate) AS FLOAT)), 0)
    FROM dbo.Sales f WHERE f.SalesChannelKey IN (1, 10);
    INSERT INTO #R VALUES ('Correlation', 'Store/Kiosk avg due days <2',
        'Physical channel orders should have near-zero fulfillment offset',
        CASE WHEN @avg_due < 2 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@avg_due AS VARCHAR) + ' days');

    -- Channel-Product eligibility
    IF OBJECT_ID(N'dbo.ProductProfile', N'U') IS NOT NULL
    BEGIN
        DECLARE @elig_pct DECIMAL(5,1);
        SELECT @elig_pct = ISNULL(
            SUM(CASE
                WHEN f.SalesChannelKey IN (1, 10)   THEN pp.EligibleStore
                WHEN f.SalesChannelKey IN (2, 6, 7) THEN pp.EligibleOnline
                WHEN f.SalesChannelKey IN (3, 8)    THEN pp.EligibleMarketplace
                WHEN f.SalesChannelKey IN (4, 9)    THEN pp.EligibleB2B
                ELSE 1 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
        FROM dbo.Sales f
        JOIN dbo.Products p        ON p.ProductKey = f.ProductKey
        JOIN dbo.ProductProfile pp ON pp.ProductKey = p.ProductKey;
        INSERT INTO #R VALUES ('Correlation', 'Channel eligibility compliance >95%',
            'Products sold through a channel should be eligible for that channel',
            CASE WHEN @elig_pct > 95 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@elig_pct AS VARCHAR) + '%');
    END

    -- Channel-Promo alignment
    IF OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
    BEGIN
        DECLARE @promo_mis DECIMAL(5,1);
        SELECT @promo_mis = ISNULL(
            SUM(CASE WHEN sc.IsDigital = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
        FROM dbo.Sales f
        JOIN dbo.Promotions pr    ON pr.PromotionKey = f.PromotionKey
        JOIN dbo.SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
        WHERE pr.PromotionCategory = 'Store' AND pr.PromotionKey > 1;
        INSERT INTO #R VALUES ('Correlation', 'Store promo on digital channel <5%',
            'Store-category promotions should not appear on digital channels',
            CASE WHEN @promo_mis < 5 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@promo_mis AS VARCHAR) + '%');
    END

    -- ================================================================
    -- REFERENTIAL INTEGRITY (Sales FKs)
    -- ================================================================

    -- Currency FK
    DECLARE @bad_curr INT;
    SELECT @bad_curr = COUNT(DISTINCT f.CurrencyKey)
    FROM dbo.Sales f LEFT JOIN dbo.Currency c ON c.CurrencyKey = f.CurrencyKey
    WHERE c.CurrencyKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'Valid CurrencyKey',
        'Every CurrencyKey in Sales must exist in Currency',
        CASE WHEN @bad_curr = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_curr AS VARCHAR) + ' orphaned keys');

    -- Sales → Customers FK
    DECLARE @bad_cust INT;
    SELECT @bad_cust = COUNT(DISTINCT f.CustomerKey)
    FROM dbo.Sales f LEFT JOIN dbo.Customers c ON c.CustomerKey = f.CustomerKey
    WHERE c.CustomerKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'Valid CustomerKey',
        'Every CustomerKey in Sales must exist in Customers',
        CASE WHEN @bad_cust = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_cust AS VARCHAR) + ' orphaned keys');

    -- Sales → Products FK
    DECLARE @bad_prod INT;
    SELECT @bad_prod = COUNT(DISTINCT f.ProductKey)
    FROM dbo.Sales f LEFT JOIN dbo.Products p ON p.ProductKey = f.ProductKey
    WHERE p.ProductKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'Valid ProductKey',
        'Every ProductKey in Sales must exist in Products',
        CASE WHEN @bad_prod = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_prod AS VARCHAR) + ' orphaned keys');

    -- ================================================================
    -- PROMOTION CHECKS
    -- ================================================================
    IF OBJECT_ID(N'dbo.Promotions', N'U') IS NOT NULL
    BEGIN
        -- Sentinel check
        DECLARE @sentinel INT;
        SELECT @sentinel = COUNT(*) FROM dbo.Promotions WHERE PromotionKey = 1 AND DiscountPct = 0;
        INSERT INTO #R VALUES ('Domain', 'No Discount sentinel at key 1',
            'PromotionKey=1 must be the No Discount placeholder with 0% discount',
            CASE WHEN @sentinel = 1 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@sentinel AS VARCHAR) + ' matching rows');

        -- DiscountPct range
        DECLARE @bad_disc INT;
        SELECT @bad_disc = COUNT(*) FROM dbo.Promotions WHERE DiscountPct < 0 OR DiscountPct > 1;
        INSERT INTO #R VALUES ('Domain', 'DiscountPct in [0,1]',
            'Promotion discount must be between 0 and 100%',
            CASE WHEN @bad_disc = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@bad_disc AS VARCHAR) + ' out of range');

        -- Promo date sanity
        DECLARE @promo_inv INT;
        SELECT @promo_inv = COUNT(*) FROM dbo.Promotions WHERE StartDate > EndDate;
        INSERT INTO #R VALUES ('DateSanity', 'Promotion StartDate <= EndDate',
            'Promotion cannot end before it starts',
            CASE WHEN @promo_inv = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@promo_inv AS VARCHAR) + ' inversions');
    END

    -- ================================================================
    -- INFO
    -- ================================================================
    DECLARE @total_sales INT;
    SELECT @total_sales = COUNT(*) FROM dbo.Sales;
    INSERT INTO #R VALUES ('Info', 'Sales row count',
        'Total sales line items',
        'INFO', FORMAT(@total_sales, 'N0'));

    DECLARE @date_range VARCHAR(50);
    SELECT @date_range = CONVERT(VARCHAR, MIN(OrderDate), 23) + ' to ' + CONVERT(VARCHAR, MAX(OrderDate), 23)
    FROM dbo.Sales;
    INSERT INTO #R VALUES ('Info', 'Sales date range',
        'Earliest to latest OrderDate',
        'INFO', @date_range);

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
