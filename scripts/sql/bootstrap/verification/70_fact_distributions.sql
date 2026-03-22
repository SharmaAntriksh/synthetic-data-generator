SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[FactDistributions]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.Sales', N'U') IS NULL
        RETURN;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- Monthly CoV
    DECLARE @cov DECIMAL(5,2);
    SELECT @cov = ISNULL(STDEV(MonthlySales) / NULLIF(AVG(MonthlySales), 0), 0)
    FROM (
        SELECT YEAR(OrderDate) * 100 + MONTH(OrderDate) AS YM, CAST(COUNT(*) AS FLOAT) AS MonthlySales
        FROM dbo.Sales GROUP BY YEAR(OrderDate), MONTH(OrderDate)
    ) m;
    INSERT INTO #R VALUES ('Distribution', 'Monthly CoV > 0.10',
        'Monthly sales volume should vary (seasonal demand); flat data suggests broken demand curve',
        CASE WHEN @cov > 0.10 THEN 'PASS' ELSE 'FAIL' END,
        'CoV = ' + CAST(@cov AS VARCHAR));

    -- Customer concentration
    DECLARE @max_cust_pct DECIMAL(5,2);
    SELECT @max_cust_pct = ISNULL(MAX(CustPct), 0)
    FROM (
        SELECT SUM(NetPrice) * 100.0 / NULLIF((SELECT SUM(NetPrice) FROM dbo.Sales), 0) AS CustPct
        FROM dbo.Sales GROUP BY CustomerKey
    ) x;
    INSERT INTO #R VALUES ('Distribution', 'No customer > 5% of revenue',
        'No single customer should dominate total revenue',
        CASE WHEN @max_cust_pct < 5.0 THEN 'PASS' ELSE 'FAIL' END,
        'max = ' + CAST(@max_cust_pct AS VARCHAR) + '%');

    -- Quantity mode
    DECLARE @qty1_pct DECIMAL(5,1);
    SELECT @qty1_pct = ISNULL(
        SUM(CASE WHEN Quantity = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
    FROM dbo.Sales;
    INSERT INTO #R VALUES ('Distribution', 'Quantity distribution peaks at 1',
        'Poisson quantity model means most lines have Qty=1 (>30%)',
        CASE WHEN @qty1_pct > 30 THEN 'PASS' ELSE 'FAIL' END,
        'Qty=1 is ' + CAST(@qty1_pct AS VARCHAR) + '%');

    -- NetPrice <= UnitPrice
    DECLARE @net_exceed INT;
    SELECT @net_exceed = COUNT(*) FROM dbo.Sales WHERE NetPrice > UnitPrice;
    INSERT INTO #R VALUES ('Domain', 'NetPrice <= UnitPrice',
        'Selling price should not exceed sticker price',
        CASE WHEN @net_exceed = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@net_exceed AS VARCHAR) + ' violations');

    -- Delivery delay rate
    DECLARE @delay_pct DECIMAL(5,1);
    SELECT @delay_pct = ISNULL(
        SUM(CAST(IsOrderDelayed AS INT)) * 100.0 / NULLIF(COUNT(*), 0), 0)
    FROM dbo.Sales;
    INSERT INTO #R VALUES ('Distribution', 'Delayed orders < 20%',
        'Order delay rate should be a minority',
        CASE WHEN @delay_pct < 20 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@delay_pct AS VARCHAR) + '% delayed');

    -- INFO: average order value
    DECLARE @aov DECIMAL(10,2);
    SELECT @aov = ISNULL(AVG(NetPrice * Quantity), 0) FROM dbo.Sales;
    INSERT INTO #R VALUES ('Info', 'Average order line value',
        'Average NetPrice * Quantity per line item',
        'INFO', '$' + CAST(@aov AS VARCHAR));

    -- INFO: discount utilization
    DECLARE @disc_pct DECIMAL(5,1);
    SELECT @disc_pct = ISNULL(
        SUM(CASE WHEN DiscountAmount > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
    FROM dbo.Sales;
    INSERT INTO #R VALUES ('Info', 'Discount utilization',
        'Percentage of line items with DiscountAmount > 0',
        'INFO', CAST(@disc_pct AS VARCHAR) + '% discounted');

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
