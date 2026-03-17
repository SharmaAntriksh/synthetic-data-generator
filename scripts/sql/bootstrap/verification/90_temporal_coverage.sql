SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[TemporalCoverage]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.Sales', N'U') IS NULL
       OR OBJECT_ID(N'dbo.Dates', N'U') IS NULL
        RETURN;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- Sales gap months
    DECLARE @gap_months INT;
    SELECT @gap_months = COUNT(*) FROM (
        SELECT d.[Year], d.[Month]
        FROM dbo.Dates d
        WHERE d.[Day] = 1
          AND d.Date BETWEEN (SELECT MIN(OrderDate) FROM dbo.Sales)
                         AND (SELECT MAX(OrderDate) FROM dbo.Sales)
          AND NOT EXISTS (
              SELECT 1 FROM dbo.Sales f
              WHERE YEAR(f.OrderDate) = d.[Year]
                AND MONTH(f.OrderDate) = d.[Month]
          )
    ) x;
    DECLARE @total_months INT;
    SELECT @total_months = COUNT(DISTINCT YEAR(OrderDate) * 100 + MONTH(OrderDate)) FROM dbo.Sales;
    INSERT INTO #R VALUES ('Temporal', 'Sales: no gap months',
        'Every month in the sales date range must have transactions',
        CASE WHEN @gap_months = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@total_months AS VARCHAR) + ' months covered, ' + CAST(@gap_months AS VARCHAR) + ' gaps');

    -- Date dim completeness
    DECLARE @actual_days INT, @expected_days INT;
    SELECT @actual_days = COUNT(*),
           @expected_days = DATEDIFF(DAY, MIN(Date), MAX(Date)) + 1
    FROM dbo.Dates;
    INSERT INTO #R VALUES ('Temporal', 'Date dim: no gaps',
        'Date table must have one row per calendar day with no missing days',
        CASE WHEN @actual_days = @expected_days THEN 'PASS' ELSE 'FAIL' END,
        CAST(@actual_days AS VARCHAR) + ' of ' + CAST(@expected_days AS VARCHAR) + ' days');

    -- FX coverage
    IF OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NOT NULL
    BEGIN
        DECLARE @fx_gaps INT;
        SELECT @fx_gaps = COUNT(*) FROM (
            SELECT DISTINCT f.OrderDate FROM dbo.Sales f
            LEFT JOIN dbo.ExchangeRates er ON er.Date = f.OrderDate
            WHERE er.Date IS NULL
        ) x;
        INSERT INTO #R VALUES ('Temporal', 'FX covers all sales dates',
            'ExchangeRates must have rates for every day with sales',
            CASE WHEN @fx_gaps = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@fx_gaps AS VARCHAR) + ' uncovered dates');

        -- INFO: FX range
        DECLARE @fx_start VARCHAR(10), @fx_end VARCHAR(10), @fx_currencies INT;
        SELECT @fx_start = CONVERT(VARCHAR, MIN(Date), 23),
               @fx_end   = CONVERT(VARCHAR, MAX(Date), 23),
               @fx_currencies = COUNT(DISTINCT ToCurrency)
        FROM dbo.ExchangeRates;
        INSERT INTO #R VALUES ('Temporal', 'FX coverage range',
            'Exchange rate date range and currency count',
            'INFO', @fx_start + ' to ' + @fx_end + ' (' + CAST(@fx_currencies AS VARCHAR) + ' currencies)');
    END

    -- Budget temporal
    IF OBJECT_ID(N'dbo.BudgetYearly', N'U') IS NOT NULL
    BEGIN
        DECLARE @min_sc INT, @max_sc INT;
        SELECT @min_sc = MIN(ScenarioCount), @max_sc = MAX(ScenarioCount)
        FROM (
            SELECT BudgetYear, COUNT(DISTINCT Scenario) AS ScenarioCount
            FROM dbo.BudgetYearly GROUP BY BudgetYear
        ) x;
        INSERT INTO #R VALUES ('Temporal', 'Budget: 3 scenarios per year',
            'Each budget year must have Low/Medium/High scenarios',
            CASE WHEN @min_sc = 3 AND @max_sc = 3 THEN 'PASS' ELSE 'FAIL' END,
            'min=' + CAST(@min_sc AS VARCHAR) + ' max=' + CAST(@max_sc AS VARCHAR) + ' per year');
    END

    -- Inventory temporal
    IF OBJECT_ID(N'dbo.InventorySnapshot', N'U') IS NOT NULL
    BEGIN
        DECLARE @inv_gaps INT;
        SELECT @inv_gaps = COUNT(*) FROM (
            SELECT d.[Year], d.[Month]
            FROM dbo.Dates d
            WHERE d.[Day] = 1
              AND d.Date BETWEEN (SELECT MIN(SnapshotDate) FROM dbo.InventorySnapshot)
                             AND (SELECT MAX(SnapshotDate) FROM dbo.InventorySnapshot)
              AND NOT EXISTS (
                  SELECT 1 FROM dbo.InventorySnapshot i
                  WHERE YEAR(i.SnapshotDate) = d.[Year]
                    AND MONTH(i.SnapshotDate) = d.[Month]
              )
        ) x;
        INSERT INTO #R VALUES ('Temporal', 'Inventory: no gap months',
            'Inventory snapshots must exist for every month in the range',
            CASE WHEN @inv_gaps = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@inv_gaps AS VARCHAR) + ' missing months');
    END

    -- INFO: overall date range
    DECLARE @sales_range VARCHAR(50);
    SELECT @sales_range = CONVERT(VARCHAR, MIN(OrderDate), 23) + ' to ' + CONVERT(VARCHAR, MAX(OrderDate), 23)
    FROM dbo.Sales;
    INSERT INTO #R VALUES ('Temporal', 'Sales date range',
        'Earliest to latest OrderDate',
        'INFO', @sales_range);

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
