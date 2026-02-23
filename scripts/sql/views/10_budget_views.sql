SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

-----------------------------------------------------------------------
-- BUDGET ENGINE (Yearly)
-----------------------------------------------------------------------
CREATE OR ALTER   VIEW [dbo].[vw_Budget]
AS
WITH
-- 1) Actuals at low grain (Country, Year, Category)
Actuals AS (
    SELECT
        g.Country,
        d.Year AS ActualYear,
        pc.Category,
        SUM(f.Quantity * f.NetPrice) AS ActualSalesAmount,
        SUM(COALESCE(f.Quantity, 0))  AS ActualSalesQuantity
    FROM dbo.vw_Sales f
    JOIN dbo.vw_Dates d ON d.Date = f.OrderDate
    JOIN dbo.vw_Stores s ON s.StoreKey = f.StoreKey
    JOIN dbo.vw_Geography g ON g.GeographyKey = s.GeographyKey
    JOIN dbo.vw_Products p ON p.ProductKey = f.ProductKey
    JOIN dbo.vw_ProductSubcategory ps ON ps.SubcategoryKey = p.SubcategoryKey
    JOIN dbo.vw_ProductCategory pc ON pc.CategoryKey = ps.CategoryKey
    GROUP BY g.Country, d.Year, pc.Category
),

YearBounds AS (
    SELECT MIN(ActualYear) AS MinSalesYear,
           MAX(ActualYear) AS MaxSalesYear
    FROM Actuals
),

-- 2) Local YoY at (Country, Category)
CountryYoY AS (
    SELECT
        a.*,
        LAG(a.ActualSalesAmount) OVER (
            PARTITION BY a.Country, a.Category
            ORDER BY a.ActualYear
        ) AS PrevSalesAmount
    FROM Actuals a
),

-- 3) Category-Year and Global-Year rolling YoY (fallback tiers)
CatYear AS (
    SELECT Category, ActualYear, SUM(ActualSalesAmount) AS CatSalesAmount
    FROM Actuals
    GROUP BY Category, ActualYear
),
CatYearYoY AS (
    SELECT
        cy.*,
        LAG(cy.CatSalesAmount) OVER (PARTITION BY cy.Category ORDER BY cy.ActualYear) AS PrevCatSalesAmount
    FROM CatYear cy
),
CatRoll AS (
    SELECT
        Category,
        ActualYear,
        AVG(
            CASE
                WHEN PrevCatSalesAmount IS NULL OR PrevCatSalesAmount = 0 THEN NULL
                ELSE
                    CASE
                        WHEN (CatSalesAmount - PrevCatSalesAmount) / PrevCatSalesAmount > 0.30 THEN 0.30
                        WHEN (CatSalesAmount - PrevCatSalesAmount) / PrevCatSalesAmount < -0.20 THEN -0.20
                        ELSE (CatSalesAmount - PrevCatSalesAmount) / PrevCatSalesAmount
                    END
            END
        ) OVER (
            PARTITION BY Category
            ORDER BY ActualYear
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS CatYoYGrowth3Yr
    FROM CatYearYoY
),
GlobalYear AS (
    SELECT ActualYear, SUM(ActualSalesAmount) AS GlobalSalesAmount
    FROM Actuals
    GROUP BY ActualYear
),
GlobalYearYoY AS (
    SELECT
        gy.*,
        LAG(gy.GlobalSalesAmount) OVER (ORDER BY gy.ActualYear) AS PrevGlobalSalesAmount
    FROM GlobalYear gy
),
GlobalRoll AS (
    SELECT
        ActualYear,
        AVG(
            CASE
                WHEN PrevGlobalSalesAmount IS NULL OR PrevGlobalSalesAmount = 0 THEN NULL
                ELSE
                    CASE
                        WHEN (GlobalSalesAmount - PrevGlobalSalesAmount) / PrevGlobalSalesAmount > 0.30 THEN 0.30
                        WHEN (GlobalSalesAmount - PrevGlobalSalesAmount) / PrevGlobalSalesAmount < -0.20 THEN -0.20
                        ELSE (GlobalSalesAmount - PrevGlobalSalesAmount) / PrevGlobalSalesAmount
                    END
            END
        ) OVER (
            ORDER BY ActualYear
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS GlobalYoYGrowth3Yr
    FROM GlobalYearYoY
),

Scenario AS (
    SELECT 'Low' AS Scenario, -0.03 AS ScenarioAdj
    UNION ALL SELECT 'Medium', 0.00
    UNION ALL SELECT 'High', 0.05
),

-- 4) Growth inputs + dynamic blend (for years where YoY exists)
Growth AS (
    SELECT
        y.Country,
        y.Category,
        y.ActualYear,
        y.ActualSalesAmount,
        y.ActualSalesQuantity,

        CASE
            WHEN y.PrevSalesAmount IS NULL OR y.PrevSalesAmount = 0 THEN NULL
            ELSE
                CASE
                    WHEN (y.ActualSalesAmount - y.PrevSalesAmount) / y.PrevSalesAmount > 0.30 THEN 0.30
                    WHEN (y.ActualSalesAmount - y.PrevSalesAmount) / y.PrevSalesAmount < -0.20 THEN -0.20
                    ELSE (y.ActualSalesAmount - y.PrevSalesAmount) / y.PrevSalesAmount
                END
        END AS LocalYoYGrowth,

        cr.CatYoYGrowth3Yr,
        gr.GlobalYoYGrowth3Yr
    FROM CountryYoY y
    LEFT JOIN CatRoll cr ON cr.Category = y.Category AND cr.ActualYear = y.ActualYear
    LEFT JOIN GlobalRoll gr ON gr.ActualYear = y.ActualYear
),

Blend AS (
    SELECT
        g.*,
        CASE WHEN g.LocalYoYGrowth     IS NULL THEN 0.00 ELSE 0.60 END AS wLocal,
        CASE WHEN g.CatYoYGrowth3Yr    IS NULL THEN 0.00 ELSE 0.30 END AS wCat,
        CASE WHEN g.GlobalYoYGrowth3Yr IS NULL THEN 0.00 ELSE 0.10 END AS wGlobal,

        CASE
            WHEN (CASE WHEN g.LocalYoYGrowth IS NULL THEN 0.00 ELSE 0.60 END
                + CASE WHEN g.CatYoYGrowth3Yr IS NULL THEN 0.00 ELSE 0.30 END
                + CASE WHEN g.GlobalYoYGrowth3Yr IS NULL THEN 0.00 ELSE 0.10 END) = 0
            THEN 0.05
            ELSE
                (COALESCE(g.LocalYoYGrowth,0)     * (CASE WHEN g.LocalYoYGrowth     IS NULL THEN 0.00 ELSE 0.60 END)
               + COALESCE(g.CatYoYGrowth3Yr,0)    * (CASE WHEN g.CatYoYGrowth3Yr    IS NULL THEN 0.00 ELSE 0.30 END)
               + COALESCE(g.GlobalYoYGrowth3Yr,0) * (CASE WHEN g.GlobalYoYGrowth3Yr IS NULL THEN 0.00 ELSE 0.10 END))
                /
                ((CASE WHEN g.LocalYoYGrowth     IS NULL THEN 0.00 ELSE 0.60 END)
               + (CASE WHEN g.CatYoYGrowth3Yr    IS NULL THEN 0.00 ELSE 0.30 END)
               + (CASE WHEN g.GlobalYoYGrowth3Yr IS NULL THEN 0.00 ELSE 0.10 END))
        END AS BaseGrowthPct
    FROM Growth g
),

-- 5) Standard budgets: BudgetYear = ActualYear + 1 (but clamped to MaxSalesYear)
StandardBudget AS (
    SELECT
        b.Country,
        b.Category,
        (b.ActualYear + 1) AS BudgetYear,
        sc.Scenario,
        sc.ScenarioAdj,

        -- jitter is keyed to BudgetYear (not ActualYear)
        ((CAST(ABS(CHECKSUM(b.Country, b.Category, (b.ActualYear + 1))) % 401 AS float) - 200.0) / 10000.0) AS JitterPct,

        b.BaseGrowthPct,
        b.ActualSalesAmount,
        b.ActualSalesQuantity,

        CAST('Standard: LY actual + blended growth + scenario + jitter' AS varchar(120)) AS BudgetMethod
    FROM Blend b
    CROSS JOIN Scenario sc
    CROSS JOIN YearBounds yb
    WHERE (b.ActualYear + 1) BETWEEN (yb.MinSalesYear - 1) AND yb.MaxSalesYear
),

-- 6) Backfill budgets for the first two years of the window
--    BudgetYear = MinSalesYear: baseline = same-year actual
--    BudgetYear = MinSalesYear-1: baseline = back-cast from first-year actual using DefaultBackcastGrowth
BackfillSeed AS (
    SELECT
        a.Country,
        a.Category,
        a.ActualYear,
        a.ActualSalesAmount,
        a.ActualSalesQuantity
    FROM Actuals a
    CROSS JOIN YearBounds yb
    WHERE a.ActualYear = yb.MinSalesYear
),
BackfillBudget AS (
    SELECT
        s.Country,
        s.Category,
        bys.BudgetYear,
        sc.Scenario,
        sc.ScenarioAdj,

        ((CAST(ABS(CHECKSUM(s.Country, s.Category, bys.BudgetYear)) % 401 AS float) - 200.0) / 10000.0) AS JitterPct,

        bys.DefaultBackcastGrowth,
        bys.BackfillMode,

        s.ActualSalesAmount,
        s.ActualSalesQuantity
    FROM BackfillSeed s
    CROSS JOIN Scenario sc
    CROSS JOIN YearBounds yb
    CROSS APPLY (
        SELECT (yb.MinSalesYear)     AS BudgetYear, CAST(0.05 AS float) AS DefaultBackcastGrowth, CAST('Backfill: same-year baseline' AS varchar(80)) AS BackfillMode
        UNION ALL
        SELECT (yb.MinSalesYear - 1) AS BudgetYear, CAST(0.05 AS float) AS DefaultBackcastGrowth, CAST('Backfill: back-cast from first-year' AS varchar(80)) AS BackfillMode
    ) bys
)

SELECT
    x.Country,
    x.Category,
    x.BudgetYear,
    x.Scenario,

    CAST(x.BudgetGrowthPct AS decimal(9,6)) AS BudgetGrowthPct,
    CAST(x.BudgetSalesAmount AS decimal(19,2)) AS BudgetSalesAmount,
    CAST(x.BudgetSalesQuantity AS decimal(19,2)) AS BudgetSalesQuantity,
    x.BudgetMethod
FROM (
    -- Standard
    SELECT
        sb.Country,
        sb.Category,
        sb.BudgetYear,
        sb.Scenario,
        (sb.BaseGrowthPct + sb.JitterPct + sb.ScenarioAdj) AS BudgetGrowthPct,
        (sb.ActualSalesAmount   * (1.0 + sb.BaseGrowthPct + sb.JitterPct + sb.ScenarioAdj)) AS BudgetSalesAmount,
        (sb.ActualSalesQuantity * (1.0 + sb.BaseGrowthPct + sb.JitterPct + sb.ScenarioAdj)) AS BudgetSalesQuantity,
        sb.BudgetMethod
    FROM StandardBudget sb

    UNION ALL

    -- Backfill: BudgetYear = MinSalesYear (same-year) and MinSalesYear-1 (back-cast)
    SELECT
        bf.Country,
        bf.Category,
        bf.BudgetYear,
        bf.Scenario,
        CASE
            WHEN bf.BackfillMode = 'Backfill: same-year baseline'
                THEN (0.0 + bf.JitterPct + bf.ScenarioAdj)
            ELSE
                (-bf.DefaultBackcastGrowth + bf.JitterPct + bf.ScenarioAdj)
        END AS BudgetGrowthPct,

        CASE
            WHEN bf.BackfillMode = 'Backfill: same-year baseline'
                THEN bf.ActualSalesAmount * (1.0 + bf.JitterPct + bf.ScenarioAdj)
            ELSE
                (bf.ActualSalesAmount / (1.0 + bf.DefaultBackcastGrowth)) * (1.0 + bf.JitterPct + bf.ScenarioAdj)
        END AS BudgetSalesAmount,

        CASE
            WHEN bf.BackfillMode = 'Backfill: same-year baseline'
                THEN bf.ActualSalesQuantity * (1.0 + bf.JitterPct + bf.ScenarioAdj)
            ELSE
                (bf.ActualSalesQuantity / (1.0 + bf.DefaultBackcastGrowth)) * (1.0 + bf.JitterPct + bf.ScenarioAdj)
        END AS BudgetSalesQuantity,

        CAST(bf.BackfillMode + ' + scenario + jitter' AS varchar(120)) AS BudgetMethod
    FROM BackfillBudget bf
) x;
GO

-----------------------------------------------------------------------
-- BUDGET ALLOCATION (Channel + Month)
-----------------------------------------------------------------------
CREATE OR ALTER   VIEW [dbo].[vw_Budget_ChannelMonth]
AS
WITH
SalesLineAgg AS (
    SELECT
        g.Country,
        pc.Category,
        d.Year  AS SalesYear,
        DATEFROMPARTS(d.Year, MONTH(d.Date), 1) AS SalesMonthStart,
        f.SalesChannelKey,
        SUM(f.Quantity * f.NetPrice) AS SalesAmount,
        SUM(COALESCE(f.Quantity,0))  AS SalesQty
    FROM dbo.vw_Sales f
    JOIN dbo.vw_Dates d ON d.Date = f.OrderDate
    JOIN dbo.vw_Stores s ON s.StoreKey = f.StoreKey
    JOIN dbo.vw_Geography g ON g.GeographyKey = s.GeographyKey
    JOIN dbo.vw_Products p ON p.ProductKey = f.ProductKey
    JOIN dbo.vw_ProductSubcategory ps ON ps.SubcategoryKey = p.SubcategoryKey
    JOIN dbo.vw_ProductCategory pc ON pc.CategoryKey = ps.CategoryKey
    GROUP BY
        g.Country, pc.Category, d.Year, DATEFROMPARTS(d.Year, MONTH(d.Date), 1), f.SalesChannelKey
),
SalesYearAgg AS (
    SELECT
        Country, Category, SalesYear, SalesChannelKey,
        SUM(SalesAmount) AS SalesAmount_Y,
        SUM(SalesQty)    AS SalesQty_Y
    FROM SalesLineAgg
    GROUP BY Country, Category, SalesYear, SalesChannelKey
),
YearBounds AS (
    SELECT MIN(SalesYear) AS MinSalesYear, MAX(SalesYear) AS MaxSalesYear
    FROM SalesYearAgg
),

-- Channel mix from actuals
ChannelMixRaw AS (
    SELECT
        y.Country, y.Category,
        y.SalesYear,
        y.SalesChannelKey,
        0.70 * y.SalesAmount_Y + 0.30 * COALESCE(y1.SalesAmount_Y, 0) AS MixAmount
    FROM SalesYearAgg y
    LEFT JOIN SalesYearAgg y1
        ON  y1.Country = y.Country
        AND y1.Category = y.Category
        AND y1.SalesChannelKey = y.SalesChannelKey
        AND y1.SalesYear = y.SalesYear - 1
),
ChannelMixShifted AS (
    SELECT
        cm.Country, cm.Category, cm.SalesYear, cm.SalesChannelKey,
        cm.MixAmount *
        CASE
            -- If these columns do not exist, replace the whole CASE with 1.0
            WHEN sc.IsDigital  = 1 THEN 1.02
            WHEN sc.IsPhysical = 1 THEN 0.98
            ELSE 1.00
        END AS MixAmountAdj
    FROM ChannelMixRaw cm
    JOIN dbo.vw_SalesChannels sc ON sc.SalesChannelKey = cm.SalesChannelKey
),
ChannelMix AS (
    SELECT
        cms.Country, cms.Category, cms.SalesYear, cms.SalesChannelKey,
        CASE
            WHEN SUM(cms.MixAmountAdj) OVER (PARTITION BY cms.Country, cms.Category, cms.SalesYear) = 0 THEN NULL
            ELSE cms.MixAmountAdj
                 / SUM(cms.MixAmountAdj) OVER (PARTITION BY cms.Country, cms.Category, cms.SalesYear)
        END AS ChannelShare
    FROM ChannelMixShifted cms
),
ChannelMixPresence AS (
    SELECT DISTINCT Country, Category, SalesYear
    FROM ChannelMix
),

-- Default channel shares (used when LY channel mix missing)
ChannelDefaults AS (
    SELECT
        sc.SalesChannelKey,
        CAST(
            CASE
                WHEN sc.IsDigital  = 1 THEN 1.02
                WHEN sc.IsPhysical = 1 THEN 0.98
                ELSE 1.00
            END AS float
        ) AS DefaultWeight
    FROM dbo.vw_SalesChannels sc
),
ChannelDefaultsNorm AS (
    SELECT
        cd.SalesChannelKey,
        cd.DefaultWeight / NULLIF(SUM(cd.DefaultWeight) OVER (), 0) AS DefaultShare
    FROM ChannelDefaults cd
),

-- Month mix from actuals
MonthMixRaw AS (
    SELECT
        m.Country, m.Category, m.SalesChannelKey,
        m.SalesYear,
        m.SalesMonthStart,
        0.70 * m.SalesAmount + 0.30 * COALESCE(m1.SalesAmount, 0) AS MonthAmount
    FROM SalesLineAgg m
    LEFT JOIN SalesLineAgg m1
        ON  m1.Country = m.Country
        AND m1.Category = m.Category
        AND m1.SalesChannelKey = m.SalesChannelKey
        AND m1.SalesYear = m.SalesYear - 1
        AND m1.SalesMonthStart = DATEADD(YEAR, -1, m.SalesMonthStart)
),
MonthMix AS (
    SELECT
        mm.Country, mm.Category, mm.SalesChannelKey, mm.SalesYear, mm.SalesMonthStart,
        CASE
            WHEN SUM(mm.MonthAmount) OVER (PARTITION BY mm.Country, mm.Category, mm.SalesChannelKey, mm.SalesYear) = 0 THEN NULL
            ELSE mm.MonthAmount
                 / SUM(mm.MonthAmount) OVER (PARTITION BY mm.Country, mm.Category, mm.SalesChannelKey, mm.SalesYear)
        END AS MonthShare
    FROM MonthMixRaw mm
),
MonthDomain AS (
    SELECT DISTINCT MONTH([Date]) AS MonthNum
    FROM dbo.Dates
),

-- Returns rate (same as your version)
ReturnLineAgg AS (
    SELECT
        g.Country,
        pc.Category,
        d.Year AS SalesYear,
        f.SalesChannelKey,
        SUM(r.ReturnQuantity * r.ReturnNetPrice) AS ReturnAmount
    FROM dbo.vw_SalesReturn r
    JOIN dbo.vw_Sales f
        ON  f.SalesOrderNumber = r.SalesOrderNumber
        AND f.SalesOrderLineNumber = r.SalesOrderLineNumber
    JOIN dbo.vw_Dates d ON d.Date = f.OrderDate
    JOIN dbo.vw_Stores s ON s.StoreKey = f.StoreKey
    JOIN dbo.vw_Geography g ON g.GeographyKey = s.GeographyKey
    JOIN dbo.vw_Products p ON p.ProductKey = f.ProductKey
    JOIN dbo.vw_ProductSubcategory ps ON ps.SubcategoryKey = p.SubcategoryKey
    JOIN dbo.vw_ProductCategory pc ON pc.CategoryKey = ps.CategoryKey
    GROUP BY g.Country, pc.Category, d.Year, f.SalesChannelKey
),
ReturnRateRaw AS (
    SELECT
        sy.Country, sy.Category, sy.SalesYear, sy.SalesChannelKey,
        sy.SalesAmount_Y AS SalesAmount,
        COALESCE(ra.ReturnAmount, 0) AS ReturnAmount,
        CASE
            WHEN sy.SalesAmount_Y = 0 THEN NULL
            ELSE COALESCE(ra.ReturnAmount, 0) / sy.SalesAmount_Y
        END AS ReturnRate
    FROM SalesYearAgg sy
    LEFT JOIN ReturnLineAgg ra
        ON  ra.Country = sy.Country
        AND ra.Category = sy.Category
        AND ra.SalesYear = sy.SalesYear
        AND ra.SalesChannelKey = sy.SalesChannelKey
),
ReturnRateSmoothed AS (
    SELECT
        r.Country, r.Category, r.SalesYear, r.SalesChannelKey,
        CASE
            WHEN rr IS NULL THEN NULL
            WHEN rr < 0 THEN 0
            WHEN rr > 0.30 THEN 0.30
            ELSE rr
        END AS ReturnRateCapped
    FROM (
        SELECT
            r0.Country, r0.Category, r0.SalesYear, r0.SalesChannelKey,
            (0.70 * r0.ReturnRate) + (0.30 * COALESCE(r1.ReturnRate, r0.ReturnRate)) AS rr
        FROM ReturnRateRaw r0
        LEFT JOIN ReturnRateRaw r1
            ON  r1.Country = r0.Country
            AND r1.Category = r0.Category
            AND r1.SalesChannelKey = r0.SalesChannelKey
            AND r1.SalesYear = r0.SalesYear - 1
    ) r
),

-- Budget base (now includes backfill years)
BudgetBase AS (
    SELECT
        b.Country,
        b.Category,
        b.BudgetYear,
        b.Scenario,
        b.BudgetSalesAmount   AS BudgetYearAmount,
        b.BudgetSalesQuantity AS BudgetYearQty,
        b.BudgetGrowthPct
    FROM dbo.vw_Budget b
),

-- Channel expansion: use LY ChannelMix if present; otherwise default shares
BudgetChannel AS (
    SELECT
        bb.Country, bb.Category, bb.BudgetYear, bb.Scenario,
        cm.SalesChannelKey,
        cm.ChannelShare,
        bb.BudgetYearAmount * cm.ChannelShare AS BudgetChannelAmount,
        bb.BudgetYearQty    * cm.ChannelShare AS BudgetChannelQty,
        bb.BudgetGrowthPct
    FROM BudgetBase bb
    JOIN ChannelMix cm
        ON  cm.Country = bb.Country
        AND cm.Category = bb.Category
        AND cm.SalesYear = bb.BudgetYear - 1

    UNION ALL

    SELECT
        bb.Country, bb.Category, bb.BudgetYear, bb.Scenario,
        cd.SalesChannelKey,
        cd.DefaultShare AS ChannelShare,
        bb.BudgetYearAmount * cd.DefaultShare AS BudgetChannelAmount,
        bb.BudgetYearQty    * cd.DefaultShare AS BudgetChannelQty,
        bb.BudgetGrowthPct
    FROM BudgetBase bb
    CROSS JOIN ChannelDefaultsNorm cd
    WHERE NOT EXISTS (
        SELECT 1
        FROM ChannelMixPresence p
        WHERE p.Country = bb.Country
          AND p.Category = bb.Category
          AND p.SalesYear = bb.BudgetYear - 1
    )
),

-- Month expansion: always produce 12 months; use LY MonthMix if present else 1/12
BudgetChannelMonth AS (
    SELECT
        bc.Country,
        bc.Category,
        bc.BudgetYear,
        bc.Scenario,
        bc.SalesChannelKey,

        DATEFROMPARTS(bc.BudgetYear, md.MonthNum, 1) AS BudgetMonthStart,

        COALESCE(
            mm.MonthShare,
            1.0 / 12.0
        ) AS MonthShare,

        bc.ChannelShare,
        (bc.BudgetChannelAmount * COALESCE(mm.MonthShare, 1.0 / 12.0)) AS BudgetGrossAmount,
        (bc.BudgetChannelQty    * COALESCE(mm.MonthShare, 1.0 / 12.0)) AS BudgetGrossQty,
        bc.BudgetGrowthPct
    FROM BudgetChannel bc
    CROSS JOIN MonthDomain md
    LEFT JOIN MonthMix mm
        ON  mm.Country = bc.Country
        AND mm.Category = bc.Category
        AND mm.SalesChannelKey = bc.SalesChannelKey
        AND mm.SalesYear = bc.BudgetYear - 1
        AND mm.SalesMonthStart = DATEFROMPARTS(bc.BudgetYear - 1, md.MonthNum, 1)
)

SELECT
    bcm.Country,
    bcm.Category,
    bcm.SalesChannelKey,
    bcm.BudgetYear,
    bcm.BudgetMonthStart,
    bcm.Scenario,

    CAST(bcm.BudgetGrowthPct AS decimal(9,6)) AS BudgetGrowthPct,
    CAST(bcm.ChannelShare AS decimal(9,6))    AS Audit_ChannelShare,
    CAST(bcm.MonthShare AS decimal(9,6))      AS Audit_MonthShare,

    CAST(COALESCE(rr.ReturnRateCapped, 0.0) AS decimal(9,6)) AS Audit_ReturnRate,
    CAST(bcm.BudgetGrossAmount AS decimal(19,2)) AS BudgetGrossAmount,
    CAST(bcm.BudgetGrossAmount * (1.0 - COALESCE(rr.ReturnRateCapped, 0.0)) AS decimal(19,2)) AS BudgetNetAmount,

    CAST(bcm.BudgetGrossQty AS decimal(19,2)) AS BudgetGrossQuantity,
    CAST(bcm.BudgetGrossQty * (1.0 - COALESCE(rr.ReturnRateCapped, 0.0)) AS decimal(19,2)) AS BudgetNetQuantity,

    CAST('Yearly budget -> channel mix (LY or default) -> seasonality (LY or flat) -> returns adj' AS varchar(140)) AS BudgetMethod
FROM BudgetChannelMonth bcm
CROSS JOIN YearBounds yb
LEFT JOIN ReturnRateSmoothed rr
    ON  rr.Country = bcm.Country
    AND rr.Category = bcm.Category
    AND rr.SalesChannelKey = bcm.SalesChannelKey
    AND rr.SalesYear = bcm.BudgetYear - 1
WHERE
    bcm.BudgetYear BETWEEN (yb.MinSalesYear - 1) AND yb.MaxSalesYear;
GO

-----------------------------------------------------------------------
-- BUDGET FX LAYER
-----------------------------------------------------------------------
CREATE OR ALTER   VIEW [dbo].[vw_Budget_ChannelMonth_FX]
AS
WITH
-- 0) Params (change reporting currency here)
Params AS (
    SELECT CAST('USD' AS varchar(10)) AS ReportCurrency
),

-- 1) Base budget (local currency amounts)
B AS (
    SELECT
        Country,
        Category,
        SalesChannelKey,
        BudgetYear,
        BudgetMonthStart,
        Scenario,
        BudgetGrowthPct,
        Audit_ChannelShare,
        Audit_MonthShare,
        Audit_ReturnRate,
        BudgetGrossAmount,
        BudgetNetAmount,
        BudgetGrossQuantity,
        BudgetNetQuantity,
        BudgetMethod
    FROM dbo.vw_Budget_ChannelMonth
),

-- 2) Country -> LocalCurrency (force 1 row per country)
-- EDIT CurrencyCode column name if yours differs
CountryCurrency AS (
    SELECT
        g.Country,
        MAX(g.ISOCode) AS LocalCurrency
    FROM dbo.Geography g
    WHERE g.Country IS NOT NULL
      AND g.ISOCode IS NOT NULL
    GROUP BY g.Country
),

-- 3) Monthly FX rates (AVG daily within month)
FxMonth AS (
    SELECT
        DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS MonthStart,
        FromCurrency,
        ToCurrency,
        AVG(Rate) AS AvgRate
    FROM dbo.ExchangeRates
    GROUP BY
        DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1),
        FromCurrency,
        ToCurrency
),

-- 4) Attach currencies
Base AS (
    SELECT
        b.*,
        cc.LocalCurrency,
        p.ReportCurrency
    FROM B b
    CROSS JOIN Params p
    LEFT JOIN CountryCurrency cc
        ON cc.Country = b.Country
),

-- 5) Resolve FX using carry-forward (latest MonthStart <= BudgetMonthStart)
FxResolved AS (
    SELECT
        x.*,

        fxD.AvgRate AS Rate_Direct_CF,
        fxI.AvgRate AS Rate_Inverse_CF,

        CASE
            WHEN x.LocalCurrency IS NULL OR x.ReportCurrency IS NULL THEN NULL
            WHEN x.LocalCurrency = x.ReportCurrency THEN 1.0
            WHEN fxD.AvgRate IS NOT NULL THEN fxD.AvgRate
            WHEN fxI.AvgRate IS NOT NULL AND fxI.AvgRate <> 0 THEN 1.0 / fxI.AvgRate
            ELSE NULL
        END AS FxRate_ToReport,

        CASE
            WHEN x.LocalCurrency IS NULL OR x.ReportCurrency IS NULL THEN 'missing-currency'
            WHEN x.LocalCurrency = x.ReportCurrency THEN 'identity'
            WHEN fxD.AvgRate IS NOT NULL THEN 'direct-carryforward'
            WHEN fxI.AvgRate IS NOT NULL THEN 'inverse-carryforward'
            ELSE 'missing-rate'
        END AS Audit_FxSource
    FROM Base x
    OUTER APPLY (
        SELECT TOP (1) fm.AvgRate
        FROM FxMonth fm
        WHERE fm.FromCurrency = x.LocalCurrency
          AND fm.ToCurrency   = x.ReportCurrency
          AND fm.MonthStart  <= x.BudgetMonthStart
        ORDER BY fm.MonthStart DESC
    ) fxD
    OUTER APPLY (
        SELECT TOP (1) fm.AvgRate
        FROM FxMonth fm
        WHERE fm.FromCurrency = x.ReportCurrency
          AND fm.ToCurrency   = x.LocalCurrency
          AND fm.MonthStart  <= x.BudgetMonthStart
        ORDER BY fm.MonthStart DESC
    ) fxI
)

SELECT
    Country,
    Category,
    SalesChannelKey,
    BudgetYear,
    BudgetMonthStart,
    Scenario,

    LocalCurrency,
    ReportCurrency,
    CAST(FxRate_ToReport AS decimal(18,6)) AS FxRate_ToReport,
    Audit_FxSource,

    -- Local amounts
    CAST(BudgetGrossAmount AS decimal(19,2)) AS BudgetGrossAmount_Local,
    CAST(BudgetNetAmount   AS decimal(19,2)) AS BudgetNetAmount_Local,

    -- Converted amounts (NULL if still no rate)
    CAST(
        CASE WHEN FxRate_ToReport IS NULL THEN NULL ELSE BudgetGrossAmount * FxRate_ToReport END
        AS decimal(19,2)
    ) AS BudgetGrossAmount_Report,

    CAST(
        CASE WHEN FxRate_ToReport IS NULL THEN NULL ELSE BudgetNetAmount * FxRate_ToReport END
        AS decimal(19,2)
    ) AS BudgetNetAmount_Report,

    -- Quantities
    CAST(BudgetGrossQuantity AS decimal(19,2)) AS BudgetGrossQuantity,
    CAST(BudgetNetQuantity   AS decimal(19,2)) AS BudgetNetQuantity,

    -- Audits
    CAST(BudgetGrowthPct     AS decimal(9,6))  AS BudgetGrowthPct,
    CAST(Audit_ChannelShare  AS decimal(9,6))  AS Audit_ChannelShare,
    CAST(Audit_MonthShare    AS decimal(9,6))  AS Audit_MonthShare,
    CAST(Audit_ReturnRate    AS decimal(9,6))  AS Audit_ReturnRate,

    BudgetMethod
FROM FxResolved;
GO
