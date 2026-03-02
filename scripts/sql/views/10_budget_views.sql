SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

-----------------------------------------------------------------------
-- BUDGET ENGINE (Yearly)
--
-- WARNING: This view is expensive (full scan of vw_Sales joined to 5
-- dimension tables). For ad-hoc queries use the materialized cache
-- table dbo.Budget_ChannelMonth produced by sp_RefreshBudgetCache.
--
-- FIXES vs original:
--  1. Business constants extracted into top-level Params CTE.
--  2. NULLIF guards on all divisors (CountryYoY, Blend denominator,
--     backfill DefaultBackcastGrowth).
--  3. Backfill DefaultBackcastGrowth references the Params CTE.
-----------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vw_Budget]
AS
WITH
-- ================================================================
-- Tunables: change these instead of editing buried CASE expressions
-- ================================================================
Params AS (
    SELECT
        CAST(0.30  AS float) AS MaxGrowthCap,         -- clamp local/cat/global YoY
        CAST(-0.20 AS float) AS MinGrowthFloor,       -- clamp local/cat/global YoY
        CAST(0.05  AS float) AS DefaultGrowthPct,      -- fallback when no YoY data exists
        CAST(0.05  AS float) AS DefaultBackcastGrowth,  -- back-cast growth for backfill years
        CAST(0.60  AS float) AS WeightLocal,           -- blend weight: country-level YoY
        CAST(0.30  AS float) AS WeightCategory,        -- blend weight: category rolling YoY
        CAST(0.10  AS float) AS WeightGlobal           -- blend weight: global rolling YoY
),

Scenario AS (
    SELECT 'Low'    AS Scenario, CAST(-0.03 AS float) AS ScenarioAdj
    UNION ALL SELECT 'Medium',   CAST( 0.00 AS float)
    UNION ALL SELECT 'High',     CAST( 0.05 AS float)
),

-- 1) Actuals at low grain (Country x Year x Category)
Actuals AS (
    SELECT
        g.Country,
        d.Year AS ActualYear,
        pc.Category,
        SUM(f.Quantity * f.NetPrice)   AS ActualSalesAmount,
        SUM(COALESCE(f.Quantity, 0))   AS ActualSalesQuantity
    FROM dbo.vw_Sales f
    JOIN dbo.vw_Dates d                ON d.Date = f.OrderDate
    JOIN dbo.vw_Stores s               ON s.StoreKey = f.StoreKey
    JOIN dbo.vw_Geography g            ON g.GeographyKey = s.GeographyKey
    JOIN dbo.vw_Products p             ON p.ProductKey = f.ProductKey
    JOIN dbo.vw_ProductSubcategory ps  ON ps.SubcategoryKey = p.SubcategoryKey
    JOIN dbo.vw_ProductCategory pc     ON pc.CategoryKey = ps.CategoryKey
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

-- 3) Category-level and Global-level rolling YoY (fallback tiers)
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
        cyy.Category,
        cyy.ActualYear,
        AVG(
            CASE
                WHEN cyy.PrevCatSalesAmount IS NULL OR cyy.PrevCatSalesAmount = 0 THEN NULL
                ELSE
                    CASE
                        WHEN (cyy.CatSalesAmount - cyy.PrevCatSalesAmount) / cyy.PrevCatSalesAmount > p.MaxGrowthCap   THEN p.MaxGrowthCap
                        WHEN (cyy.CatSalesAmount - cyy.PrevCatSalesAmount) / cyy.PrevCatSalesAmount < p.MinGrowthFloor THEN p.MinGrowthFloor
                        ELSE (cyy.CatSalesAmount - cyy.PrevCatSalesAmount) / cyy.PrevCatSalesAmount
                    END
            END
        ) OVER (
            PARTITION BY cyy.Category
            ORDER BY cyy.ActualYear
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS CatYoYGrowth3Yr
    FROM CatYearYoY cyy
    CROSS JOIN Params p
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
        gyy.ActualYear,
        AVG(
            CASE
                WHEN gyy.PrevGlobalSalesAmount IS NULL OR gyy.PrevGlobalSalesAmount = 0 THEN NULL
                ELSE
                    CASE
                        WHEN (gyy.GlobalSalesAmount - gyy.PrevGlobalSalesAmount) / gyy.PrevGlobalSalesAmount > p.MaxGrowthCap   THEN p.MaxGrowthCap
                        WHEN (gyy.GlobalSalesAmount - gyy.PrevGlobalSalesAmount) / gyy.PrevGlobalSalesAmount < p.MinGrowthFloor THEN p.MinGrowthFloor
                        ELSE (gyy.GlobalSalesAmount - gyy.PrevGlobalSalesAmount) / gyy.PrevGlobalSalesAmount
                    END
            END
        ) OVER (
            ORDER BY gyy.ActualYear
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS GlobalYoYGrowth3Yr
    FROM GlobalYearYoY gyy
    CROSS JOIN Params p
),

-- 4) Growth inputs + dynamic blend
--    FIX: NULLIF on PrevSalesAmount + on Blend denominator
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
                    WHEN (y.ActualSalesAmount - y.PrevSalesAmount) / y.PrevSalesAmount > p.MaxGrowthCap   THEN p.MaxGrowthCap
                    WHEN (y.ActualSalesAmount - y.PrevSalesAmount) / y.PrevSalesAmount < p.MinGrowthFloor THEN p.MinGrowthFloor
                    ELSE (y.ActualSalesAmount - y.PrevSalesAmount) / y.PrevSalesAmount
                END
        END AS LocalYoYGrowth,

        cr.CatYoYGrowth3Yr,
        gr.GlobalYoYGrowth3Yr
    FROM CountryYoY y
    CROSS JOIN Params p
    LEFT JOIN CatRoll cr    ON cr.Category = y.Category AND cr.ActualYear = y.ActualYear
    LEFT JOIN GlobalRoll gr ON gr.ActualYear = y.ActualYear
),

Blend AS (
    SELECT
        g.*,
        CASE
            WHEN (CASE WHEN g.LocalYoYGrowth     IS NULL THEN 0.00 ELSE p.WeightLocal    END
                + CASE WHEN g.CatYoYGrowth3Yr    IS NULL THEN 0.00 ELSE p.WeightCategory END
                + CASE WHEN g.GlobalYoYGrowth3Yr IS NULL THEN 0.00 ELSE p.WeightGlobal   END) = 0
            THEN p.DefaultGrowthPct
            ELSE
                (COALESCE(g.LocalYoYGrowth,0)     * (CASE WHEN g.LocalYoYGrowth     IS NULL THEN 0.00 ELSE p.WeightLocal    END)
               + COALESCE(g.CatYoYGrowth3Yr,0)    * (CASE WHEN g.CatYoYGrowth3Yr    IS NULL THEN 0.00 ELSE p.WeightCategory END)
               + COALESCE(g.GlobalYoYGrowth3Yr,0) * (CASE WHEN g.GlobalYoYGrowth3Yr IS NULL THEN 0.00 ELSE p.WeightGlobal   END))
                /
                NULLIF(
                    (CASE WHEN g.LocalYoYGrowth     IS NULL THEN 0.00 ELSE p.WeightLocal    END)
                  + (CASE WHEN g.CatYoYGrowth3Yr    IS NULL THEN 0.00 ELSE p.WeightCategory END)
                  + (CASE WHEN g.GlobalYoYGrowth3Yr IS NULL THEN 0.00 ELSE p.WeightGlobal   END)
                , 0)
        END AS BaseGrowthPct
    FROM Growth g
    CROSS JOIN Params p
),

-- 5) Standard budgets: BudgetYear = ActualYear + 1 (clamped to MaxSalesYear)
StandardBudget AS (
    SELECT
        b.Country,
        b.Category,
        (b.ActualYear + 1) AS BudgetYear,
        sc.Scenario,
        sc.ScenarioAdj,
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
BackfillSeed AS (
    SELECT
        a.Country, a.Category, a.ActualYear,
        a.ActualSalesAmount, a.ActualSalesQuantity
    FROM Actuals a
    CROSS JOIN YearBounds yb
    WHERE a.ActualYear = yb.MinSalesYear
),
BackfillBudget AS (
    SELECT
        s.Country, s.Category,
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
    CROSS JOIN Params p
    CROSS APPLY (
        SELECT yb.MinSalesYear       AS BudgetYear, p.DefaultBackcastGrowth, CAST('Backfill: same-year baseline'       AS varchar(80)) AS BackfillMode
        UNION ALL
        SELECT (yb.MinSalesYear - 1) AS BudgetYear, p.DefaultBackcastGrowth, CAST('Backfill: back-cast from first-year' AS varchar(80)) AS BackfillMode
    ) bys
)

SELECT
    x.Country, x.Category, x.BudgetYear, x.Scenario,
    CAST(x.BudgetGrowthPct     AS decimal(9,6))  AS BudgetGrowthPct,
    CAST(x.BudgetSalesAmount   AS decimal(19,2))  AS BudgetSalesAmount,
    CAST(x.BudgetSalesQuantity AS decimal(19,2))  AS BudgetSalesQuantity,
    x.BudgetMethod
FROM (
    SELECT
        sb.Country, sb.Category, sb.BudgetYear, sb.Scenario,
        (sb.BaseGrowthPct + sb.JitterPct + sb.ScenarioAdj) AS BudgetGrowthPct,
        (sb.ActualSalesAmount   * (1.0 + sb.BaseGrowthPct + sb.JitterPct + sb.ScenarioAdj)) AS BudgetSalesAmount,
        (sb.ActualSalesQuantity * (1.0 + sb.BaseGrowthPct + sb.JitterPct + sb.ScenarioAdj)) AS BudgetSalesQuantity,
        sb.BudgetMethod
    FROM StandardBudget sb

    UNION ALL

    SELECT
        bf.Country, bf.Category, bf.BudgetYear, bf.Scenario,
        CASE
            WHEN bf.BackfillMode = 'Backfill: same-year baseline'
                THEN (0.0 + bf.JitterPct + bf.ScenarioAdj)
            ELSE (-bf.DefaultBackcastGrowth + bf.JitterPct + bf.ScenarioAdj)
        END,
        CASE
            WHEN bf.BackfillMode = 'Backfill: same-year baseline'
                THEN bf.ActualSalesAmount * (1.0 + bf.JitterPct + bf.ScenarioAdj)
            ELSE (bf.ActualSalesAmount / NULLIF(1.0 + bf.DefaultBackcastGrowth, 0)) * (1.0 + bf.JitterPct + bf.ScenarioAdj)
        END,
        CASE
            WHEN bf.BackfillMode = 'Backfill: same-year baseline'
                THEN bf.ActualSalesQuantity * (1.0 + bf.JitterPct + bf.ScenarioAdj)
            ELSE (bf.ActualSalesQuantity / NULLIF(1.0 + bf.DefaultBackcastGrowth, 0)) * (1.0 + bf.JitterPct + bf.ScenarioAdj)
        END,
        CAST(bf.BackfillMode + ' + scenario + jitter' AS varchar(120))
    FROM BackfillBudget bf
) x;
GO

-----------------------------------------------------------------------
-- BUDGET ALLOCATION (Channel + Month)
--
-- WARNING: Also expensive - full scan of vw_Sales for channel/month
-- mix plus a reference to vw_Budget. Use the cache tables for queries.
--
-- FIXES vs original:
--  1. MonthDomain uses safe VALUES(1..12) instead of SELECT DISTINCT
--     MONTH from dbo.Dates (avoids edge cases with partial-year data).
--  2. Digital/physical channel shift factors reference AllocParams CTE.
--  3. NULLIF on all windowed denominators.
-----------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vw_Budget_ChannelMonth]
AS
WITH
AllocParams AS (
    SELECT
        CAST(0.70 AS float) AS WeightCurrentYear,
        CAST(0.30 AS float) AS WeightPriorYear,
        CAST(1.02 AS float) AS DigitalShift,
        CAST(0.98 AS float) AS PhysicalShift,
        CAST(0.30 AS float) AS MaxReturnRate
),

SalesLineAgg AS (
    SELECT
        g.Country, pc.Category,
        d.Year AS SalesYear,
        DATEFROMPARTS(d.Year, MONTH(d.Date), 1) AS SalesMonthStart,
        f.SalesChannelKey,
        SUM(f.Quantity * f.NetPrice)   AS SalesAmount,
        SUM(COALESCE(f.Quantity, 0))   AS SalesQty
    FROM dbo.vw_Sales f
    JOIN dbo.vw_Dates d                ON d.Date = f.OrderDate
    JOIN dbo.vw_Stores s               ON s.StoreKey = f.StoreKey
    JOIN dbo.vw_Geography g            ON g.GeographyKey = s.GeographyKey
    JOIN dbo.vw_Products p             ON p.ProductKey = f.ProductKey
    JOIN dbo.vw_ProductSubcategory ps  ON ps.SubcategoryKey = p.SubcategoryKey
    JOIN dbo.vw_ProductCategory pc     ON pc.CategoryKey = ps.CategoryKey
    GROUP BY g.Country, pc.Category, d.Year, DATEFROMPARTS(d.Year, MONTH(d.Date), 1), f.SalesChannelKey
),

SalesYearAgg AS (
    SELECT Country, Category, SalesYear, SalesChannelKey,
        SUM(SalesAmount) AS SalesAmount_Y, SUM(SalesQty) AS SalesQty_Y
    FROM SalesLineAgg
    GROUP BY Country, Category, SalesYear, SalesChannelKey
),

YearBounds AS (
    SELECT MIN(SalesYear) AS MinSalesYear, MAX(SalesYear) AS MaxSalesYear
    FROM SalesYearAgg
),

ChannelMixRaw AS (
    SELECT
        y.Country, y.Category, y.SalesYear, y.SalesChannelKey,
        ap.WeightCurrentYear * y.SalesAmount_Y + ap.WeightPriorYear * COALESCE(y1.SalesAmount_Y, 0) AS MixAmount
    FROM SalesYearAgg y
    CROSS JOIN AllocParams ap
    LEFT JOIN SalesYearAgg y1
        ON  y1.Country = y.Country AND y1.Category = y.Category
        AND y1.SalesChannelKey = y.SalesChannelKey AND y1.SalesYear = y.SalesYear - 1
),

ChannelMixShifted AS (
    SELECT
        cm.Country, cm.Category, cm.SalesYear, cm.SalesChannelKey,
        cm.MixAmount *
        CASE
            WHEN sc.IsDigital  = 1 THEN ap.DigitalShift
            WHEN sc.IsPhysical = 1 THEN ap.PhysicalShift
            ELSE 1.00
        END AS MixAmountAdj
    FROM ChannelMixRaw cm
    CROSS JOIN AllocParams ap
    JOIN dbo.vw_SalesChannels sc ON sc.SalesChannelKey = cm.SalesChannelKey
),

ChannelMix AS (
    SELECT
        cms.Country, cms.Category, cms.SalesYear, cms.SalesChannelKey,
        cms.MixAmountAdj
            / NULLIF(SUM(cms.MixAmountAdj) OVER (PARTITION BY cms.Country, cms.Category, cms.SalesYear), 0) AS ChannelShare
    FROM ChannelMixShifted cms
),

ChannelMixPresence AS (
    SELECT DISTINCT Country, Category, SalesYear FROM ChannelMix
),

ChannelDefaults AS (
    SELECT sc.SalesChannelKey,
        CAST(CASE WHEN sc.IsDigital = 1 THEN ap.DigitalShift WHEN sc.IsPhysical = 1 THEN ap.PhysicalShift ELSE 1.00 END AS float) AS DefaultWeight
    FROM dbo.vw_SalesChannels sc
    CROSS JOIN AllocParams ap
),
ChannelDefaultsNorm AS (
    SELECT cd.SalesChannelKey,
        cd.DefaultWeight / NULLIF(SUM(cd.DefaultWeight) OVER (), 0) AS DefaultShare
    FROM ChannelDefaults cd
),

MonthMixRaw AS (
    SELECT
        m.Country, m.Category, m.SalesChannelKey, m.SalesYear, m.SalesMonthStart,
        ap.WeightCurrentYear * m.SalesAmount + ap.WeightPriorYear * COALESCE(m1.SalesAmount, 0) AS MonthAmount
    FROM SalesLineAgg m
    CROSS JOIN AllocParams ap
    LEFT JOIN SalesLineAgg m1
        ON  m1.Country = m.Country AND m1.Category = m.Category
        AND m1.SalesChannelKey = m.SalesChannelKey
        AND m1.SalesYear = m.SalesYear - 1
        AND m1.SalesMonthStart = DATEADD(YEAR, -1, m.SalesMonthStart)
),
MonthMix AS (
    SELECT
        mm.Country, mm.Category, mm.SalesChannelKey, mm.SalesYear, mm.SalesMonthStart,
        mm.MonthAmount
            / NULLIF(SUM(mm.MonthAmount) OVER (PARTITION BY mm.Country, mm.Category, mm.SalesChannelKey, mm.SalesYear), 0) AS MonthShare
    FROM MonthMixRaw mm
),

-- FIX: Safe month generator (does not depend on dbo.Dates content)
MonthDomain AS (
    SELECT MonthNum FROM (VALUES (1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12)) AS v(MonthNum)
),

ReturnLineAgg AS (
    SELECT
        g.Country, pc.Category, d.Year AS SalesYear, f.SalesChannelKey,
        SUM(r.ReturnQuantity * r.ReturnNetPrice) AS ReturnAmount
    FROM dbo.vw_SalesReturn r
    JOIN dbo.vw_Sales f
        ON  f.SalesOrderNumber = r.SalesOrderNumber AND f.SalesOrderLineNumber = r.SalesOrderLineNumber
    JOIN dbo.vw_Dates d                ON d.Date = f.OrderDate
    JOIN dbo.vw_Stores s               ON s.StoreKey = f.StoreKey
    JOIN dbo.vw_Geography g            ON g.GeographyKey = s.GeographyKey
    JOIN dbo.vw_Products p             ON p.ProductKey = f.ProductKey
    JOIN dbo.vw_ProductSubcategory ps  ON ps.SubcategoryKey = p.SubcategoryKey
    JOIN dbo.vw_ProductCategory pc     ON pc.CategoryKey = ps.CategoryKey
    GROUP BY g.Country, pc.Category, d.Year, f.SalesChannelKey
),
ReturnRateRaw AS (
    SELECT
        sy.Country, sy.Category, sy.SalesYear, sy.SalesChannelKey,
        COALESCE(ra.ReturnAmount, 0) / NULLIF(sy.SalesAmount_Y, 0) AS ReturnRate
    FROM SalesYearAgg sy
    LEFT JOIN ReturnLineAgg ra
        ON  ra.Country = sy.Country AND ra.Category = sy.Category
        AND ra.SalesYear = sy.SalesYear AND ra.SalesChannelKey = sy.SalesChannelKey
),
ReturnRateSmoothed AS (
    SELECT
        r.Country, r.Category, r.SalesYear, r.SalesChannelKey,
        CASE
            WHEN rr IS NULL THEN NULL
            WHEN rr < 0 THEN 0
            WHEN rr > ap.MaxReturnRate THEN ap.MaxReturnRate
            ELSE rr
        END AS ReturnRateCapped
    FROM (
        SELECT r0.Country, r0.Category, r0.SalesYear, r0.SalesChannelKey,
            (0.70 * r0.ReturnRate) + (0.30 * COALESCE(r1.ReturnRate, r0.ReturnRate)) AS rr
        FROM ReturnRateRaw r0
        LEFT JOIN ReturnRateRaw r1
            ON  r1.Country = r0.Country AND r1.Category = r0.Category
            AND r1.SalesChannelKey = r0.SalesChannelKey AND r1.SalesYear = r0.SalesYear - 1
    ) r
    CROSS JOIN AllocParams ap
),

BudgetBase AS (
    SELECT b.Country, b.Category, b.BudgetYear, b.Scenario,
        b.BudgetSalesAmount AS BudgetYearAmount, b.BudgetSalesQuantity AS BudgetYearQty, b.BudgetGrowthPct
    FROM dbo.vw_Budget b
),

BudgetChannel AS (
    SELECT bb.Country, bb.Category, bb.BudgetYear, bb.Scenario,
        cm.SalesChannelKey, cm.ChannelShare,
        bb.BudgetYearAmount * cm.ChannelShare AS BudgetChannelAmount,
        bb.BudgetYearQty    * cm.ChannelShare AS BudgetChannelQty,
        bb.BudgetGrowthPct
    FROM BudgetBase bb
    JOIN ChannelMix cm ON cm.Country = bb.Country AND cm.Category = bb.Category AND cm.SalesYear = bb.BudgetYear - 1

    UNION ALL

    SELECT bb.Country, bb.Category, bb.BudgetYear, bb.Scenario,
        cd.SalesChannelKey, cd.DefaultShare,
        bb.BudgetYearAmount * cd.DefaultShare,
        bb.BudgetYearQty    * cd.DefaultShare,
        bb.BudgetGrowthPct
    FROM BudgetBase bb
    CROSS JOIN ChannelDefaultsNorm cd
    WHERE NOT EXISTS (
        SELECT 1 FROM ChannelMixPresence p
        WHERE p.Country = bb.Country AND p.Category = bb.Category AND p.SalesYear = bb.BudgetYear - 1
    )
),

BudgetChannelMonth AS (
    SELECT
        bc.Country, bc.Category, bc.BudgetYear, bc.Scenario, bc.SalesChannelKey,
        DATEFROMPARTS(bc.BudgetYear, md.MonthNum, 1) AS BudgetMonthStart,
        COALESCE(mm.MonthShare, 1.0 / 12.0) AS MonthShare,
        bc.ChannelShare,
        (bc.BudgetChannelAmount * COALESCE(mm.MonthShare, 1.0 / 12.0)) AS BudgetGrossAmount,
        (bc.BudgetChannelQty    * COALESCE(mm.MonthShare, 1.0 / 12.0)) AS BudgetGrossQty,
        bc.BudgetGrowthPct
    FROM BudgetChannel bc
    CROSS JOIN MonthDomain md
    LEFT JOIN MonthMix mm
        ON  mm.Country = bc.Country AND mm.Category = bc.Category
        AND mm.SalesChannelKey = bc.SalesChannelKey
        AND mm.SalesYear = bc.BudgetYear - 1
        AND mm.SalesMonthStart = DATEFROMPARTS(bc.BudgetYear - 1, md.MonthNum, 1)
)

SELECT
    bcm.Country, bcm.Category, bcm.SalesChannelKey,
    bcm.BudgetYear, bcm.BudgetMonthStart, bcm.Scenario,

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
    ON  rr.Country = bcm.Country AND rr.Category = bcm.Category
    AND rr.SalesChannelKey = bcm.SalesChannelKey AND rr.SalesYear = bcm.BudgetYear - 1
WHERE bcm.BudgetYear BETWEEN (yb.MinSalesYear - 1) AND yb.MaxSalesYear;
GO

-----------------------------------------------------------------------
-- BUDGET FX LAYER
--
-- Sales transactions are stored in USD. Budget amounts from
-- vw_Budget_ChannelMonth are therefore also in USD.
--
-- This view converts USD budget amounts TO each country's local
-- currency using ExchangeRates (stored as USD -> Local, i.e.
-- Rate = units of local currency per 1 USD).
--
-- Column semantics (corrected):
--   BudgetGrossAmount_Report  = USD amount (the reporting currency, unchanged)
--   BudgetGrossAmount_Local   = converted to country's local currency
--   FxRate_UsdToLocal         = the conversion multiplier applied
--
-- FIX: Corrected FX direction (was treating budget as local and
--      converting to USD; budget IS USD, we convert to local).
-- FIX: Carry-forward via OUTER APPLY for months where exact FX
--      is missing (uses most recent available rate).
-----------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vw_Budget_ChannelMonth_FX]
AS
WITH
B AS (
    SELECT Country, Category, SalesChannelKey, BudgetYear, BudgetMonthStart, Scenario,
        BudgetGrowthPct, Audit_ChannelShare, Audit_MonthShare, Audit_ReturnRate,
        BudgetGrossAmount, BudgetNetAmount, BudgetGrossQuantity, BudgetNetQuantity, BudgetMethod
    FROM dbo.vw_Budget_ChannelMonth
),

CountryCurrency AS (
    SELECT g.Country, MAX(g.ISOCode) AS LocalCurrency
    FROM dbo.Geography g
    WHERE g.Country IS NOT NULL AND g.ISOCode IS NOT NULL
    GROUP BY g.Country
),

-- Monthly FX rates (AVG daily within month)
-- ExchangeRates invariant: FromCurrency=USD, ToCurrency=Local, Rate=Local per 1 USD
FxMonth AS (
    SELECT
        DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS MonthStart,
        FromCurrency, ToCurrency,
        AVG(Rate) AS AvgRate
    FROM dbo.ExchangeRates
    GROUP BY DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1), FromCurrency, ToCurrency
),

Base AS (
    SELECT b.*, cc.LocalCurrency
    FROM B b
    LEFT JOIN CountryCurrency cc ON cc.Country = b.Country
),

-- Resolve FX: USD -> Local with carry-forward
FxResolved AS (
    SELECT
        x.*,

        CASE
            WHEN x.LocalCurrency IS NULL                       THEN NULL
            WHEN x.LocalCurrency = 'USD'                       THEN 1.0
            WHEN fxD.AvgRate IS NOT NULL                       THEN fxD.AvgRate
            WHEN fxI.AvgRate IS NOT NULL AND fxI.AvgRate <> 0  THEN 1.0 / fxI.AvgRate
            ELSE NULL
        END AS FxRate_UsdToLocal,

        CASE
            WHEN x.LocalCurrency IS NULL                       THEN 'missing-currency'
            WHEN x.LocalCurrency = 'USD'                       THEN 'identity'
            WHEN fxD.AvgRate IS NOT NULL                       THEN 'direct'
            WHEN fxI.AvgRate IS NOT NULL                       THEN 'inverse'
            ELSE 'missing-rate'
        END AS Audit_FxSource
    FROM Base x
    -- Direct: USD -> Local
    -- Carry-forward (latest rate on or before budget month);
    -- if budget month predates all FX data, carry-backward (earliest available rate).
    OUTER APPLY (
        SELECT TOP (1) fm.AvgRate
        FROM FxMonth fm
        WHERE fm.FromCurrency = 'USD'
          AND fm.ToCurrency   = x.LocalCurrency
        ORDER BY
            CASE WHEN fm.MonthStart <= x.BudgetMonthStart THEN 0 ELSE 1 END,
            ABS(DATEDIFF(MONTH, fm.MonthStart, x.BudgetMonthStart))
    ) fxD
    -- Inverse fallback: Local -> USD (invert to get USD -> Local)
    -- Same bidirectional logic: prefer carry-forward, fall back to carry-backward.
    OUTER APPLY (
        SELECT TOP (1) fm.AvgRate
        FROM FxMonth fm
        WHERE fm.FromCurrency = x.LocalCurrency
          AND fm.ToCurrency   = 'USD'
        ORDER BY
            CASE WHEN fm.MonthStart <= x.BudgetMonthStart THEN 0 ELSE 1 END,
            ABS(DATEDIFF(MONTH, fm.MonthStart, x.BudgetMonthStart))
    ) fxI
)

SELECT
    Country, Category, SalesChannelKey, BudgetYear, BudgetMonthStart, Scenario,

    LocalCurrency,
    CAST('USD' AS varchar(10)) AS ReportCurrency,
    CAST(FxRate_UsdToLocal AS decimal(18,6)) AS FxRate_UsdToLocal,
    Audit_FxSource,

    -- Report amounts (USD - unchanged)
    CAST(BudgetGrossAmount AS decimal(19,2)) AS BudgetGrossAmount_Report,
    CAST(BudgetNetAmount   AS decimal(19,2)) AS BudgetNetAmount_Report,

    -- Local currency amounts (converted from USD)
    CAST(CASE WHEN FxRate_UsdToLocal IS NULL THEN NULL ELSE BudgetGrossAmount * FxRate_UsdToLocal END AS decimal(19,2)) AS BudgetGrossAmount_Local,
    CAST(CASE WHEN FxRate_UsdToLocal IS NULL THEN NULL ELSE BudgetNetAmount   * FxRate_UsdToLocal END AS decimal(19,2)) AS BudgetNetAmount_Local,

    CAST(BudgetGrossQuantity AS decimal(19,2)) AS BudgetGrossQuantity,
    CAST(BudgetNetQuantity   AS decimal(19,2)) AS BudgetNetQuantity,

    CAST(BudgetGrowthPct     AS decimal(9,6))  AS BudgetGrowthPct,
    CAST(Audit_ChannelShare  AS decimal(9,6))  AS Audit_ChannelShare,
    CAST(Audit_MonthShare    AS decimal(9,6))  AS Audit_MonthShare,
    CAST(Audit_ReturnRate    AS decimal(9,6))  AS Audit_ReturnRate,

    BudgetMethod
FROM FxResolved;
GO
