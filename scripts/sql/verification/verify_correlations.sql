-- ============================================================================
-- VERIFY BUSINESS CORRELATIONS IN GENERATED SALES DATA
-- Run after loading generated data into SQL Server.
-- Each query returns evidence of correlation (or lack thereof).
-- ============================================================================


-- ============================================================================
-- 1. StoreKey <-> SalesChannelKey
--    Physical stores should skew toward physical channels (1=Store, 10=Kiosk)
--    Online/Fulfillment stores should skew toward digital (2,3,6,7,8)
-- ============================================================================

-- 1a. Channel distribution by StoreType (should NOT be uniform)
SELECT
    s.StoreType,
    sc.SalesChannel,
    COUNT(*)                                         AS SalesCount,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY s.StoreType) AS DECIMAL(5,1)) AS PctOfStoreType
FROM Sales f
JOIN Stores s         ON s.StoreKey = f.StoreKey
JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
GROUP BY s.StoreType, sc.SalesChannel
ORDER BY s.StoreType, SalesCount DESC;

-- 1b. Quick check: do Convenience stores ever sell via Marketplace?
--     (Should be rare or zero with correlation; common without)
SELECT
    s.StoreType,
    sc.SalesChannel,
    COUNT(*) AS Cnt
FROM Sales f
JOIN Stores s         ON s.StoreKey = f.StoreKey
JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
WHERE s.StoreType = 'Convenience'
  AND sc.SalesChannel IN ('Marketplace', 'SocialCommerce', 'Web', 'MobileApp')
GROUP BY s.StoreType, sc.SalesChannel;

-- 1c. Chi-square independence check (simplified):
--     If correlated, the ratio of Store-channel sales for physical vs online
--     store types should differ significantly.
SELECT
    StoreGroup,
    SUM(CASE WHEN ChannelGroup = 'Physical' THEN 1 ELSE 0 END) AS PhysicalChannelSales,
    SUM(CASE WHEN ChannelGroup = 'Digital'  THEN 1 ELSE 0 END) AS DigitalChannelSales,
    CAST(SUM(CASE WHEN ChannelGroup = 'Physical' THEN 1 ELSE 0 END) * 100.0
       / NULLIF(COUNT(*), 0) AS DECIMAL(5,1))                   AS PctPhysical
FROM (
    SELECT
        CASE WHEN s.StoreType IN ('Online', 'Fulfillment') THEN 'Online/Fulfillment'
             ELSE 'Physical Store' END                          AS StoreGroup,
        CASE WHEN sc.IsPhysical = 1 THEN 'Physical'
             WHEN sc.IsDigital = 1  THEN 'Digital'
             ELSE 'Other' END                                   AS ChannelGroup
    FROM Sales f
    JOIN Stores s         ON s.StoreKey = f.StoreKey
    JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
) x
GROUP BY StoreGroup;
-- EXPECTED: Physical Stores should show >50% PhysicalChannelSales
--           Online/Fulfillment should show >70% DigitalChannelSales
-- NO CORRELATION: both groups ~equal split


-- ============================================================================
-- 2. CustomerKey <-> StoreKey (Geographic Affinity)
--    Customers should mostly buy from stores in their own country.
-- ============================================================================

-- 2a. % of sales where customer and store are in the same country
SELECT
    COUNT(*)                                                        AS TotalSales,
    SUM(CASE WHEN cg.Country = sg.Country THEN 1 ELSE 0 END)       AS SameCountrySales,
    CAST(SUM(CASE WHEN cg.Country = sg.Country THEN 1 ELSE 0 END)
         * 100.0 / NULLIF(COUNT(*), 0) AS DECIMAL(5,1))            AS PctSameCountry
FROM Sales f
JOIN Customers c  ON c.CustomerKey = f.CustomerKey AND c.IsCurrent = 1
JOIN Stores s     ON s.StoreKey = f.StoreKey
JOIN Geography cg ON cg.GeographyKey = c.GeographyKey
JOIN Geography sg ON sg.GeographyKey = s.GeographyKey;
-- EXPECTED WITH CORRELATION: ~60-75% same country
-- NO CORRELATION: proportional to store count per country (likely <30% for multi-country data)

-- 2b. Top customer-store country pairs
SELECT TOP 20
    cg.Country AS CustomerCountry,
    sg.Country AS StoreCountry,
    COUNT(*)   AS SalesCount,
    CASE WHEN cg.Country = sg.Country THEN 'MATCH' ELSE 'CROSS' END AS GeoMatch
FROM Sales f
JOIN Customers c  ON c.CustomerKey = f.CustomerKey AND c.IsCurrent = 1
JOIN Stores s     ON s.StoreKey = f.StoreKey
JOIN Geography cg ON cg.GeographyKey = c.GeographyKey
JOIN Geography sg ON sg.GeographyKey = s.GeographyKey
GROUP BY cg.Country, sg.Country
ORDER BY SalesCount DESC;

-- 2c. Per-country local purchase rate
SELECT
    cg.Country,
    COUNT(*)                                                        AS TotalSales,
    SUM(CASE WHEN cg.Country = sg.Country THEN 1 ELSE 0 END)       AS LocalSales,
    CAST(SUM(CASE WHEN cg.Country = sg.Country THEN 1 ELSE 0 END)
         * 100.0 / NULLIF(COUNT(*), 0) AS DECIMAL(5,1))            AS PctLocal
FROM Sales f
JOIN Customers c  ON c.CustomerKey = f.CustomerKey AND c.IsCurrent = 1
JOIN Stores s     ON s.StoreKey = f.StoreKey
JOIN Geography cg ON cg.GeographyKey = c.GeographyKey
JOIN Geography sg ON sg.GeographyKey = s.GeographyKey
GROUP BY cg.Country
ORDER BY TotalSales DESC;
-- EXPECTED: each country should show 60-75% local


-- ============================================================================
-- 3. SalesChannelKey <-> DeliveryDate
--    Store/Kiosk channels should have 0-day DueDate offset.
--    Online channels should have 2-5 day offset.
--    B2B should have 5-10 day offset.
-- ============================================================================

-- 3a. Average due-date offset by channel
SELECT
    sc.SalesChannel,
    sc.TypicalFulfillmentDays,
    COUNT(*)                                                               AS SalesCount,
    AVG(CAST(DATEDIFF(DAY, f.OrderDate, f.DueDate) AS FLOAT))             AS AvgDueDays,
    MIN(DATEDIFF(DAY, f.OrderDate, f.DueDate))                            AS MinDueDays,
    MAX(DATEDIFF(DAY, f.OrderDate, f.DueDate))                            AS MaxDueDays
FROM Sales f
JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
GROUP BY sc.SalesChannel, sc.TypicalFulfillmentDays
ORDER BY AvgDueDays;
-- EXPECTED: Store/Kiosk AvgDueDays ~0-1, Online ~2-4, B2B ~6-8
-- NO CORRELATION: all channels show AvgDueDays ~5 (uniform 3-7)

-- 3b. Distribution of due-date offsets for physical vs digital channels
SELECT
    ChannelType,
    DueDays,
    COUNT(*) AS Cnt
FROM (
    SELECT
        CASE WHEN sc.IsPhysical = 1 THEN 'Physical'
             WHEN sc.IsDigital  = 1 THEN 'Digital'
             WHEN sc.IsB2B      = 1 THEN 'B2B'
             ELSE 'Other' END                               AS ChannelType,
        DATEDIFF(DAY, f.OrderDate, f.DueDate)               AS DueDays
    FROM Sales f
    JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
) x
GROUP BY ChannelType, DueDays
ORDER BY ChannelType, DueDays;
-- EXPECTED: Physical peaks at 0, Digital peaks at 2-4, B2B peaks at 5-8

-- 3c. Same-day delivery should only come from physical channels
SELECT
    sc.SalesChannel,
    sc.IsPhysical,
    COUNT(*) AS SameDayDeliveries
FROM Sales f
JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
WHERE f.DueDate = f.OrderDate
GROUP BY sc.SalesChannel, sc.IsPhysical
ORDER BY SameDayDeliveries DESC;
-- EXPECTED: overwhelmingly Store + Kiosk


-- ============================================================================
-- 4. SalesChannelKey <-> ProductKey (Channel Eligibility)
--    Products marked EligibleOnline=0 should rarely appear in Online sales.
--    Products marked EligibleStore=0 should rarely appear in Store sales.
-- ============================================================================

-- 4a. Ineligible product sales by channel (should be very low %)
SELECT
    ChannelGroup,
    COUNT(*)                                                                AS TotalSales,
    SUM(CASE WHEN IsEligible = 0 THEN 1 ELSE 0 END)                        AS IneligibleSales,
    CAST(SUM(CASE WHEN IsEligible = 0 THEN 1 ELSE 0 END) * 100.0
         / NULLIF(COUNT(*), 0) AS DECIMAL(5,2))                             AS PctIneligible
FROM (
    SELECT
        CASE
            WHEN f.SalesChannelKey IN (1, 10)      THEN 'Store'
            WHEN f.SalesChannelKey IN (2, 6, 7)    THEN 'Online'
            WHEN f.SalesChannelKey IN (3, 8)       THEN 'Marketplace'
            WHEN f.SalesChannelKey IN (4, 9)       THEN 'B2B'
            ELSE 'Other'
        END AS ChannelGroup,
        CASE
            WHEN f.SalesChannelKey IN (1, 10)   AND pp.EligibleStore       = 1 THEN 1
            WHEN f.SalesChannelKey IN (2, 6, 7) AND pp.EligibleOnline      = 1 THEN 1
            WHEN f.SalesChannelKey IN (3, 8)    AND pp.EligibleMarketplace  = 1 THEN 1
            WHEN f.SalesChannelKey IN (4, 9)    AND pp.EligibleB2B         = 1 THEN 1
            ELSE 0
        END AS IsEligible
    FROM Sales f
    JOIN Products p        ON p.ProductKey = f.ProductKey
    JOIN ProductProfile pp ON pp.ProductKey = p.ProductKey
) x
GROUP BY ChannelGroup
ORDER BY PctIneligible DESC;
-- EXPECTED WITH CORRELATION: PctIneligible < 5% (soft penalty, not hard block)
-- NO CORRELATION: PctIneligible matches overall ineligibility rate (~15-25%)

-- 4b. Store-only products appearing in digital channels
SELECT
    sc.SalesChannel,
    p.ProductName,
    pp.EligibleStore,
    pp.EligibleOnline,
    COUNT(*) AS SalesCount
FROM Sales f
JOIN Products p        ON p.ProductKey = f.ProductKey AND p.IsCurrent = 1
JOIN ProductProfile pp ON pp.ProductKey = p.ProductKey
JOIN SalesChannels sc  ON sc.SalesChannelKey = f.SalesChannelKey
WHERE pp.EligibleOnline = 0
  AND sc.IsDigital = 1
GROUP BY sc.SalesChannel, p.ProductName, pp.EligibleStore, pp.EligibleOnline
ORDER BY SalesCount DESC;
-- EXPECTED: very few rows (soft penalty allows ~5%)

-- 4c. Eligibility compliance rate by channel
SELECT
    sc.SalesChannel,
    COUNT(*) AS TotalSales,
    SUM(CASE
        WHEN sc.SalesChannelKey IN (1, 10)   THEN pp.EligibleStore
        WHEN sc.SalesChannelKey IN (2, 6, 7) THEN pp.EligibleOnline
        WHEN sc.SalesChannelKey IN (3, 8)    THEN pp.EligibleMarketplace
        WHEN sc.SalesChannelKey IN (4, 9)    THEN pp.EligibleB2B
        ELSE 1
    END) AS EligibleSales,
    CAST(SUM(CASE
        WHEN sc.SalesChannelKey IN (1, 10)   THEN pp.EligibleStore
        WHEN sc.SalesChannelKey IN (2, 6, 7) THEN pp.EligibleOnline
        WHEN sc.SalesChannelKey IN (3, 8)    THEN pp.EligibleMarketplace
        WHEN sc.SalesChannelKey IN (4, 9)    THEN pp.EligibleB2B
        ELSE 1
    END) * 100.0 / NULLIF(COUNT(*), 0) AS DECIMAL(5,1)) AS CompliancePct
FROM Sales f
JOIN Products p        ON p.ProductKey = f.ProductKey
JOIN ProductProfile pp ON pp.ProductKey = p.ProductKey
JOIN SalesChannels sc  ON sc.SalesChannelKey = f.SalesChannelKey
GROUP BY sc.SalesChannel
ORDER BY CompliancePct;
-- EXPECTED: >95% compliance across all channels


-- ============================================================================
-- 5. SalesChannelKey <-> PromotionKey (Channel-Targeted Promos)
--    Promotions with PromotionCategory='Store' should only appear
--    with physical channels. 'Online' promos only with digital channels.
-- ============================================================================

-- 5a. Promo category vs channel type alignment
SELECT
    pr.PromotionCategory,
    CASE WHEN sc.IsPhysical = 1 THEN 'Physical'
         WHEN sc.IsDigital  = 1 THEN 'Digital'
         WHEN sc.IsB2B      = 1 THEN 'B2B'
         ELSE 'Other' END                                   AS ChannelType,
    COUNT(*)                                                 AS SalesCount
FROM Sales f
JOIN Promotions pr     ON pr.PromotionKey = f.PromotionKey
JOIN SalesChannels sc  ON sc.SalesChannelKey = f.SalesChannelKey
WHERE pr.PromotionKey > 1  -- exclude "No Discount"
GROUP BY pr.PromotionCategory,
         CASE WHEN sc.IsPhysical = 1 THEN 'Physical'
              WHEN sc.IsDigital  = 1 THEN 'Digital'
              WHEN sc.IsB2B      = 1 THEN 'B2B'
              ELSE 'Other' END
ORDER BY pr.PromotionCategory, SalesCount DESC;
-- EXPECTED: 'Store' category -> mostly Physical channel
--           'Online' category -> mostly Digital channel
-- NO CORRELATION: uniform distribution across channel types

-- 5b. Misaligned promos (Store promo on Digital channel or vice versa)
SELECT
    'Store promo on Digital channel' AS MismatchType,
    COUNT(*) AS MismatchCount,
    CAST(COUNT(*) * 100.0 / NULLIF(
        (SELECT COUNT(*) FROM Sales f2
         JOIN Promotions pr2 ON pr2.PromotionKey = f2.PromotionKey
         WHERE pr2.PromotionCategory = 'Store' AND pr2.PromotionKey > 1), 0
    ) AS DECIMAL(5,1)) AS PctOfCategoryTotal
FROM Sales f
JOIN Promotions pr    ON pr.PromotionKey = f.PromotionKey
JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
WHERE pr.PromotionCategory = 'Store'
  AND pr.PromotionKey > 1
  AND sc.IsDigital = 1

UNION ALL

SELECT
    'Online promo on Physical channel' AS MismatchType,
    COUNT(*) AS MismatchCount,
    CAST(COUNT(*) * 100.0 / NULLIF(
        (SELECT COUNT(*) FROM Sales f2
         JOIN Promotions pr2 ON pr2.PromotionKey = f2.PromotionKey
         WHERE pr2.PromotionCategory = 'Online' AND pr2.PromotionKey > 1), 0
    ) AS DECIMAL(5,1)) AS PctOfCategoryTotal
FROM Sales f
JOIN Promotions pr    ON pr.PromotionKey = f.PromotionKey
JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
WHERE pr.PromotionCategory = 'Online'
  AND pr.PromotionKey > 1
  AND sc.IsPhysical = 1;
-- EXPECTED WITH CORRELATION: 0% mismatch
-- NO CORRELATION: ~50% mismatch (random assignment)


-- ============================================================================
-- BONUS: Combined correlation health scorecard
-- ============================================================================
SELECT
    'Store-Channel'      AS Correlation,
    'Physical stores use physical channels >50%' AS Expectation,
    CAST((
        SELECT SUM(CASE WHEN sc.IsPhysical = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)
        FROM Sales f
        JOIN Stores s         ON s.StoreKey = f.StoreKey
        JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
        WHERE s.StoreType NOT IN ('Online', 'Fulfillment')
    ) AS DECIMAL(5,1)) AS ActualPct

UNION ALL

SELECT
    'Customer-Store Geo',
    'Same-country sales >60%',
    CAST((
        SELECT SUM(CASE WHEN cg.Country = sg.Country THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)
        FROM Sales f
        JOIN Customers c  ON c.CustomerKey = f.CustomerKey AND c.IsCurrent = 1
        JOIN Stores s     ON s.StoreKey = f.StoreKey
        JOIN Geography cg ON cg.GeographyKey = c.GeographyKey
        JOIN Geography sg ON sg.GeographyKey = s.GeographyKey
    ) AS DECIMAL(5,1))

UNION ALL

SELECT
    'Channel-Delivery',
    'Store/Kiosk avg due days <2',
    CAST((
        SELECT AVG(CAST(DATEDIFF(DAY, f.OrderDate, f.DueDate) AS FLOAT))
        FROM Sales f
        WHERE f.SalesChannelKey IN (1, 10)
    ) AS DECIMAL(5,1))

UNION ALL

SELECT
    'Channel-Product',
    'Channel eligibility compliance >95%',
    CAST((
        SELECT SUM(CASE
            WHEN f.SalesChannelKey IN (1, 10)   THEN pp.EligibleStore
            WHEN f.SalesChannelKey IN (2, 6, 7) THEN pp.EligibleOnline
            WHEN f.SalesChannelKey IN (3, 8)    THEN pp.EligibleMarketplace
            WHEN f.SalesChannelKey IN (4, 9)    THEN pp.EligibleB2B
            ELSE 1 END) * 100.0 / NULLIF(COUNT(*), 0)
        FROM Sales f
        JOIN Products p        ON p.ProductKey = f.ProductKey
        JOIN ProductProfile pp ON pp.ProductKey = p.ProductKey
    ) AS DECIMAL(5,1))

UNION ALL

SELECT
    'Channel-Promo',
    'Store promo on digital channel = 0%',
    CAST(ISNULL((
        SELECT SUM(CASE WHEN sc.IsDigital = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)
        FROM Sales f
        JOIN Promotions pr    ON pr.PromotionKey = f.PromotionKey
        JOIN SalesChannels sc ON sc.SalesChannelKey = f.SalesChannelKey
        WHERE pr.PromotionCategory = 'Store' AND pr.PromotionKey > 1
    ), 0) AS DECIMAL(5,1));
