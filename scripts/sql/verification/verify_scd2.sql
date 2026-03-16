-- ============================================================================
-- VERIFY SCD TYPE 2 BEHAVIOUR FOR CUSTOMERS AND PRODUCTS
-- Run after loading generated data into SQL Server.
-- Assumes customers.scd2.enabled=true and products.scd2.enabled=true in config.
-- ============================================================================


-- ############################################################################
-- CUSTOMER SCD2 — Life Event Engine
-- ############################################################################

-- ============================================================================
-- 1. Basic SCD2 Structure
-- ============================================================================

-- 1a. Version distribution — how many customers have multiple versions?
SELECT
    VersionCount,
    COUNT(*)                                                    AS CustomerCount,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,1)) AS PctOfTotal
FROM (
    SELECT CustomerID, COUNT(*) AS VersionCount
    FROM Customers
    GROUP BY CustomerID
) x
GROUP BY VersionCount
ORDER BY VersionCount;
-- EXPECTED: ~85% at version 1 (unchanged), 15% with 2-4 versions
--           (governed by scd2.change_rate, default 0.15)

-- 1b. Only one IsCurrent=1 row per CustomerID
SELECT
    CustomerID,
    SUM(CAST(IsCurrent AS INT)) AS CurrentCount
FROM Customers
GROUP BY CustomerID
HAVING SUM(CAST(IsCurrent AS INT)) <> 1;
-- EXPECTED: zero rows (exactly one current version per customer)

-- 1c. Version numbers are sequential per CustomerID
SELECT c.CustomerID, c.VersionNumber
FROM Customers c
WHERE NOT EXISTS (
    SELECT 1 FROM Customers c2
    WHERE c2.CustomerID = c.CustomerID
      AND c2.VersionNumber = c.VersionNumber - 1
)
AND c.VersionNumber > 1;
-- EXPECTED: zero rows (no gaps in version sequence)


-- ============================================================================
-- 2. Date Chain Integrity
-- ============================================================================

-- 2a. EffectiveStartDate < EffectiveEndDate for all non-current rows
SELECT CustomerID, VersionNumber, EffectiveStartDate, EffectiveEndDate
FROM Customers
WHERE IsCurrent = 0
  AND EffectiveStartDate >= EffectiveEndDate;
-- EXPECTED: zero rows

-- 2b. Current rows should have EffectiveEndDate = '9999-12-31'
SELECT CustomerID, VersionNumber, EffectiveEndDate
FROM Customers
WHERE IsCurrent = 1
  AND EffectiveEndDate <> '9999-12-31';
-- EXPECTED: zero rows

-- 2c. No gaps or overlaps in date chain per CustomerID
--     (previous version EndDate + 1 day = next version StartDate)
SELECT
    c1.CustomerID,
    c1.VersionNumber                                    AS PrevVersion,
    c1.EffectiveEndDate                                 AS PrevEnd,
    c2.VersionNumber                                    AS NextVersion,
    c2.EffectiveStartDate                               AS NextStart,
    DATEDIFF(DAY, c1.EffectiveEndDate, c2.EffectiveStartDate) AS GapDays
FROM Customers c1
JOIN Customers c2
  ON c2.CustomerID = c1.CustomerID
 AND c2.VersionNumber = c1.VersionNumber + 1
WHERE DATEDIFF(DAY, c1.EffectiveEndDate, c2.EffectiveStartDate) <> 1;
-- EXPECTED: zero rows (each next version starts exactly 1 day after prev ends)

-- 2d. First version EffectiveStartDate matches CustomerStartDate
SELECT CustomerID, VersionNumber, EffectiveStartDate, CustomerStartDate
FROM Customers
WHERE VersionNumber = 1
  AND CAST(EffectiveStartDate AS DATE) <> CAST(CustomerStartDate AS DATE);
-- EXPECTED: zero rows


-- ============================================================================
-- 3. Life Event Coherence
-- ============================================================================

-- 3a. What attributes changed between versions? (shows life event variety)
SELECT
    ChangeType,
    COUNT(*)                                                    AS Occurrences,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,1)) AS PctOfChanges
FROM (
    SELECT
        c2.CustomerID,
        CASE
            WHEN c2.YearlyIncome > c1.YearlyIncome AND c2.MaritalStatus = c1.MaritalStatus
                THEN 'Career Growth'
            WHEN c1.MaritalStatus IN ('Single', 'Divorced') AND c2.MaritalStatus = 'Married'
                THEN 'Marriage'
            WHEN c1.MaritalStatus = 'Married' AND c2.MaritalStatus = 'Divorced'
                THEN 'Divorce'
            WHEN c2.NumberOfChildren > c1.NumberOfChildren
                THEN 'Family Growth'
            WHEN c1.HomeOwnership IN ('Rent', 'Other') AND c2.HomeOwnership = 'Own'
                THEN 'Home Purchase'
            WHEN c2.GeographyKey <> c1.GeographyKey
                THEN 'Relocation'
            WHEN c2.LoyaltyTierKey > c1.LoyaltyTierKey
                THEN 'Tier Upgrade'
            ELSE 'Other / Combined'
        END AS ChangeType
    FROM Customers c1
    JOIN Customers c2
      ON c2.CustomerID = c1.CustomerID
     AND c2.VersionNumber = c1.VersionNumber + 1
) x
GROUP BY ChangeType
ORDER BY Occurrences DESC;
-- EXPECTED: mix of all event types, Career Growth ~30%, Marriage ~20%, etc.

-- 3b. Organizations should NOT have SCD2 versions
SELECT CustomerID, COUNT(*) AS VersionCount
FROM Customers
WHERE CustomerType = 'Organization'
GROUP BY CustomerID
HAVING COUNT(*) > 1;
-- EXPECTED: zero rows (orgs stay at version 1)

-- 3c. Marital status transitions should be valid
--     (Single->Married OK, Married->Divorced OK, but not Single->Divorced directly)
SELECT
    c1.CustomerID,
    c1.MaritalStatus AS FromStatus,
    c2.MaritalStatus AS ToStatus
FROM Customers c1
JOIN Customers c2
  ON c2.CustomerID = c1.CustomerID
 AND c2.VersionNumber = c1.VersionNumber + 1
WHERE c1.MaritalStatus = 'Single' AND c2.MaritalStatus = 'Divorced';
-- EXPECTED: zero rows (can't divorce without being married first)

-- 3d. NumberOfChildren should never decrease
SELECT
    c1.CustomerID,
    c1.VersionNumber,
    c1.NumberOfChildren AS PrevChildren,
    c2.NumberOfChildren AS NextChildren
FROM Customers c1
JOIN Customers c2
  ON c2.CustomerID = c1.CustomerID
 AND c2.VersionNumber = c1.VersionNumber + 1
WHERE c2.NumberOfChildren < c1.NumberOfChildren;
-- EXPECTED: zero rows


-- ============================================================================
-- 4. Key Integrity After SCD2 Expansion
-- ============================================================================

-- 4a. CustomerKey is unique across all version rows
SELECT CustomerKey, COUNT(*) AS Cnt
FROM Customers
GROUP BY CustomerKey
HAVING COUNT(*) > 1;
-- EXPECTED: zero rows

-- 4b. CustomerProfile links to IsCurrent=1 row only
SELECT
    cp.CustomerKey,
    c.IsCurrent
FROM CustomerProfile cp
LEFT JOIN Customers c ON c.CustomerKey = cp.CustomerKey
WHERE c.IsCurrent IS NULL OR c.IsCurrent <> 1;
-- EXPECTED: zero rows

-- 4c. OrganizationProfile links to IsCurrent=1 row only
SELECT
    op.CustomerKey,
    c.IsCurrent
FROM OrganizationProfile op
LEFT JOIN Customers c ON c.CustomerKey = op.CustomerKey
WHERE c.IsCurrent IS NULL OR c.IsCurrent <> 1;
-- EXPECTED: zero rows

-- 4d. Sales fact should join to customers (IsCurrent=1 for point-in-time reporting)
SELECT
    COUNT(*)                                                        AS TotalSales,
    SUM(CASE WHEN c.CustomerKey IS NULL THEN 1 ELSE 0 END)         AS OrphanedSales,
    SUM(CASE WHEN c.IsCurrent = 1 THEN 1 ELSE 0 END)              AS CurrentVersionSales
FROM Sales f
LEFT JOIN Customers c ON c.CustomerKey = f.CustomerKey;
-- EXPECTED: zero orphaned sales, all join to IsCurrent=1


-- ############################################################################
-- PRODUCT SCD2 — Price Revision Engine
-- ############################################################################

-- ============================================================================
-- 5. Basic SCD2 Structure
-- ============================================================================

-- 5a. Version distribution for products
SELECT
    VersionCount,
    COUNT(*)                                                    AS ProductCount,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,1)) AS PctOfTotal
FROM (
    SELECT ProductID, COUNT(*) AS VersionCount
    FROM Products
    GROUP BY ProductID
) x
GROUP BY VersionCount
ORDER BY VersionCount;
-- EXPECTED: mix of 1-4 versions (governed by scd2.max_versions)

-- 5b. Only one IsCurrent=1 row per ProductID
SELECT
    ProductID,
    SUM(CAST(IsCurrent AS INT)) AS CurrentCount
FROM Products
GROUP BY ProductID
HAVING SUM(CAST(IsCurrent AS INT)) <> 1;
-- EXPECTED: zero rows

-- 5c. Version numbers are sequential per ProductID
SELECT p.ProductID, p.VersionNumber
FROM Products p
WHERE NOT EXISTS (
    SELECT 1 FROM Products p2
    WHERE p2.ProductID = p.ProductID
      AND p2.VersionNumber = p.VersionNumber - 1
)
AND p.VersionNumber > 1;
-- EXPECTED: zero rows


-- ============================================================================
-- 6. Date Chain Integrity (Products)
-- ============================================================================

-- 6a. EffectiveStartDate < EffectiveEndDate for non-current rows
SELECT ProductID, VersionNumber, EffectiveStartDate, EffectiveEndDate
FROM Products
WHERE IsCurrent = 0
  AND EffectiveStartDate >= EffectiveEndDate;
-- EXPECTED: zero rows

-- 6b. Current rows have EffectiveEndDate = '9999-12-31'
SELECT ProductID, VersionNumber, EffectiveEndDate
FROM Products
WHERE IsCurrent = 1
  AND EffectiveEndDate <> '9999-12-31';
-- EXPECTED: zero rows

-- 6c. No gaps or overlaps in date chain per ProductID
SELECT
    p1.ProductID,
    p1.VersionNumber                                    AS PrevVersion,
    p1.EffectiveEndDate                                 AS PrevEnd,
    p2.VersionNumber                                    AS NextVersion,
    p2.EffectiveStartDate                               AS NextStart,
    DATEDIFF(DAY, p1.EffectiveEndDate, p2.EffectiveStartDate) AS GapDays
FROM Products p1
JOIN Products p2
  ON p2.ProductID = p1.ProductID
 AND p2.VersionNumber = p1.VersionNumber + 1
WHERE DATEDIFF(DAY, p1.EffectiveEndDate, p2.EffectiveStartDate) <> 1;
-- EXPECTED: zero rows


-- ============================================================================
-- 7. Price Revision Coherence
-- ============================================================================

-- 7a. Prices should change between versions (that's the point of product SCD2)
SELECT
    COUNT(*) AS TotalTransitions,
    SUM(CASE WHEN p2.ListPrice <> p1.ListPrice THEN 1 ELSE 0 END) AS PriceChanged,
    CAST(SUM(CASE WHEN p2.ListPrice <> p1.ListPrice THEN 1 ELSE 0 END)
         * 100.0 / NULLIF(COUNT(*), 0) AS DECIMAL(5,1))           AS PctPriceChanged
FROM Products p1
JOIN Products p2
  ON p2.ProductID = p1.ProductID
 AND p2.VersionNumber = p1.VersionNumber + 1;
-- EXPECTED: ~100% of transitions have a price change

-- 7b. Price drift magnitude (should be within configured drift range, default ±5%)
SELECT
    AVG(ABS(PriceDriftPct))                    AS AvgDriftPct,
    MIN(PriceDriftPct)                         AS MinDriftPct,
    MAX(PriceDriftPct)                         AS MaxDriftPct,
    STDEV(PriceDriftPct)                       AS StdevDriftPct
FROM (
    SELECT
        p1.ProductID,
        p1.VersionNumber,
        (p2.ListPrice - p1.ListPrice) * 100.0 / NULLIF(p1.ListPrice, 0) AS PriceDriftPct
    FROM Products p1
    JOIN Products p2
      ON p2.ProductID = p1.ProductID
     AND p2.VersionNumber = p1.VersionNumber + 1
    WHERE p1.ListPrice > 0
) x;
-- EXPECTED: AvgDriftPct ~2-3%, MinDriftPct ~-5%, MaxDriftPct ~+10%
--           (drift range is [-price_drift, +price_drift*2], default [-0.05, +0.10])

-- 7c. UnitCost should never exceed ListPrice in any version
SELECT ProductID, VersionNumber, ListPrice, UnitCost
FROM Products
WHERE UnitCost > ListPrice;
-- EXPECTED: zero rows

-- 7d. Non-price attributes should NOT change between versions
SELECT COUNT(*) AS AttributeChanges
FROM Products p1
JOIN Products p2
  ON p2.ProductID = p1.ProductID
 AND p2.VersionNumber = p1.VersionNumber + 1
WHERE p1.ProductName <> p2.ProductName
   OR p1.SubcategoryKey <> p2.SubcategoryKey
   OR p1.BrandName <> p2.BrandName;
-- EXPECTED: zero (only prices change between product versions)


-- ============================================================================
-- 8. Key Integrity After SCD2 Expansion (Products)
-- ============================================================================

-- 8a. ProductKey is unique across all version rows
SELECT ProductKey, COUNT(*) AS Cnt
FROM Products
GROUP BY ProductKey
HAVING COUNT(*) > 1;
-- EXPECTED: zero rows

-- 8b. ProductProfile links to current version only
SELECT
    pp.ProductKey,
    p.IsCurrent
FROM ProductProfile pp
LEFT JOIN Products p ON p.ProductKey = pp.ProductKey
WHERE p.IsCurrent IS NULL OR p.IsCurrent <> 1;
-- EXPECTED: zero rows

-- 8c. Sales fact should join to products
SELECT
    COUNT(*)                                                        AS TotalSales,
    SUM(CASE WHEN p.ProductKey IS NULL THEN 1 ELSE 0 END)          AS OrphanedSales
FROM Sales f
LEFT JOIN Products p ON p.ProductKey = f.ProductKey;
-- EXPECTED: zero orphaned sales


-- ############################################################################
-- 9. COMBINED SCD2 HEALTH SCORECARD
-- ############################################################################
SELECT
    'Customer: unique IsCurrent per ID'   AS [Check],
    'Exactly one IsCurrent=1 row per CustomerID; FAIL = duplicate or missing current version' AS [Description],
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS Result
FROM (
    SELECT CustomerID FROM Customers
    GROUP BY CustomerID HAVING SUM(CAST(IsCurrent AS INT)) <> 1
) x

UNION ALL

SELECT
    'Customer: no date chain gaps',
    'Each version EndDate+1 day = next version StartDate; FAIL = gap or overlap in version timeline',
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM Customers c1
JOIN Customers c2
  ON c2.CustomerID = c1.CustomerID AND c2.VersionNumber = c1.VersionNumber + 1
WHERE DATEDIFF(DAY, c1.EffectiveEndDate, c2.EffectiveStartDate) <> 1

UNION ALL

SELECT
    'Customer: orgs have no versions',
    'Organization customers stay at version 1; FAIL = org has SCD2 life events (should be individual-only)',
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM (
    SELECT CustomerID FROM Customers WHERE CustomerType = 'Organization'
    GROUP BY CustomerID HAVING COUNT(*) > 1
) x

UNION ALL

SELECT
    'Customer: children never decrease',
    'NumberOfChildren only increases across versions (family growth); FAIL = child count went down',
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM Customers c1
JOIN Customers c2
  ON c2.CustomerID = c1.CustomerID AND c2.VersionNumber = c1.VersionNumber + 1
WHERE c2.NumberOfChildren < c1.NumberOfChildren

UNION ALL

SELECT
    'Product: unique IsCurrent per ID',
    'Exactly one IsCurrent=1 row per ProductID; FAIL = duplicate or missing current version',
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM (
    SELECT ProductID FROM Products
    GROUP BY ProductID HAVING SUM(CAST(IsCurrent AS INT)) <> 1
) x

UNION ALL

SELECT
    'Product: no date chain gaps',
    'Each version EndDate+1 day = next version StartDate; FAIL = gap or overlap in price revision timeline',
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM Products p1
JOIN Products p2
  ON p2.ProductID = p1.ProductID AND p2.VersionNumber = p1.VersionNumber + 1
WHERE DATEDIFF(DAY, p1.EffectiveEndDate, p2.EffectiveStartDate) <> 1

UNION ALL

SELECT
    'Product: UnitCost <= ListPrice',
    'Margin must be non-negative in every version; FAIL = cost exceeds selling price',
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM Products WHERE UnitCost > ListPrice

UNION ALL

SELECT
    'Product: non-price attrs stable',
    'Only ListPrice/UnitCost change between versions; FAIL = name, brand, or subcategory changed',
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM Products p1
JOIN Products p2
  ON p2.ProductID = p1.ProductID AND p2.VersionNumber = p1.VersionNumber + 1
WHERE p1.ProductName <> p2.ProductName
   OR p1.SubcategoryKey <> p2.SubcategoryKey;
