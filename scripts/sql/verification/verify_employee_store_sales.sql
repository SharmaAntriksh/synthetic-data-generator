-- ============================================================================
-- VERIFY EMPLOYEE -> EMPLOYEE_STORE_ASSIGNMENTS -> SALES RELATIONSHIP
-- Run after loading generated data into SQL Server.
-- Spots (StoreKey, EmployeeKey, OrderDate) combos in Sales
-- that have no matching effective-dated row in EmployeeStoreAssignments.
--
-- Supports both sales_output modes:
--   sales        -> single Sales table
--   sales_order  -> SalesOrderHeader + SalesOrderDetail tables
-- ============================================================================

-- Detect which sales table exists
DECLARE @has_sales  BIT = CASE WHEN OBJECT_ID('dbo.Sales') IS NOT NULL THEN 1 ELSE 0 END;
DECLARE @has_header BIT = CASE WHEN OBJECT_ID('dbo.SalesOrderHeader') IS NOT NULL THEN 1 ELSE 0 END;


-- ############################################################################
-- 1. ORPHAN DETECTION — Sales with no valid assignment
-- ############################################################################

-- 1a. Sales where (EmployeeKey, StoreKey, OrderDate) has no
--     matching assignment row (the core check)
IF @has_sales = 1
BEGIN
    SELECT
        f.EmployeeKey,
        f.StoreKey,
        f.OrderDate,
        COUNT(*)                                                        AS SalesRows
    FROM Sales f
    LEFT JOIN EmployeeStoreAssignments esa
      ON  esa.EmployeeKey = f.EmployeeKey
     AND esa.StoreKey     = f.StoreKey
     AND f.OrderDate     >= esa.StartDate
     AND (esa.EndDate IS NULL OR f.OrderDate <= esa.EndDate)
    WHERE f.EmployeeKey > 0
      AND esa.EmployeeKey IS NULL
    GROUP BY f.EmployeeKey, f.StoreKey, f.OrderDate
    ORDER BY SalesRows DESC;
    -- EXPECTED: zero rows — every sale should map to an active assignment
END
ELSE IF @has_header = 1
BEGIN
    SELECT
        h.EmployeeKey,
        h.StoreKey,
        h.OrderDate,
        COUNT(*)                                                        AS SalesRows
    FROM SalesOrderHeader h
    LEFT JOIN EmployeeStoreAssignments esa
      ON  esa.EmployeeKey = h.EmployeeKey
     AND esa.StoreKey     = h.StoreKey
     AND h.OrderDate     >= esa.StartDate
     AND (esa.EndDate IS NULL OR h.OrderDate <= esa.EndDate)
    WHERE h.EmployeeKey > 0
      AND esa.EmployeeKey IS NULL
    GROUP BY h.EmployeeKey, h.StoreKey, h.OrderDate
    ORDER BY SalesRows DESC;
    -- EXPECTED: zero rows — every sale should map to an active assignment
END

-- 1b. Summary: how many sales rows are orphaned vs covered?
IF @has_sales = 1
BEGIN
    SELECT
        COUNT(*)                                                        AS TotalSales,
        SUM(CASE WHEN esa.EmployeeKey IS NOT NULL THEN 1 ELSE 0 END)   AS CoveredSales,
        SUM(CASE WHEN esa.EmployeeKey IS NULL
                  AND f.EmployeeKey > 0 THEN 1 ELSE 0 END)  AS OrphanedSales,
        SUM(CASE WHEN f.EmployeeKey <= 0 THEN 1 ELSE 0 END) AS NoSalesperson,
        CAST(SUM(CASE WHEN esa.EmployeeKey IS NULL
                       AND f.EmployeeKey > 0 THEN 1 ELSE 0 END)
             * 100.0 / NULLIF(COUNT(*), 0) AS DECIMAL(5,2))            AS OrphanedPct
    FROM Sales f
    LEFT JOIN EmployeeStoreAssignments esa
      ON  esa.EmployeeKey = f.EmployeeKey
     AND esa.StoreKey     = f.StoreKey
     AND f.OrderDate     >= esa.StartDate
     AND (esa.EndDate IS NULL OR f.OrderDate <= esa.EndDate);
    -- EXPECTED: OrphanedSales = 0, NoSalesperson = 0 (or very few)
END
ELSE IF @has_header = 1
BEGIN
    SELECT
        COUNT(*)                                                        AS TotalSales,
        SUM(CASE WHEN esa.EmployeeKey IS NOT NULL THEN 1 ELSE 0 END)   AS CoveredSales,
        SUM(CASE WHEN esa.EmployeeKey IS NULL
                  AND h.EmployeeKey > 0 THEN 1 ELSE 0 END)  AS OrphanedSales,
        SUM(CASE WHEN h.EmployeeKey <= 0 THEN 1 ELSE 0 END) AS NoSalesperson,
        CAST(SUM(CASE WHEN esa.EmployeeKey IS NULL
                       AND h.EmployeeKey > 0 THEN 1 ELSE 0 END)
             * 100.0 / NULLIF(COUNT(*), 0) AS DECIMAL(5,2))            AS OrphanedPct
    FROM SalesOrderHeader h
    LEFT JOIN EmployeeStoreAssignments esa
      ON  esa.EmployeeKey = h.EmployeeKey
     AND esa.StoreKey     = h.StoreKey
     AND h.OrderDate     >= esa.StartDate
     AND (esa.EndDate IS NULL OR h.OrderDate <= esa.EndDate);
    -- EXPECTED: OrphanedSales = 0, NoSalesperson = 0 (or very few)
END


-- ############################################################################
-- 2. REFERENTIAL INTEGRITY
-- ############################################################################

-- 2a. Sales referencing an EmployeeKey that doesn't exist in Employees
IF @has_sales = 1
BEGIN
    SELECT DISTINCT f.EmployeeKey
    FROM Sales f
    LEFT JOIN Employees e ON e.EmployeeKey = f.EmployeeKey
    WHERE f.EmployeeKey > 0
      AND e.EmployeeKey IS NULL;
    -- EXPECTED: zero rows
END
ELSE IF @has_header = 1
BEGIN
    SELECT DISTINCT h.EmployeeKey
    FROM SalesOrderHeader h
    LEFT JOIN Employees e ON e.EmployeeKey = h.EmployeeKey
    WHERE h.EmployeeKey > 0
      AND e.EmployeeKey IS NULL;
    -- EXPECTED: zero rows
END

-- 2b. EmployeeStoreAssignments referencing an EmployeeKey not in Employees
SELECT DISTINCT esa.EmployeeKey
FROM EmployeeStoreAssignments esa
LEFT JOIN Employees e ON e.EmployeeKey = esa.EmployeeKey
WHERE e.EmployeeKey IS NULL;
-- EXPECTED: zero rows

-- 2c. EmployeeStoreAssignments referencing a StoreKey not in Stores
SELECT DISTINCT esa.StoreKey
FROM EmployeeStoreAssignments esa
LEFT JOIN Stores s ON s.StoreKey = esa.StoreKey
WHERE s.StoreKey IS NULL;
-- EXPECTED: zero rows

-- 2d. Sales with EmployeeKey = -1 or 0 (unassigned)
IF @has_sales = 1
BEGIN
    SELECT
        COUNT(*)                                                        AS UnassignedSales,
        CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Sales) AS DECIMAL(5,2)) AS PctUnassigned
    FROM Sales
    WHERE EmployeeKey <= 0;
    -- EXPECTED: zero or near-zero
END
ELSE IF @has_header = 1
BEGIN
    SELECT
        COUNT(*)                                                        AS UnassignedSales,
        CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM SalesOrderHeader) AS DECIMAL(5,2)) AS PctUnassigned
    FROM SalesOrderHeader
    WHERE EmployeeKey <= 0;
    -- EXPECTED: zero or near-zero
END


-- ############################################################################
-- 3. ASSIGNMENT DATE COHERENCE
-- ############################################################################

-- 3a. Assignment StartDate should be >= employee HireDate
SELECT
    esa.EmployeeKey,
    esa.StoreKey,
    esa.StartDate       AS AssignmentStart,
    e.HireDate
FROM EmployeeStoreAssignments esa
JOIN Employees e ON e.EmployeeKey = esa.EmployeeKey
WHERE esa.StartDate < e.HireDate;
-- EXPECTED: zero rows

-- 3b. Assignment EndDate should be <= employee TerminationDate (if terminated)
SELECT
    esa.EmployeeKey,
    esa.StoreKey,
    esa.EndDate          AS AssignmentEnd,
    e.TerminationDate
FROM EmployeeStoreAssignments esa
JOIN Employees e ON e.EmployeeKey = esa.EmployeeKey
WHERE e.TerminationDate IS NOT NULL
  AND esa.EndDate IS NOT NULL
  AND esa.EndDate > e.TerminationDate;
-- EXPECTED: zero rows

-- 3c. Assignment StartDate < EndDate (no zero-length or inverted windows)
SELECT EmployeeKey, StoreKey, StartDate, EndDate
FROM EmployeeStoreAssignments
WHERE EndDate IS NOT NULL
  AND StartDate >= EndDate;
-- EXPECTED: zero rows

-- 3d. No overlapping assignments for the same employee at the same store
SELECT
    a.EmployeeKey,
    a.StoreKey,
    a.StartDate  AS Window1Start,
    a.EndDate    AS Window1End,
    b.StartDate  AS Window2Start,
    b.EndDate    AS Window2End
FROM EmployeeStoreAssignments a
JOIN EmployeeStoreAssignments b
  ON  b.EmployeeKey = a.EmployeeKey
 AND b.StoreKey     = a.StoreKey
 AND b.StartDate    > a.StartDate
WHERE a.EndDate IS NOT NULL
  AND b.StartDate <= a.EndDate;
-- EXPECTED: zero rows


-- ############################################################################
-- 4. ROLE & MANAGER CONSTRAINTS
-- ############################################################################

-- 4a. Store Managers (EmployeeKey 30M-40M) should NOT appear in Sales
IF @has_sales = 1
BEGIN
    SELECT DISTINCT f.EmployeeKey
    FROM Sales f
    WHERE f.EmployeeKey >= 30000000
      AND f.EmployeeKey <  40000000;
    -- EXPECTED: zero rows (managers are excluded from salesperson sampling)
END
ELSE IF @has_header = 1
BEGIN
    SELECT DISTINCT h.EmployeeKey
    FROM SalesOrderHeader h
    WHERE h.EmployeeKey >= 30000000
      AND h.EmployeeKey <  40000000;
    -- EXPECTED: zero rows (managers are excluded from salesperson sampling)
END

-- 4b. Every employee in Sales should have a sales-eligible role in the bridge
IF @has_sales = 1
BEGIN
    SELECT DISTINCT f.EmployeeKey, f.StoreKey
    FROM Sales f
    WHERE f.EmployeeKey > 0
      AND NOT EXISTS (
        SELECT 1
        FROM EmployeeStoreAssignments esa
        WHERE esa.EmployeeKey = f.EmployeeKey
          AND esa.StoreKey    = f.StoreKey
          AND esa.RoleAtStore = 'Sales Associate'
      );
    -- EXPECTED: zero rows (only Sales Associates should appear in sales)
END
ELSE IF @has_header = 1
BEGIN
    SELECT DISTINCT h.EmployeeKey, h.StoreKey
    FROM SalesOrderHeader h
    WHERE h.EmployeeKey > 0
      AND NOT EXISTS (
        SELECT 1
        FROM EmployeeStoreAssignments esa
        WHERE esa.EmployeeKey = h.EmployeeKey
          AND esa.StoreKey    = h.StoreKey
          AND esa.RoleAtStore = 'Sales Associate'
      );
    -- EXPECTED: zero rows (only Sales Associates should appear in sales)
END


-- ############################################################################
-- 5. COVERAGE — Every store should have salesperson coverage
-- ############################################################################

-- 5a. Stores with sales but no assignments at all
IF @has_sales = 1
BEGIN
    SELECT DISTINCT f.StoreKey
    FROM Sales f
    WHERE NOT EXISTS (
        SELECT 1
        FROM EmployeeStoreAssignments esa
        WHERE esa.StoreKey = f.StoreKey
    );
    -- EXPECTED: zero rows
END
ELSE IF @has_header = 1
BEGIN
    SELECT DISTINCT h.StoreKey
    FROM SalesOrderHeader h
    WHERE NOT EXISTS (
        SELECT 1
        FROM EmployeeStoreAssignments esa
        WHERE esa.StoreKey = h.StoreKey
    );
    -- EXPECTED: zero rows
END

-- 5b. Stores with sales on dates where no salesperson was assigned
IF @has_sales = 1
BEGIN
    SELECT
        f.StoreKey,
        f.OrderDate,
        COUNT(*) AS SalesOnDate
    FROM Sales f
    WHERE f.EmployeeKey > 0
      AND NOT EXISTS (
        SELECT 1
        FROM EmployeeStoreAssignments esa
        WHERE esa.StoreKey  = f.StoreKey
          AND f.OrderDate  >= esa.StartDate
          AND (esa.EndDate IS NULL OR f.OrderDate <= esa.EndDate)
          AND esa.RoleAtStore = 'Sales Associate'
      )
    GROUP BY f.StoreKey, f.OrderDate
    ORDER BY SalesOnDate DESC;
    -- EXPECTED: zero rows
END
ELSE IF @has_header = 1
BEGIN
    SELECT
        h.StoreKey,
        h.OrderDate,
        COUNT(*) AS SalesOnDate
    FROM SalesOrderHeader h
    WHERE h.EmployeeKey > 0
      AND NOT EXISTS (
        SELECT 1
        FROM EmployeeStoreAssignments esa
        WHERE esa.StoreKey  = h.StoreKey
          AND h.OrderDate  >= esa.StartDate
          AND (esa.EndDate IS NULL OR h.OrderDate <= esa.EndDate)
          AND esa.RoleAtStore = 'Sales Associate'
      )
    GROUP BY h.StoreKey, h.OrderDate
    ORDER BY SalesOnDate DESC;
    -- EXPECTED: zero rows
END

-- 5c. Assignment density — how many salesperson-assignments per store?
SELECT
    StoreKey,
    COUNT(*)                                                     AS TotalAssignments,
    SUM(CASE WHEN RoleAtStore = 'Sales Associate' THEN 1 ELSE 0 END) AS SalesAssignments,
    COUNT(DISTINCT EmployeeKey)                                  AS UniqueEmployees
FROM EmployeeStoreAssignments
GROUP BY StoreKey
ORDER BY SalesAssignments ASC;
-- INFO: review stores with very few sales assignments for coverage risk


-- ############################################################################
-- 6. EMPLOYEE -> STORE ASSIGNMENT -> SALES HEALTH SCORECARD
-- ############################################################################
IF @has_sales = 1
BEGIN
    SELECT
        'No orphaned sales (employee+store+date)' AS [Check],
        'Every sale with a salesperson has a matching effective-dated assignment' AS [Description],
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS Result
    FROM Sales f
    LEFT JOIN EmployeeStoreAssignments esa
      ON  esa.EmployeeKey = f.EmployeeKey
     AND esa.StoreKey     = f.StoreKey
     AND f.OrderDate     >= esa.StartDate
     AND (esa.EndDate IS NULL OR f.OrderDate <= esa.EndDate)
    WHERE f.EmployeeKey > 0
      AND esa.EmployeeKey IS NULL

    UNION ALL

    SELECT
        'All salesperson keys exist in Employees',
        'EmployeeKey in Sales references a valid Employees row',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM (
        SELECT DISTINCT f.EmployeeKey
        FROM Sales f
        LEFT JOIN Employees e ON e.EmployeeKey = f.EmployeeKey
        WHERE f.EmployeeKey > 0 AND e.EmployeeKey IS NULL
    ) x

    UNION ALL

    SELECT
        'All assignment keys exist in Employees',
        'EmployeeKey in bridge table references a valid Employees row',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM (
        SELECT DISTINCT esa.EmployeeKey
        FROM EmployeeStoreAssignments esa
        LEFT JOIN Employees e ON e.EmployeeKey = esa.EmployeeKey
        WHERE e.EmployeeKey IS NULL
    ) x

    UNION ALL

    SELECT
        'No managers in sales',
        'Store Manager keys (30M-40M range) should not appear as EmployeeKey',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM Sales
    WHERE EmployeeKey >= 30000000
      AND EmployeeKey <  40000000

    UNION ALL

    SELECT
        'Assignment dates within employment',
        'Assignment StartDate >= HireDate and EndDate <= TerminationDate',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM (
        SELECT esa.EmployeeKey
        FROM EmployeeStoreAssignments esa
        JOIN Employees e ON e.EmployeeKey = esa.EmployeeKey
        WHERE esa.StartDate < e.HireDate
           OR (e.TerminationDate IS NOT NULL AND esa.EndDate IS NOT NULL
               AND esa.EndDate > e.TerminationDate)
    ) x

    UNION ALL

    SELECT
        'No overlapping same-store assignments',
        'An employee should not have overlapping date windows at the same store',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM EmployeeStoreAssignments a
    JOIN EmployeeStoreAssignments b
      ON  b.EmployeeKey = a.EmployeeKey
     AND b.StoreKey     = a.StoreKey
     AND b.StartDate    > a.StartDate
    WHERE a.EndDate IS NOT NULL
      AND b.StartDate <= a.EndDate

    UNION ALL

    SELECT
        'No unassigned salesperson keys',
        'EmployeeKey should be > 0 (not -1 or 0)',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM Sales
    WHERE EmployeeKey <= 0;
END
ELSE IF @has_header = 1
BEGIN
    SELECT
        'No orphaned sales (employee+store+date)' AS [Check],
        'Every sale with a salesperson has a matching effective-dated assignment' AS [Description],
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS Result
    FROM SalesOrderHeader h
    LEFT JOIN EmployeeStoreAssignments esa
      ON  esa.EmployeeKey = h.EmployeeKey
     AND esa.StoreKey     = h.StoreKey
     AND h.OrderDate     >= esa.StartDate
     AND (esa.EndDate IS NULL OR h.OrderDate <= esa.EndDate)
    WHERE h.EmployeeKey > 0
      AND esa.EmployeeKey IS NULL

    UNION ALL

    SELECT
        'All salesperson keys exist in Employees',
        'EmployeeKey in SalesOrderHeader references a valid Employees row',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM (
        SELECT DISTINCT h.EmployeeKey
        FROM SalesOrderHeader h
        LEFT JOIN Employees e ON e.EmployeeKey = h.EmployeeKey
        WHERE h.EmployeeKey > 0 AND e.EmployeeKey IS NULL
    ) x

    UNION ALL

    SELECT
        'All assignment keys exist in Employees',
        'EmployeeKey in bridge table references a valid Employees row',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM (
        SELECT DISTINCT esa.EmployeeKey
        FROM EmployeeStoreAssignments esa
        LEFT JOIN Employees e ON e.EmployeeKey = esa.EmployeeKey
        WHERE e.EmployeeKey IS NULL
    ) x

    UNION ALL

    SELECT
        'No managers in sales',
        'Store Manager keys (30M-40M range) should not appear as EmployeeKey',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM SalesOrderHeader
    WHERE EmployeeKey >= 30000000
      AND EmployeeKey <  40000000

    UNION ALL

    SELECT
        'Assignment dates within employment',
        'Assignment StartDate >= HireDate and EndDate <= TerminationDate',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM (
        SELECT esa.EmployeeKey
        FROM EmployeeStoreAssignments esa
        JOIN Employees e ON e.EmployeeKey = esa.EmployeeKey
        WHERE esa.StartDate < e.HireDate
           OR (e.TerminationDate IS NOT NULL AND esa.EndDate IS NOT NULL
               AND esa.EndDate > e.TerminationDate)
    ) x

    UNION ALL

    SELECT
        'No overlapping same-store assignments',
        'An employee should not have overlapping date windows at the same store',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM EmployeeStoreAssignments a
    JOIN EmployeeStoreAssignments b
      ON  b.EmployeeKey = a.EmployeeKey
     AND b.StoreKey     = a.StoreKey
     AND b.StartDate    > a.StartDate
    WHERE a.EndDate IS NOT NULL
      AND b.StartDate <= a.EndDate

    UNION ALL

    SELECT
        'No unassigned salesperson keys',
        'EmployeeKey should be > 0 (not -1 or 0)',
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM SalesOrderHeader
    WHERE EmployeeKey <= 0;
END
