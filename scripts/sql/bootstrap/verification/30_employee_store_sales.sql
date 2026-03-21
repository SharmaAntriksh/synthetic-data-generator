SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[EmployeeStoreSales]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.EmployeeStoreAssignments', N'U') IS NULL
       OR OBJECT_ID(N'dbo.Sales', N'U') IS NULL
       OR OBJECT_ID(N'dbo.Employees', N'U') IS NULL
        RETURN;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- Orphaned sales
    DECLARE @orphaned INT;
    SELECT @orphaned = COUNT(DISTINCT CAST(f.SalesPersonEmployeeKey AS VARCHAR) + '-' + CAST(f.StoreKey AS VARCHAR))
    FROM dbo.Sales f
    LEFT JOIN dbo.EmployeeStoreAssignments esa
      ON  esa.EmployeeKey = f.SalesPersonEmployeeKey
     AND esa.StoreKey     = f.StoreKey
     AND f.OrderDate     >= esa.StartDate
     AND (esa.EndDate IS NULL OR f.OrderDate <= esa.EndDate)
    WHERE f.SalesPersonEmployeeKey > 0
      AND esa.EmployeeKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'No orphaned sales (employee+store+date)',
        'Every sale with a salesperson has a matching effective-dated assignment',
        CASE WHEN @orphaned = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@orphaned AS VARCHAR) + ' orphaned combos');

    -- Salesperson FK
    DECLARE @bad_sp INT;
    SELECT @bad_sp = COUNT(DISTINCT f.SalesPersonEmployeeKey)
    FROM dbo.Sales f LEFT JOIN dbo.Employees e ON e.EmployeeKey = f.SalesPersonEmployeeKey
    WHERE f.SalesPersonEmployeeKey > 0 AND e.EmployeeKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'All salesperson keys exist in Employees',
        'SalesPersonEmployeeKey in Sales references a valid Employees row',
        CASE WHEN @bad_sp = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_sp AS VARCHAR) + ' invalid keys');

    -- Assignment FK
    DECLARE @bad_asgn INT;
    SELECT @bad_asgn = COUNT(DISTINCT esa.EmployeeKey)
    FROM dbo.EmployeeStoreAssignments esa LEFT JOIN dbo.Employees e ON e.EmployeeKey = esa.EmployeeKey
    WHERE e.EmployeeKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'All assignment keys exist in Employees',
        'EmployeeKey in bridge table references a valid Employees row',
        CASE WHEN @bad_asgn = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_asgn AS VARCHAR) + ' invalid keys');

    -- No managers in sales
    DECLARE @mgr_sales INT;
    SELECT @mgr_sales = COUNT(DISTINCT SalesPersonEmployeeKey) FROM dbo.Sales
    WHERE SalesPersonEmployeeKey >= 30000000 AND SalesPersonEmployeeKey < 40000000;
    INSERT INTO #R VALUES ('Domain', 'No managers in sales',
        'Store Manager keys (30M-40M range) should not appear as SalesPersonEmployeeKey',
        CASE WHEN @mgr_sales = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@mgr_sales AS VARCHAR) + ' manager keys found');

    -- Assignment dates within employment
    DECLARE @bad_dates INT;
    SELECT @bad_dates = COUNT(*) FROM dbo.EmployeeStoreAssignments esa
    JOIN dbo.Employees e ON e.EmployeeKey = esa.EmployeeKey
    WHERE esa.StartDate < e.HireDate
       OR (e.TerminationDate IS NOT NULL AND esa.EndDate IS NOT NULL
           AND esa.EndDate > e.TerminationDate);
    INSERT INTO #R VALUES ('DateSanity', 'Assignment dates within employment',
        'Assignment StartDate >= HireDate and EndDate <= TerminationDate',
        CASE WHEN @bad_dates = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_dates AS VARCHAR) + ' violations');

    -- No overlapping same-store assignments
    DECLARE @overlaps INT;
    SELECT @overlaps = COUNT(*) FROM dbo.EmployeeStoreAssignments a
    JOIN dbo.EmployeeStoreAssignments b
      ON  b.EmployeeKey = a.EmployeeKey
     AND b.StoreKey     = a.StoreKey
     AND b.StartDate    > a.StartDate
    WHERE a.EndDate IS NOT NULL AND b.StartDate <= a.EndDate;
    INSERT INTO #R VALUES ('DateSanity', 'No overlapping same-store assignments',
        'An employee should not have overlapping date windows at the same store',
        CASE WHEN @overlaps = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@overlaps AS VARCHAR) + ' overlaps');

    -- Unassigned salesperson keys
    DECLARE @unassigned INT;
    SELECT @unassigned = COUNT(*) FROM dbo.Sales WHERE SalesPersonEmployeeKey <= 0;
    DECLARE @total_sales INT;
    SELECT @total_sales = COUNT(*) FROM dbo.Sales;
    INSERT INTO #R VALUES ('Domain', 'No unassigned salesperson keys',
        'SalesPersonEmployeeKey should be > 0 (not -1 or 0)',
        CASE WHEN @unassigned = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@unassigned AS VARCHAR) + ' of ' + CAST(@total_sales AS VARCHAR) + ' unassigned');

    -- Employee hierarchy
    DECLARE @bad_parent INT;
    SELECT @bad_parent = COUNT(*) FROM dbo.Employees e
    LEFT JOIN dbo.Employees p ON p.EmployeeKey = e.ParentEmployeeKey
    WHERE e.OrgLevel > 1 AND p.EmployeeKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'Employee hierarchy valid',
        'Every non-CEO employee must reference a valid ParentEmployeeKey',
        CASE WHEN @bad_parent = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_parent AS VARCHAR) + ' broken links');

    -- Employee date sanity
    DECLARE @emp_date_inv INT;
    SELECT @emp_date_inv = COUNT(*) FROM dbo.Employees
    WHERE TerminationDate IS NOT NULL AND HireDate > TerminationDate;
    INSERT INTO #R VALUES ('DateSanity', 'Employee HireDate <= TerminationDate',
        'Employee cannot be terminated before being hired',
        CASE WHEN @emp_date_inv = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@emp_date_inv AS VARCHAR) + ' inversions');

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
