SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[Stores]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.Stores', N'U') IS NULL
        RETURN;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- StoreType domain
    DECLARE @bad_type INT;
    SELECT @bad_type = COUNT(*) FROM dbo.Stores
    WHERE StoreType NOT IN ('Supermarket', 'Convenience', 'Online', 'Hypermarket', 'Fulfillment');
    INSERT INTO #R VALUES ('Domain', 'Valid StoreType',
        'Must be Supermarket/Convenience/Online/Hypermarket/Fulfillment',
        CASE WHEN @bad_type = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_type AS VARCHAR) + ' invalid');

    -- RevenueClass domain
    DECLARE @bad_rc INT;
    SELECT @bad_rc = COUNT(*) FROM dbo.Stores WHERE RevenueClass NOT IN ('A', 'B', 'C');
    INSERT INTO #R VALUES ('Domain', 'Valid RevenueClass',
        'Must be A, B, or C',
        CASE WHEN @bad_rc = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_rc AS VARCHAR) + ' invalid');

    -- Open stores have employees
    DECLARE @no_staff INT;
    SELECT @no_staff = COUNT(*) FROM dbo.Stores WHERE Status = 'Open' AND EmployeeCount <= 0;
    INSERT INTO #R VALUES ('Domain', 'Open stores have employees',
        'Open stores must have EmployeeCount > 0',
        CASE WHEN @no_staff = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@no_staff AS VARCHAR) + ' open stores with 0 staff');

    -- OpeningDate <= ClosingDate
    DECLARE @date_inv INT;
    SELECT @date_inv = COUNT(*) FROM dbo.Stores
    WHERE ClosingDate IS NOT NULL AND OpeningDate > ClosingDate;
    INSERT INTO #R VALUES ('DateSanity', 'OpeningDate <= ClosingDate',
        'Store cannot close before it opens',
        CASE WHEN @date_inv = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@date_inv AS VARCHAR) + ' inversions');

    -- EmployeeCount matches actual
    IF OBJECT_ID(N'dbo.Employees', N'U') IS NOT NULL
    BEGIN
        DECLARE @count_mismatch INT;
        SELECT @count_mismatch = COUNT(*) FROM dbo.Stores s
        LEFT JOIN (
            SELECT StoreKey, COUNT(*) AS Cnt FROM dbo.Employees
            WHERE StoreKey IS NOT NULL AND StoreKey > 0 GROUP BY StoreKey
        ) e ON e.StoreKey = s.StoreKey
        WHERE s.Status = 'Open'
          AND ABS(s.EmployeeCount - ISNULL(e.Cnt, 0)) > 0;
        INSERT INTO #R VALUES ('Referential', 'EmployeeCount matches actual',
            'Stores.EmployeeCount must match count of employee rows',
            CASE WHEN @count_mismatch = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@count_mismatch AS VARCHAR) + ' mismatched');

        -- Closed stores have no active employees
        DECLARE @closed_active INT;
        SELECT @closed_active = COUNT(DISTINCT s.StoreKey) FROM dbo.Stores s
        JOIN dbo.Employees e ON e.StoreKey = s.StoreKey AND e.IsActive = 1
        WHERE s.Status = 'Closed';
        INSERT INTO #R VALUES ('Referential', 'Closed stores have no active staff',
            'No active employees should be assigned to closed stores',
            CASE WHEN @closed_active = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@closed_active AS VARCHAR) + ' closed stores with active staff');
    END

    -- Geography FK
    DECLARE @bad_geo INT;
    SELECT @bad_geo = COUNT(DISTINCT s.GeographyKey)
    FROM dbo.Stores s LEFT JOIN dbo.Geography g ON g.GeographyKey = s.GeographyKey
    WHERE g.GeographyKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'Valid GeographyKey',
        'Every store GeographyKey must exist in Geography',
        CASE WHEN @bad_geo = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_geo AS VARCHAR) + ' orphaned keys');

    -- StoreNumber uniqueness
    DECLARE @dup_sn INT;
    SELECT @dup_sn = COUNT(*) FROM (
        SELECT StoreNumber FROM dbo.Stores GROUP BY StoreNumber HAVING COUNT(*) > 1
    ) x;
    INSERT INTO #R VALUES ('Uniqueness', 'Unique StoreNumber',
        'StoreNumber should be unique across all stores',
        CASE WHEN @dup_sn = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@dup_sn AS VARCHAR) + ' duplicates');

    -- INFO: status distribution
    DECLARE @open INT, @closed INT, @other INT;
    SELECT @open = SUM(CASE WHEN Status = 'Open' THEN 1 ELSE 0 END),
           @closed = SUM(CASE WHEN Status = 'Closed' THEN 1 ELSE 0 END),
           @other = SUM(CASE WHEN Status NOT IN ('Open', 'Closed') THEN 1 ELSE 0 END)
    FROM dbo.Stores;
    INSERT INTO #R VALUES ('Info', 'Store status distribution',
        'Open / Closed / Other',
        'INFO', CAST(@open AS VARCHAR) + ' open / ' + CAST(@closed AS VARCHAR) + ' closed / ' + CAST(@other AS VARCHAR) + ' other');

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
