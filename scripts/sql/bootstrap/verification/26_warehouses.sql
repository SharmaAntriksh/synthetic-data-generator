SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[Warehouses]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.Warehouses', N'U') IS NULL
        RETURN;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- WarehouseType domain
    DECLARE @bad_type INT;
    SELECT @bad_type = COUNT(*) FROM dbo.Warehouses
    WHERE WarehouseType NOT IN ('Distribution Center', 'Regional Hub', 'Fulfillment Center');
    INSERT INTO #R VALUES ('Domain', 'Valid WarehouseType',
        'Must be Distribution Center/Regional Hub/Fulfillment Center',
        CASE WHEN @bad_type = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_type AS VARCHAR) + ' invalid');

    -- Positive Capacity
    DECLARE @bad_cap INT;
    SELECT @bad_cap = COUNT(*) FROM dbo.Warehouses WHERE Capacity <= 0;
    INSERT INTO #R VALUES ('Range', 'Positive Capacity',
        'All warehouses must have Capacity > 0',
        CASE WHEN @bad_cap = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_cap AS VARCHAR) + ' invalid');

    -- Positive SquareFootage
    DECLARE @bad_sqft INT;
    SELECT @bad_sqft = COUNT(*) FROM dbo.Warehouses WHERE SquareFootage <= 0;
    INSERT INTO #R VALUES ('Range', 'Positive SquareFootage',
        'All warehouses must have SquareFootage > 0',
        CASE WHEN @bad_sqft = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_sqft AS VARCHAR) + ' invalid');

    -- Unique WarehouseKey
    DECLARE @dup_key INT;
    SELECT @dup_key = COUNT(*) - COUNT(DISTINCT WarehouseKey) FROM dbo.Warehouses;
    INSERT INTO #R VALUES ('Uniqueness', 'Unique WarehouseKey',
        'No duplicate WarehouseKey values',
        CASE WHEN @dup_key = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@dup_key AS VARCHAR) + ' duplicates');

    -- Valid GeographyKey (orphan check)
    DECLARE @orphan_geo INT = 0;
    IF OBJECT_ID(N'dbo.Geography', N'U') IS NOT NULL
    BEGIN
        SELECT @orphan_geo = COUNT(*)
        FROM dbo.Warehouses w
        WHERE NOT EXISTS (SELECT 1 FROM dbo.Geography g WHERE g.GeographyKey = w.GeographyKey);
    END;
    INSERT INTO #R VALUES ('FK', 'Valid GeographyKey',
        'Every Warehouse GeographyKey must exist in Geography',
        CASE WHEN @orphan_geo = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@orphan_geo AS VARCHAR) + ' orphaned keys');

    -- Every open store has a warehouse (cross-dimension)
    DECLARE @stores_no_wh INT = 0;
    IF OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
    AND COL_LENGTH(N'dbo.Stores', N'WarehouseKey') IS NOT NULL
    BEGIN
        SELECT @stores_no_wh = COUNT(*)
        FROM dbo.Stores s
        WHERE s.WarehouseKey IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM dbo.Warehouses w WHERE w.WarehouseKey = s.WarehouseKey);
    END;
    INSERT INTO #R VALUES ('FK', 'Store WarehouseKey valid',
        'Every Store WarehouseKey must exist in Warehouses',
        CASE WHEN @stores_no_wh = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@stores_no_wh AS VARCHAR) + ' orphaned keys');

    SELECT * FROM #R;
    DROP TABLE #R;
END;
GO
