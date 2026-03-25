SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[Products]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.Products', N'U') IS NULL
        RETURN;

    DECLARE @has_scd2 BIT = 0;
    IF COL_LENGTH('dbo.Products', 'VersionNumber') IS NOT NULL
    BEGIN
        IF EXISTS (SELECT 1 FROM dbo.Products WHERE VersionNumber > 1)
            SET @has_scd2 = 1;
    END

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- ================================================================
    -- GENERAL CHECKS
    -- ================================================================

    -- IsCurrent uniqueness
    DECLARE @dup_current INT;
    SELECT @dup_current = COUNT(*) FROM (
        SELECT ProductID FROM dbo.Products WHERE IsCurrent = 1
        GROUP BY ProductID HAVING COUNT(*) > 1
    ) x;
    INSERT INTO #R VALUES ('Uniqueness', 'IsCurrent unique per ProductID',
        'No duplicate IsCurrent=1 rows for same ProductID',
        CASE WHEN @dup_current = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@dup_current AS VARCHAR) + ' duplicates');

    -- UnitCost <= ListPrice
    DECLARE @neg_margin INT;
    SELECT @neg_margin = COUNT(*) FROM dbo.Products WHERE UnitCost > ListPrice;
    INSERT INTO #R VALUES ('Domain', 'UnitCost <= ListPrice',
        'Cost must not exceed selling price in any version',
        CASE WHEN @neg_margin = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@neg_margin AS VARCHAR) + ' negative margin');

    -- Positive prices
    DECLARE @bad_price INT;
    SELECT @bad_price = COUNT(*) FROM dbo.Products WHERE ListPrice <= 0 OR UnitCost < 0;
    INSERT INTO #R VALUES ('Domain', 'Positive prices',
        'ListPrice must be > 0 and UnitCost must be >= 0',
        CASE WHEN @bad_price = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_price AS VARCHAR) + ' invalid');

    -- Subcategory FK
    DECLARE @bad_subcat INT;
    SELECT @bad_subcat = COUNT(DISTINCT p.SubcategoryKey)
    FROM dbo.Products p LEFT JOIN dbo.ProductSubcategory ps ON ps.SubcategoryKey = p.SubcategoryKey
    WHERE ps.SubcategoryKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'Valid SubcategoryKey',
        'Every product SubcategoryKey must exist in ProductSubcategory',
        CASE WHEN @bad_subcat = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_subcat AS VARCHAR) + ' orphaned keys');

    -- Subcategory -> Category chain
    DECLARE @bad_cat INT;
    SELECT @bad_cat = COUNT(DISTINCT ps.CategoryKey)
    FROM dbo.ProductSubcategory ps LEFT JOIN dbo.ProductCategory pc ON pc.CategoryKey = ps.CategoryKey
    WHERE pc.CategoryKey IS NULL;
    INSERT INTO #R VALUES ('Referential', 'Subcategory -> Category valid',
        'Every subcategory CategoryKey must exist in ProductCategory',
        CASE WHEN @bad_cat = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_cat AS VARCHAR) + ' broken links');

    -- ProductProfile coverage
    IF OBJECT_ID(N'dbo.ProductProfile', N'U') IS NOT NULL
    BEGIN
        DECLARE @missing_pp INT;
        SELECT @missing_pp = COUNT(*) FROM dbo.Products p
        LEFT JOIN dbo.ProductProfile pp ON pp.ProductKey = p.ProductKey
        WHERE pp.ProductKey IS NULL;
        INSERT INTO #R VALUES ('Referential', 'ProductProfile covers all products',
            'Every product (all SCD2 versions) must have a ProductProfile row',
            CASE WHEN @missing_pp = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@missing_pp AS VARCHAR) + ' missing');

        -- ABC classification domain
        DECLARE @bad_abc INT;
        SELECT @bad_abc = COUNT(*) FROM dbo.ProductProfile WHERE ABCClassification NOT IN ('A', 'B', 'C');
        INSERT INTO #R VALUES ('Domain', 'Valid ABC classification',
            'Must be A, B, or C',
            CASE WHEN @bad_abc = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@bad_abc AS VARCHAR) + ' invalid');
    END

    -- INFO: counts
    DECLARE @total INT;
    SELECT @total = COUNT(*) FROM dbo.Products WHERE IsCurrent = 1;
    INSERT INTO #R VALUES ('Info', 'Product count',
        'Total IsCurrent=1 products',
        'INFO', CAST(@total AS VARCHAR) + ' products');

    -- ================================================================
    -- SCD2 CHECKS
    -- ================================================================
    IF @has_scd2 = 1
    BEGIN
        -- Date chain gaps
        DECLARE @chain_gaps INT;
        SELECT @chain_gaps = COUNT(*) FROM dbo.Products p1
        JOIN dbo.Products p2
          ON p2.ProductID = p1.ProductID AND p2.VersionNumber = p1.VersionNumber + 1
        WHERE DATEDIFF(DAY, p1.EffectiveEndDate, p2.EffectiveStartDate) <> 1;
        INSERT INTO #R VALUES ('SCD2', 'No date chain gaps',
            'Each version EndDate+1 day = next version StartDate',
            CASE WHEN @chain_gaps = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@chain_gaps AS VARCHAR) + ' gaps');

        -- Non-price attrs stable
        DECLARE @attr_changes INT;
        SELECT @attr_changes = COUNT(*) FROM dbo.Products p1
        JOIN dbo.Products p2
          ON p2.ProductID = p1.ProductID AND p2.VersionNumber = p1.VersionNumber + 1
        WHERE p1.ProductName <> p2.ProductName
           OR p1.SubcategoryKey <> p2.SubcategoryKey;
        INSERT INTO #R VALUES ('SCD2', 'Non-price attrs stable',
            'Only ListPrice/UnitCost should change between versions',
            CASE WHEN @attr_changes = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@attr_changes AS VARCHAR) + ' attribute changes');

        -- Version distribution (INFO)
        DECLARE @v1 INT, @v2plus INT;
        SELECT @v1 = SUM(CASE WHEN vc = 1 THEN 1 ELSE 0 END),
               @v2plus = SUM(CASE WHEN vc > 1 THEN 1 ELSE 0 END)
        FROM (SELECT ProductID, COUNT(*) AS vc FROM dbo.Products GROUP BY ProductID) x;
        INSERT INTO #R VALUES ('SCD2', 'Version distribution',
            'Products with 1 version vs 2+ versions',
            'INFO', CAST(@v1 AS VARCHAR) + ' unchanged / ' + CAST(@v2plus AS VARCHAR) + ' with revisions');
    END

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
