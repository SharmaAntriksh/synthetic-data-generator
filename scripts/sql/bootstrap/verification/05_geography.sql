SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[Geography]
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID(N'dbo.Geography', N'U') IS NULL
        RETURN;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- Row count
    DECLARE @total INT;
    SELECT @total = COUNT(*) FROM dbo.Geography;
    INSERT INTO #R VALUES ('Info', 'Row count',
        'Total geography rows',
        'INFO', CAST(@total AS VARCHAR) + ' cities');

    -- Unique City+State+Country
    DECLARE @dupes INT;
    SELECT @dupes = COUNT(*) FROM (
        SELECT City, State, Country FROM dbo.Geography
        GROUP BY City, State, Country HAVING COUNT(*) > 1
    ) x;
    INSERT INTO #R VALUES ('Uniqueness', 'Unique City+State+Country',
        'No duplicate city/state/country combinations',
        CASE WHEN @dupes = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@dupes AS VARCHAR) + ' duplicates');

    -- No NULL ISOCode
    DECLARE @null_iso INT;
    SELECT @null_iso = COUNT(*) FROM dbo.Geography WHERE ISOCode IS NULL OR LEN(RTRIM(ISOCode)) = 0;
    INSERT INTO #R VALUES ('Domain', 'ISOCode populated',
        'Every row must have a non-empty ISOCode (currency code)',
        CASE WHEN @null_iso = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@null_iso AS VARCHAR) + ' missing');

    -- Latitude in valid range [-90, 90]
    DECLARE @bad_lat INT;
    SELECT @bad_lat = COUNT(*) FROM dbo.Geography
    WHERE Latitude IS NULL OR Latitude < -90 OR Latitude > 90;
    INSERT INTO #R VALUES ('Domain', 'Valid Latitude',
        'Latitude must be between -90 and 90',
        CASE WHEN @bad_lat = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_lat AS VARCHAR) + ' invalid');

    -- Longitude in valid range [-180, 180]
    DECLARE @bad_lon INT;
    SELECT @bad_lon = COUNT(*) FROM dbo.Geography
    WHERE Longitude IS NULL OR Longitude < -180 OR Longitude > 180;
    INSERT INTO #R VALUES ('Domain', 'Valid Longitude',
        'Longitude must be between -180 and 180',
        CASE WHEN @bad_lon = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@bad_lon AS VARCHAR) + ' invalid');

    -- Population > 0
    DECLARE @zero_pop INT;
    SELECT @zero_pop = COUNT(*) FROM dbo.Geography
    WHERE Population IS NULL OR Population <= 0;
    INSERT INTO #R VALUES ('Domain', 'Positive Population',
        'Every city must have Population > 0',
        CASE WHEN @zero_pop = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@zero_pop AS VARCHAR) + ' invalid');

    -- Timezone non-empty
    DECLARE @empty_tz INT;
    SELECT @empty_tz = COUNT(*) FROM dbo.Geography
    WHERE Timezone IS NULL OR LEN(RTRIM(Timezone)) = 0;
    INSERT INTO #R VALUES ('Domain', 'Timezone populated',
        'Every city must have a non-empty timezone',
        CASE WHEN @empty_tz = 0 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@empty_tz AS VARCHAR) + ' missing');

    -- Multiple continents represented
    DECLARE @n_continents INT;
    SELECT @n_continents = COUNT(DISTINCT Continent) FROM dbo.Geography;
    INSERT INTO #R VALUES ('Coverage', 'Multiple continents',
        'Geography should span at least 2 continents',
        CASE WHEN @n_continents >= 2 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@n_continents AS VARCHAR) + ' continents');

    -- Multiple countries represented
    DECLARE @n_countries INT;
    SELECT @n_countries = COUNT(DISTINCT Country) FROM dbo.Geography;
    INSERT INTO #R VALUES ('Coverage', 'Multiple countries',
        'Geography should span at least 3 countries',
        CASE WHEN @n_countries >= 3 THEN 'PASS' ELSE 'FAIL' END,
        CAST(@n_countries AS VARCHAR) + ' countries');

    -- INFO: continent distribution
    DECLARE @cont_dist VARCHAR(500);
    SELECT @cont_dist = STRING_AGG(Continent + ': ' + CAST(Cnt AS VARCHAR), ', ')
    FROM (
        SELECT Continent, COUNT(*) AS Cnt FROM dbo.Geography GROUP BY Continent
    ) x;
    INSERT INTO #R VALUES ('Info', 'Continent distribution',
        'Cities per continent',
        'INFO', ISNULL(@cont_dist, 'N/A'));

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
