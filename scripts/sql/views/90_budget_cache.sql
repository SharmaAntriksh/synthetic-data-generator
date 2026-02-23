CREATE OR ALTER PROCEDURE dbo.sp_RefreshBudgetCache
    @RebuildIfSchemaChanged bit = 1,
    @Target varchar(10) = 'FX',     -- 'LOCAL' | 'FX' | 'BOTH'
    @CreateCCI bit = 1
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    IF @Target NOT IN ('LOCAL','FX','BOTH')
        THROW 50010, 'Invalid @Target. Use LOCAL, FX, or BOTH.', 1;

    IF OBJECT_ID(N'dbo.vw_Budget_ChannelMonth', N'V') IS NULL
        THROW 50001, 'Missing view dbo.vw_Budget_ChannelMonth.', 1;

    -- If FX requested, require ExchangeRates + currency mapping availability
    IF @Target IN ('FX','BOTH') AND OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NULL
        THROW 50003, 'Missing table dbo.ExchangeRates required for FX cache.', 1;

    BEGIN TRY
        BEGIN TRAN;

        ----------------------------------------------------------------------
        -- 1) LOCAL cache (heavy) - build only if requested or needed by FX
        ----------------------------------------------------------------------
        IF @Target IN ('LOCAL','BOTH','FX')
        BEGIN
            DECLARE @needLocalRebuild bit =
                CASE
                    WHEN OBJECT_ID(N'dbo.Budget_ChannelMonth', 'U') IS NULL THEN 1
                    ELSE 0
                END;

            IF @needLocalRebuild = 1
            BEGIN
                SELECT *
                INTO dbo.Budget_ChannelMonth
                FROM dbo.vw_Budget_ChannelMonth;

                IF @CreateCCI = 1
                    CREATE CLUSTERED COLUMNSTORE INDEX CCI_Budget_ChannelMonth
                    ON dbo.Budget_ChannelMonth;
            END
            ELSE
            BEGIN
                TRUNCATE TABLE dbo.Budget_ChannelMonth;

                INSERT INTO dbo.Budget_ChannelMonth
                SELECT *
                FROM dbo.vw_Budget_ChannelMonth;
            END
        END

        ----------------------------------------------------------------------
        -- 2) FX cache (cheap if built from local cache)
        ----------------------------------------------------------------------
        IF @Target IN ('FX','BOTH')
        BEGIN
            -- Ensure local exists (FX depends on it)
            IF OBJECT_ID(N'dbo.Budget_ChannelMonth', 'U') IS NULL
                THROW 50011, 'Local cache dbo.Budget_ChannelMonth not available for FX build.', 1;

            -- Recreate FX table on first run; after that do truncate+insert
            IF OBJECT_ID(N'dbo.Budget_ChannelMonth_FX', 'U') IS NULL
            BEGIN
                -- Build monthly FX rates
                ;WITH FxMonth AS (
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
                CountryCurrency AS (
                    -- TODO: adjust if you use CurrencyKey instead of CurrencyCode
                    SELECT g.Country, MAX(g.CurrencyCode) AS LocalCurrency
                    FROM dbo.Geography g
                    WHERE g.Country IS NOT NULL AND g.CurrencyCode IS NOT NULL
                    GROUP BY g.Country
                )
                SELECT
                    b.Country,
                    b.Category,
                    b.SalesChannelKey,
                    b.BudgetYear,
                    b.BudgetMonthStart,
                    b.Scenario,

                    cc.LocalCurrency,
                    CAST('USD' AS varchar(10)) AS ReportCurrency,

                    CAST(NULL AS decimal(18,6)) AS FxRate_ToReport,
                    CAST(NULL AS varchar(40))   AS Audit_FxSource,

                    b.BudgetGrossAmount AS BudgetGrossAmount_Local,
                    b.BudgetNetAmount   AS BudgetNetAmount_Local,

                    CAST(NULL AS decimal(19,2)) AS BudgetGrossAmount_Report,
                    CAST(NULL AS decimal(19,2)) AS BudgetNetAmount_Report,

                    b.BudgetGrossQuantity,
                    b.BudgetNetQuantity,

                    b.BudgetGrowthPct,
                    b.Audit_ChannelShare,
                    b.Audit_MonthShare,
                    b.Audit_ReturnRate,
                    b.BudgetMethod
                INTO dbo.Budget_ChannelMonth_FX
                FROM dbo.Budget_ChannelMonth b
                LEFT JOIN CountryCurrency cc
                    ON cc.Country = b.Country;

                IF @CreateCCI = 1
                    CREATE CLUSTERED COLUMNSTORE INDEX CCI_Budget_ChannelMonth_FX
                    ON dbo.Budget_ChannelMonth_FX;
            END
            ELSE
            BEGIN
                TRUNCATE TABLE dbo.Budget_ChannelMonth_FX;

                ;WITH FxMonth AS (
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
                CountryCurrency AS (
                    SELECT g.Country, MAX(g.CurrencyCode) AS LocalCurrency
                    FROM dbo.Geography g
                    WHERE g.Country IS NOT NULL AND g.CurrencyCode IS NOT NULL
                    GROUP BY g.Country
                )
                INSERT INTO dbo.Budget_ChannelMonth_FX (
                    Country, Category, SalesChannelKey, BudgetYear, BudgetMonthStart, Scenario,
                    LocalCurrency, ReportCurrency, FxRate_ToReport, Audit_FxSource,
                    BudgetGrossAmount_Local, BudgetNetAmount_Local,
                    BudgetGrossAmount_Report, BudgetNetAmount_Report,
                    BudgetGrossQuantity, BudgetNetQuantity,
                    BudgetGrowthPct, Audit_ChannelShare, Audit_MonthShare, Audit_ReturnRate,
                    BudgetMethod
                )
                SELECT
                    b.Country,
                    b.Category,
                    b.SalesChannelKey,
                    b.BudgetYear,
                    b.BudgetMonthStart,
                    b.Scenario,
                    cc.LocalCurrency,
                    CAST('USD' AS varchar(10)) AS ReportCurrency,

                    CASE
                        WHEN cc.LocalCurrency IS NULL THEN NULL
                        WHEN cc.LocalCurrency = 'USD' THEN 1.0
                        WHEN fxD.AvgRate IS NOT NULL THEN fxD.AvgRate
                        WHEN fxI.AvgRate IS NOT NULL AND fxI.AvgRate <> 0 THEN 1.0 / fxI.AvgRate
                        ELSE NULL
                    END AS FxRate_ToReport,

                    CASE
                        WHEN cc.LocalCurrency IS NULL THEN 'missing-currency'
                        WHEN cc.LocalCurrency = 'USD' THEN 'identity'
                        WHEN fxD.AvgRate IS NOT NULL THEN 'direct'
                        WHEN fxI.AvgRate IS NOT NULL THEN 'inverse'
                        ELSE 'missing-rate'
                    END AS Audit_FxSource,

                    b.BudgetGrossAmount,
                    b.BudgetNetAmount,

                    CAST(
                        CASE
                            WHEN cc.LocalCurrency IS NULL THEN NULL
                            WHEN cc.LocalCurrency = 'USD' THEN b.BudgetGrossAmount
                            WHEN fxD.AvgRate IS NOT NULL THEN b.BudgetGrossAmount * fxD.AvgRate
                            WHEN fxI.AvgRate IS NOT NULL AND fxI.AvgRate <> 0 THEN b.BudgetGrossAmount / fxI.AvgRate
                            ELSE NULL
                        END AS decimal(19,2)
                    ) AS BudgetGrossAmount_Report,

                    CAST(
                        CASE
                            WHEN cc.LocalCurrency IS NULL THEN NULL
                            WHEN cc.LocalCurrency = 'USD' THEN b.BudgetNetAmount
                            WHEN fxD.AvgRate IS NOT NULL THEN b.BudgetNetAmount * fxD.AvgRate
                            WHEN fxI.AvgRate IS NOT NULL AND fxI.AvgRate <> 0 THEN b.BudgetNetAmount / fxI.AvgRate
                            ELSE NULL
                        END AS decimal(19,2)
                    ) AS BudgetNetAmount_Report,

                    b.BudgetGrossQuantity,
                    b.BudgetNetQuantity,

                    b.BudgetGrowthPct,
                    b.Audit_ChannelShare,
                    b.Audit_MonthShare,
                    b.Audit_ReturnRate,
                    b.BudgetMethod
                FROM dbo.Budget_ChannelMonth b
                LEFT JOIN CountryCurrency cc
                    ON cc.Country = b.Country
                LEFT JOIN FxMonth fxD
                    ON fxD.MonthStart = b.BudgetMonthStart
                   AND fxD.FromCurrency = cc.LocalCurrency
                   AND fxD.ToCurrency = 'USD'
                LEFT JOIN FxMonth fxI
                    ON fxI.MonthStart = b.BudgetMonthStart
                   AND fxI.FromCurrency = 'USD'
                   AND fxI.ToCurrency = cc.LocalCurrency;
            END
        END

        COMMIT;
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0 ROLLBACK;
        THROW;
    END CATCH
END;
GO