-----------------------------------------------------------------------
-- sp_RefreshBudgetCache
--
-- Materializes the expensive budget views into cache tables with
-- optional clustered columnstore indexes.
-----------------------------------------------------------------------
CREATE OR ALTER PROCEDURE dbo.sp_RefreshBudgetCache
    @Target    varchar(10) = 'FX',     -- 'LOCAL' | 'FX' | 'BOTH'
    @CreateCCI bit         = 1
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    IF @Target NOT IN ('LOCAL','FX','BOTH')
        THROW 50010, 'Invalid @Target. Use LOCAL, FX, or BOTH.', 1;

    IF OBJECT_ID(N'dbo.vw_Budget_ChannelMonth', N'V') IS NULL
        THROW 50001, 'Missing view dbo.vw_Budget_ChannelMonth.', 1;

    IF @Target IN ('FX','BOTH') AND OBJECT_ID(N'dbo.vw_Budget_ChannelMonth_FX', N'V') IS NULL
        THROW 50002, 'Missing view dbo.vw_Budget_ChannelMonth_FX required for FX cache.', 1;

    IF @Target IN ('FX','BOTH') AND OBJECT_ID(N'dbo.ExchangeRates', N'U') IS NULL
        THROW 50003, 'Missing table dbo.ExchangeRates required for FX cache.', 1;

    DECLARE @ts datetime2(3) = SYSUTCDATETIME();

    BEGIN TRY
        BEGIN TRAN;

        ------------------------------------------------------------------
        -- 1) LOCAL cache
        ------------------------------------------------------------------
        IF @Target IN ('LOCAL','BOTH','FX')
        BEGIN
            IF OBJECT_ID(N'dbo.Budget_ChannelMonth', 'U') IS NULL
            BEGIN
                -- First run: create table from view
                SELECT *, @ts AS LastRefreshedUtc
                INTO dbo.Budget_ChannelMonth
                FROM dbo.vw_Budget_ChannelMonth;

                -- CCI in TRY/CATCH so Express edition does not abort
                IF @CreateCCI = 1
                BEGIN
                    BEGIN TRY
                        CREATE CLUSTERED COLUMNSTORE INDEX CCI_Budget_ChannelMonth
                        ON dbo.Budget_ChannelMonth;
                    END TRY
                    BEGIN CATCH
                        PRINT 'WARNING: CCI creation failed on Budget_ChannelMonth: '
                            + ERROR_MESSAGE();
                    END CATCH
                END
            END
            ELSE
            BEGIN
                TRUNCATE TABLE dbo.Budget_ChannelMonth;

                -- If LastRefreshedUtc column exists, include it
                IF COL_LENGTH(N'dbo.Budget_ChannelMonth', N'LastRefreshedUtc') IS NOT NULL
                BEGIN
                    -- Dynamic SQL because the column may not exist on older schemas
                    EXEC sp_executesql N'
                        INSERT INTO dbo.Budget_ChannelMonth
                        SELECT *, @ts AS LastRefreshedUtc
                        FROM dbo.vw_Budget_ChannelMonth;',
                        N'@ts datetime2(3)', @ts = @ts;
                END
                ELSE
                BEGIN
                    INSERT INTO dbo.Budget_ChannelMonth
                    SELECT *
                    FROM dbo.vw_Budget_ChannelMonth;
                END
            END
        END

        ------------------------------------------------------------------
        -- 2) FX cache - reads from the FX VIEW (single source of truth,
        --    no duplicated FX logic in this procedure)
        ------------------------------------------------------------------
        IF @Target IN ('FX','BOTH')
        BEGIN
            -- Ensure local cache exists (FX view depends on the channel-month data)
            IF OBJECT_ID(N'dbo.Budget_ChannelMonth', 'U') IS NULL
                THROW 50011, 'Local cache dbo.Budget_ChannelMonth not available for FX build.', 1;

            IF OBJECT_ID(N'dbo.Budget_ChannelMonth_FX', 'U') IS NULL
            BEGIN
                -- FIX: First run now reads from the view (populates FX columns)
                SELECT *, @ts AS LastRefreshedUtc
                INTO dbo.Budget_ChannelMonth_FX
                FROM dbo.vw_Budget_ChannelMonth_FX;

                IF @CreateCCI = 1
                BEGIN
                    BEGIN TRY
                        CREATE CLUSTERED COLUMNSTORE INDEX CCI_Budget_ChannelMonth_FX
                        ON dbo.Budget_ChannelMonth_FX;
                    END TRY
                    BEGIN CATCH
                        PRINT 'WARNING: CCI creation failed on Budget_ChannelMonth_FX: '
                            + ERROR_MESSAGE();
                    END CATCH
                END
            END
            ELSE
            BEGIN
                TRUNCATE TABLE dbo.Budget_ChannelMonth_FX;

                IF COL_LENGTH(N'dbo.Budget_ChannelMonth_FX', N'LastRefreshedUtc') IS NOT NULL
                BEGIN
                    EXEC sp_executesql N'
                        INSERT INTO dbo.Budget_ChannelMonth_FX
                        SELECT *, @ts AS LastRefreshedUtc
                        FROM dbo.vw_Budget_ChannelMonth_FX;',
                        N'@ts datetime2(3)', @ts = @ts;
                END
                ELSE
                BEGIN
                    INSERT INTO dbo.Budget_ChannelMonth_FX
                    SELECT *
                    FROM dbo.vw_Budget_ChannelMonth_FX;
                END
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
