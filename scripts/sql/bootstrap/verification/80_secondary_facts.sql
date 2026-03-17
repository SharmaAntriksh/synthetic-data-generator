SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

CREATE OR ALTER PROCEDURE [verify].[SecondaryFacts]
AS
BEGIN
    SET NOCOUNT ON;

    CREATE TABLE #R (
        Category    VARCHAR(50)  NOT NULL,
        [Check]     VARCHAR(200) NOT NULL,
        Description VARCHAR(500) NOT NULL,
        Result      VARCHAR(10)  NOT NULL,
        ActualValue VARCHAR(100) NOT NULL
    );

    -- ================================================================
    -- BUDGET
    -- ================================================================
    IF OBJECT_ID(N'dbo.BudgetYearly', N'U') IS NOT NULL
    BEGIN
        DECLARE @scenarios INT;
        SELECT @scenarios = COUNT(DISTINCT Scenario) FROM dbo.BudgetYearly;
        INSERT INTO #R VALUES ('Budget', '3 scenarios exist',
            'Must have exactly Low/Medium/High scenarios',
            CASE WHEN @scenarios = 3 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@scenarios AS VARCHAR) + ' scenarios');

        DECLARE @neg_budget INT;
        SELECT @neg_budget = COUNT(*) FROM dbo.BudgetYearly WHERE BudgetSalesAmount <= 0;
        INSERT INTO #R VALUES ('Budget', 'No negative amounts',
            'BudgetSalesAmount must be positive',
            CASE WHEN @neg_budget = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@neg_budget AS VARCHAR) + ' negative');

        DECLARE @budget_rows INT;
        SELECT @budget_rows = COUNT(*) FROM dbo.BudgetYearly;
        INSERT INTO #R VALUES ('Budget', 'Budget row count',
            'Total BudgetYearly rows', 'INFO', FORMAT(@budget_rows, 'N0'));
    END

    -- ================================================================
    -- INVENTORY
    -- ================================================================
    IF OBJECT_ID(N'dbo.InventorySnapshot', N'U') IS NOT NULL
    BEGIN
        DECLARE @neg_qoh INT;
        SELECT @neg_qoh = COUNT(*) FROM dbo.InventorySnapshot WHERE QuantityOnHand < 0;
        INSERT INTO #R VALUES ('Inventory', 'No negative QOH',
            'QuantityOnHand must be >= 0',
            CASE WHEN @neg_qoh = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@neg_qoh AS VARCHAR) + ' negative');

        DECLARE @days_no_flag INT;
        SELECT @days_no_flag = COUNT(*) FROM dbo.InventorySnapshot
        WHERE DaysOutOfStock > 0 AND StockoutFlag = 0;
        INSERT INTO #R VALUES ('Inventory', 'DaysOutOfStock implies StockoutFlag',
            'If DaysOutOfStock > 0 then StockoutFlag must be 1',
            CASE WHEN @days_no_flag = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@days_no_flag AS VARCHAR) + ' inconsistent');

        DECLARE @inv_rows INT;
        SELECT @inv_rows = COUNT(*) FROM dbo.InventorySnapshot;
        INSERT INTO #R VALUES ('Inventory', 'Inventory row count',
            'Total InventorySnapshot rows', 'INFO', FORMAT(@inv_rows, 'N0'));
    END

    -- ================================================================
    -- RETURNS
    -- ================================================================
    IF OBJECT_ID(N'dbo.SalesReturn', N'U') IS NOT NULL
       AND OBJECT_ID(N'dbo.Sales', N'U') IS NOT NULL
    BEGIN
        DECLARE @ret_rate DECIMAL(5,2);
        SELECT @ret_rate = ISNULL(
            COUNT(DISTINCT r.SalesOrderNumber) * 100.0
            / NULLIF((SELECT COUNT(DISTINCT SalesOrderNumber) FROM dbo.Sales), 0), 0)
        FROM dbo.SalesReturn r;
        INSERT INTO #R VALUES ('Returns', 'Return rate within 1-10%',
            'Return rate should match config (~3%)',
            CASE WHEN @ret_rate BETWEEN 1.0 AND 10.0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@ret_rate AS VARCHAR) + '%');

        DECLARE @ret_rows INT;
        SELECT @ret_rows = COUNT(*) FROM dbo.SalesReturn;
        INSERT INTO #R VALUES ('Returns', 'Return row count',
            'Total SalesReturn rows', 'INFO', FORMAT(@ret_rows, 'N0'));
    END

    -- ================================================================
    -- WISHLISTS
    -- ================================================================
    IF OBJECT_ID(N'dbo.CustomerWishlists', N'U') IS NOT NULL
    BEGIN
        DECLARE @bad_wish INT;
        SELECT @bad_wish = COUNT(*) FROM dbo.CustomerWishlists w
        LEFT JOIN dbo.Products p ON p.ProductKey = w.ProductKey
        WHERE p.ProductKey IS NULL;
        INSERT INTO #R VALUES ('Wishlists', 'Valid product refs',
            'Every wishlist ProductKey must exist in Products',
            CASE WHEN @bad_wish = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@bad_wish AS VARCHAR) + ' orphaned');

        DECLARE @wish_cust INT, @wish_total INT;
        SELECT @wish_cust = COUNT(DISTINCT CustomerKey) FROM dbo.CustomerWishlists;
        SELECT @wish_total = COUNT(DISTINCT CustomerKey) FROM dbo.Customers WHERE IsCurrent = 1;
        INSERT INTO #R VALUES ('Wishlists', 'Participation rate',
            'Percentage of customers with wishlists', 'INFO',
            CAST(@wish_cust AS VARCHAR) + ' of ' + CAST(@wish_total AS VARCHAR)
            + ' (' + CAST(CAST(@wish_cust * 100.0 / NULLIF(@wish_total, 0) AS DECIMAL(5,1)) AS VARCHAR) + '%)');
    END

    -- ================================================================
    -- COMPLAINTS
    -- ================================================================
    IF OBJECT_ID(N'dbo.Complaints', N'U') IS NOT NULL
    BEGIN
        DECLARE @comp_date_inv INT;
        SELECT @comp_date_inv = COUNT(*) FROM dbo.Complaints
        WHERE ResolutionDate IS NOT NULL AND ResolutionDate < ComplaintDate;
        INSERT INTO #R VALUES ('Complaints', 'ResolutionDate >= ComplaintDate',
            'Complaint cannot be resolved before it was filed',
            CASE WHEN @comp_date_inv = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@comp_date_inv AS VARCHAR) + ' inversions');

        DECLARE @resolution_pct DECIMAL(5,1);
        SELECT @resolution_pct = ISNULL(
            SUM(CASE WHEN Status = 'Resolved' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
        FROM dbo.Complaints;
        INSERT INTO #R VALUES ('Complaints', 'Resolution rate',
            'Percentage of complaints resolved', 'INFO',
            CAST(@resolution_pct AS VARCHAR) + '% resolved');
    END

    -- ================================================================
    -- SUBSCRIPTIONS
    -- ================================================================
    IF OBJECT_ID(N'dbo.CustomerSubscriptions', N'U') IS NOT NULL
    BEGIN
        DECLARE @bad_plan INT;
        SELECT @bad_plan = COUNT(*) FROM dbo.CustomerSubscriptions s
        LEFT JOIN dbo.Plans p ON p.PlanKey = s.PlanKey
        WHERE p.PlanKey IS NULL;
        INSERT INTO #R VALUES ('Subscriptions', 'Valid PlanKey refs',
            'Every subscription PlanKey must exist in Plans',
            CASE WHEN @bad_plan = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@bad_plan AS VARCHAR) + ' orphaned');

        DECLARE @sub_date_inv INT;
        SELECT @sub_date_inv = COUNT(*) FROM dbo.CustomerSubscriptions
        WHERE SubscribedDate IS NOT NULL AND CancelledDate < SubscribedDate;
        INSERT INTO #R VALUES ('Subscriptions', 'CancelledDate >= SubscribedDate',
            'Subscription cannot be cancelled before it started',
            CASE WHEN @sub_date_inv = 0 THEN 'PASS' ELSE 'FAIL' END,
            CAST(@sub_date_inv AS VARCHAR) + ' inversions');

        DECLARE @churn_pct DECIMAL(5,1);
        SELECT @churn_pct = ISNULL(
            SUM(CASE WHEN Status = 'Cancelled' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0)
        FROM dbo.CustomerSubscriptions;
        INSERT INTO #R VALUES ('Subscriptions', 'Churn rate',
            'Percentage of subscriptions cancelled', 'INFO',
            CAST(@churn_pct AS VARCHAR) + '% churned');
    END

    SELECT Category, [Check], Description, Result, ActualValue FROM #R;
    DROP TABLE #R;
END;
GO
