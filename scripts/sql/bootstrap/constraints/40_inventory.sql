/*
  40_inventory.sql – Fact table constraints: InventorySnapshot
  Aligned to src/utils/static_schemas.py (FACT_SCHEMAS["InventorySnapshot"])

  Design notes
  ────────────
  • Composite PK on (ProductKey, StoreKey, SnapshotDate) — one row per
    product per store per snapshot period.

  • NONCLUSTERED PK to coexist with clustered columnstore index.

  • FK to Products and Stores dimension tables.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

-----------------------------------------------------------------------
-- 1. INVENTORYSNAPSHOT: PRIMARY KEY
-----------------------------------------------------------------------

IF OBJECT_ID(N'dbo.InventorySnapshot', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.key_constraints
    WHERE name = N'PK_InventorySnapshot'
      AND parent_object_id = OBJECT_ID(N'dbo.InventorySnapshot')
)
BEGIN
    ALTER TABLE dbo.InventorySnapshot
    ADD CONSTRAINT PK_InventorySnapshot
        PRIMARY KEY NONCLUSTERED ([ProductKey], [StoreKey], [SnapshotDate]);
END;
GO

-----------------------------------------------------------------------
-- 2. INVENTORYSNAPSHOT: FOREIGN KEYS
-----------------------------------------------------------------------
SET NOCOUNT ON;
SET XACT_ABORT ON;

-- InventorySnapshot -> Products
IF OBJECT_ID(N'dbo.InventorySnapshot', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Products', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_InventorySnapshot_Products'
      AND parent_object_id = OBJECT_ID(N'dbo.InventorySnapshot')
)
BEGIN
    ALTER TABLE dbo.InventorySnapshot WITH CHECK
    ADD CONSTRAINT FK_InventorySnapshot_Products
        FOREIGN KEY ([ProductKey])
        REFERENCES dbo.Products ([ProductKey]);

    ALTER TABLE dbo.InventorySnapshot CHECK CONSTRAINT FK_InventorySnapshot_Products;
END;
GO

-- InventorySnapshot -> Stores
IF OBJECT_ID(N'dbo.InventorySnapshot', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Stores', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_InventorySnapshot_Stores'
      AND parent_object_id = OBJECT_ID(N'dbo.InventorySnapshot')
)
BEGIN
    ALTER TABLE dbo.InventorySnapshot WITH CHECK
    ADD CONSTRAINT FK_InventorySnapshot_Stores
        FOREIGN KEY ([StoreKey])
        REFERENCES dbo.Stores ([StoreKey]);

    ALTER TABLE dbo.InventorySnapshot CHECK CONSTRAINT FK_InventorySnapshot_Stores;
END;
GO
