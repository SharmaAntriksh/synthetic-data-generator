/*
  40_inventory.sql – Fact table constraints: InventorySnapshot
  Aligned to src/utils/static_schemas.py (FACT_SCHEMAS["InventorySnapshot"])

  Design notes
  ────────────
  • Composite PK on (ProductKey, WarehouseKey, SnapshotDate) — one row per
    product per warehouse per snapshot period.

  • NONCLUSTERED PK to coexist with clustered columnstore index.

  • FK to Products and Warehouses dimension tables.
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
        PRIMARY KEY NONCLUSTERED ([ProductKey], [WarehouseKey], [SnapshotDate]);
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

-- InventorySnapshot -> Warehouses
IF OBJECT_ID(N'dbo.InventorySnapshot', N'U') IS NOT NULL
AND OBJECT_ID(N'dbo.Warehouses', N'U') IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys
    WHERE name = N'FK_InventorySnapshot_Warehouses'
      AND parent_object_id = OBJECT_ID(N'dbo.InventorySnapshot')
)
BEGIN
    ALTER TABLE dbo.InventorySnapshot WITH CHECK
    ADD CONSTRAINT FK_InventorySnapshot_Warehouses
        FOREIGN KEY ([WarehouseKey])
        REFERENCES dbo.Warehouses ([WarehouseKey]);

    ALTER TABLE dbo.InventorySnapshot CHECK CONSTRAINT FK_InventorySnapshot_Warehouses;
END;
GO
