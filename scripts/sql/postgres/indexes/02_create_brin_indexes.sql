/*
  02_create_brin_indexes.sql – BRIN indexes on date columns (Postgres).

  BRIN (Block Range INdex) stores per-page-range min/max metadata. For
  columns whose physical heap order matches the logical column order,
  BRIN gives near-btree selectivity at a tiny fraction of the storage
  cost — typically <100 KB per index regardless of table size.

  When BRIN helps here
  ────────────────────
  • Sales / SalesReturn / SalesOrderHeader / SalesOrderDetail are loaded
    via COPY in chronological order, so OrderDate / DueDate / DeliveryDate
    / ReturnDate columns are naturally clustered in the heap.
  • InventorySnapshot is generated month-by-month, so SnapshotDate is
    monotonically increasing in heap order.

  Postgres-specific: there is no SQL Server equivalent (CCI provides a
  similar but more complete form of segment elimination). These add no
  measurable maintenance cost and cost only a few KB per index, so we
  ship them alongside the btree indexes rather than as an opt-in.
*/

DO $$
BEGIN
    IF to_regclass('"public"."Sales"') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS "BRIN_Sales_OrderDate"
            ON "public"."Sales" USING BRIN ("OrderDate");
        CREATE INDEX IF NOT EXISTS "BRIN_Sales_DueDate"
            ON "public"."Sales" USING BRIN ("DueDate");
        CREATE INDEX IF NOT EXISTS "BRIN_Sales_DeliveryDate"
            ON "public"."Sales" USING BRIN ("DeliveryDate");
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('"public"."SalesReturn"') IS NOT NULL
       AND EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='SalesReturn' AND column_name='ReturnDate')
    THEN
        CREATE INDEX IF NOT EXISTS "BRIN_SalesReturn_ReturnDate"
            ON "public"."SalesReturn" USING BRIN ("ReturnDate");
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('"public"."SalesOrderHeader"') IS NOT NULL
       AND EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='SalesOrderHeader' AND column_name='OrderDate')
    THEN
        CREATE INDEX IF NOT EXISTS "BRIN_SalesOrderHeader_OrderDate"
            ON "public"."SalesOrderHeader" USING BRIN ("OrderDate");
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('"public"."SalesOrderDetail"') IS NOT NULL THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='SalesOrderDetail' AND column_name='DueDate') THEN
            CREATE INDEX IF NOT EXISTS "BRIN_SalesOrderDetail_DueDate"
                ON "public"."SalesOrderDetail" USING BRIN ("DueDate");
        END IF;
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='SalesOrderDetail' AND column_name='DeliveryDate') THEN
            CREATE INDEX IF NOT EXISTS "BRIN_SalesOrderDetail_DeliveryDate"
                ON "public"."SalesOrderDetail" USING BRIN ("DeliveryDate");
        END IF;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('"public"."InventorySnapshot"') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS "BRIN_InventorySnapshot_SnapshotDate"
            ON "public"."InventorySnapshot" USING BRIN ("SnapshotDate");
    END IF;
END $$;
