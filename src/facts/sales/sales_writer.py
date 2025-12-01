import os
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.logging_utils import info, skip, done


def merge_parquet_files(parquet_files, merged_file, delete_after=False):
    """
    Fully optimized parquet merger:
    - STREAMS row-groups instead of loading entire tables
    - Handles schema mismatches safely
    - Does not blow memory for large datasets
    - Logs progress using info/skip/done
    """

    # Keep only files that exist
    parquet_files = [p for p in parquet_files if os.path.exists(p)]

    if not parquet_files:
        skip("No parquet chunk files to merge")
        return None

    parquet_files = sorted(parquet_files)
    info(f"Merging {len(parquet_files)} chunks → {os.path.basename(merged_file)}...")

    # ------------------------------------------------------------------
    # STEP 1: Read schema from the FIRST file only (safe & fast)
    # ------------------------------------------------------------------
    first = parquet_files[0]
    first_reader = pq.ParquetFile(first)
    schema = first_reader.schema_arrow

    # Create writer for the merged file
    # Enable dictionary encoding for low-cardinality columns only
    # Exclude high-cardinality ones to avoid memory blow-up
    dict_cols = [
        c for c in schema.names
        if c not in ["SalesOrderNumber", "CustomerKey"]
    ]

    writer = pq.ParquetWriter(
        merged_file,
        schema,
        compression="snappy",           # or "lz4" to match chunks
        use_dictionary=dict_cols,       # selective dictionary encoding
        write_statistics=True,          # reduces final file size
    )


    # ------------------------------------------------------------------
    # STEP 2: STREAM each chunk row-group-by-row-group
    # ------------------------------------------------------------------
    for path in parquet_files:
        reader = pq.ParquetFile(path)

        # Validate / reconcile schema
        if reader.schema_arrow != schema:
            # Align schema (by selecting columns in consistent order)
            table = reader.read()
            table = table.select(schema.names)
            writer.write_table(table)
            continue

        # Fast path: stream each row group from this file
        for i in range(reader.num_row_groups):
            rg = reader.read_row_group(i)
            writer.write_table(rg)

    writer.close()

    # ------------------------------------------------------------------
    # STEP 3: Delete chunks if requested
    # ------------------------------------------------------------------
    if delete_after:
        for path in parquet_files:
            try:
                os.remove(path)
            except Exception:
                pass

    done(f"Merged chunks → {os.path.basename(merged_file)}")
    return merged_file
