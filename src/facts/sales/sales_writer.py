# Sales fact writer (Parquet / Delta)
# Pure I/O layer: does NOT interpret or modify business logic

import os
import shutil
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import info, skip, done


# Columns we never dictionary-encode
DICT_EXCLUDE = {"SalesOrderNumber", "CustomerKey"}

# Columns that must always exist in Sales
REQUIRED_PRICING_COLS = {
    "UnitPrice",
    "NetPrice",
    "UnitCost",
    "DiscountAmount",
}


# ----------------------------------------------------------------------
# PARQUET MERGER
# ----------------------------------------------------------------------
def merge_parquet_files(parquet_files, merged_file, delete_after=False):
    """
    Optimized parquet merger:
    - Streams row-groups (constant memory)
    - Handles schema mismatches safely
    - No pandas, no Arrow dataset
    """

    parquet_files = [p for p in parquet_files if os.path.exists(p)]
    if not parquet_files:
        skip("No parquet chunk files to merge")
        return None

    parquet_files = sorted(parquet_files)
    info(f"Merging {len(parquet_files)} chunks → {os.path.basename(merged_file)}")

    readers = [(p, pq.ParquetFile(p)) for p in parquet_files]

    # ------------------------------------------------------------------
    # Canonical schema (first file wins by design)
    # ------------------------------------------------------------------
    schema = readers[0][1].schema_arrow

    missing = REQUIRED_PRICING_COLS - set(schema.names)
    if missing:
        raise RuntimeError(f"Missing required pricing columns: {missing}")

    dict_cols = [c for c in schema.names if c not in DICT_EXCLUDE]

    writer = pq.ParquetWriter(
        merged_file,
        schema,
        compression="snappy",
        use_dictionary=dict_cols,
        write_statistics=True,
    )

    try:
        for path, reader in readers:
            # Schema mismatch: project to canonical schema
            if reader.schema_arrow != schema:
                for i in range(reader.num_row_groups):
                    batch = reader.read_row_group(i).select(schema.names)
                    writer.write_table(batch)
                continue

            # Fast path: identical schema
            for i in range(reader.num_row_groups):
                writer.write_table(reader.read_row_group(i))
    finally:
        writer.close()

    if delete_after:
        for path, _ in readers:
            try:
                os.remove(path)
            except Exception:
                pass

    done(f"Merged chunks → {os.path.basename(merged_file)}")
    return merged_file


# ----------------------------------------------------------------------
# DELTA-PARQUET PARTITION WRITER
# ----------------------------------------------------------------------
def write_delta_partitioned(parts_folder, delta_output_folder, partition_cols):
    """
    Convert worker parquet parts into a partitioned Delta table.

    - Arrow-only (no pandas, no dataset)
    - Uses deltalake.write_deltalake
    - Scales via append-style writes
    """

    parts_folder = os.path.abspath(parts_folder)
    delta_output_folder = os.path.abspath(delta_output_folder)

    if not os.path.exists(parts_folder):
        raise FileNotFoundError(f"Parts folder not found: {parts_folder}")

    part_files = sorted(
        os.path.join(parts_folder, f)
        for f in os.listdir(parts_folder)
        if f.endswith(".parquet")
    )
    if not part_files:
        raise RuntimeError("No delta part files found.")

    # ------------------------------------------------------------------
    # Validate schema once (first file)
    # ------------------------------------------------------------------
    first_schema = pq.ParquetFile(part_files[0]).schema_arrow

    missing = REQUIRED_PRICING_COLS - set(first_schema.names)
    if missing:
        raise RuntimeError(f"Missing required pricing columns: {missing}")

    if partition_cols is None:
        partition_cols = []

    missing = [c for c in partition_cols if c not in first_schema.names]
    if missing:
        raise RuntimeError(f"Partition columns missing from schema: {missing}")

    os.makedirs(delta_output_folder, exist_ok=True)

    try:
        from deltalake import write_deltalake
    except Exception as e:
        raise RuntimeError(
            "deltalake is required for Delta output"
        ) from e

    info(f"[DELTA] Writing {len(part_files)} parts using Arrow → Delta")

    first = True
    for pf in part_files:
        try:
            table = pq.read_table(pf)
        except Exception as ex:
            raise RuntimeError(f"Failed to read part file {pf}: {ex}") from ex

        # Optional stable partition ordering
        if partition_cols:
            try:
                sort_keys = [(c, "ascending") for c in partition_cols]
                table = table.sort_by(sort_keys)
            except Exception as ex:
                raise RuntimeError(f"Failed to sort table: {ex}") from ex

        write_deltalake(
            delta_output_folder,
            table,
            mode="overwrite" if first else "append",
            partition_by=partition_cols,
        )
        first = False

    # Cleanup only the parts folder that was used
    try:
        shutil.rmtree(parts_folder, ignore_errors=True)
    except Exception:
        pass

    return
