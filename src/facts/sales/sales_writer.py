# Sales fact writer (Parquet / Delta)
# Pure I/O layer: does NOT interpret or modify business logic


import os
import shutil
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import info, skip, done


# ----------------------------------------------------------------------
# PARQUET MERGER (unchanged, safe)
# ----------------------------------------------------------------------
def merge_parquet_files(parquet_files, merged_file, delete_after=False):
    """
    Fully optimized parquet merger:
    - STREAMS row-groups instead of loading entire tables
    - Handles schema mismatches safely
    - Does not blow memory for large datasets
    - Logs progress using info/skip/done
    """

    parquet_files = [p for p in parquet_files if os.path.exists(p)]

    if not parquet_files:
        skip("No parquet chunk files to merge")
        return None

    parquet_files = sorted(parquet_files)
    info(f"Merging {len(parquet_files)} chunks → {os.path.basename(merged_file)}...")

    # Schema from first file
    first_reader = pq.ParquetFile(parquet_files[0])
    schema = first_reader.schema_arrow

    required_cols = {
        "UnitPrice",
        "NetPrice",
        "UnitCost",
        "DiscountAmount",
    }
    missing = required_cols - set(schema.names)
    if missing:
        raise RuntimeError(f"Missing required pricing columns: {missing}")

    dict_cols = [
        c for c in schema.names
        if c not in ["SalesOrderNumber", "CustomerKey"]
    ]

    writer = pq.ParquetWriter(
        merged_file,
        schema,
        compression="snappy",
        use_dictionary=dict_cols,
        write_statistics=True,
    )

    # Stream each row group
    for path in parquet_files:
        reader = pq.ParquetFile(path)

        if reader.schema_arrow != schema:
            table = reader.read().select(schema.names)
            writer.write_table(table)
            continue

        for i in range(reader.num_row_groups):
            writer.write_table(reader.read_row_group(i))

    writer.close()

    if delete_after:
        for path in parquet_files:
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
    Convert worker parquet parts into a fully partitioned Delta table.

    - Uses deltalake.write_deltalake() (REQUIRED). No pyarrow.dataset fallback.
    - Reads worker part files (parquet) and concatenates using pyarrow.
    - Sorts by partition_cols if provided for cleaner partition files.
    - Writes a single clean Delta commit with partition_by=partition_cols.
    - Cleans up _tmp_parts after successful write.
    """

    # info("[DELTA] Assembling final partitioned dataset...")

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
        raise RuntimeError("No delta part files found for deltaparquet output.")

    # Read schema / preview from first part file (NO Arrow dataset)
    first_file = part_files[0]

    schema = pq.ParquetFile(first_file).schema_arrow
    required_cols = {
        "UnitPrice",
        "NetPrice",
        "UnitCost",
        "DiscountAmount",
    }
    missing = required_cols - set(schema.names)
    if missing:
        raise RuntimeError(f"Missing required pricing columns: {missing}")

    # info(f"[DELTA] dataset schema fields: {schema.names}")

    if partition_cols is None:
        partition_cols = []

    # Validate partition columns exist in schema
    missing = [c for c in partition_cols if c not in schema.names]
    if missing:
        raise RuntimeError(f"Partition columns missing from dataset schema: {missing}")

    # Ensure output folder exists and is empty (safe overwrite)
    if os.path.exists(delta_output_folder):
        # do not blindly delete entire folder - we will overwrite via write_deltalake
        pass
    else:
        os.makedirs(delta_output_folder, exist_ok=True)

    # Ensure deltalake is installed
    try:
        from deltalake import write_deltalake
    except Exception as e:
        raise RuntimeError(
            "deltalake is not installed. Delta output is required; "
            "pyarrow fallback is intentionally disabled to avoid corrupted output."
        ) from e

    # ------------------------------------------------------------------
    # Arrow-native concat (NO pandas) — scalable & faster
    # ------------------------------------------------------------------
    info(f"[DELTA] Reading {len(part_files)} part files using pyarrow...")

    tables = []
    for pf in part_files:
        try:
            tables.append(pq.read_table(pf))
        except Exception as ex:
            raise RuntimeError(f"Failed to read part file {pf}: {ex}") from ex

    # Concatenate Arrow tables
    try:
        combined = pa.concat_tables(tables, promote_options="default")
    except Exception as ex:
        raise RuntimeError(f"Failed to concat Arrow tables: {ex}") from ex

    # Optional sort for stable partitions (Arrow compute)
    if partition_cols:
        info(f"[DELTA] Sorting combined table by: {partition_cols}")
        try:
            sort_keys = [(c, "ascending") for c in partition_cols]
            combined = combined.sort_by(sort_keys)
        except Exception as ex:
            raise RuntimeError(f"Failed to sort Arrow table: {ex}") from ex


    # Final write: single clean delta commit
    # info("[DELTA] Writing real Delta Lake using deltalake.write_deltalake()")
    try:
        write_deltalake(
            str(delta_output_folder),
            combined,
            mode="overwrite",
            partition_by=partition_cols,
        )
    except Exception as ex:
        raise RuntimeError(f"Failed to write delta table: {ex}") from ex

    # done("[DELTA] Real Delta table written cleanly (sorted partitions).")

    # Remove temporary parts folder if present (cleanup)
    tmp_parts = os.path.join(os.path.dirname(parts_folder), "_tmp_parts")
    if os.path.exists(tmp_parts):
        try:
            shutil.rmtree(tmp_parts, ignore_errors=True)
            # info("Cleaning delta _tmp_parts")
        except Exception:
            pass

    return
