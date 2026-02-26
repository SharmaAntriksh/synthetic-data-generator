import os, sys, csv
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

root = r"C:\Users\antsharma\Downloads\synthetic-data-generator\generated_datasets\2026-02-26 03_54_20 PM Customers 80 Sales 10K PARQUET"

parquet_paths = []
for dp, _, fns in os.walk(root):
    for fn in fns:
        if fn.lower().endswith(".parquet"):
            parquet_paths.append(os.path.join(dp, fn))

print("Root:", root)
print("Found parquet files:", len(parquet_paths))

if not parquet_paths:
    print("No .parquet files found under root. Check the path and folder structure.")
    sys.exit(0)

issues = []

def type_flags(t: pa.DataType):
    flags = []
    if pa.types.is_null(t):
        flags.append("NULL_TYPE")
    if (pa.types.is_struct(t) or pa.types.is_list(t) or pa.types.is_large_list(t)
        or pa.types.is_map(t) or pa.types.is_union(t)):
        flags.append("NESTED")
    if pa.types.is_dictionary(t):
        flags.append("DICTIONARY")
    if pa.types.is_decimal(t):
        flags.append(f"DECIMAL({t.precision},{t.scale})")
    if pa.types.is_timestamp(t) and t.tz is not None:
        flags.append(f"TIMESTAMP_TZ({t.tz})")
    if pa.types.is_duration(t) or pa.types.is_time32(t) or pa.types.is_time64(t):
        flags.append("TIME/DURATION")
    if pa.types.is_large_string(t) or pa.types.is_large_binary(t):
        flags.append("LARGE_*")
    return flags

for path in parquet_paths:
    try:
        pf = pq.ParquetFile(path)
        nrows = pf.metadata.num_rows
        schema = pf.schema_arrow

        if nrows == 0:
            issues.append((path, nrows, "", "", "ZERO_ROWS_FILE"))

        # Flag “hard”/edge types in schema
        for field in schema:
            flags = type_flags(field.type)
            if flags:
                issues.append((path, nrows, field.name, str(field.type), ";".join(flags)))

        # Flag columns where the FIRST sample chunk is entirely null (Power BI sampling issue)
        if nrows > 0 and pf.num_row_groups > 0:
            t0 = pf.read_row_group(0)
            sample = t0.slice(0, min(1024, t0.num_rows))
            if sample.num_rows > 0:
                for name in sample.schema.names:
                    col = sample[name]
                    v = pc.all(pc.is_null(col))
                    # pc.all(...) can return null if empty; guard with is_valid
                    if v.is_valid and v.as_py() is True:
                        issues.append((path, nrows, name, str(col.type), "FIRST_1024_ALL_NULL"))

    except Exception as e:
        issues.append((path, None, "", "", f"READ_ERROR:{type(e).__name__}:{e}"))

print("Issues found:", len(issues))
for row in issues[:50]:
    print(row)

out = os.path.join(root, "_parquet_diagnostics.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "rows", "column", "arrow_type", "flags"])
    w.writerows(issues)

print("Wrote report:", out)