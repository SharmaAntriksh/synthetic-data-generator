"""Compact small Delta Lake files into fewer, larger ones.

Usage:
    python scripts/optimize_delta.py <dataset_folder>
    python scripts/optimize_delta.py generated_datasets/2026-03-29_run
    python scripts/optimize_delta.py <dataset_folder> --compression ZSTD

Scans for all Delta tables in the dataset folder and runs OPTIMIZE
(file compaction) on each. Safe to run multiple times.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_VALID_COMPRESSIONS = ("UNCOMPRESSED", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD", "LZ4_RAW")


def find_delta_tables(root: Path) -> list[Path]:
    """Find directories containing _delta_log/ (Delta Lake tables)."""
    tables = []
    for delta_log in root.rglob("_delta_log"):
        if delta_log.is_dir():
            tables.append(delta_log.parent)
    return sorted(tables)


def optimize_table(
    table_path: Path,
    target_size: int,
    max_tasks: int | None = None,
    compression: str | None = None,
) -> dict:
    """Run OPTIMIZE on a single Delta table. Returns stats."""
    from deltalake import DeltaTable, WriterProperties

    dt = DeltaTable(str(table_path))

    def _total_size(uris):
        total = 0
        for uri in uris:
            p = Path(unquote(uri.replace("file://", "")))
            if p.exists():
                total += p.stat().st_size
        return total

    uris_before = dt.file_uris()
    files_before = len(uris_before)
    size_before = _total_size(uris_before)

    writer_props = WriterProperties(compression=compression) if compression else None

    t0 = time.time()
    dt.optimize.compact(
        target_size=target_size,
        max_concurrent_tasks=max_tasks,
        writer_properties=writer_props,
    )
    dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
    elapsed = time.time() - t0

    dt = DeltaTable(str(table_path))
    uris_after = dt.file_uris()
    files_after = len(uris_after)
    size_after = _total_size(uris_after)

    return {
        "files_before": files_before,
        "files_after": files_after,
        "size_before_mb": size_before / 1024 / 1024,
        "size_after_mb": size_after / 1024 / 1024,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize Delta Lake tables")
    parser.add_argument("folder", help="Dataset folder to scan for Delta tables")
    parser.add_argument("--target-size", type=int, default=256,
                        help="Target file size in MB (default: 256)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max concurrent compaction tasks (default: CPU count)")
    parser.add_argument("--min-files", type=int, default=5,
                        help="Skip tables with this many files or fewer (default: 5)")
    parser.add_argument(
        "--compression", type=str.upper, default=None, metavar="CODEC",
        choices=_VALID_COMPRESSIONS,
        help=f"Recompress during compaction (default: keep existing). "
             f"Valid: {', '.join(_VALID_COMPRESSIONS)}",
    )
    args = parser.parse_args()

    from deltalake import DeltaTable

    root = Path(args.folder)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        return 1

    tables = find_delta_tables(root)
    if not tables:
        print(f"No Delta tables found in {root}")
        return 0

    print(f"Dataset: {root.name}")
    print(f"Tables:  {len(tables)}")
    print(f"Options: target_size={args.target_size} MB  min_files={args.min_files}"
          f"{f'  compression={args.compression}' if args.compression else ''}")
    print()

    errors = 0
    skipped = 0
    t_start = time.time()

    for table_path in tables:
        rel = table_path.relative_to(root)
        folder = rel.parts[0] if len(rel.parts) > 1 else ""
        name = rel.parts[-1]

        try:
            dt = DeltaTable(str(table_path))
            n_files = len(dt.file_uris())
        except Exception:
            n_files = 0

        if n_files <= args.min_files:
            skipped += 1
            continue

        try:
            stats = optimize_table(
                table_path, args.target_size * 1024 * 1024, args.max_tasks,
                compression=args.compression,
            )
            pct = (1 - stats["size_after_mb"] / stats["size_before_mb"]) * 100 if stats["size_before_mb"] > 0 else 0
            print(f"  {folder:12s} {name:30s}"
                  f"  {stats['size_before_mb']:7.1f} MB -> {stats['size_after_mb']:7.1f} MB ({pct:4.1f}% smaller)"
                  f"  {stats['files_before']:>4d} -> {stats['files_after']:<4d} files"
                  f"  {stats['elapsed_s']:.1f}s")
        except Exception as e:
            print(f"  {folder:12s} {name:30s}  Error: {e}")
            errors += 1

    elapsed = time.time() - t_start
    print()
    if skipped:
        print(f"Skipped {skipped} table(s) with {args.min_files} or fewer files")
    if errors:
        print(f"Failed: {errors} table(s)")
    print(f"Total: {len(tables) - skipped - errors} table(s) optimized in {elapsed:.1f}s")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
