"""Compact small Delta Lake files into fewer, larger ones.

Usage:
    python scripts/optimize_delta.py <dataset_folder>
    python scripts/optimize_delta.py generated_datasets/2026-03-29_run

Scans for all Delta tables in the dataset folder and runs OPTIMIZE
(file compaction) on each. Safe to run multiple times.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def find_delta_tables(root: Path) -> list[Path]:
    """Find directories containing _delta_log/ (Delta Lake tables)."""
    tables = []
    for delta_log in root.rglob("_delta_log"):
        if delta_log.is_dir():
            tables.append(delta_log.parent)
    return sorted(tables)


def optimize_table(table_path: Path, target_size: int, max_tasks: int | None = None) -> dict:
    """Run OPTIMIZE on a single Delta table. Returns stats."""
    from deltalake import DeltaTable

    dt = DeltaTable(str(table_path))

    def _total_size(uris):
        total = 0
        for uri in uris:
            p = Path(uri.replace("file://", ""))
            if p.exists():
                total += p.stat().st_size
        return total

    files_before = len(dt.file_uris())
    size_before = _total_size(dt.file_uris())

    t0 = time.time()
    dt.optimize.compact(
        target_size=target_size,
        max_concurrent_tasks=max_tasks,
    )
    dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
    elapsed = time.time() - t0

    # Reload to get post-optimize stats
    dt = DeltaTable(str(table_path))
    files_after = len(dt.file_uris())
    size_after = _total_size(dt.file_uris())

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
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        return 1

    tables = find_delta_tables(root)
    if not tables:
        print(f"No Delta tables found in {root}")
        return 0

    print(f"Found {len(tables)} Delta table(s):\n")

    for table_path in tables:
        rel = table_path.relative_to(root)

        try:
            from deltalake import DeltaTable
            dt = DeltaTable(str(table_path))
            n_files = len(dt.file_uris())
        except Exception:
            n_files = 0

        if n_files <= args.min_files:
            print(f"  Skipping: {rel} ({n_files} file(s))")
            continue

        print(f"  Optimizing: {rel} ({n_files} files)")

        try:
            stats = optimize_table(table_path, args.target_size * 1024 * 1024, args.max_tasks)
            print(f"    Files: {stats['files_before']} -> {stats['files_after']}")
            print(f"    Size:  {stats['size_before_mb']:.1f} MB -> {stats['size_after_mb']:.1f} MB")
            print(f"    Time:  {stats['elapsed_s']:.1f}s")
        except Exception as e:
            print(f"    Error: {e}")
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
