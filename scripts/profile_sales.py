"""Profile a single sales worker chunk to identify hot paths.

Usage:
    python scripts/profile_sales.py [--rows 1000000] [--format parquet]

Outputs:
    - scripts/profile_output/sales_profile.prof  (cProfile binary)
    - scripts/profile_output/sales_top50.txt     (top 50 by cumulative time)
    - scripts/profile_output/sales_top50_tottime.txt (top 50 by total time)

View interactively:
    pip install snakeviz
    snakeviz scripts/profile_output/sales_profile.prof
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import io
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Profile sales generation")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Total sales rows")
    parser.add_argument("--format", type=str, default="parquet", help="Output format")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (1 for clean profile)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "scripts" / "profile_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.engine.runners.pipeline_runner import run_pipeline, PipelineOverrides

    overrides = PipelineOverrides(
        sales_rows=args.rows,
        file_format=args.format,
        workers=args.workers,
    )

    prof = cProfile.Profile()
    prof.enable()

    try:
        run_pipeline(
            config_path=str(PROJECT_ROOT / "config.yaml"),
            models_config_path=str(PROJECT_ROOT / "models.yaml"),
            overrides=overrides,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving partial profile.")
    finally:
        prof.disable()

    # Save binary profile (for snakeviz / pstats interactive)
    prof_path = out_dir / "sales_profile.prof"
    prof.dump_stats(str(prof_path))
    print(f"\nProfile saved: {prof_path}")

    # Save top-50 by cumulative time
    for sort_key, suffix in [("cumulative", "cumtime"), ("tottime", "tottime")]:
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf)
        stats.sort_stats(sort_key)
        stats.print_stats(50)
        txt_path = out_dir / f"sales_top50_{suffix}.txt"
        txt_path.write_text(buf.getvalue())
        print(f"Top 50 by {sort_key}: {txt_path}")

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("TOP 20 BY TOTAL TIME (self time, excludes sub-calls)")
    print("=" * 70)
    stats = pstats.Stats(prof, stream=sys.stdout)
    stats.sort_stats("tottime")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
