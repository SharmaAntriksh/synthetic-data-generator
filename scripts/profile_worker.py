"""Profile the sales generation hot path IN-PROCESS.

Monkey-patches the multiprocessing pool to run workers in the main
process so cProfile can see inside build_chunk_table.

Usage:
    python scripts/profile_worker.py [--rows 500000]

Outputs:
    - scripts/profile_output/worker_profile.prof
    - scripts/profile_output/worker_top50_tottime.txt
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import io
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _fake_imap_unordered(*, tasks, task_fn, spec, initializer=None, initargs=()):
    """Replace multiprocessing pool with in-process sequential execution.

    Matches the signature of src.utils.pool.iter_imap_unordered but runs
    the worker function directly in the current process so cProfile can
    see into build_chunk_table.
    """
    if initializer:
        initializer(*initargs)

    for task in tasks:
        result = task_fn(task)
        yield result


def main():
    parser = argparse.ArgumentParser(description="Profile sales worker in-process")
    parser.add_argument("--rows", type=int, default=500_000, help="Sales rows to generate")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "scripts" / "profile_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch the pool to run in-process
    with patch("src.utils.pool.iter_imap_unordered", _fake_imap_unordered):
        from src.engine.runners.pipeline_runner import run_pipeline, PipelineOverrides

        overrides = PipelineOverrides(
            sales_rows=args.rows,
            file_format="parquet",
            workers=1,
        )

        print(f"\nProfiling sales pipeline with {args.rows} rows (in-process, no multiprocessing)...")
        print("=" * 70)

        prof = cProfile.Profile()
        prof.enable()

        try:
            run_pipeline(
                config_path=str(PROJECT_ROOT / "config.yaml"),
                models_config_path=str(PROJECT_ROOT / "models.yaml"),
                overrides=overrides,
                only="sales",
            )
        except KeyboardInterrupt:
            print("\nInterrupted — saving partial profile.")
        finally:
            prof.disable()

    # Save outputs
    prof_path = out_dir / "worker_profile.prof"
    prof.dump_stats(str(prof_path))
    print(f"\nProfile saved: {prof_path}")

    for sort_key, suffix in [("tottime", "tottime"), ("cumulative", "cumtime")]:
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf)
        stats.sort_stats(sort_key)
        stats.print_stats(50)
        txt_path = out_dir / f"worker_top50_{suffix}.txt"
        txt_path.write_text(buf.getvalue())
        print(f"Top 50 by {sort_key}: {txt_path}")

    print("\n" + "=" * 70)
    print("TOP 30 BY TOTAL TIME (self time, where CPU actually spends time)")
    print("=" * 70)
    stats = pstats.Stats(prof, stream=sys.stdout)
    stats.sort_stats("tottime")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
