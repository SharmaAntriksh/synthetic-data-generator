from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PoolRunSpec:
    processes: int
    chunksize: int = 1
    maxtasksperchild: Optional[int] = None
    timeout_s: Optional[float] = None  # per-task timeout (None disables)
    poll_interval_s: float = 0.05
    label: str = ""


def iter_imap_unordered(
    *,
    tasks: Sequence[Any],
    task_fn: Callable[[Any], Any],
    spec: PoolRunSpec,
    initializer: Optional[Callable[..., Any]] = None,
    initargs: Tuple[Any, ...] = (),
) -> Iterator[Any]:
    """
    Generic multiprocessing runner.

    - 'tasks' must be pickleable (dict/tuple/list of primitives is ideal)
    - 'task_fn' and 'initializer' must be top-level functions (importable) for Windows spawn
    - Yields results in completion order (unordered).

    Improvements over raw Pool.imap_unordered:
      - Optional per-task timeout (spec.timeout_s)
      - Worker health checks (fail fast if a worker exits unexpectedly)
      - Robust termination/join on failure
    """
    import time
    import multiprocessing as mp

    if spec.processes <= 0:
        raise ValueError("spec.processes must be >= 1")

    timeout_s = spec.timeout_s if (spec.timeout_s is None or spec.timeout_s > 0) else None
    poll = float(spec.poll_interval_s) if spec.poll_interval_s and spec.poll_interval_s > 0 else 0.05

    with Pool(
        processes=spec.processes,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=spec.maxtasksperchild,
    ) as pool:
        pending = []
        start_times = {}

        try:
            for i, t in enumerate(tasks):
                ar = pool.apply_async(task_fn, (t,))
                pending.append(ar)
                start_times[ar] = time.monotonic()

            while pending:
                # Fail-fast if any worker has died.
                for p in getattr(pool, "_pool", []) or []:
                    if p.exitcode is not None and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker process exited unexpectedly (exitcode={p.exitcode})."
                            + (f" label={spec.label!r}" if spec.label else "")
                        )

                ready = [ar for ar in pending if ar.ready()]
                if ready:
                    for ar in ready:
                        pending.remove(ar)
                        start_times.pop(ar, None)
                        yield ar.get()
                    continue

                if timeout_s is not None:
                    now = time.monotonic()
                    # If any task exceeds the timeout, terminate pool and fail.
                    for ar in list(pending):
                        st = start_times.get(ar, now)
                        if (now - st) > float(timeout_s):
                            raise RuntimeError(
                                f"Task timed out after {timeout_s} seconds" + (f" label={spec.label!r}" if spec.label else "")
                            )

                time.sleep(poll)

        except Exception:
            # Ensure pool is torn down promptly (important for stuck workers).
            try:
                pool.terminate()
            finally:
                pool.join()
            raise
