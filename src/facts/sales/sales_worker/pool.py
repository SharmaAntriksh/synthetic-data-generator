from __future__ import annotations

import time
from collections import deque
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


def _check_worker_health(pool: Pool, label: str) -> None:
    """Raise immediately if any worker process has crashed."""
    for p in getattr(pool, "_pool", ()) or ():
        if p.exitcode is not None and p.exitcode != 0:
            tag = f" label={label!r}" if label else ""
            raise RuntimeError(
                f"Worker process exited unexpectedly (exitcode={p.exitcode}).{tag}"
            )


def _iter_fast(
    pool: Pool,
    tasks: Sequence[Any],
    task_fn: Callable[[Any], Any],
    spec: PoolRunSpec,
) -> Iterator[Any]:
    """Yield results via imap_unordered — no timeout overhead, native batching."""
    yield from pool.imap_unordered(task_fn, tasks, chunksize=spec.chunksize)


def _iter_with_timeout(
    pool: Pool,
    tasks: Sequence[Any],
    task_fn: Callable[[Any], Any],
    spec: PoolRunSpec,
) -> Iterator[Any]:
    """Sliding-window submission with per-task timeout enforcement.

    Instead of submitting every task upfront, only ``window_size`` tasks are
    in-flight at once.  This caps IPC queue depth and serialised-payload memory
    to a bounded multiple of the worker count.
    """
    timeout_s = spec.timeout_s
    poll = max(spec.poll_interval_s, 0.001)
    window_size = spec.processes * 2

    task_iter = iter(tasks)
    pending: deque[Tuple[Any, float]] = deque()  # (AsyncResult, submit_time)
    exhausted = False

    def _fill_window() -> None:
        nonlocal exhausted
        while len(pending) < window_size and not exhausted:
            t = next(task_iter, _SENTINEL)
            if t is _SENTINEL:
                exhausted = True
                break
            ar = pool.apply_async(task_fn, (t,))
            pending.append((ar, time.monotonic()))

    _fill_window()

    while pending:
        _check_worker_health(pool, spec.label)

        reaped = False
        now = time.monotonic()
        remaining: deque[Tuple[Any, float]] = deque()

        for ar, submit_t in pending:
            if ar.ready():
                yield ar.get()
                reaped = True
            else:
                if timeout_s is not None and (now - submit_t) > timeout_s:
                    tag = f" label={spec.label!r}" if spec.label else ""
                    raise RuntimeError(
                        f"Task timed out after {timeout_s} seconds{tag}"
                    )
                remaining.append((ar, submit_t))

        pending = remaining

        if reaped:
            _fill_window()
            continue

        time.sleep(poll)


_SENTINEL = object()


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

    When no per-task timeout is configured (the common case), delegates directly
    to ``Pool.imap_unordered`` which batches tasks via *chunksize*, avoiding the
    overhead of individual ``apply_async`` calls and busy-wait polling.

    When a timeout is set, uses a bounded sliding window of ``apply_async``
    submissions so that at most ``2 × processes`` payloads sit in the IPC queue
    at any time.
    """
    if spec.processes <= 0:
        raise ValueError("spec.processes must be >= 1")

    use_timeout = spec.timeout_s is not None and spec.timeout_s > 0

    with Pool(
        processes=spec.processes,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=spec.maxtasksperchild,
    ) as pool:
        try:
            if use_timeout:
                yield from _iter_with_timeout(pool, tasks, task_fn, spec)
            else:
                yield from _iter_fast(pool, tasks, task_fn, spec)
        except Exception:
            try:
                pool.terminate()
            finally:
                pool.join()
            raise
