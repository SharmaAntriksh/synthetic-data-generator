from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PoolRunSpec:
    processes: int
    chunksize: int = 1
    maxtasksperchild: Optional[int] = None
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
    - yields results from Pool.imap_unordered(task_fn, tasks)

    This is mechanics only: no assumptions about the task payload or return type.
    """
    if spec.processes <= 0:
        raise ValueError("spec.processes must be >= 1")

    with Pool(
        processes=spec.processes,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=spec.maxtasksperchild,
    ) as pool:
        # NOTE: Pool.imap_unordered chunksize is an int >= 1
        for result in pool.imap_unordered(task_fn, tasks, chunksize=max(1, int(spec.chunksize))):
            yield result


__all__ = ["PoolRunSpec", "iter_imap_unordered"]
