"""One replaceable background calculation with cooperative cancellation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Executor
from functools import partial
from threading import Event

from opengraph import events
from opengraph.engine.cancellation import Cancelled, cancellable
from opengraph.graph.rendering import worker_pool

log = logging.getLogger(__name__)


class _Job:
    def __init__(
        self,
        key: object,
        fn: Callable[[], object],
        lifetime: Event,
        finished: Callable[[_Job], None],
    ) -> None:
        self.key, self.fn, self.lifetime = key, fn, lifetime
        self.finished = finished
        self.cancelled = Event()
        self.value: object = None

    def obsolete(self) -> bool:
        return self.cancelled.is_set() or self.lifetime.is_set()

    def run(self) -> None:
        try:
            with cancellable(self.obsolete):
                self.value = self.fn()
        except Cancelled:
            pass
        except Exception:
            log.exception("Background graph analysis failed")
        events.post(partial(self.finished, self))


class LatestTask:
    """Runs the latest requested calculation; ``ready`` gets ``(key, value)``.

    Jobs run on ``pool`` (the geometry workers by default); ``value`` is None if the
    calculation raised (the exception is logged).
    """

    def __init__(self, pool: Executor | None = None) -> None:
        self.ready = events.Signal()
        self._pool = pool
        self._active: _Job | None = None
        self._pending: tuple[object, Callable[[], object]] | None = None
        self._lifetime = Event()

    def request(self, key: object, fn: Callable[[], object]) -> None:
        if self._lifetime.is_set():
            return
        if self._active is not None and self._active.key == key and not self._active.obsolete():
            return
        self.cancel()
        self._pending = key, fn
        self._pump()

    def _pump(self) -> None:
        if self._active is not None or self._pending is None:
            return
        key, fn = self._pending
        self._pending = None
        job = self._active = _Job(key, fn, self._lifetime, self._completed)
        (self._pool or worker_pool()).submit(job.run)

    def _completed(self, job: _Job) -> None:
        self._active = None
        if not job.obsolete():
            self.ready.emit(job.key, job.value)
        self._pump()

    def cancel(self) -> None:
        self._pending = None
        if self._active is not None:
            self._active.cancelled.set()

    def stop(self) -> None:
        self._lifetime.set()
        self.cancel()
