"""Byte-bounded whole-shard read-ahead for a sequential Parquet consumer."""

import logging
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from operator import index
from threading import Condition, Thread

from filelock import FileLock, Timeout


class SequentialShardPrefill:
    def __init__(
        self, directory, budget, parallelism, order, shard_size, acquire, active_shards
    ):
        self.directory = directory
        self.budget = budget
        self.parallelism = parallelism
        self.active_shards = active_shards
        self._order = order
        self.count = len(order)
        self.shard_size = shard_size
        self.acquire = acquire
        self.condition = Condition()
        self.first = None
        self.stopped = False
        self.error = None
        self.thread = None
        self.ready_range = None

    def shard_at(self, position):
        """Return the physical shard ID at a read-ahead position."""
        if not 0 <= position < self.count:
            raise IndexError("Shard order position is out of bounds")
        return self._order[position]

    def advance(self, first):
        """Retire earlier shards and refill from this position in the order."""
        first = index(first)
        if not 0 <= first <= self.count:
            raise IndexError("Shard order position is out of bounds")
        with self.condition:
            if self.error is not None:
                raise self.error
            if self.first is not None and first < self.first:
                raise ValueError("Start a new prefill cursor when rewinding.")
            self.first = first
            if self.thread is None and not self.stopped:
                self.thread = Thread(
                    target=self._run, name="webshart-prefill", daemon=True
                )
                self.thread.start()
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join()

    def _run(self):
        lock = FileLock(self.directory / ".webshart-prefill.lock")
        try:
            while True:
                try:
                    lock.acquire(timeout=0)
                    break
                except Timeout:
                    with self.condition:
                        if self.condition.wait_for(lambda: self.stopped, timeout=1):
                            return
            with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
                self._fill(pool)
        except Exception as exc:
            with self.condition:
                logging.getLogger(__name__).exception("Sequential shard prefill failed")
                self.error = exc.with_traceback(None)
        finally:
            lock.release()

    def _fill(self, pool):
        resident = deque()
        running = set()
        used = 0
        next_shard = self.first
        checked_first = None
        fits = False
        reported_full = False
        while True:
            with self.condition:
                if self.stopped:
                    return
                first = self.first
                while resident and resident[0][0] < first and resident[0][2].done():
                    used -= resident.popleft()[1]
                if first != checked_first:
                    active_end = min(self.count, first + self.active_shards)
                    fits = (
                        sum(
                            self.shard_size(self._order[position])
                            for position in range(first, active_end)
                        )
                        <= self.budget
                    )
                    checked_first = first
                if not fits and not running:
                    resident.clear()
                    used = 0
                    next_shard = first
                next_shard = max(next_shard, first)
                while (
                    fits and next_shard < self.count and len(running) < self.parallelism
                ):
                    shard = self._order[next_shard]
                    size = self.shard_size(shard)
                    if used + size > self.budget:
                        break
                    # The future owns the read lease until the consumer moves past it.
                    future = pool.submit(self.acquire, shard)
                    running.add(future)
                    resident.append((next_shard, size, future))
                    used += size
                    next_shard += 1
                    del future
                if not running:
                    self.ready_range = (first, next_shard)
                    if not reported_full and resident:
                        logging.getLogger(__name__).info(
                            "Sequential shard prefill holds %d shards (%d of %d bytes).",
                            len(resident),
                            used,
                            self.budget,
                        )
                        reported_full = True
                    self.condition.wait_for(lambda: self.stopped or self.first != first)
                    continue
            done, running = wait(running, return_when=FIRST_COMPLETED)
            while done:
                done.pop().result()
