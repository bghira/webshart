import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Event

import pytest

from webshart import TransformCache


def test_transform_cache_reads_in_order_and_keeps_namespaces_separate(tmp_path):
    cache = TransformCache(tmp_path, "resolution-128")
    requests = [("first-v1", 2), ("second-v1", 0), ("first-v1", 2), ("first-v1", 1)]
    assert cache.read_batch(requests) == [None] * len(requests)
    assert not list(tmp_path.iterdir())

    cache.enqueue("first-v1", [(2, b"first")])
    cache.enqueue("second-v1", [(0, b"second")])
    cache.flush()
    assert cache.read_batch(requests) == [b"first", b"second", b"first", None]
    assert cache.read_batch([("first-v2", 2)]) == [None]
    assert len(list(tmp_path.rglob("*.sqlite3"))) == 2

    other_recipe = TransformCache(tmp_path, "resolution-256")
    try:
        assert other_recipe.read_batch(requests) == [None] * len(requests)
    finally:
        other_recipe.close()

    cache.enqueue("first-v1", [(1, b"last")])
    cache.close()
    cache.close()
    with pytest.raises(OSError, match="closed"):
        cache.read_batch(requests)
    with pytest.raises(OSError, match="closed"):
        cache.enqueue("first-v1", [(4, b"too late")])

    reopened = TransformCache(tmp_path, "resolution-128")
    try:
        assert reopened.read_batch(requests) == [b"first", b"second", b"first", b"last"]
    finally:
        reopened.close()


def test_transform_cache_reports_worker_failure_on_flush(tmp_path):
    directory = tmp_path / "cache"
    cache = TransformCache(directory, "recipe")
    directory.rmdir()
    directory.write_bytes(b"cannot create shard directories here")
    cache.enqueue("shard", [(0, b"transform")])

    with pytest.raises(OSError, match="transform cache write failed"):
        cache.flush()
    with pytest.raises(OSError, match="transform cache write failed"):
        cache.enqueue("shard", [(1, b"transform")])
    with pytest.raises(OSError, match="transform cache write failed"):
        cache.close()


def test_transform_cache_enforces_queue_and_batch_limits(tmp_path):
    with pytest.raises(ValueError, match="queue limits"):
        TransformCache(tmp_path, "recipe", queue_batches=0)
    cache = TransformCache(tmp_path, "recipe", queue_batches=1, max_batch_bytes=128)
    try:
        with pytest.raises(ValueError, match="max_batch_bytes"):
            cache.enqueue("shard", [(0, b"x" * 128)])
        with pytest.raises(ValueError, match="nonnegative"):
            cache.enqueue("shard", [(-1, b"x")])
        with pytest.raises(ValueError, match="nonnegative"):
            cache.read_batch([("shard", -1)])

        cache.enqueue("shard", [(0, b"x")])
        cache.flush()
        path = next(tmp_path.rglob("*.sqlite3"))
        cache.enqueue("other", [(0, b"y")])
        cache.flush()
        submitted_two = Event()

        def enqueue_until_blocked():
            cache.enqueue("shard", [(1, b"x")])
            cache.enqueue("shard", [(2, b"x")])
            submitted_two.set()
            cache.enqueue("shard", [(3, b"x")])

        with closing(sqlite3.connect(path)) as lock, ThreadPoolExecutor(max_workers=2) as pool:
            lock.execute("BEGIN EXCLUSIVE")
            pending = pool.submit(enqueue_until_blocked)
            try:
                assert submitted_two.wait(2)
                assert not pending.done()
                read = pool.submit(cache.read_batch, [("other", 0)])
                assert read.result(timeout=2) == [b"y"]
            finally:
                lock.rollback()
            pending.result(timeout=2)

        cache.flush()
        assert cache.read_batch([("shard", row) for row in range(4)]) == [b"x"] * 4
    finally:
        cache.close()


def test_transform_cache_two_processes_can_fill_the_same_shard(tmp_path):
    worker = """
import sys
from webshart import TransformCache
cache = TransformCache(sys.argv[1], "recipe", queue_batches=2)
rank = int(sys.argv[2])
for start in range(1, 101, 10):
    cache.enqueue("shared", [(0, b"same")] + [
        (2 * row + rank, bytes([rank])) for row in range(start, start + 10)
    ])
cache.close()
"""

    def write(rank):
        subprocess.run(
            [sys.executable, "-c", worker, str(tmp_path), str(rank)],
            check=True,
            timeout=20,
            capture_output=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, range(2)))

    cache = TransformCache(tmp_path, "recipe")
    try:
        requests = [("shared", row) for row in [0, *range(2, 202)]]
        assert cache.read_batch(requests) == [b"same", *([b"\x00", b"\x01"] * 100)]
    finally:
        cache.close()
