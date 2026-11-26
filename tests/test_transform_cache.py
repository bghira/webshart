import os
import sqlite3
import subprocess
import sys
import time
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


def test_read_recovers_interrupted_shard_transaction(tmp_path):
    with closing(TransformCache(tmp_path, "recipe")) as cache:
        cache.enqueue("shard", [(row, bytes(8192)) for row in range(64)])
        cache.flush()

    (path,) = tmp_path.rglob("*.sqlite3")
    writer = """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute("PRAGMA cache_size=4")
db.execute("BEGIN IMMEDIATE")
db.execute("UPDATE transforms SET value=randomblob(8192)")
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", writer, str(path)], check=True, timeout=20)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as reader:
        with pytest.raises(sqlite3.OperationalError) as error:
            reader.execute("SELECT value FROM transforms LIMIT 1")
        assert error.value.sqlite_errorname == "SQLITE_READONLY_ROLLBACK"

    with closing(TransformCache(tmp_path, "recipe")) as cache:
        assert cache.read_batch([("shard", 0), ("shard", 63)]) == [bytes(8192)] * 2


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


def test_transform_cache_budget_evicts_across_recipes_and_keeps_recent_hits(tmp_path):
    first = TransformCache(tmp_path, "first")
    first.enqueue("recent", [(0, b"recent")])
    first.close()
    recent = next(tmp_path.rglob("*.sqlite3"))
    shard_bytes = recent.stat().st_size
    second = TransformCache(tmp_path, "second")
    second.enqueue("old", [(0, b"old")])
    second.close()
    old = next(path for path in tmp_path.rglob("*.sqlite3") if path != recent)
    os.utime(recent, (1, 1))
    os.utime(old, (2, 2))

    limit = 2 * shard_bytes
    first = TransformCache(tmp_path, "first", max_disk_bytes=limit)
    second = TransformCache(tmp_path, "second", max_disk_bytes=limit)
    try:
        assert first.read_batch([("recent", 0)]) == [b"recent"]
        second.enqueue("new", [(0, b"new")])
        second.flush()

        assert first.read_batch([("recent", 0)]) == [b"recent"]
        assert second.read_batch([("old", 0), ("new", 0)]) == [None, b"new"]
        assert sum(path.stat().st_size for path in tmp_path.rglob("*.sqlite3")) <= limit
    finally:
        first.close()
        second.close()


def test_transform_cache_automatically_drops_oversized_database(tmp_path):
    cache = TransformCache(tmp_path, "recipe")
    cache.enqueue("small", [(0, b"keep")])
    cache.close()
    small = next(tmp_path.rglob("*.sqlite3"))
    cache = TransformCache(tmp_path, "recipe", max_disk_bytes=small.stat().st_size)
    try:
        cache.enqueue("oversized", [(0, b"x" * 65536)])
        deadline = time.monotonic() + 15
        saw_oversized = False
        while time.monotonic() < deadline:
            oversized = set(tmp_path.rglob("*.sqlite3")) - {small}
            if oversized:
                saw_oversized = True
            elif saw_oversized:
                break
            time.sleep(0.02)
        assert saw_oversized and not oversized
        assert cache.read_batch([("small", 0), ("oversized", 0)]) == [b"keep", None]
    finally:
        cache.close()


@pytest.mark.parametrize("access", ["read", "write"])
def test_transform_cache_budget_preserves_active_database_in_another_process(tmp_path, access):
    fcntl = pytest.importorskip("fcntl")
    cache = TransformCache(tmp_path, "recipe")
    cache.enqueue("shard", [(0, b"existing")])
    cache.close()
    path = next(tmp_path.rglob("*.sqlite3"))
    limit = path.stat().st_size - 1
    gate = tmp_path / ".webshart-transform-locks" / path.stem[:3]
    worker = """
import sys
from webshart import TransformCache
cache = TransformCache(sys.argv[1], "recipe")
if sys.argv[2] == "read":
    assert cache.read_batch([("shard", 0)]) == [b"existing"]
else:
    cache.enqueue("shard", [(1, b"new")])
cache.close()
"""
    with closing(sqlite3.connect(path)) as blocker:
        blocker.execute("BEGIN EXCLUSIVE")
        process = subprocess.Popen([sys.executable, "-c", worker, str(tmp_path), access])
        with gate.open("rb+") as probe:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    break
                fcntl.flock(probe, fcntl.LOCK_UN)
                time.sleep(0.01)
            else:
                blocker.rollback()
                process.wait(timeout=10)
                pytest.fail("child process did not acquire its cache lock")
        evictor = None
        try:
            evictor = TransformCache(tmp_path, "recipe", max_disk_bytes=limit)
            evictor.flush()
            assert path.exists()
        finally:
            blocker.rollback()
            assert process.wait(timeout=10) == 0
            if evictor is not None:
                evictor.close()
    assert not path.exists()


def test_transform_cache_budget_cleans_orphaned_sidecars_on_reopen(tmp_path):
    cache = TransformCache(tmp_path, "recipe")
    cache.enqueue("shard", [(0, b"old")])
    cache.close()
    path = next(tmp_path.rglob("*.sqlite3"))
    path.unlink()
    orphan = path.with_suffix(".sqlite3-journal")
    orphan.write_bytes(b"simulated interrupted eviction")
    unrelated = path.parent / "other.sqlite3"
    unrelated.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="max_disk_bytes"):
        TransformCache(tmp_path, "recipe", max_disk_bytes=0)
    cache = TransformCache(tmp_path, "recipe", max_disk_bytes=1 << 20)
    try:
        assert not orphan.exists()
        assert unrelated.read_bytes() == b"unrelated"
        assert cache.read_batch([("shard", 0)]) == [None]
        cache.enqueue("shard", [(0, b"new")])
        cache.flush()
        assert cache.read_batch([("shard", 0)]) == [b"new"]
    finally:
        cache.close()
