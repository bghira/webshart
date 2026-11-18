"""Ownership races and publication fencing do not require cloud credentials."""

import concurrent.futures
import importlib.util
import json
from io import BytesIO
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "shard_queue", Path(__file__).parents[1] / "python/webshart/shard_queue.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Conflict, LeaseLost, ShardQueue = module.Conflict, module.LeaseLost, module.ShardQueue


class Store:
    def __init__(self):
        self.data, self.counter, self.lock = {}, 0, threading.Lock()

    def get(self, key):
        with self.lock:
            value = self.data.get(key)
            return (json.loads(value[0]), value[1]) if value else None

    def put(self, key, value, expected):
        with self.lock:
            old = self.data.get(key)
            if (old[1] if old else None) != expected:
                raise Conflict(key)
            self.counter += 1
            etag = str(self.counter)
            self.data[key] = json.dumps(value), etag
            return etag


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000
        self.store = Store()
        self.queue = ShardQueue(
            self.store,
            "capture",
            100,
            200,
            source="example/images",
            revision="fixed",
            ttl=30,
            clock=lambda: self.now,
        )
        self.queue.initialize()

    def test_simultaneous_claim_has_one_owner(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            results = list(
                pool.map(lambda n: self.queue.claim(100, str(n)), range(100))
            )
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_expired_owner_cannot_publish_or_overwrite_new_attempt(self):
        old = self.queue.claim(100, "first")
        self.now += 31
        new = self.queue.claim(100, "second")
        self.assertNotEqual(old.output_prefix, new.output_prefix)
        with self.assertRaises(LeaseLost):
            self.queue.renew(old)
        with self.assertRaises(LeaseLost):
            self.queue.complete(old, old.output_prefix + "manifest.json", "a" * 64)
        self.queue.complete(new, new.output_prefix + "manifest.json", "b" * 64)
        self.now += 100
        self.assertIsNone(self.queue.claim(100, "third"))

    def test_cannot_complete_after_expiry_even_without_replacement(self):
        old = self.queue.claim(101, "first")
        self.now += 30
        with self.assertRaises(LeaseLost):
            self.queue.complete(old, old.output_prefix + "manifest.json", "a" * 64)

    def test_renew_and_resume_require_identity(self):
        lease = self.queue.claim(102, "first")
        self.now += 20
        renewed = self.queue.resume(102, "first", lease.token)
        self.assertEqual(renewed.expires_at, self.now + 30)
        self.assertIsNone(self.queue.claim(102, "first"))
        with self.assertRaises(LeaseLost):
            self.queue.resume(102, "second", lease.token)

    def test_completion_is_idempotent_but_manifest_cannot_change(self):
        lease = self.queue.claim(103, "first")
        key = lease.output_prefix + "manifest.json"
        first = self.queue.complete(lease, key, "a" * 64)
        self.assertEqual(self.queue.complete(lease, key, "a" * 64), first)
        with self.assertRaises(LeaseLost):
            self.queue.complete(lease, key, "b" * 64)
        with self.assertRaises(ValueError):
            self.queue.complete(lease, "another/manifest.json", "a" * 64)

    def test_queue_configuration_and_bounds_are_immutable(self):
        self.queue.initialize()
        with self.assertRaises(ValueError):
            ShardQueue(
                self.store,
                "capture",
                99,
                200,
                source="example/images",
                revision="fixed",
            ).initialize()
        with self.assertRaises(ValueError):
            self.queue.claim(99, "worker")


class S3StoreTests(unittest.TestCase):
    @staticmethod
    def error(status):
        error = RuntimeError("object request failed")
        error.response = {"ResponseMetadata": {"HTTPStatusCode": status}}
        return error

    def test_conditional_writes_preserve_etags(self):
        client = Mock()
        client.put_object.return_value = {"ETag": '"updated"'}
        store = module.S3Store(client, "captures")
        for expected in (None, '"previous"'):
            with self.subTest(expected=expected):
                self.assertEqual(
                    store.put("claim.json", {"status": "leased"}, expected), '"updated"'
                )
                kwargs = client.put_object.call_args.kwargs
                self.assertEqual(kwargs["Bucket"], "captures")
                self.assertEqual(json.loads(kwargs["Body"]), {"status": "leased"})
                self.assertEqual(
                    kwargs.get("IfNoneMatch"), "*" if expected is None else None
                )
                self.assertEqual(kwargs.get("IfMatch"), expected)

    def test_conditional_failures_are_conflicts_and_other_errors_propagate(self):
        client = Mock()
        store = module.S3Store(client, "captures")
        for status in (409, 412, 403, 500):
            with self.subTest(status=status):
                client.put_object.side_effect = self.error(status)
                expected = Conflict if status in (409, 412) else RuntimeError
                with self.assertRaises(expected):
                    store.put("claim.json", {}, '"previous"')

    def test_reads_close_body_and_only_missing_objects_return_none(self):
        client = Mock()
        store = module.S3Store(client, "captures")
        body = BytesIO(b'{"status":"done"}')
        client.get_object.return_value = {"Body": body, "ETag": '"current"'}
        self.assertEqual(store.get("claim.json"), ({"status": "done"}, '"current"'))
        self.assertTrue(body.closed)
        client.get_object.side_effect = self.error(404)
        self.assertIsNone(store.get("claim.json"))
        client.get_object.side_effect = self.error(403)
        with self.assertRaises(RuntimeError):
            store.get("claim.json")

    def test_invalid_json_still_closes_the_response_body(self):
        client = Mock()
        body = BytesIO(b"invalid")
        client.get_object.return_value = {"Body": body, "ETag": '"current"'}
        with self.assertRaises(json.JSONDecodeError):
            module.S3Store(client, "captures").get("claim.json")
        self.assertTrue(body.closed)


if __name__ == "__main__":
    unittest.main()
