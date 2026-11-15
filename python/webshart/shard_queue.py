"""Conditional object-store leases for independently captured source shards.

Capture outputs must live beneath ``Lease.output_prefix``. Publishing the
manifest through ``complete`` fences out workers whose lease has expired.
Clocks on workers must be synchronized. No database files are shared.
"""

from dataclasses import dataclass
import json
import time
import uuid


class Conflict(RuntimeError):
    """A conditional write lost a race."""


class LeaseLost(RuntimeError):
    """The worker no longer owns the shard."""


class S3Store:
    """Small adapter for an existing boto3-compatible client and bucket."""

    def __init__(self, client, bucket):
        self.client, self.bucket = client, bucket

    def get(self, key):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            if (
                getattr(error, "response", {})
                .get("ResponseMetadata", {})
                .get("HTTPStatusCode")
                == 404
            ):
                return None
            raise
        body = response["Body"]
        try:
            return json.loads(body.read()), response["ETag"]
        finally:
            body.close()

    def put(self, key, value, expected):
        condition = {"IfNoneMatch": "*"} if expected is None else {"IfMatch": expected}
        try:
            response = self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
                ContentType="application/json",
                **condition,
            )
        except Exception as error:
            if getattr(error, "response", {}).get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            ) in (409, 412):
                raise Conflict(key) from error
            raise
        return response["ETag"]


@dataclass(frozen=True)
class Lease:
    shard: int
    owner: str
    token: str
    expires_at: float
    etag: str
    output_prefix: str


class ShardQueue:
    """Claim, renew, and publish shards using conditional object writes.

    A claim is kept until its captured bytes have been uploaded, including time
    spent accumulating an upload batch. Expired owners cannot publish a result.
    Each attempt has a unique output prefix, so stale uploads cannot overwrite
    another attempt. Readers consume only manifests referenced by done claims.
    """

    def __init__(
        self, store, prefix, start, stop, *, source, revision, ttl=1800, clock=time.time
    ):
        if not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
            raise ValueError("Queue prefix must be a relative object prefix")
        if not 0 <= start < stop or ttl <= 0:
            raise ValueError("Invalid shard range or lease duration")
        self.store, self.prefix = store, prefix.rstrip("/") + "/"
        self.start, self.stop, self.ttl, self.clock = start, stop, ttl, clock
        self.config = {
            "version": 1,
            "start": start,
            "stop": stop,
            "source": source,
            "revision": revision,
        }

    def initialize(self):
        key = self.prefix + "config.json"
        try:
            self.store.put(key, self.config, None)
        except Conflict:
            current = self.store.get(key)
            if current is None or current[0] != self.config:
                raise ValueError("Queue configuration differs from existing ownership")

    def key(self, shard):
        if not self.start <= shard < self.stop:
            raise ValueError("Shard is outside queue ownership")
        return self.prefix + f"claims/{shard:06d}.json"

    def _record(self, shard, owner):
        if not owner:
            raise ValueError("Owner must be nonempty")
        token = uuid.uuid4().hex
        return {
            "shard": shard,
            "owner": owner,
            "token": token,
            "status": "leased",
            "expires_at": self.clock() + self.ttl,
            "output_prefix": self.prefix + f"attempts/{shard:06d}/{token}/",
        }

    @staticmethod
    def _lease(record, etag):
        return Lease(
            **{
                k: record[k]
                for k in ("shard", "owner", "token", "expires_at", "output_prefix")
            },
            etag=etag,
        )

    def claim(self, shard, owner):
        key, record = self.key(shard), self._record(shard, owner)
        try:
            return self._lease(record, self.store.put(key, record, None))
        except Conflict:
            current = self.store.get(key)
        if current is None:
            return None
        old, etag = current
        if old["status"] == "done" or old["expires_at"] > self.clock():
            return None
        try:
            return self._lease(record, self.store.put(key, record, etag))
        except Conflict:
            return None

    def _owned(self, lease):
        current = self.store.get(self.key(lease.shard))
        if current is None:
            raise LeaseLost(lease.shard)
        value, etag = current
        if (
            value["status"] != "leased"
            or value["token"] != lease.token
            or value["owner"] != lease.owner
            or value["output_prefix"] != lease.output_prefix
            or value["expires_at"] <= self.clock()
        ):
            raise LeaseLost(lease.shard)
        return value, etag

    def renew(self, lease):
        value, etag = self._owned(lease)
        value = dict(value, expires_at=self.clock() + self.ttl)
        try:
            etag = self.store.put(self.key(lease.shard), value, etag)
        except Conflict as error:
            raise LeaseLost(lease.shard) from error
        return self._lease(value, etag)

    def resume(self, shard, owner, token):
        """Recover an unexpired claim only when local attempt state survived."""
        current = self.store.get(self.key(shard))
        if current is None:
            raise LeaseLost(shard)
        lease = self._lease(*current)
        if lease.owner != owner or lease.token != token:
            raise LeaseLost(shard)
        return self.renew(lease)

    def complete(self, lease, manifest_key, manifest_sha256):
        if (
            not manifest_key.startswith(lease.output_prefix)
            or ".." in manifest_key.split("/")
            or len(manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in manifest_sha256)
        ):
            raise ValueError(
                "Manifest must belong to this attempt and have a SHA-256 digest"
            )
        current = self.store.get(self.key(lease.shard))
        if current and current[0].get("status") == "done":
            value = current[0]
            if (
                value["token"] == lease.token
                and value["owner"] == lease.owner
                and value.get("manifest") == manifest_key
                and value.get("manifest_sha256") == manifest_sha256
            ):
                return value
            raise LeaseLost(lease.shard)
        value, etag = self._owned(lease)
        value = dict(
            value,
            status="done",
            manifest=manifest_key,
            manifest_sha256=manifest_sha256,
            completed_at=self.clock(),
        )
        try:
            self.store.put(self.key(lease.shard), value, etag)
        except Conflict as error:
            raise LeaseLost(lease.shard) from error
        return value
