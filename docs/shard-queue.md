# Shard claims

`webshart.shard_queue` coordinates independent source shards through conditional
object-store writes. Fetching, image probing, and recompression continue to use
the native scanner. The queue does not share SQLite databases between workers.

```python
import boto3
from webshart.shard_queue import ShardQueue, S3Store

queue = ShardQueue(
    S3Store(boto3.client("s3"), "image-capture"),
    "capture-v1", 0, 1000,
    source="example/images", revision="pinned-revision",
)
queue.initialize()
lease = queue.claim(42, "worker-1")
```

The client must implement conditional `PutObject` with `IfNoneMatch` and
`IfMatch`, and return object ETags. Install `boto3` separately when using the
adapter above. Use a separate queue prefix for each source revision and shard
assignment. Initialization rejects an incompatible existing configuration.

A successful claim returns a token and an attempt-specific `output_prefix`.
Keep clocks synchronized and renew the claim well before its expiry. Continue
renewing while captured data waits for an upload batch. A live claim cannot be
claimed again, even by the same owner; use `resume` with the saved token only
when the corresponding local attempt state survived.

Upload all completed shard data under `lease.output_prefix`, verify the uploaded
objects, and then upload a manifest referencing those objects. Publish that
manifest with `queue.complete(lease, manifest_key, manifest_sha256)`. A successful
completion is idempotent for the same manifest and digest. The worker must verify
that its source scan is complete before publishing; the queue does not interpret
application-specific manifests.

Readers consume only manifests referenced by claims whose status is `done`.
An expired worker cannot renew or publish. Another worker can claim the shard
with a new token and output prefix; uploads from the previous attempt cannot
overwrite the new attempt's objects. Uncommitted objects may remain after a
worker failure and should be retained until their ownership has been reconciled
before cleanup.

Existing fixed-range workers must relinquish the new queue's source range before
queue workers start. Renewals and completion for a lease should be serialized
within a worker. If a conditional write loses a race, stop publishing for that
lease and reconcile ownership rather than writing unconditionally.
