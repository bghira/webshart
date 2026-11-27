"""Aspect-bucket iteration over logical samples in TAR and Parquet shards."""

from collections import Counter, defaultdict, deque
from hashlib import sha256
from pathlib import Path
import json
import random
from functools import cached_property

from ._webshart import TarDataLoader, _shuffled_shard_order
from .parquet import ParquetDataLoader, ParquetDataset, ParquetSample, _integer


class BucketDataLoader:
    def __init__(
        self,
        dataset_or_path,
        key="aspect",
        target_pixel_area=None,
        target_resolution_multiple=64,
        round_to=2,
        sampling_strategy="sequential",
        load_file_data=True,
        max_file_size=50_000_000,
        hf_token=None,
        chunk_size_mb=10,
        lazy_load=True,
        shard_batch_size=10,
        batch_size=None,
        *,
        shuffle_seed=None,
        format="tar",
    ):
        from . import discover_dataset

        if isinstance(dataset_or_path, (str, Path)):
            dataset_or_path = discover_dataset(
                str(dataset_or_path), hf_token=hf_token, format=format
            )
        if sampling_strategy not in ("sequential", "random_within_buckets", "fully_random"):
            raise ValueError("Unknown bucket sampling_strategy")
        self.parquet = isinstance(dataset_or_path, ParquetDataset)
        self.dataset = dataset_or_path
        self.loader = (
            ParquetDataLoader(dataset_or_path)
            if self.parquet
            else TarDataLoader(
                dataset_or_path,
                load_file_data=load_file_data,
                max_file_size=max_file_size,
                hf_token=hf_token,
                chunk_size_mb=chunk_size_mb,
            )
        )
        self.count = dataset_or_path.num_shards
        self.count = self.count() if callable(self.count) else self.count
        self.geometry = dict(
            key=key,
            target_pixel_area=target_pixel_area,
            target_resolution_multiple=target_resolution_multiple,
            round_to=round_to,
        )
        if self.parquet:
            self.geometry["metadata_fields"] = []
        self.strategy = sampling_strategy
        self.lazy_load = lazy_load
        self.group = _integer(shard_batch_size, "shard_batch_size", 1)
        self.batch_size = batch_size
        self.seed = shuffle_seed
        self.max_file_size = max_file_size
        self.load_file_data = load_file_data
        self.prefill = None
        self.reset()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        self._batch_size = None if size is None else _integer(size, "batch_size", 1)

    def reset(self):
        self.close()
        self.order = (
            range(self.count) if self.seed is None else _shuffled_shard_order(self.count, self.seed)
        )
        self.stream_seed = self.seed if self.seed is not None else random.getrandbits(64)
        self.group_start = None
        self.group_offset = 0
        self.group_revision = None
        self.prior_counts = {}
        self._open_prefill()
        self.processed = 0
        self.counts = Counter()
        self.pending = defaultdict(deque)
        self.current = None
        self.refs = self._groups()
        self.lookahead = next(self.refs, None)

    def _groups(self, start=0, skip=0):
        size = (
            self.group if self.lazy_load and self.strategy != "fully_random" else max(1, self.count)
        )
        for first in range(start, self.count, size):
            self.group_start = first
            self.prior_counts = dict(self.counts)
            rng = random.Random(f"{self.stream_seed}:{first}")
            digest = sha256()
            buckets = defaultdict(list)
            for slot in range(first, min(first + size, self.count)):
                shard = self.order[slot]
                (info,) = self.loader.list_shard_sample_aspect_buckets([shard], **self.geometry)
                for key, entries in sorted(info["buckets"].items()):
                    refs = [
                        (shard, row["sample_idx"], first)
                        for row in entries
                        if (row.get("size") or row.get("stored_bytes") or 0) <= self.max_file_size
                    ]
                    digest.update(json.dumps([key, refs]).encode())
                    buckets[key].extend(refs)
                    self.counts[key] += len(refs)
                self.processed += 1
            if self.strategy == "fully_random":
                mixed = [(k, ref) for k in sorted(buckets) for ref in buckets[k]]
                rng.shuffle(mixed)
                ordered = mixed
            else:
                if self.strategy == "random_within_buckets":
                    for k in sorted(buckets):
                        rng.shuffle(buckets[k])
                ordered = ((k, ref) for k in sorted(buckets) for ref in buckets[k])
            self.group_revision = digest.hexdigest()
            for self.group_offset, item in enumerate(ordered):
                if self.group_offset >= skip:
                    yield item
            skip = 0
        self.group_start = None

    def _open_prefill(self):
        self.prefill_position = -1
        if self.load_file_data and self.lazy_load and self.strategy != "fully_random":
            self.prefill = self.dataset.sequential_prefill(
                active_shards=self.group, shuffle_seed=self.seed
            )

    def close(self):
        if self.prefill is not None:
            self.prefill.close()
            self.prefill = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @cached_property
    def source_identity(self):
        if self.parquet:
            return self.dataset.fingerprint
        return sha256(json.dumps(self.dataset.shard_locations()).encode()).hexdigest()

    def _layout(self):
        return [
            self.source_identity,
            self.geometry,
            self.strategy,
            self.group,
            self.lazy_load,
            self.batch_size,
            self.seed,
            self.max_file_size,
        ]

    def state_dict(self):
        return dict(
            version=1,
            layout=self._layout(),
            seed=self.stream_seed,
            group=self.group_start,
            offset=self.group_offset,
            revision=self.group_revision,
            current=self.current,
            counts=dict(self.counts if self.group_start is None else self.prior_counts),
            pending={k: list(v) for k, v in self.pending.items()},
        )

    def load_state_dict(self, state):
        if state.get("version") != 1 or state["layout"] != self._layout():
            raise ValueError("Bucket checkpoint requires the same dataset and sampling settings")
        self.close()
        start = state["group"]
        self.stream_seed = state["seed"]
        self.counts = Counter(state["counts"])
        self.pending = defaultdict(
            deque, {k: deque(map(tuple, v)) for k, v in state["pending"].items()}
        )
        self.current = state["current"]
        self.processed = self.count if start is None else start
        self.group_start = start
        self.group_offset = state["offset"]
        self.group_revision = state["revision"]
        if start is not None:
            _integer(start, "group")
            _integer(self.group_offset, "offset")
            if start >= self.count:
                raise ValueError("Invalid bucket checkpoint group")
        if start is not None or self.pending:
            self._open_prefill()
        self.refs = iter(()) if start is None else self._groups(start, self.group_offset)
        self.lookahead = next(self.refs, None)
        if self.group_start != start or self.group_revision != state["revision"]:
            raise ValueError("Bucket membership changed since the checkpoint")

    def _pop(self):
        item = self.lookahead
        self.lookahead = next(self.refs, None)
        return item

    def _load(self, refs):
        if self.prefill is not None:
            first = max(group for _, _, group in refs)
            if first > self.prefill_position:
                self.prefill.advance(first)
                self.prefill_position = first
        refs = [(shard, row) for shard, row, _ in refs]
        if self.parquet:
            if self.load_file_data:
                return [s for s in self.loader.load_samples(refs) if s.size <= self.max_file_size]
            result = []
            for shard, row in refs:
                metadata = self.loader.get_metadata(shard, row)
                result.append(
                    ParquetSample(f"{metadata['source_path']}/{row:012d}", None, metadata, {})
                )
            return result
        return [
            entry
            for shard, row in refs
            if (entry := self.loader.load_sample(shard, row, load_sidecars=False)) is not None
        ]

    def __iter__(self):
        return self

    def __next__(self):
        while self.pending or self.lookahead is not None:
            if self.pending:
                key = next(iter(self.pending))
                ref = self.pending[key].popleft()
                if not self.pending[key]:
                    del self.pending[key]
            else:
                key, ref = self._pop()
            if samples := self._load([ref]):
                self.current = key
                return samples[0]
        self.close()
        raise StopIteration

    def next_batch(self):
        size = self.batch_size or 1
        while self.pending or self.lookahead is not None:
            key = next((k for k, rows in self.pending.items() if len(rows) >= size), None)
            while key is None and self.lookahead is not None:
                k, ref = self._pop()
                self.pending[k].append(ref)
                if len(self.pending[k]) >= size:
                    key = k
            if key is None:
                key = next(iter(self.pending))
            rows = self.pending[key]
            refs = [rows.popleft() for _ in range(min(size, len(rows)))]
            if not rows:
                del self.pending[key]
            if samples := self._load(refs):
                self.current = key
                return samples
        self.close()
        return None

    def iter_batches(self):
        if self.batch_size is None:
            raise ValueError("batch_size must be set to use iter_batches()")
        return iter(self.next_batch, None)

    def get_current_bucket(self):
        return (
            self.current
            if self.current is not None
            else (self.lookahead[0] if self.lookahead else None)
        )

    def skip_to_bucket(self, bucket_key):
        self.reset()
        while self.lookahead is not None:
            if self.lookahead[0] == bucket_key:
                return
            self._pop()
        raise ValueError(f"Bucket {bucket_key!r} not found")

    def get_bucket_stats(self):
        counts = self.counts
        return dict(
            num_buckets=len(counts),
            sampling_strategy=self.strategy.title().replace("_", ""),
            lazy_load=self.lazy_load,
            shards_processed=self.processed,
            total_shards=self.count,
            batch_size=self.batch_size,
            total_files_loaded=counts.total(),
            min_files_per_bucket=min(counts.values(), default=0),
            max_files_per_bucket=max(counts.values(), default=0),
            avg_files_per_bucket=counts.total() / len(counts) if counts else 0,
            bucket_details=dict(counts),
        )
