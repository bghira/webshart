"""Resumable concurrent image geometry scans with row-keyed Parquet overlays."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from .parquet import MANIFEST, ParquetDataset, _integer
from .url_images import URLImageClient, URLImageOptions


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def scan_url_metadata(
    dataset: ParquetDataset,
    destination: str,
    *,
    workers: int = 32,
    batch_size: int = 1000,
    max_samples: int | None = None,
    start_shard: int = 0,
    stop_shard: int | None = None,
    url_options: URLImageOptions | None = None,
    max_pixels: int = 64_000_000,
) -> dict:
    """Fetch and discard images, checkpointing geometry and availability per batch.

    ``max_samples`` limits new attempts in this invocation. Resume with the same
    source revision, shard range and fetch options. Worker count and invocation
    limits may change. Use disjoint shard ranges and separate destinations for
    multiple processes. Completed attempts, including failures, are not retried
    on resume. Failed attempts can be rescanned into a fresh destination.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("URL metadata scans require webshart[scan]") from error
    if not isinstance(dataset, ParquetDataset) or not dataset.image_url_column:
        raise ValueError("Scan requires a ParquetDataset with image_url_column")
    if dataset._metadata_repo is not None:
        raise ValueError("Scan the source dataset without an existing metadata overlay")
    workers = _integer(workers, "workers", 1)
    batch_size = _integer(batch_size, "batch_size", 1)
    max_pixels = _integer(max_pixels, "max_pixels", 1)
    if max_samples is not None:
        _integer(max_samples, "max_samples", 1)
    start_shard = _integer(start_shard, "start_shard")
    stop_shard = (
        dataset.num_shards if stop_shard is None else _integer(stop_shard, "stop_shard")
    )
    if not 0 <= start_shard < stop_shard <= dataset.num_shards:
        raise ValueError("Invalid shard range")
    options = url_options or URLImageOptions()
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "scan-state.json"
    # A separate SQLite transaction is a process lock released automatically on exit.
    lock = sqlite3.connect(root / ".scan-lock.sqlite", timeout=0)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        if not state_path.exists() and any(
            p.name not in {".scan-lock.sqlite", ".scan-lock.sqlite-journal"}
            for p in root.iterdir()
        ):
            raise ValueError("New scan destination must be empty")
        source = {
            "repo_id": dataset._source_repo.source,
            "revision": dataset.source_revision,
        }
        shards = []
        for index in range(start_shard, stop_shard):
            record = dataset._shard(index)
            shard = {"path": record["path"], "size": record["size"]}
            if dataset._source_repo.local:
                with open(dataset._source_repo.location(shard["path"]), "rb") as stream:
                    shard["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            shards.append(shard)
        if dataset._source_repo.local:
            source = {"kind": "local"}
        identity = {
            "version": 1,
            "source": source,
            "shards": shards,
            "start_shard": start_shard,
            "stop_shard": stop_shard,
            "image_url_column": dataset.image_url_column,
            "caption_column": dataset.caption_column,
            "options": asdict(options),
            "max_pixels": max_pixels,
        }
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["identity"] != identity:
                raise ValueError(
                    "Scan source or options changed; use a new destination"
                )
        else:
            state = {"identity": identity}
            _atomic_json(state_path, state)
        schema = pa.schema(
            [
                ("source_row", pa.int64()),
                ("width", pa.int64()),
                ("height", pa.int64()),
                ("url_status", pa.string()),
                ("url_http_status", pa.int64()),
                ("download_bytes", pa.int64()),
                ("download_sha256", pa.string()),
            ]
        )

        def checkpoint(index):
            return root / "checkpoints" / f"{index:06d}.json"

        def read_progress(index):
            path = checkpoint(index)
            return json.loads(path.read_text()) if path.exists() else None

        attempted = 0
        client = URLImageClient(options)
        for index in range(start_shard, stop_shard):
            if max_samples is not None and attempted >= max_samples:
                break
            progress = read_progress(index) or {
                "next_row": 0,
                "parts": [],
                "counts": {},
            }
            for part in progress["parts"]:
                path = root / part["path"]
                if not path.is_file() or path.stat().st_size != part["size"]:
                    raise ValueError("Missing or changed scan checkpoint part")
                with path.open("rb") as stream:
                    if (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        != part["sha256"]
                    ):
                        raise ValueError("Changed scan checkpoint part")
            reader = dataset._file(index)
            count = reader.info()["num_rows"]
            progress["num_rows"] = count
            stop = (
                count
                if max_samples is None
                else min(count, progress["next_row"] + max_samples - attempted)
            )
            rows = reader.rows(
                columns=[dataset.image_url_column],
                start=progress["next_row"],
                stop=stop,
            )
            while progress["next_row"] < stop:
                start = progress["next_row"]
                size = min(batch_size, stop - start)
                batch = [
                    (start + offset, next(rows)[dataset.image_url_column])
                    for offset in range(size)
                ]
                results = client.scan(
                    [url for _, url in batch], workers=workers, max_pixels=max_pixels
                )
                for (row, _), result in zip(batch, results, strict=True):
                    result["source_row"] = row
                relative = f"parts/{index:06d}/{start:012d}.parquet"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                pq.write_table(pa.Table.from_pylist(results, schema=schema), temporary)
                temporary.replace(path)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                progress["parts"].append(
                    {"path": relative, "size": path.stat().st_size, "sha256": digest}
                )
                progress["next_row"] += size
                attempted += size
                counts = Counter(progress["counts"])
                counts.update(
                    f"http_{row['url_http_status']}"
                    if row["url_status"] == "http_error"
                    else row["url_status"]
                    for row in results
                )
                progress["counts"] = dict(counts)
                checkpoint(index).parent.mkdir(exist_ok=True)
                _atomic_json(checkpoint(index), progress)
            if count == 0:
                checkpoint(index).parent.mkdir(exist_ok=True)
                _atomic_json(checkpoint(index), progress)
        manifest = {
            "format": "webshart-parquet",
            "version": 1,
            "source": source,
            "metadata_mode": "overlay",
            "columns": {
                "image_url": dataset.image_url_column,
                "captions": dataset.caption_column,
                "width": "width",
                "height": "height",
            },
            "shards": [],
        }
        total = 0
        counts = Counter()
        complete = True
        for index in range(start_shard, stop_shard):
            shard = dict(dataset._shard(index))
            progress = read_progress(index)
            if progress:
                total += progress["next_row"]
                counts.update(progress["counts"])
                complete = complete and progress["next_row"] == progress["num_rows"]
            else:
                complete = False
            if progress and progress["parts"]:
                relative = f"metadata/{index:06d}.parquet"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                with pq.ParquetWriter(temporary, schema) as writer:
                    for part in progress["parts"]:
                        writer.write_table(pq.ParquetFile(root / part["path"]).read())
                temporary.replace(path)
                shard["metadata_path"] = relative
            manifest["shards"].append(shard)
        _atomic_json(root / MANIFEST, manifest)
        summary = {
            "attempted": total,
            "new_attempts": attempted,
            "counts": dict(counts),
            "complete": complete,
        }
        _atomic_json(root / "scan-summary.json", summary)
        return summary
    finally:
        lock.close()
