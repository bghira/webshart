"""Resumable concurrent image geometry scans with row-keyed Parquet overlays."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import tempfile
import random
import time

from .parquet import MANIFEST, ParquetDataset, _integer
from .url_images import URLImageClient, URLImageOptions


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _upload_capture(api, **kwargs):
    # Retry commit contention and transient service failures without refetching.
    for attempt in range(8):
        try:
            return api.upload_file(**kwargs)
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if attempt == 7 or status is None or not (status in (408, 409, 429) or 500 <= status < 600):
                raise
            time.sleep(min(30, 2 ** attempt) + random.random())


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
    capture_images: bool = False,
    recompress_images: bool = False,
    recompress_workers: int = 2,
    upload_repo: str | None = None,
    upload_repo_type: str = "dataset",
    upload_endpoint: str | None = None,
    upload_token: str | None = None,
    keep_local_images: bool = True,
) -> dict:
    """Checkpoint geometry and availability, optionally capturing image Parquet shards.

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
    recompress_workers = _integer(recompress_workers, "recompress_workers", 1)
    if recompress_images and not capture_images:
        raise ValueError("recompress_images requires capture_images")
    if upload_repo and not capture_images:
        raise ValueError("upload_repo requires capture_images")
    if not keep_local_images and not upload_repo:
        raise ValueError("Discarding local images requires upload_repo")
    if upload_repo_type not in {"dataset", "model"}:
        raise ValueError("upload_repo_type must be dataset or model")
    api = None
    if upload_repo:
        from huggingface_hub import HfApi
        api = HfApi(endpoint=upload_endpoint, token=upload_token)
        api.repo_info(upload_repo, repo_type=upload_repo_type)
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
        if dataset._source_repo.endpoint:
            source["endpoint"] = dataset._source_repo.endpoint
        if dataset._source_repo.repo_type != "dataset":
            source["repo_type"] = dataset._source_repo.repo_type
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
        if capture_images:
            identity["capture"] = {
                "format": "parquet-image-bytes-v1",
                "upload_repo": upload_repo,
                "upload_repo_type": upload_repo_type,
                "upload_endpoint": api.endpoint if api else None,
                "keep_local_images": keep_local_images,
            }
        if recompress_images:
            identity["recompression"] = "jxl-jpeg-lossless-effort3-verified-v1"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["identity"] != identity:
                raise ValueError(
                    "Scan source or options changed; use a new destination"
                )
        else:
            state = {"identity": identity}
            _atomic_json(state_path, state)
        # A previous process may have exited while native writers were staging bytes.
        for stale in root.glob(".images-*"):
            if stale.is_dir():
                shutil.rmtree(stale)
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

        storage_fields = [
            ("stored_format", pa.string()), ("stored_bytes", pa.int64()),
            ("stored_sha256", pa.string()), ("recompression_status", pa.string()),
        ]
        if recompress_images:
            schema = pa.schema(list(schema) + storage_fields)
        image_schema = pa.schema(list(schema) + ([] if recompress_images else storage_fields) + [
            ("source_shard", pa.int64()), ("url", pa.string()),
            ("captions", pa.string()),
            ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ])

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
            if capture_images and keep_local_images:
                for part in progress["parts"]:
                    image_path = root / part["image_path"]
                    if not image_path.is_file() or image_path.stat().st_size != part["image_size"]:
                        raise ValueError("Missing or changed captured image part")
                    with image_path.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != part["image_sha256"]:
                            raise ValueError("Changed captured image part")
            if not keep_local_images:
                for part in progress["parts"]:
                    if part.get("image_path"):
                        (root / part["image_path"]).unlink(missing_ok=True)
            reader = dataset._file(index)
            count = reader.info()["num_rows"]
            progress["num_rows"] = count
            stop = (
                count
                if max_samples is None
                else min(count, progress["next_row"] + max_samples - attempted)
            )
            rows = reader.rows(
                columns=list(dict.fromkeys([dataset.image_url_column] +
                    ([dataset.caption_column] if capture_images and dataset.caption_column else []))),
                start=progress["next_row"],
                stop=stop,
            )
            while progress["next_row"] < stop:
                start = progress["next_row"]
                size = min(batch_size, stop - start)
                relative = f"parts/{index:06d}/{start:012d}.parquet"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image_relative = f"data/{index:06d}/{start:012d}.parquet"
                image_path = root / image_relative
                # A pending upload keeps its original batch boundaries on resume.
                pending = None
                if capture_images and image_path.is_file() and path.is_file():
                    pending = pq.read_table(path).to_pylist()
                    if not pending or len(pending) > stop - start:
                        pending = None
                    elif [row["source_row"] for row in pending] != list(range(start, start + len(pending))):
                        raise ValueError("Invalid staged capture row identities")
                if pending is not None:
                    size = len(pending)
                batch = [next(rows) for _ in range(size)]
                if pending is not None:
                    results = pending
                else:
                    with tempfile.TemporaryDirectory(prefix=".images-", dir=root) as staging:
                        scan_kwargs = {"workers": workers, "max_pixels": max_pixels}
                        if capture_images:
                            scan_kwargs["capture_directory"] = staging
                        if recompress_images:
                            scan_kwargs.update(recompress_images=True, recompress_workers=recompress_workers)
                        results = client.scan(
                            [row[dataset.image_url_column] for row in batch], **scan_kwargs
                        )
                        for offset, result in enumerate(results):
                            result["source_row"] = start + offset
                        if capture_images:
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            temporary = image_path.with_suffix(".tmp")
                            captured_schema = image_schema.with_metadata({
                                "webshart.source": json.dumps(source),
                                "webshart.source_path": dataset._shard(index)["path"],
                            })
                            with pq.ParquetWriter(temporary, captured_schema) as writer:
                                captured_rows, captured_bytes = [], 0
                                for offset, (row, result) in enumerate(zip(batch, results, strict=True)):
                                    blob = Path(staging) / f"{offset}.bin"
                                    captured = dict(result, source_shard=index,
                                        url=row[dataset.image_url_column],
                                        captions=row.get(dataset.caption_column),
                                        image={"bytes": blob.read_bytes(), "path": None} if blob.exists() else None)
                                    captured_rows.append(captured)
                                    captured_bytes += blob.stat().st_size if blob.exists() else 0
                                    if captured_bytes >= 16 * 1024 * 1024 or len(captured_rows) >= 32:
                                        writer.write_table(pa.Table.from_pylist(captured_rows, schema=image_schema))
                                        captured_rows, captured_bytes = [], 0
                                if captured_rows:
                                    writer.write_table(pa.Table.from_pylist(captured_rows, schema=image_schema))
                            temporary.replace(image_path)
                        temporary = path.with_suffix(".tmp")
                        pq.write_table(pa.Table.from_pylist(results, schema=schema), temporary)
                        temporary.replace(path)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                part = {"path": relative, "size": path.stat().st_size, "sha256": digest}
                if capture_images:
                    with image_path.open("rb") as stream:
                        image_digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    part.update(image_path=image_relative, image_sha256=image_digest,
                                image_size=image_path.stat().st_size)
                    if api:
                        commit = _upload_capture(api,
                            repo_id=upload_repo, repo_type=upload_repo_type,
                            path_in_repo=image_relative, path_or_fileobj=str(image_path),
                            commit_message=f"Capture shard {index} rows {start}-{start + size}",
                        )
                        part["upload_commit"] = commit.oid
                progress["parts"].append(part)
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
                if capture_images and not keep_local_images:
                    image_path.unlink(missing_ok=True)
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
