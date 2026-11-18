"""Authenticated S3 transport and captured-Parquet training reads."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from webshart import ParquetDataset, ParquetDataLoader
from webshart._webshart import _ParquetFile, _S3Store


@contextmanager
def s3_server(objects, hashes=None):
    calls = []
    etags = {
        key: '"' + hashlib.md5(data).hexdigest() + '"' for key, data in objects.items()
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            key = self.path.removeprefix("/test-bucket/")
            calls.append((self.command, key, dict(self.headers)))
            if not self.headers.get("Authorization", "").startswith(
                "AWS4-HMAC-SHA256 "
            ):
                self.send_error(403)
                return
            if key not in objects:
                self.send_error(404)
                return
            if self.headers.get("If-Match", etags[key]) != etags[key]:
                self.send_error(412)
                return
            data = objects[key]
            span = self.headers.get("Range")
            self.send_response(206 if span else 200)
            if span:
                start, end = (int(v) for v in span.removeprefix("bytes=").split("-"))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                data = data[start : end + 1]
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", etags[key])
            if hashes and key in hashes:
                self.send_header("x-amz-meta-sha256", hashes[key])
            self.send_header("Last-Modified", "Mon, 01 Jan 2024 00:00:00 GMT")
            self.end_headers()
            if body:
                self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    options = dict(
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        allow_http=True,
        aws_access_key_id="test-access",
        aws_secret_access_key="test-secret",
        aws_session_token="test-session",
        max_connections=4,
    )
    try:
        yield options, calls, etags
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def capture(tmp_path):
    path = tmp_path / "part.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    image=dict(bytes=b"first", path=None),
                    captions="one",
                    width=8,
                    height=4,
                    url_status="ok",
                    stored_bytes=5,
                    stored_format="jxl",
                ),
                dict(
                    image=None,
                    captions="dead",
                    width=None,
                    height=None,
                    url_status="http_error",
                    stored_bytes=None,
                    stored_format=None,
                ),
                dict(
                    image=dict(bytes=b"third", path=None),
                    captions="three",
                    width=4,
                    height=4,
                    url_status="ok",
                    stored_bytes=5,
                    stored_format="original",
                ),
                dict(
                    image=dict(bytes=b"fourth", path=None),
                    captions="four",
                    width=4,
                    height=4,
                    url_status="ok",
                    stored_bytes=6,
                    stored_format="original",
                ),
            ]
        ),
        path,
        row_group_size=2,
    )
    manifest = (
        json.dumps(
            dict(key="parts/part.parquet", size_bytes=path.stat().st_size, num_rows=4)
        )
        + "\n"
    ).encode()
    descriptor = json.dumps(
        dict(
            format="parquet-image-capture",
            version=2,
            files_manifest="catalog/manifest.jsonl",
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        )
    ).encode()
    return {
        "parts/part.parquet": path.read_bytes(),
        "combined/dataset.json": descriptor,
        "catalog/manifest.jsonl": manifest,
    }


def _open_shared_capture(options, cache):
    dataset = ParquetDataset(
        "s3://test-bucket",
        manifest="combined/dataset.json",
        s3_options=options,
        catalogue_cache_dir=cache,
    )
    return dataset.fingerprint, dataset.physical_row_count


def test_capture_catalogue_is_published_once_and_reused_across_processes(tmp_path):
    objects = capture(tmp_path)
    cache = tmp_path / "shared"
    with s3_server(objects) as (options, calls, etags):
        with ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(_open_shared_capture, options, cache) for _ in range(8)
            ]
            opened = [future.result(timeout=60) for future in futures]
        assert len(set(opened)) == 1
        assert opened[0][1] == 4
        assert (
            sum(
                method == "GET" and key == "catalog/manifest.jsonl"
                for method, key, _ in calls
            )
            == 1
        )
        assert len(list(cache.glob("*.sqlite3"))) == 1

        dataset = ParquetDataset(
            "s3://test-bucket",
            manifest="combined/dataset.json",
            s3_options=options,
            catalogue_cache_dir=cache,
        )
        assert ParquetDataLoader(dataset).load_sample(0, 0).data == b"first"
        assert "etag" in dataset.get_shard_info(0)
        other = ParquetDataset(
            "s3://test-bucket",
            manifest="combined/dataset.json",
            s3_options=options,
            catalogue_cache_dir=cache,
        )
        assert "etag" not in other.get_shard_info(0)
        assert (
            sum(
                method == "GET" and key == "catalog/manifest.jsonl"
                for method, key, _ in calls
            )
            == 1
        )

        manifest_key = "catalog/manifest.jsonl"
        replaced = objects[manifest_key].replace(b'"num_rows": 4', b'"num_rows": 5')
        descriptor = json.loads(objects["combined/dataset.json"])
        descriptor["manifest_sha256"] = hashlib.sha256(replaced).hexdigest()
        for key, data in (
            (manifest_key, replaced),
            ("combined/dataset.json", json.dumps(descriptor).encode()),
        ):
            objects[key] = data
            etags[key] = '"' + hashlib.md5(data).hexdigest() + '"'
        revised = _open_shared_capture(options, cache)
        assert revised[1] == 5
        assert revised[0] != opened[0][0]
        assert dataset.physical_row_count == 4
        assert len(list(cache.glob("*.sqlite3"))) == 2


def test_failed_capture_index_build_does_not_publish(tmp_path):
    objects = capture(tmp_path)
    objects["catalog/manifest.jsonl"] += b"\n"
    cache = tmp_path / "shared"
    with s3_server(objects) as (options, _, _):
        with pytest.raises(ValueError, match="Manifest SHA-256"):
            _open_shared_capture(options, cache)
    assert not list(cache.glob("*.sqlite3"))


def test_signed_capture_discovery_projection_batch_and_cache(tmp_path):
    objects = capture(tmp_path)
    with s3_server(objects) as (options, calls, _):
        dataset = ParquetDataset(
            "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
        )
        assert dataset.num_shards == 1
        assert dataset.physical_row_count == len(dataset) == 4
        assert dataset.get_shard_position("parts/part.parquet") == 0
        assert dataset.get_shard_position("parts/missing.parquet") is None
        assert not any(key == "parts/part.parquet" for _, key, _ in calls)
        assert dataset.count_rows(exclude_null_images=True) == 3
        assert all(
            headers.get("range")
            for method, key, headers in calls
            if method == "GET" and key == "parts/part.parquet"
        )
        loader = ParquetDataLoader(dataset)
        assert [r["sample_idx"] for r in loader.list_samples_in_shard(0)] == [0, 2, 3]
        assert set(loader.list_shard_sample_aspect_buckets([0])[0]["buckets"]) == {
            "2.00",
            "1.00",
        }
        assert loader.get_metadata(0, 2)["captions"] == "three"
        assert [
            s.data for s in loader.load_samples([(0, 3), (0, 0), (0, 3), (0, 1)])
        ] == [b"fourth", b"first", b"fourth", None]
        assert all(h.get("x-amz-security-token") == "test-session" for _, _, h in calls)
        assert any(
            h.get("range") and h.get("if-match")
            for method, key, h in calls
            if method == "GET" and key.endswith(".parquet")
        )
        dataset.enable_shard_cache(str(tmp_path / "cache"), 0.01)
        assert loader.load_sample(0, 2).data == b"third"
        before = len(
            [
                1
                for method, key, _ in calls
                if method == "GET" and key.endswith(".parquet")
            ]
        )
        dataset._files.clear()
        assert loader.load_sample(0, 3).data == b"fourth"
        assert (
            len(
                [
                    1
                    for method, key, _ in calls
                    if method == "GET" and key.endswith(".parquet")
                ]
            )
            == before
        )
        assert len(list((tmp_path / "cache").glob("*.parquet"))) == 1
        stats = dataset.io_stats()
        assert stats["head_requests"] == sum(method == "HEAD" for method, _, _ in calls)
        assert stats["range_requests"] == sum(
            method == "GET" and "range" in headers for method, _, headers in calls
        )
        assert stats["downloads"] == sum(
            method == "GET" and "range" not in headers for method, _, headers in calls
        )
        assert stats["download_bytes"] == len(objects["catalog/manifest.jsonl"]) + len(
            objects["parts/part.parquet"]
        )
        assert stats["range_bytes"] > 0
        assert stats["list_operations"] == 0


def test_s3_projection_reuses_a_shard_populated_by_another_reader(tmp_path):
    objects = capture(tmp_path)
    cache = tmp_path / "cache"
    path = "parts/part.parquet"
    with s3_server(objects) as (options, calls, etags):

        def open_dataset():
            dataset = ParquetDataset(
                "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
            )
            dataset.enable_shard_cache(str(cache), 0.01)
            return dataset

        def projection(dataset):
            return dataset.read_metadata_rows(0, [3, 0, 3], fields=["captions"])

        projecting = open_dataset()
        before = len(calls)
        expected = projection(projecting)
        assert [row["captions"] for row in expected] == ["four", "one", "four"]
        assert all(set(row) == {"captions", "source_path", "source_row"} for row in expected)
        reads = [
            headers
            for method, key, headers in calls[before:]
            if method == "GET" and key == path
        ]
        assert reads and all(headers.get("range") for headers in reads)
        assert not list(cache.glob("*.parquet"))

        payload_reader = open_dataset()
        assert ParquetDataLoader(payload_reader).load_sample(0, 3).data == b"fourth"
        assert len(list(cache.glob("*.parquet"))) == 1

        before = len(calls)
        assert projection(projecting) == expected
        assert len(calls) == before

        reopened = open_dataset()
        revision = projecting.source_revision_for_shard(0)
        reopened.bind_source_revisions({0: revision}.get)
        assert reopened.source_revision_for_shard(0) == revision
        before = len(calls)
        assert projection(reopened) == expected
        assert len(calls) == before

        etags[path] = '"replacement"'
        replaced = open_dataset()
        replaced.source_revision_for_shard(0)
        before = len(calls)
        assert projection(replaced) == expected
        assert any(method == "GET" and key == path for method, key, _ in calls[before:])
        assert len(list(cache.glob("*.parquet"))) == 1
        replaced.bind_source_revisions({0: revision}.get)
        with pytest.raises(ValueError, match="accepted dataset view"):
            projection(replaced)


def test_s3_limits_missing_and_etag_change(tmp_path):
    objects = capture(tmp_path)
    with s3_server(objects) as (options, calls, etags):
        store = _S3Store("test-bucket", **options)
        with pytest.raises(FileNotFoundError):
            store.head("missing")
        with pytest.raises(ValueError, match="max_bytes"):
            store.read("parts/part.parquet", max_bytes=1)
        reader = _ParquetFile("s3://test-bucket/parts/part.parquet", s3=store)
        etags["parts/part.parquet"] = '"changed"'
        with pytest.raises(OSError, match="S3"):
            list(reader.rows(columns=["image"]))
        with pytest.raises(ValueError, match="SHA-256"):
            key = "combined/dataset.json"
            descriptor = json.loads(objects[key])
            objects[key] = json.dumps(
                {**descriptor, "manifest_sha256": "0" * 64}
            ).encode()
            ParquetDataset("s3://test-bucket", manifest=key, s3_options=options)


def test_batch_reads_share_row_groups_and_preserve_order(tmp_path):
    objects = capture(tmp_path)
    path = tmp_path / "local.parquet"
    path.write_bytes(objects["parts/part.parquet"])
    reader = _ParquetFile(str(path))
    before = reader.bytes_read
    rows = reader.rows_at([3, 2, 3, 0], columns=["image"], parallel_reads=2)
    batch_bytes = reader.bytes_read - before
    assert [row["image"]["bytes"] for row in rows] == [
        b"fourth",
        b"third",
        b"fourth",
        b"first",
    ]
    before = reader.bytes_read
    for index in [3, 2, 3, 0]:
        list(reader.rows(columns=["image"], start=index, stop=index + 1))
    assert reader.bytes_read - before > batch_bytes
    with pytest.raises(IndexError):
        reader.rows_at([4])
    with pytest.raises(ValueError):
        reader.rows_at([0], parallel_reads=0)


def test_capture_catalogue_supports_parallel_reads(tmp_path):
    objects = capture(tmp_path)
    with s3_server(objects) as (options, _, _):
        dataset = ParquetDataset(
            "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
        )
        start = threading.Barrier(8)

        def visit(worker):
            start.wait(timeout=10)
            for index in range(200):
                assert dataset.get_shard_info(0)["num_rows"] == 4
                if (index + worker) % 32 == 0:
                    row = dataset.read_metadata_rows(0, [0], fields=("captions",))
                    assert row[0]["captions"] == "one"

        with ThreadPoolExecutor(max_workers=8) as executor:
            for _ in executor.map(visit, range(8)):
                pass


def test_capture_manifest_rejects_unsafe_paths_and_wrong_sizes(tmp_path):
    objects = capture(tmp_path)
    objects["catalog/manifest.jsonl"] = (
        b'{"key":"../escape.parquet","size_bytes":10,"num_rows":1}\n'
    )
    descriptor = json.loads(objects["combined/dataset.json"])
    descriptor.pop("manifest_sha256")
    objects["combined/dataset.json"] = json.dumps(descriptor).encode()
    with s3_server(objects) as (options, _, _):
        with pytest.raises(ValueError, match="Unsafe"):
            ParquetDataset(
                "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
            )


def test_metadata_uses_ranges_without_downloading_image_cache(tmp_path):
    with s3_server(capture(tmp_path)) as (options, calls, _):
        dataset = ParquetDataset(
            "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
        )
        dataset.enable_shard_cache(str(tmp_path / "cache"), 0.01)
        loader = ParquetDataLoader(dataset)
        assert loader.get_metadata(0, 0)["width"] == 8
        assert not list((tmp_path / "cache").glob("*.parquet"))
        assert all(
            headers.get("range")
            for method, key, headers in calls
            if method == "GET" and key.endswith(".parquet")
        )
        assert loader.load_sample(0, 0).data == b"first"
        assert len(list((tmp_path / "cache").glob("*.parquet"))) == 1


def test_capture_manifest_size_is_verified_on_first_open(tmp_path):
    objects = capture(tmp_path)
    descriptor = json.loads(objects["combined/dataset.json"])
    descriptor.pop("manifest_sha256")
    objects["combined/dataset.json"] = json.dumps(descriptor).encode()
    row = json.loads(objects["catalog/manifest.jsonl"])
    row["size_bytes"] += 1
    objects["catalog/manifest.jsonl"] = json.dumps(row).encode()
    with s3_server(objects) as (options, _, _):
        dataset = ParquetDataset(
            "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
        )
        with pytest.raises(ValueError, match="size"):
            dataset.get_metadata(0, 0)


def test_metadata_prefetch_coalesces_columns_and_respects_budget(tmp_path):
    objects = capture(tmp_path)
    with s3_server(objects) as (options, calls, _):
        reader = _ParquetFile(
            "s3://test-bucket/parts/part.parquet", s3=_S3Store("test-bucket", **options)
        )
        calls.clear()
        columns = [
            "captions",
            "width",
            "height",
            "url_status",
            "stored_bytes",
            "stored_format",
        ]
        assert reader.prefetch_columns(columns, max_bytes=0) is False
        assert calls == []
        assert reader.prefetch_columns(columns, parallel_reads=2)
        gets = sum(method == "GET" for method, _, _ in calls)
        assert gets == 2
        rows = reader.rows_at([3, 0, 2], columns=columns)
        assert [row["captions"] for row in rows] == ["four", "one", "three"]
        assert sum(method == "GET" for method, _, _ in calls) == gets


def test_projected_scan_coalesces_remote_metadata_ranges(tmp_path):
    with s3_server(capture(tmp_path)) as (options, calls, _):
        dataset = ParquetDataset(
            "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
        )
        dataset.get_metadata(0, 0)
        calls.clear()
        rows = list(
            ParquetDataLoader(dataset).iter_samples_in_shard(
                0, metadata_fields=["captions"], batch_size=1
            )
        )
        assert [row["captions"] for row in rows] == ["one", "three", "four"]
        gets = [headers for method, _, headers in calls if method == "GET"]
        assert len(gets) <= 5
        assert all(headers.get("range") for headers in gets)


def test_explicit_credentials_do_not_inherit_environment_session_token(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AWS_SESSION_TOKEN", "ambient-session")
    with s3_server(capture(tmp_path)) as (options, calls, _):
        options.pop("aws_session_token")
        _S3Store("test-bucket", **options).head("parts/part.parquet")
        assert "x-amz-security-token" not in calls[-1][2]


def test_environment_credentials_are_used_when_keys_are_omitted(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "environment-session")
    with s3_server(capture(tmp_path)) as (options, calls, _):
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
            options.pop(key)
        _S3Store("test-bucket", **options).head("parts/part.parquet")
        assert calls[-1][2]["x-amz-security-token"] == "environment-session"
        assert "Credential=environment-access/" in calls[-1][2]["authorization"]


def test_capture_hash_is_checked_on_new_process_equivalent_open(tmp_path):
    objects = capture(tmp_path)
    path = "parts/part.parquet"
    manifest = "catalog/manifest.jsonl"
    descriptor = "combined/dataset.json"
    original = hashlib.sha256(objects[path]).hexdigest()
    hashes = {path: original}
    record = json.loads(objects[manifest])
    objects[manifest] = (json.dumps({**record, "sha256": original}) + "\n").encode()
    objects[descriptor] = json.dumps(
        {
            **json.loads(objects[descriptor]),
            "manifest_sha256": hashlib.sha256(objects[manifest]).hexdigest(),
        }
    ).encode()
    with s3_server(objects, hashes) as (options, calls, etags):

        def reopen():
            return ParquetDataset(
                "s3://test-bucket", manifest=descriptor, s3_options=options
            )

        first = reopen()
        assert first.get_shard_info(0)["sha256"] == original
        assert first.read_metadata_rows(0, [2], fields=["width"])[0]["width"] == 4
        hashes[path] = "0" * 64
        etags[path] = '"replacement"'
        second = reopen()
        assert first.fingerprint == second.fingerprint
        calls.clear()
        with pytest.raises(ValueError, match="SHA-256 metadata"):
            second.read_metadata_rows(0, [2], fields=["width"])
        assert [(method, key) for method, key, _ in calls] == [("HEAD", path)]


def test_prepared_etag_pin_survives_reopen_without_object_checksum(tmp_path):
    objects = capture(tmp_path)
    with s3_server(objects) as (options, _, etags):

        def open_source():
            return ParquetDataset(
                "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
            )

        first = open_source()
        revision = first.source_revision_for_shard(0)
        second = open_source()
        etags["parts/part.parquet"] = '"new-object"'
        second.bind_source_revisions({0: revision}.get)
        with pytest.raises(ValueError, match="accepted dataset view"):
            second.read_metadata_rows(0, [0], fields=["width"])
