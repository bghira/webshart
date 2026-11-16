"""Authenticated S3 transport and captured-Parquet training reads."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from webshart import ParquetDataset, ParquetDataLoader
from webshart._webshart import _ParquetFile, _S3Store


@contextmanager
def s3_server(objects):
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


def test_signed_capture_discovery_projection_batch_and_cache(tmp_path):
    objects = capture(tmp_path)
    with s3_server(objects) as (options, calls, _):
        dataset = ParquetDataset(
            "s3://test-bucket", manifest="combined/dataset.json", s3_options=options
        )
        assert dataset.num_shards == 1
        assert not any(key == "parts/part.parquet" for _, key, _ in calls)
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
            store.read_jsonl("catalog/manifest.jsonl", expected_sha256="0" * 64)


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
