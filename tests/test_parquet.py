from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace
import sys

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

import webshart
from webshart._webshart import _ParquetFile


def write_source(root, rows=None, *, row_group_size=2, **kwargs):
    path = root / "task" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        rows
        if rows is not None
        else [
            {
                "id": 0,
                "image": b"first",
                "source_image": b"original",
                "captions": {"text": "one", "tags": ["a", "b"]},
                "width": 4,
                "height": 2,
            },
            {
                "id": 0,
                "image": b"second",
                "source_image": None,
                "captions": {"text": "two", "tags": []},
                "width": 8,
                "height": 4,
            },
            {
                "id": 1,
                "image": b"third",
                "source_image": b"other",
                "captions": {"text": "three", "tags": None},
                "width": 2,
                "height": 2,
            },
        ]
    )
    parquet.write_table(
        arrow.Table.from_pylist(rows), path, row_group_size=row_group_size, **kwargs
    )
    return path


def write_metadata(
    root, source_path, rows, *, mode="overlay", kind="json", **manifest_options
):
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / f"custom.{kind}"
    if kind == "json":
        metadata_path.write_text(json.dumps(rows))
    else:
        parquet.write_table(
            arrow.Table.from_pylist(rows),
            metadata_path,
            compression="zstd",
            row_group_size=1,
        )
    manifest = {
        "format": "webshart-parquet",
        "version": 1,
        "metadata_mode": mode,
        "shards": [
            {
                "path": "task/part.parquet",
                "size": source_path.stat().st_size,
                "num_rows": parquet.read_metadata(source_path).num_rows,
                "metadata_path": metadata_path.name,
                "metadata_sha256": hashlib.sha256(
                    metadata_path.read_bytes()
                ).hexdigest(),
            }
        ],
        **manifest_options,
    }
    (root / "webshart-parquet.json").write_text(json.dumps(manifest))
    return manifest


def test_native_nested_captions_pair_payloads_and_row_identity(tmp_path):
    write_source(tmp_path)
    dataset = webshart.discover_dataset(str(tmp_path), format="parquet")
    assert isinstance(dataset, webshart.ParquetDataset)
    assert len(dataset) == 3
    assert dataset.num_shards == 1
    assert len(set(dataset.list_samples_in_shard(0))) == 3
    loader = webshart.ParquetDataLoader(dataset, image_columns=["source_image"])
    entries = list(loader)
    assert [entry.data for entry in entries] == [b"first", b"second", b"third"]
    assert entries[0].caption == {"text": "one", "tags": ["a", "b"]}
    assert entries[0].payloads["source_image"] == b"original"
    assert entries[1].payloads["source_image"] is None
    assert entries[0].aspect_ratio == 2
    assert entries[0].size == 5
    assert "image" not in entries[0].metadata
    assert "source_image" not in entries[0].metadata
    assert loader.load_sample(0, 2).data == b"third"
    assert [entry.data for entry in loader.iter_shard(0, 1, 3)] == [b"second", b"third"]
    assert list(loader.iter_shard(0, 3, 3)) == []
    assert entries[0].path == "task/part.parquet/000000000000"


@pytest.mark.parametrize("kind", ["json", "parquet"])
def test_custom_metadata_joins_by_row_not_id_or_order(tmp_path, kind):
    source = tmp_path / "source"
    path = write_source(source)
    metadata = tmp_path / "metadata"
    write_metadata(
        metadata,
        path,
        [
            {"source_row": 2, "captions": {"text": "changed third"}, "custom": "three"},
            {"source_row": 0, "captions": {"text": "changed first"}, "custom": "one"},
        ],
        kind=kind,
    )
    dataset = webshart.discover_parquet_dataset(str(source), metadata=str(metadata))
    loader = webshart.ParquetDataLoader(dataset)
    assert loader.load_sample(0, 0).captions == {"text": "changed first"}
    assert loader.get_metadata(0, 0)["width"] == 4
    assert loader.get_metadata(0, 1)["captions"]["text"] == "two"
    assert loader.get_metadata(0, 2)["custom"] == "three"
    assert loader.get_metadata(0, 0)["id"] == 0
    assert loader.get_metadata(0, 1)["id"] == 0
    changed = loader.get_metadata(0, 0)
    changed["captions"]["text"] = "must not mutate cache"
    assert loader.get_metadata(0, 0)["captions"]["text"] == "changed first"


def test_json_caption_objects_and_mixed_lists_are_not_stringified(tmp_path):
    source = tmp_path / "source"
    path = write_source(source)
    metadata = tmp_path / "metadata"
    values = [{"scene": {"lighting": "soft"}}, [{"text": "structured"}, "plain"], []]
    write_metadata(
        metadata,
        path,
        [
            {"source_row": index, "captions": json.dumps(value)}
            for index, value in enumerate(values)
        ],
        kind="parquet",
        json_columns=["captions"],
        mode="replace",
    )
    loader = webshart.ParquetDataLoader(
        webshart.discover_parquet_dataset(str(source), metadata=str(metadata))
    )
    entries = list(loader)
    assert [entry.captions for entry in entries] == values
    assert entries[1].caption == {"text": "structured"}
    assert entries[2].caption is None


def test_native_lists_of_objects_and_huggingface_image_structs(tmp_path):
    write_source(
        tmp_path,
        [
            {
                "image": {"bytes": b"embedded", "path": "image.jpg"},
                "captions": [{"text": "one"}, {"text": "two"}],
            },
            {
                "image": {
                    "bytes": None,
                    "path": "https://invalid.example/do-not-fetch",
                },
                "captions": [],
            },
        ],
    )
    loader = webshart.ParquetDataLoader(
        webshart.discover_parquet_dataset(str(tmp_path))
    )
    entries = list(loader)
    assert entries[0].data == b"embedded"
    assert entries[0].captions == [{"text": "one"}, {"text": "two"}]
    assert entries[1].data is None
    assert entries[1].size == 0
    assert entries[1].payloads["image"]["path"].startswith("https://invalid.example")


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"source_row": 0}, {"source_row": 0}], "Duplicate"),
        ([{"source_row": 3}], "out-of-bounds"),
        ([{"source_row": -1}], "source_row"),
        ([{"source_row": True}], "source_row"),
        ([{"id": 0}], "source_row"),
        ([{"source_row": 0, "source_path": "wrong.parquet"}], "source_path"),
        ([{"source_row": 0, "image": "not payload"}], "payload"),
    ],
)
def test_invalid_custom_metadata_is_rejected(tmp_path, rows, message):
    path = write_source(tmp_path / "source")
    write_metadata(tmp_path / "metadata", path, rows)
    dataset = webshart.discover_parquet_dataset(
        str(tmp_path / "source"), metadata=str(tmp_path / "metadata")
    )
    with pytest.raises(ValueError, match=message):
        dataset.get_metadata(0, 0)


def test_replace_metadata_requires_complete_coverage(tmp_path):
    path = write_source(tmp_path / "source")
    write_metadata(tmp_path / "metadata", path, [{"source_row": 0}], mode="replace")
    dataset = webshart.discover_parquet_dataset(
        str(tmp_path / "source"), metadata=str(tmp_path / "metadata")
    )
    with pytest.raises(ValueError, match="cover every"):
        dataset.get_metadata(0, 0)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"path": "../part.parquet"}, "Unsafe"),
        ({"metadata_path": "/tmp/metadata.json"}, "Unsafe"),
        ({"size": 1}, "size mismatch"),
        ({"num_rows": 100}, "row count"),
        ({"metadata_sha256": "0" * 64}, "SHA-256 mismatch"),
    ],
)
def test_manifest_validation(tmp_path, change, message):
    path = write_source(tmp_path / "source")
    manifest = write_metadata(tmp_path / "metadata", path, [{"source_row": 0}])
    manifest["shards"][0].update(change)
    (tmp_path / "metadata" / "webshart-parquet.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        dataset = webshart.discover_parquet_dataset(
            str(tmp_path / "source"), metadata=str(tmp_path / "metadata")
        )
        dataset.get_metadata(0, 0)


def test_metadata_aliases_custom_caption_override_and_cache_bound(tmp_path):
    path = write_source(
        tmp_path / "source",
        [
            {
                "image": b"img",
                "instruction": "original",
                "image_width": 10,
                "image_height": 20,
            }
        ],
    )
    write_metadata(
        tmp_path / "metadata",
        path,
        [{"source_row": 0, "captions": {"edited": True}}],
        columns={
            "captions": "instruction",
            "width": "image_width",
            "height": "image_height",
        },
    )
    dataset = webshart.discover_parquet_dataset(
        str(tmp_path / "source"), metadata=str(tmp_path / "metadata")
    )
    entry = webshart.ParquetDataLoader(dataset).load_sample(0, 0)
    assert entry.captions == {"edited": True}
    assert (entry.width, entry.height) == (10, 20)
    second = tmp_path / "source" / "other.parquet"
    second.write_bytes(path.read_bytes())
    uncoupled = webshart.discover_parquet_dataset(
        str(tmp_path / "source"), caption_column="instruction"
    )
    for index in range(2):
        uncoupled.get_metadata(index, 0)
    assert len(uncoupled._files) == len(uncoupled._metadata_cache) == 1
    assert uncoupled.get_metadata(0, 0)["captions"] == "original"


def test_bounds_columns_corruption_and_concurrent_reads(tmp_path):
    path = write_source(tmp_path)
    dataset = webshart.discover_parquet_dataset(str(tmp_path))
    loader = webshart.ParquetDataLoader(dataset)
    for shard, row in [(-1, 0), (1, 0), (0, -1), (0, 3)]:
        with pytest.raises(IndexError):
            loader.load_sample(shard, row)
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert list(
            executor.map(lambda index: loader.load_sample(0, index).data, [2, 0, 1])
        ) == [b"third", b"first", b"second"]
    reader = _ParquetFile(str(path))
    with pytest.raises(ValueError, match="Unknown"):
        reader.rows(columns=["missing"])
    with pytest.raises(ValueError, match="Duplicate"):
        reader.rows(columns=["id", "id"])
    with pytest.raises(IndexError):
        reader.rows(start=2, stop=1)
    with pytest.raises(ValueError, match="SHA-256"):
        reader.verify_sha256("0" * 64)
    path.write_bytes(b"not parquet")
    with pytest.raises(OSError):
        _ParquetFile(str(path))


@contextmanager
def range_server(content, mode="range"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            begin, end = self.headers["Range"].removeprefix("bytes=").split("-")
            begin, end = int(begin), int(end)
            requests.append((begin, end))
            if mode == "ignore":
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                return
            body = content[begin : end + 1]
            self.send_response(206)
            first = begin + 1 if mode == "wrong" else begin
            self.send_header("Content-Range", f"bytes {first}-{end}/{len(content)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/part.parquet", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_remote_page_streaming_does_not_download_whole_column_chunk(tmp_path):
    rows = [
        {"image": bytes([index]) * 100_000, "captions": f"row {index}"}
        for index in range(100)
    ]
    path = write_source(
        tmp_path,
        rows,
        row_group_size=100,
        compression="none",
        use_dictionary=False,
        data_page_size=100_000,
        write_batch_size=1,
    )
    with range_server(path.read_bytes()) as (url, requests):
        reader = _ParquetFile(url, size=path.stat().st_size)
        metadata = list(reader.rows(columns=["captions"]))
        assert len(metadata) == 100
        assert reader.bytes_read < 50_000
        entries = reader.rows(columns=["image"], start=0, stop=2)
        assert [row["image"] for row in entries] == [rows[0]["image"], rows[1]["image"]]
        assert reader.bytes_read < 500_000
        assert max(end - begin + 1 for begin, end in requests) <= 100_010
        assert reader.bytes_read < path.stat().st_size // 10


@pytest.mark.parametrize("mode", ["ignore", "wrong"])
def test_http_range_failures_are_not_treated_as_full_files(tmp_path, mode):
    path = write_source(tmp_path)
    with range_server(path.read_bytes(), mode) as (url, requests):
        with pytest.raises(OSError, match="range request failed"):
            _ParquetFile(url, size=path.stat().st_size)
        assert len(requests) == 1


def test_large_page_guard(tmp_path):
    path = write_source(
        tmp_path, [{"image": b"x" * 100_000}], compression="none", use_dictionary=False
    )
    reader = _ParquetFile(str(path), max_page_bytes=10_000)
    with pytest.raises(OSError, match="max_page_bytes"):
        next(reader.rows())


def test_small_remote_metadata_is_buffered_once_and_hash_verified(tmp_path):
    path = write_source(
        tmp_path,
        [{"source_row": index, "captions": f"row {index}"} for index in range(20)],
        row_group_size=1,
    )
    content = path.read_bytes()
    with range_server(content) as (url, requests):
        reader = _ParquetFile(url, size=len(content), cache_bytes=64 * 1024 * 1024)
        reader.verify_sha256(hashlib.sha256(content).hexdigest())
        assert list(reader.rows()) == parquet.read_table(path).to_pylist()
        assert requests == [(0, len(content) - 1)]
        assert reader.bytes_read == len(content)
        with pytest.raises(ValueError, match="SHA-256"):
            reader.verify_sha256("0" * 64)
    with range_server(content) as (url, requests):
        reader = _ParquetFile(url, size=len(content), cache_bytes=1)
        assert len(list(reader.rows())) == 20
        assert len(requests) > 1


def test_explicit_parquet_dispatch_does_not_change_tar(tmp_path):
    with pytest.raises(ValueError, match="format"):
        webshart.discover_dataset(str(tmp_path), format="csv")
    with pytest.raises(ValueError, match="Parquet-specific"):
        webshart.discover_dataset(str(tmp_path), image_column="image")
    with pytest.raises(ValueError, match="No Parquet"):
        webshart.discover_parquet_dataset(str(tmp_path))


def test_hub_metadata_pins_source_and_metadata_revisions(monkeypatch):
    from webshart.parquet import _Repository

    source_revision = "a" * 40
    metadata_revision = "b" * 40
    manifest = {
        "format": "webshart-parquet",
        "version": 1,
        "source": {"repo_id": "owner/source", "revision": source_revision},
        "columns": {"image": "edited_image"},
        "metadata_mode": "replace",
        "shards": [
            {
                "path": "part.parquet",
                "size": 1000,
                "num_rows": 3,
                "metadata_path": "custom.parquet",
            }
        ],
    }
    calls = []

    class FakeApi:
        def __init__(self, token):
            assert token == "test-token"

        def dataset_info(self, source, revision, files_metadata):
            calls.append((source, revision, files_metadata))
            names = (
                ["webshart-parquet.json", "custom.parquet"]
                if source == "owner/metadata"
                else ["part.parquet"]
            )
            return SimpleNamespace(
                sha=metadata_revision
                if source == "owner/metadata"
                else source_revision,
                siblings=[SimpleNamespace(rfilename=name, size=1000) for name in names],
            )

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setitem(sys.modules, "hf_xet", None)
    monkeypatch.setattr(_Repository, "read_json", lambda self, path: manifest)
    dataset = webshart.discover_parquet_dataset(
        "owner/source",
        metadata="owner/metadata",
        hf_token="test-token",
        metadata_revision="release",
    )
    assert calls == [
        ("owner/metadata", "release", True),
        ("owner/source", source_revision, True),
    ]
    assert dataset.source_revision == source_revision
    assert dataset.metadata_revision == metadata_revision
    assert dataset.image_column == "edited_image"
    assert len(dataset) == 3
    with pytest.raises(ValueError, match="different source"):
        webshart.discover_parquet_dataset(
            "owner/wrong", metadata="owner/metadata", hf_token="test-token"
        )
    with pytest.raises(ValueError, match="conflicts"):
        webshart.discover_parquet_dataset(
            "owner/source",
            metadata="owner/metadata",
            hf_token="test-token",
            revision="c" * 40,
        )
    manifest["source"]["revision"] = "main"
    with pytest.raises(ValueError, match="commit SHA"):
        webshart.discover_parquet_dataset(
            "owner/source", metadata="owner/metadata", hf_token="test-token"
        )


@pytest.mark.parametrize("codec", ["snappy", "zstd", "gzip", "brotli", "lz4"])
def test_compressed_parquet_matches_arrow(tmp_path, codec):
    path = write_source(tmp_path, compression=codec)
    assert list(_ParquetFile(str(path)).rows()) == parquet.read_table(path).to_pylist()
