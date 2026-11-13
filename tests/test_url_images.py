from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import webshart
from webshart.url_images import fetch_image
import webshart.url_scan as scan_module


@pytest.fixture
def server():
    content = (Path(__file__).parent / "fixtures" / "url-image.png").read_bytes()
    hits, headers = Counter(), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            hits[self.path] += 1
            headers.append(dict(self.headers))
            if self.path == "/missing":
                self.send_error(404)
                return
            if self.path == "/retry" and hits[self.path] == 1:
                self.send_error(503)
                return
            if self.path in {"/redirect", "/bad-redirect"}:
                self.send_response(302)
                self.send_header(
                    "Location", "/image" if self.path == "/redirect" else "file:secret"
                )
                self.end_headers()
                return
            if self.path == "/slow":
                hits["inflight"] += 1
                hits["peak"] = max(hits["peak"], hits["inflight"])
                time.sleep(0.15)
                hits["inflight"] -= 1
            data = b"<html>not an image</html>" if self.path == "/html" else content
            self.send_response(200)
            if self.path != "/no-length":
                self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                if self.path == "/slow-body":
                    time.sleep(0.15)
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", content, hits, headers
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def source(root, base):
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"url": base + path, "re_caption": f"caption {i}", "key": "duplicate"}
                for i, path in enumerate(["/image", "/missing", "/html", "/redirect"])
            ]
        ),
        root / "part.parquet",
        row_group_size=2,
    )
    return webshart.discover_parquet_dataset(
        str(root), image_url_column="url", caption_column="re_caption"
    )


def test_url_opt_in_and_metadata_never_fetch(tmp_path, server, monkeypatch):
    base, content, hits, headers = server
    monkeypatch.setenv("HF_TOKEN", "test-token-never-send")
    dataset = source(tmp_path / "source", base)
    loader = webshart.ParquetDataLoader(dataset)
    assert loader.get_metadata(0, 0)["captions"] == "caption 0"
    assert loader.load_sample(0, 0).data is None
    assert list(loader.iter_shard(0, stop=2))[1].data is None
    assert not hits
    loader = webshart.ParquetDataLoader(
        dataset, fetch_external_urls=True, on_url_error="return"
    )
    samples = list(loader)
    assert samples[0].data == samples[3].data == content
    assert samples[0].payloads["url"] == base + "/image"
    assert samples[1].data is None and samples[1].error == "http_error (HTTP 404)"
    assert samples[1].metadata["source_row"] == 1
    assert samples[3].caption == "caption 3"
    assert all("Authorization" not in h and "Cookie" not in h for h in headers)
    assert (
        webshart.ParquetDataLoader(dataset, fetch_external_urls=True)
        .load_sample(0, 0)
        .data
        == content
    )
    with pytest.raises(webshart.URLImageError, match="HTTP 404"):
        webshart.ParquetDataLoader(dataset, fetch_external_urls=True).load_sample(0, 1)


def test_http_bounds_retries_redirects_and_timeouts(server):
    base, content, hits, _ = server
    options = webshart.URLImageOptions(timeout=1, retries=1)
    assert fetch_image(base + "/retry", options) == content
    assert hits["/retry"] == 2
    with pytest.raises(webshart.URLImageError, match="404"):
        fetch_image(base + "/missing", options)
    assert hits["/missing"] == 1
    for path in ("/image", "/no-length"):
        with pytest.raises(webshart.URLImageError, match="too_large"):
            fetch_image(base + path, webshart.URLImageOptions(max_bytes=4))
    with pytest.raises(webshart.URLImageError):
        fetch_image(base + "/bad-redirect", options)
    with pytest.raises(webshart.URLImageError, match="timeout"):
        fetch_image(base + "/slow", webshart.URLImageOptions(timeout=0.01, retries=0))
    with pytest.raises(webshart.URLImageError, match="timeout"):
        fetch_image(
            base + "/slow-body", webshart.URLImageOptions(timeout=0.01, retries=0)
        )
    for url in (
        None,
        "",
        "file:secret",
        "ftp://example.org/image",
        "https://u:p@example.org/a",
    ):
        with pytest.raises(webshart.URLImageError, match="invalid_url"):
            fetch_image(url, options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_bytes": 0},
        {"retries": -1},
        {"retries": True},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        webshart.URLImageOptions(**kwargs)


def test_scan_resume_geometry_overlay_and_status(tmp_path, server):
    base, content, hits, _ = server
    dataset = source(tmp_path / "source", base)
    output = tmp_path / "metadata"
    options = webshart.URLImageOptions(retries=0)
    first = webshart.scan_url_metadata(
        dataset,
        str(output),
        workers=4,
        batch_size=1,
        max_samples=2,
        url_options=options,
    )
    assert first == {
        "attempted": 2,
        "new_attempts": 2,
        "counts": {"ok": 1, "http_404": 1},
        "complete": False,
    }
    second = webshart.scan_url_metadata(
        dataset, str(output), workers=2, batch_size=2, url_options=options
    )
    assert second["complete"] and second["counts"] == {
        "ok": 2,
        "http_404": 1,
        "invalid_image": 1,
    }
    assert hits["/missing"] == 1
    before = hits.copy()
    assert (
        webshart.scan_url_metadata(dataset, str(output), url_options=options)[
            "new_attempts"
        ]
        == 0
    )
    assert hits == before
    loaded = webshart.discover_parquet_dataset(
        str(tmp_path / "source"), metadata=str(output)
    )
    loader = webshart.ParquetDataLoader(loaded)
    sample = loader.load_sample(0, 0)
    assert sample.data is None and sample.width == 17 and sample.height == 9
    assert sample.caption == "caption 0"
    assert sample.metadata["download_bytes"] == len(content)
    assert loader.get_metadata(0, 1)["url_status"] == "http_error"
    assert loader.get_metadata(0, 2)["width"] is None
    assert hits == before
    for path in output.rglob("*.json"):
        assert str(tmp_path) not in path.read_text()
    with pytest.raises(ValueError, match="changed"):
        webshart.scan_url_metadata(
            dataset, str(output), url_options=webshart.URLImageOptions(retries=2)
        )


def test_scan_checkpoint_survives_interruption(tmp_path, server, monkeypatch):
    dataset = source(tmp_path / "source", server[0])
    output = tmp_path / "metadata"
    original = scan_module.URLImageClient.scan

    def interrupt(self, urls, **kwargs):
        if urls[0].endswith("/html"):
            raise RuntimeError("interrupted")
        return original(self, urls, **kwargs)

    monkeypatch.setattr(scan_module.URLImageClient, "scan", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        webshart.scan_url_metadata(dataset, str(output), batch_size=2)
    monkeypatch.setattr(scan_module.URLImageClient, "scan", original)
    result = webshart.scan_url_metadata(dataset, str(output), batch_size=2)
    assert result["new_attempts"] == 2 and result["complete"]
    assert server[2]["/missing"] == 1


def test_overlay_cannot_replace_url(tmp_path, server):
    dataset = source(tmp_path / "source", server[0])
    output = tmp_path / "metadata"
    webshart.scan_url_metadata(dataset, str(output), max_samples=1)
    path = output / "metadata" / "000000.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"source_row": 0, "url": server[0] + "/missing"}]), path
    )
    loaded = webshart.discover_parquet_dataset(
        str(tmp_path / "source"), metadata=str(output)
    )
    with pytest.raises(ValueError, match="payload"):
        loaded.get_metadata(0, 0)


def test_native_scan_concurrency_order_and_missing_values(server):
    from webshart.url_images import URLImageClient

    base, _, hits, _ = server
    client = URLImageClient(webshart.URLImageOptions(retries=0))
    results = client.scan([base + "/slow"] * 6, workers=2)
    assert all(row["width"] == 17 for row in results)
    assert hits["peak"] == 2
    results = client.scan(
        [base + "/slow", None, base + "/missing", base + "/html"], workers=4
    )
    assert [row["url_status"] for row in results] == [
        "ok",
        "invalid_url",
        "http_error",
        "invalid_image",
    ]
    assert (
        client.scan([base + "/image"], max_pixels=10)[0]["url_status"]
        == "too_many_pixels"
    )


def test_missing_checkpoint_part_fails_before_more_fetches(tmp_path, server):
    dataset = source(tmp_path / "source", server[0])
    output = tmp_path / "metadata"
    webshart.scan_url_metadata(dataset, str(output), max_samples=1)
    next((output / "parts").rglob("*.parquet")).unlink()
    before = server[2].copy()
    with pytest.raises(ValueError, match="checkpoint part"):
        webshart.scan_url_metadata(dataset, str(output))
    assert server[2] == before


def test_empty_source_shard_completes(tmp_path):
    source_path = tmp_path / "source"
    source_path.mkdir()
    pq.write_table(
        pa.table({"url": pa.array([], type=pa.string())}), source_path / "empty.parquet"
    )
    dataset = webshart.discover_parquet_dataset(
        str(source_path), image_url_column="url"
    )
    output = tmp_path / "metadata"
    assert webshart.scan_url_metadata(dataset, str(output))["complete"]
    assert webshart.scan_url_metadata(dataset, str(output))["new_attempts"] == 0


def test_native_metadata_cache_respects_limit_and_avoids_rereads(tmp_path, server):
    dataset = source(tmp_path / "source", server[0])
    reader = dataset._file(0)
    before = reader.bytes_read
    assert not reader.cache_source(1)
    assert reader.bytes_read == before
    assert reader.cache_source(dataset.get_shard_info(0)["size"])
    cached = reader.bytes_read
    assert list(reader.rows(columns=["url"], stop=1))[0]["url"].endswith("/image")
    assert reader.cache_source(dataset.get_shard_info(0)["size"])
    assert reader.bytes_read == cached


def test_capture_upload_retry_and_cleanup(tmp_path, server, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub

    base, content, hits, _ = server
    dataset = source(tmp_path / "source", base)
    uploaded = {}
    calls = []

    class FakeHub:
        endpoint = "https://hub.example"

        def __init__(self, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            pass

        def upload_file(self, **kwargs):
            calls.append(kwargs["path_in_repo"])
            if len(calls) == 1:
                raise OSError("upload interrupted")
            uploaded[kwargs["path_in_repo"]] = Path(kwargs["path_or_fileobj"]).read_bytes()
            return SimpleNamespace(oid="commit-id")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHub)
    destination = tmp_path / "scan"
    options = dict(capture_images=True, upload_repo="owner/dataset", keep_local_images=False)
    with pytest.raises(OSError, match="upload interrupted"):
        webshart.scan_url_metadata(dataset, str(destination), **options)
    before = hits.copy()
    assert list(destination.glob("data/*/*.parquet"))
    result = webshart.scan_url_metadata(dataset, str(destination), batch_size=2, **options)
    assert result["complete"] and hits == before
    assert not list(destination.glob("data/*/*.parquet"))
    assert not list(destination.glob(".images-*"))
    rows = pq.read_table(pa.BufferReader(next(iter(uploaded.values())))).to_pylist()
    assert [r["source_row"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["image"]["bytes"] == content
    assert rows[0]["captions"] == "caption 0"
    assert rows[0]["width"] == 17 and rows[0]["height"] == 9
    assert rows[1]["image"] is None and rows[2]["image"] is None
    assert rows[3]["image"]["bytes"] == content
    webshart.scan_url_metadata(dataset, str(destination), **options)
    assert len(calls) == 2 and hits == before
    with pytest.raises(ValueError, match="options changed"):
        webshart.scan_url_metadata(dataset, str(destination))


def test_capture_local_and_option_validation(tmp_path, server):
    base, content, _, _ = server
    dataset = source(tmp_path / "source", base)
    destination = tmp_path / "scan"
    with pytest.raises(ValueError, match="requires capture_images"):
        webshart.scan_url_metadata(dataset, str(destination), upload_repo="owner/dataset")
    with pytest.raises(ValueError, match="requires upload_repo"):
        webshart.scan_url_metadata(dataset, str(destination), keep_local_images=False)
    result = webshart.scan_url_metadata(dataset, str(destination), capture_images=True, batch_size=2)
    assert result["complete"]
    parts = sorted(destination.glob("data/*/*.parquet"))
    assert len(parts) == 2
    assert pq.read_table(parts[0])["image"][0].as_py()["bytes"] == content
    captured = webshart.discover_parquet_dataset(str(destination / "data"))
    sample = webshart.ParquetDataLoader(captured).load_sample(0, 0)
    assert sample.data == content and sample.captions == "caption 0"
    assert sample.width == 17 and sample.height == 9
    parts[0].unlink()
    with pytest.raises(ValueError, match="captured image part"):
        webshart.scan_url_metadata(dataset, str(destination), capture_images=True)


def test_capture_upload_retries_transient_commit_errors(monkeypatch):
    from types import SimpleNamespace

    attempts, sleeps = [], []

    class Conflict(Exception):
        response = SimpleNamespace(status_code=409)

    class Hub:
        def upload_file(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) < 3:
                raise Conflict()
            return "committed"

    monkeypatch.setattr(scan_module.time, "sleep", sleeps.append)
    assert scan_module._upload_capture(Hub(), path_in_repo="data/part.parquet") == "committed"
    assert len(sleeps) == 2 and len(attempts) == 3
    assert 1 <= sleeps[0] < 2 and 2 <= sleeps[1] < 3
