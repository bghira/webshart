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


def test_overlay_cannot_replace_url(tmp_path, server):
    dataset = source(tmp_path / "source", server[0])
    output = tmp_path / "metadata"
    output.mkdir()
    path = output / "overlay.parquet"
    (output / "webshart-parquet.json").write_text(json.dumps({
        "format": "webshart-parquet", "version": 1,
        "columns": {"image_url": "url"},
        "shards": [{**dataset.get_shard_info(0), "metadata_path": path.name}],
    }))
    pq.write_table(
        pa.Table.from_pylist([{"source_row": 0, "url": server[0] + "/missing"}]), path
    )
    loaded = webshart.discover_parquet_dataset(
        str(tmp_path / "source"), metadata=str(output)
    )
    with pytest.raises(ValueError, match="payload"):
        loaded.get_metadata(0, 0)


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
