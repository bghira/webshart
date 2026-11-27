import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from webshart.repack import repack_parquet


def test_repack_preserves_sparse_pairs_and_metadata(tmp_path):
    source, dest = tmp_path / "source.parquet", tmp_path / "prepared.parquet"
    table = pa.table({
        "image": [bytes([i % 251]) * 16000 for i in range(301)],
        "reference": [None if i % 5 == 0 else bytes([i % 251]) * 29 for i in range(301)],
        "caption": ["caption"] * 301,
        "score": [float("nan"), None, -0.0, 0.25] * 75 + [1.0],
    }).replace_schema_metadata({b"annotation": b"test"})
    pq.write_table(table, source, row_group_size=137)
    assert repack_parquet(source, dest, ["image", "reference"])["rows"] == len(table)
    with pq.ParquetFile(dest) as f:
        assert f.read().drop(["score"]).equals(table.drop(["score"]), check_metadata=True)
        assert f.read()["score"].combine_chunks().view(pa.int64()).equals(table["score"].combine_chunks().view(pa.int64()))
        assert f.num_row_groups > 1
        for i in range(f.num_row_groups):
            group = f.metadata.row_group(i)
            for col in (0, 1):
                assert group.column(col).dictionary_page_offset is None
                assert group.column(col).has_offset_index
    with pytest.raises(FileExistsError):
        repack_parquet(source, dest, ["image"])
    with pytest.raises(ValueError, match="binary"):
        repack_parquet(source, tmp_path / "invalid.parquet", ["caption"])
    assert not (tmp_path / "invalid.parquet").exists()


def test_shared_repacked_cache_reuses_pinned_rows_and_evicts_between_batches(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from functools import partial

    import webshart

    root = tmp_path / "original"
    root.mkdir()
    rows = pa.table({"image": [bytes([n]) * 4096 for n in range(20)]})
    for name in ("a", "b"):
        pq.write_table(rows, root / (name + ".parquet"), compression="none")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root))
    thread = Thread(target=server.serve_forever)
    thread.start()
    with ExitStack() as cleanup:
        cleanup.callback(thread.join)
        cleanup.callback(server.server_close)
        cleanup.callback(server.shutdown)
        ds = webshart.discover_parquet_dataset(str(root))
        # Model a resolved, immutable Hub URL without needing Hub credentials.
        monkeypatch.setattr(ds._source_repo, "local", False)
        monkeypatch.setattr(ds._source_repo, "location", lambda path: f"http://127.0.0.1:{server.server_port}/{path}")
        ds.enable_shard_cache(tmp_path / "raw", 1)
        ds.enable_repacked_cache(tmp_path / "derived", 1 / 1024**3)
        loader = webshart.ParquetDataLoader(ds)
        first = ds._prefill_shard(0)
        assert loader.load_samples([(0, 12)], metadata_fields=())[0].data == rows["image"][12].as_py()
        assert ds._prefill_shard(0).path == first.path
        path = first.path
        del first
        ds._prefill_shard(1)
        assert not path.exists()
        # The original revision remains readable when a derived entry was evicted.
        assert [s.data for s in loader.load_samples([(1, 4), (0, 12)], metadata_fields=())] == [
            rows["image"][4].as_py(), rows["image"][12].as_py(),
        ]
