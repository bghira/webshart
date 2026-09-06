import copy
import io
import json
import tarfile
from functools import partial
from http.server import SimpleHTTPRequestHandler
from multiprocessing import get_context
from socketserver import TCPServer

import pytest

import webshart


def serve_metadata(source, connection):
    class MetadataHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            connection.send((self.path, self.headers.get("Authorization")))
            super().do_GET()

        def log_message(self, *_args):
            pass

    with TCPServer(
        ("127.0.0.1", 0), partial(MetadataHandler, directory=str(source))
    ) as server:
        connection.send(server.server_address[1])
        server.serve_forever()


def write_shard(root, name, fields, *, list_format=False):
    tar_path = root / f"{name}.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as archive:
        for filename in fields:
            payload = filename.encode()
            member = tarfile.TarInfo(filename)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(tar_path) as archive:
        files = {
            member.name: {
                "offset": member.offset_data,
                "length": member.size,
                **fields[member.name],
            }
            for member in archive
        }
    if list_format:
        files = [{"filename": filename, **entry} for filename, entry in files.items()]
    metadata = {"filesize": tar_path.stat().st_size, "files": files, "custom": [1, 2]}
    index_path = root / f"{name}.json"
    index_path.write_text(json.dumps(metadata), encoding="utf-8")
    return index_path, metadata


@pytest.mark.parametrize("list_format", [False, True])
@pytest.mark.parametrize("captions", ["a café", ["first", "second"], [], None])
def test_field_rename_persists_and_preserves_payload_and_metadata(
    tmp_path, list_format, captions
):
    index_path, original = write_shard(
        tmp_path,
        "shard-0000",
        {
            "sample.webp": {"captions": captions, "quality": 0.9},
            "missing.webp": {"tags": ["keep"]},
        },
        list_format=list_format,
    )
    tar_path = tmp_path / "shard-0000.tar"
    original_tar = tar_path.read_bytes()
    original_mode = index_path.stat().st_mode
    dataset = webshart.discover_dataset(str(tmp_path))

    assert dataset.field_rename("captions", "v1_captions") == 1

    expected = copy.deepcopy(original)
    sample = expected["files"][0] if list_format else expected["files"]["sample.webp"]
    sample["v1_captions"] = sample.pop("captions")
    assert json.loads(index_path.read_text()) == expected
    assert index_path.stat().st_mode == original_mode
    assert tar_path.read_bytes() == original_tar
    for source in [dataset, webshart.discover_dataset(str(tmp_path))]:
        loader = webshart.TarDataLoader(source)
        metadata = loader.get_metadata(0)["sample.webp"]
        assert metadata["v1_captions"] == captions
        assert metadata["quality"] == 0.9
        assert "captions" not in metadata
        samples = {entry.path: entry for entry in loader}
        assert samples["sample.webp"].metadata["v1_captions"] == captions
        assert samples["sample.webp"].data == b"sample.webp"
        entry = loader.load_sample(0, 1)
        assert entry.metadata["v1_captions"] == captions
        assert entry.captions is None


def test_field_rename_preflights_all_shards_before_replacing_any(tmp_path):
    first, _ = write_shard(tmp_path, "shard-0000", {"a.webp": {"captions": "a"}})
    second, _ = write_shard(
        tmp_path,
        "shard-0001",
        {"b.webp": {"captions": "b", "v1_captions": "keep"}},
    )
    originals = [first.read_bytes(), second.read_bytes()]
    dataset = webshart.discover_dataset(str(tmp_path))

    with pytest.raises(ValueError, match="already exists.*overwrite=True"):
        dataset.field_rename("captions", "v1_captions")

    assert [first.read_bytes(), second.read_bytes()] == originals
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "shard-0000.json",
        "shard-0000.tar",
        "shard-0001.json",
        "shard-0001.tar",
    ]
    assert dataset.field_rename("captions", "v1_captions", overwrite=True) == 2
    assert json.loads(second.read_text())["files"]["b.webp"]["v1_captions"] == "b"


def test_field_rename_invalidates_loaded_and_disk_metadata_cache(tmp_path):
    source = tmp_path / "source"
    index_path, _ = write_shard(source, "shard-0000", {"a.webp": {"captions": "a"}})
    cache = tmp_path / "cache"
    dataset = webshart.discover_dataset(str(source))
    dataset.enable_metadata_cache(str(cache), init_shard_count=1)
    assert dataset.get_cache_stats()["cached_shards"] == 1

    assert dataset.field_rename("captions", "v1_captions") == 1

    assert dataset.get_cache_stats()["cached_shards"] == 0
    assert "v1_captions" in json.loads(index_path.read_text())["files"]["a.webp"]
    loader = webshart.TarDataLoader(dataset, load_file_data=False)
    assert loader.load_sample(0, 0).metadata["v1_captions"] == "a"
    rediscovered = webshart.discover_dataset(str(source))
    rediscovered.enable_metadata_cache(str(cache), init_shard_count=1)
    reloaded = webshart.TarDataLoader(rediscovered, load_file_data=False)
    assert reloaded.load_sample(0, 0).metadata["v1_captions"] == "a"
    assert "captions" not in reloaded.get_metadata(0)["a.webp"]


def test_field_rename_supports_custom_fields_and_repeated_renames(tmp_path):
    index_path, _ = write_shard(
        tmp_path,
        "shard-0000",
        {"a.webp": {"captions": ["a"], "extra": {"captions": "nested"}}},
    )
    dataset = webshart.discover_dataset(str(tmp_path))

    assert dataset.field_rename("captions", "v1_captions") == 1
    assert dataset.field_rename("v1_captions", "v2_captions") == 1
    assert dataset.field_rename("extra", "attributes") == 1

    stored = json.loads(index_path.read_text())["files"]["a.webp"]
    assert stored["v2_captions"] == ["a"]
    assert stored["attributes"] == {"captions": "nested"}
    assert "captions" not in stored
    assert "v1_captions" not in stored
    before = index_path.stat().st_mtime_ns
    assert dataset.field_rename("missing", "unused") == 0
    assert dataset.field_rename("v2_captions", "v2_captions") == 0
    assert dataset.field_rename("captions", "v1_captions") == 0
    assert index_path.stat().st_mtime_ns == before
    assert dataset.field_rename("v2_captions", "captions") == 1
    loader = webshart.TarDataLoader(dataset, load_file_data=False)
    assert loader.load_sample(0, 0).captions == ["a"]


@pytest.mark.parametrize(
    "old_name,new_name",
    [("", "new"), ("captions", " "), ("offset", "old_offset"), ("captions", "size")],
)
def test_field_rename_rejects_empty_or_structural_field_names(
    tmp_path, old_name, new_name
):
    index_path, _ = write_shard(tmp_path, "shard-0000", {"a.webp": {"captions": "a"}})
    original = index_path.read_bytes()
    dataset = webshart.discover_dataset(str(tmp_path))

    with pytest.raises(ValueError):
        dataset.field_rename(old_name, new_name)

    assert index_path.read_bytes() == original


def test_field_rename_uses_separate_metadata_location(tmp_path):
    source = tmp_path / "source"
    index_path, _ = write_shard(source, "shard-0000", {"a.webp": {"captions": "a"}})
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    moved = index_path.rename(metadata_dir / index_path.name)
    dataset = webshart.discover_dataset(str(source), metadata=str(metadata_dir))

    assert dataset.field_rename("captions", "v1_captions") == 1
    assert json.loads(moved.read_text())["files"]["a.webp"]["v1_captions"] == "a"
    assert not index_path.exists()


def test_field_rename_export_preserves_nested_names_and_untouched_shards(tmp_path):
    source = tmp_path / "source"
    first, _ = write_shard(source, "nested/shard.v1", {"a.webp": {"captions": "a"}})
    second, _ = write_shard(source, "shard-0001", {"b.webp": {}})
    original = first.read_bytes()
    destination = tmp_path / "export"
    dataset = webshart.discover_dataset(str(source))
    dataset.enable_metadata_cache(str(tmp_path / "cache"), init_shard_count=2)

    assert dataset.field_rename("captions", "v1_captions", destination=destination) == 1

    assert first.read_bytes() == original
    assert json.loads((destination / "shard-0001.json").read_text()) == json.loads(
        second.read_text()
    )
    stored = json.loads((destination / "nested/shard.v1.json").read_text())
    assert stored["files"]["a.webp"]["v1_captions"] == "a"
    assert dataset.get_shard_info(0)["json_path"] == str(
        destination / "nested/shard.v1.json"
    )
    loader = webshart.TarDataLoader(dataset, load_file_data=False)
    assert loader.load_sample(0, 0).metadata["v1_captions"] == "a"


def test_field_rename_invalid_later_shard_leaves_all_indexes_unchanged(tmp_path):
    first, _ = write_shard(tmp_path, "shard-0000", {"a.webp": {"captions": "a"}})
    second, _ = write_shard(tmp_path, "shard-0001", {"b.webp": {"captions": "b"}})
    second.write_text("{broken json")
    original = first.read_bytes()
    dataset = webshart.discover_dataset(str(tmp_path))

    with pytest.raises(Exception, match="JSON parsing error"):
        dataset.field_rename("captions", "v1_captions")

    assert first.read_bytes() == original
    assert second.read_text() == "{broken json"


def test_field_rename_remote_metadata_requires_export_and_preserves_custom_keys(
    tmp_path,
):
    source = tmp_path / "source"
    index_path, original = write_shard(
        source, "shard-0000", {"a.webp": {"captions": "a", "tags": ["keep"]}}
    )
    context = get_context("spawn")
    requests, connection = context.Pipe(duplex=False)
    server = context.Process(
        target=serve_metadata, args=(source, connection), daemon=True
    )
    server.start()
    connection.close()
    try:
        assert requests.poll(
            10
        ), f"Metadata server did not start: pid={server.pid}, exitcode={server.exitcode}"
        port = requests.recv()
        dataset = webshart.discover_dataset(
            str(source),
            metadata=f"http://127.0.0.1:{port}",
            hf_token="test-token",
        )
        with pytest.raises(ValueError, match="provide a local destination"):
            dataset.field_rename("captions", "v1_captions")
        assert not requests.poll()
        destination = tmp_path / "export"
        assert (
            dataset.field_rename("captions", "v1_captions", destination=destination)
            == 1
        )
        assert requests.poll(10)
        assert requests.recv() == ("/shard-0000.json", "Bearer test-token")
        assert json.loads(index_path.read_text()) == original
        exported = json.loads((destination / index_path.name).read_text())
        assert exported["files"]["a.webp"]["v1_captions"] == "a"
        assert exported["files"]["a.webp"]["tags"] == ["keep"]
        loader = webshart.TarDataLoader(dataset, load_file_data=False)
        assert loader.load_sample(0, 0).metadata["v1_captions"] == "a"
    finally:
        server.terminate()
        server.join(timeout=10)
        requests.close()
        connection.close()
