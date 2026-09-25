import hashlib
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from urllib.request import Request, urlopen

import pytest

import webshart


REPO = "webshart/pseudo-camera-10k-structured"
REVISION = "3ad79f91c50aab9dc63cbae9daf2ec2c474c34af"
pytestmark = pytest.mark.skipif(
    os.environ.get("WEBSHART_TEST_LIVE_XET") != "1",
    reason="opt-in Hub integration tests download a 287 MiB shard",
)


@pytest.fixture
def remote_dataset(monkeypatch):
    monkeypatch.setitem(sys.modules, "hf_xet", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.delenv("WEBSHART_DISABLE_XET", raising=False)
    return webshart.discover_dataset(REPO, subfolder="data")


def assert_image(entry):
    assert entry.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(entry.data) == entry.size
    assert (entry.width, entry.height) == struct.unpack(">II", entry.data[16:24])
    assert isinstance(entry.captions, dict)


def test_native_sample_reader_batch_iteration_and_python_threads(remote_dataset):
    loader = webshart.TarDataLoader(remote_dataset, buffer_size=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        samples = list(executor.map(lambda index: loader.load_sample(0, index), [0, 1]))
    for entry in samples:
        assert_image(entry)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/shard-00000.tar"
        request = Request(
            url,
            headers={"Range": f"bytes={entry.offset}-{entry.offset + entry.size - 1}"},
        )
        with urlopen(request, timeout=60) as response:
            assert response.status == 206
            assert response.read() == entry.data
    entries = list(islice(loader, 2))
    assert [entry.data for entry in entries] == [entry.data for entry in samples]
    reader = remote_dataset.open_shard(0)
    assert reader.read_sample(0) == samples[0].data


def test_native_shard_cache_checksums_and_warm_reads(remote_dataset, tmp_path, capfd):
    remote_dataset.enable_shard_cache(
        str(tmp_path), cache_limit_gb=1, parallel_downloads=2
    )
    loader = webshart.TarDataLoader(remote_dataset)
    entry = loader.load_sample(18, 0)
    assert_image(entry)
    cached = tmp_path / "shard-00018.tar"
    assert cached.stat().st_size == 301424640
    with cached.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == (
            "990c661ebaa713d3e5a0c76f781e9bc6cff22e4273a1d09228283ab270ce79d6"
        )
    assert "native Xet" in capfd.readouterr().out
    assert loader.load_sample(18, 0).data == entry.data
    assert "native Xet" not in capfd.readouterr().out


def test_explicit_http_mode_preserves_sample_bytes(remote_dataset, monkeypatch):
    loader = webshart.TarDataLoader(remote_dataset)
    native = loader.load_sample(0, 0)
    monkeypatch.setenv("WEBSHART_DISABLE_XET", "1")
    ordinary = loader.load_sample(0, 0)
    assert ordinary.data == native.data
    assert json.loads(json.dumps(ordinary.captions)) == native.captions


@pytest.mark.parametrize("buffer_size", [1, 2])
def test_native_iterator_reports_invalid_ranges(tmp_path, buffer_size):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "shard-00000.json").write_text(
        json.dumps(
            {
                "filesize": 1073090560,
                "files": {
                    f"sample-{index}.jpg": {"offset": 1073090561, "length": 10}
                    for index in range(2)
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = webshart.discover_dataset(
        REPO, subfolder="data", metadata=str(tmp_path)
    )
    loader = webshart.TarDataLoader(dataset, buffer_size=buffer_size)
    with pytest.raises(Exception, match="Requested byte range exceeds Xet file size"):
        next(loader)
