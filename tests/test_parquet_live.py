import hashlib
import os
import sys
from urllib.request import Request, urlopen

import pyarrow.parquet as parquet
import pytest

import webshart


pytestmark = pytest.mark.skipif(
    os.environ.get("WEBSHART_TEST_LIVE_PARQUET") != "1",
    reason="opt-in ScaleEdit integration downloads the 54 MB count-change shard",
)
REPO = "InternVL-U/ScaleEdit-12M"
REVISION = "05d2eea7b80ba72c077d39be43b7aca1a7857e6a"
SUBFOLDER = "3.4_count_change"
FILENAME = SUBFOLDER + "/count_change_0000.parquet"


def test_native_scaleedit_bytes_match_independent_arrow_read(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    monkeypatch.setitem(sys.modules, "hf_xet", None)
    monkeypatch.delenv("WEBSHART_DISABLE_XET", raising=False)
    dataset = webshart.discover_parquet_dataset(
        REPO,
        revision=REVISION,
        subfolder=SUBFOLDER,
        image_column="edited_image",
        caption_column="edit_instruction",
        width_column="edited_image_width",
        height_column="edited_image_height",
    )
    loader = webshart.ParquetDataLoader(dataset, image_columns=["source_image"])
    entries = list(loader.iter_shard(0, stop=3))
    source = tmp_path / "reference.parquet"
    request = Request(
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILENAME}"
    )
    with urlopen(request, timeout=120) as response, source.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    expected = parquet.read_table(source).slice(0, 3).to_pylist()
    for entry, reference in zip(entries, expected, strict=True):
        assert entry.data == reference["edited_image"]
        assert entry.payloads["source_image"] == reference["source_image"]
        assert entry.captions == reference["edit_instruction"]
        assert entry.width == reference["edited_image_width"]
    assert (
        hashlib.sha256(loader.load_sample(0, 457).data).digest()
        == hashlib.sha256(
            parquet.read_table(source, columns=["edited_image"])["edited_image"][
                457
            ].as_py()
        ).digest()
    )
