"""Opt-in end-to-end URL metadata and payload test against pinned Recap rows."""

import os

import pytest

import webshart


pytestmark = pytest.mark.skipif(
    os.environ.get("WEBSHART_TEST_LIVE_RECAP") != "1",
    reason="opt-in test reads remote Parquet and attempts 32 external image URLs",
)


def test_recap_scan_and_online_payload(tmp_path):
    pytest.importorskip("huggingface_hub")
    repo = "UCSC-VLAA/Recap-DataComp-1B"
    dataset = webshart.discover_parquet_dataset(
        repo,
        revision="457c44d98651bcfdfb3cc8695f5e60a0d2705e78",
        subfolder="data/train_data",
        image_url_column="url",
        caption_column="re_caption",
    )
    summary = webshart.scan_url_metadata(
        dataset,
        str(tmp_path),
        workers=16,
        batch_size=16,
        max_samples=32,
        stop_shard=1,
        url_options=webshart.URLImageOptions(timeout=10, retries=0),
    )
    assert summary["attempted"] == sum(summary["counts"].values()) == 32
    assert summary["counts"].get("ok", 0) > 0
    overlay = webshart.discover_parquet_dataset(repo, metadata=str(tmp_path))
    loader = webshart.ParquetDataLoader(
        overlay,
        fetch_external_urls=True,
        on_url_error="return",
        url_options=webshart.URLImageOptions(timeout=10, retries=0),
    )
    for row in range(32):
        metadata = loader.get_metadata(0, row)
        if metadata["url_status"] != "ok":
            continue
        sample = loader.load_sample(0, row)
        if sample.error is None:
            assert sample.data and sample.width > 0 and sample.height > 0
            assert sample.captions == metadata["re_caption"]
            assert sample.metadata["source_row"] == row
            break
    else:
        pytest.fail("None of the successfully probed URLs could be loaded again")
