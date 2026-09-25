import io
import json
import tarfile

import pytest

import webshart


STRUCTURED_CAPTION = {
    "high_level_description": "A café beside a river.",
    "style_description": {"lighting": "daylight", "color_palette": ["#AABBCC"]},
    "compositional_deconstruction": {
        "elements": [{"type": "obj", "bbox": [1, 2, 30, 40], "desc": "a café"}]
    },
}
CAPTION_VALUES = [
    STRUCTURED_CAPTION,
    [STRUCTURED_CAPTION],
    [STRUCTURED_CAPTION, "alternate caption"],
    ["single alternative"],
]


def write_tar(root, sidecar=None):
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / "shard.tar", "w") as archive:
        members = {"sample.jpg": b"image bytes"}
        if sidecar is not None:
            members.update(sidecar)
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    webshart.MetadataExtractor().extract_metadata(
        source=str(root), destination=str(root), max_workers=1
    )


@pytest.mark.parametrize("captions", CAPTION_VALUES)
def test_native_caption_writer_reader_cache_and_rename(tmp_path, captions):
    source = tmp_path / "source"
    write_tar(source)
    index_path = source / "shard.json"
    assert webshart.write_captions_to_metadata(index_path, {"sample": captions}) == 1
    assert (
        json.loads(index_path.read_text())["files"]["sample.jpg"]["captions"]
        == captions
    )
    dataset = webshart.discover_dataset(str(source))
    dataset.enable_metadata_cache(str(tmp_path / "cache"), init_shard_count=1)
    loader = webshart.TarDataLoader(dataset)
    entry = loader.load_sample(0, 0)
    first = captions[0] if isinstance(captions, list) else captions
    assert entry.captions == captions
    if isinstance(captions, dict):
        assert list(entry.captions) == list(captions)
    assert entry.caption == first
    assert loader.load_caption(0, 0) == first
    assert entry.metadata["captions"] == captions
    assert next(loader).captions == captions
    assert loader.get_metadata(0)["sample.jpg"]["captions"] == captions
    rediscovered = webshart.discover_dataset(str(source))
    rediscovered.enable_metadata_cache(str(tmp_path / "cache"), init_shard_count=1)
    assert webshart.TarDataLoader(rediscovered).load_sample(0, 0).captions == captions
    assert dataset.field_rename("captions", "v1_captions") == 1
    assert (
        webshart.TarDataLoader(dataset).load_sample(0, 0).metadata["v1_captions"]
        == captions
    )


@pytest.mark.parametrize("captions", CAPTION_VALUES)
def test_json_text_sidecar_loading_and_coalescing(tmp_path, captions):
    source = tmp_path / "source"
    write_tar(source, {"sample.txt": json.dumps(captions).encode()})
    dataset = webshart.discover_dataset(str(source))
    loader = webshart.TarDataLoader(dataset, load_file_data=False)
    assert loader.load_sample(0, 0).captions == captions
    first = captions[0] if isinstance(captions, list) else captions
    assert loader.load_caption(0, 0) == first
    export = tmp_path / "export"
    loader.coalesce_caption_metadata(destination=str(export))
    assert (
        json.loads((export / "shard.json").read_text())["files"]["sample.jpg"][
            "captions"
        ]
        == captions
    )


@pytest.mark.parametrize("captions", CAPTION_VALUES)
@pytest.mark.parametrize("suffix", [".txt", ".json"])
def test_optimize_dataset_preserves_native_captions(tmp_path, captions, suffix):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.jpg").write_bytes(b"image bytes")
    (source / f"sample{suffix}").write_text(json.dumps(captions), encoding="utf-8")
    output = tmp_path / "output"
    result = webshart.optimize_dataset(
        source, destination=output, include_image_geometry=False
    )
    assert result["captioned_samples"] == 1
    assert result["uncaptioned_samples"] == 0
    dataset = webshart.discover_dataset(str(output / "webshart"))
    entry = webshart.TarDataLoader(dataset).load_sample(0, 0)
    assert entry.captions == captions
    assert entry.data == b"image bytes"


def test_json_sidecar_caption_field_preserves_objects_and_mixed_variants(tmp_path):
    captions = [STRUCTURED_CAPTION, "alternate caption"]
    write_tar(tmp_path, {"sample.json": json.dumps({"captions": captions}).encode()})
    dataset = webshart.discover_dataset(str(tmp_path))
    assert webshart.TarDataLoader(dataset).load_sample(0, 0).captions == captions


@pytest.mark.parametrize(
    "caption", ["{unfinished prose", "[a short description]", "[1, 2, 3]"]
)
def test_non_caption_json_and_plain_text_remain_strings(tmp_path, caption):
    write_tar(tmp_path, {"sample.txt": caption.encode()})
    dataset = webshart.discover_dataset(str(tmp_path))
    assert webshart.TarDataLoader(dataset).load_sample(0, 0).captions == caption
