<img width="1530" height="492" alt="image" src="https://github.com/user-attachments/assets/ebf0d101-eae7-4908-bb73-a264bf89a479" />

Fast dataloader and conversion utility for webdataset tar shards. Rust core with Python bindings.

Built for streaming large video and image datasets, but handles any byte data.

## Install

```bash
pip install webshart
```

## What is this?

Webshart is a fast reader for webdataset tar files with separate JSON index files. This format enables random access to any file in the dataset without downloading the entire archive.

**The indexed format** provides massive performance benefits:

- **Random access**: Jump to any file instantly
- **Selective downloads**: Only fetch the files you need
- **True parallelism**: Read from multiple shards simultaneously
- **Cloud-optimized**: Works efficiently with HTTP range requests
- **Aspect bucketing**: Optionally include image geometry hints `width`, `height` and `aspect` for the ability to bucket images by shape
- **Logical sample APIs**: Treat `image.ext` + `image.json` pairs as one sample while still allowing raw file access
- **Caption metadata**: Store captions under `captions` as strings, native JSON objects, or mixed lists of both
- **Custom DataLoader**: Includes state dict methods on the DataLoader so that you can resume training deterministically
- **Rate-limit friendly**: Local caching allows high-frequency random seeking without encountering storage provider rate limits
- **Instant start-up** with pre-sorted aspect buckets

**Growing ecosystem**: While not all datasets use this format yet, you can easily create indices for any tar-based dataset (see below).

## Quick Start

```python
import webshart

# Find your dataset
dataset = webshart.discover_dataset(
    source="laion/conceptual-captions-12m-webdataset",
    # we're able to upload metadata separately so that we reduce load on huggingface infra.
    metadata="webshart/conceptual-captions-12m-webdataset-metadata",
)
print(f"Found {dataset.num_shards} shards")

loader = webshart.TarDataLoader(dataset)

# File-oriented access is still available.
files = dataset.list_files_in_shard(0)

# Sample-oriented access skips paired JSON sidecars.
samples = dataset.list_samples_in_shard(0)
entry = loader.load_sample(0, 0)
print(entry.path, entry.captions, entry.json_metadata)
```

### Paired datasets

Two datasets can remain independently loadable while also exposing an opt-in
join by logical sample key. This works especially well for preference,
reference, and slider-training data stored in two subfolders of one repository:

```python
paired = webshart.discover_paired_dataset(
    "webshart/suno-various-94k",
    left_subfolder="original",
    right_subfolder="covers",
)

print(paired.num_pairs)
print(paired.get_pair(0))

loader = webshart.PairedTarDataLoader(paired)
sample = loader.load_pair(0)
print(sample.key, sample.left, sample.right)
```

The normal contract is unchanged: calling `discover_dataset(...,
subfolder="original")` or `subfolder="covers"` returns a standalone dataset.
Pair indexing is lazy, preserves left-dataset order, and validates identical key
sets by default. Pass `strict=False` to use only the intersection and inspect
`unmatched_left` / `unmatched_right`.

`max_file_size` is a visibility limit for loader APIs. Files larger than the
configured limit are omitted from iteration, batches, direct sample loading,
and aspect buckets instead of being returned with empty data. Direct
`load_sample()` calls return `None` for an oversized sample. The loader's
`list_samples_in_shard()` returns dictionaries containing `sample_idx` and
`filename`, so filtered listings retain the stable index required by
`load_sample()`.

## Native Hugging Face Xet Downloads

Hugging Face files stored in Xet are read directly by webshart's Rust transport.
Neither `hf_xet` nor `huggingface_hub` is needed for dataset reads. Non-Xet files
and other HTTP servers continue to use ordinary HTTP downloads and byte ranges.

The transport supports full-shard caching, individual sample/sidecar reads, and
batched range reads. It reconstructs files using concurrent 16 MiB windows,
decodes uncompressed, LZ4, and byte-grouped LZ4 chunks, and handles CAS v2
multi-range responses and the documented v1 compatibility path. Buffers and
request concurrency are bounded rather than proportional to the entire shard.
Full-shard downloads validate the size and Hub-provided SHA-256 before entering
the cache; partial reads validate ranges, chunk sizes, and reconstruction lengths.
Transient requests are retried, expired credentials/transfer URLs are refreshed,
and native Xet errors are reported rather than silently bypassing verification.

```python
dataset = webshart.discover_dataset("webshart/pseudo-camera-10k-structured", subfolder="data")
dataset.enable_shard_cache("cache/shards", cache_limit_gb=25, parallel_downloads=4)
loader = webshart.TarDataLoader(dataset)
loader.prepare_shards_ahead(4)
sample = loader.load_sample(0, 0)
```

`parallel_downloads` limits concurrent whole shards in each cache; prefetch must
be scheduled explicitly. Xet independently permits up to eight simultaneous
chunk transfers per process, including multiple transfers within one shard.
Set `WEBSHART_XET_CONCURRENCY=1..64` before the first read to change that limit.
`load_sample()` releases the Python GIL during I/O. Without a shard cache, only
the requested sample ranges are reconstructed, not the entire archive.

Set `WEBSHART_DISABLE_XET=1` to explicitly use the ordinary HTTP path instead.
`HF_ENDPOINT` selects the Hub origin recognized by the Xet transport for custom
resolve URLs; it does not change repository discovery's Hub endpoint.

Implemented from the public [download protocol](https://huggingface.co/docs/xet/download-protocol),
[authentication specification](https://huggingface.co/docs/xet/auth), and
[xorb format](https://huggingface.co/docs/xet/xorb). Run the opt-in live test with
`cargo test --lib xet::tests::live_public_hub_shard -- --ignored --nocapture`;
it downloads a 287 MiB public shard, verifies its SHA-256, and compares partial
reads with independent HTTP ranges.
`WEBSHART_TEST_LIVE_XET=1 .venv/bin/python -m pytest -q tests/test_xet_live.py`
also exercises the Python SDK, cache, batched reads, and ordinary HTTP opt-out
with both Hugging Face Python libraries blocked from import.

## Common Patterns

For real-world, working examples:

- [Use as a DataLoader](/examples/dataloader.py)
- [Retrieve data subset/range](/examples/retrieve_range.py)
- [Get dataset statistics without downloading](/examples/dataset_stats.py)
- [List aspect buckets](/examples/aspect_bucketing.py)
- [Write captions into metadata](/examples/write_captions_to_metadata.py)

## Creating Indices for / Converting Existing Datasets

Any tar-based webdataset can benefit from indexing! Webshart includes tools to generate indices:

A command-line tool that auto-discovers tars to process:

```bash
% webshart extract-metadata \
    --source laion/conceptual-captions-12m-webdataset \
    --destination laion_output/ \
    --checkpoint-dir ./laion_output/checkpoints \
    --max-workers 2 \
    --include-image-geometry
```

Or, if you prefer/require direct-integration to an existing Python application, [use the API](/examples/metadata_extractor.py)

### Uploading Indices to HuggingFace

Once you've generated indices, share them with the community:

```bash
# Upload all JSON files to your dataset
huggingface-cli upload --repo-type=dataset \
    username/dataset-name \
    ./indices/ \
    --include "*.json" \
    --path-in-repo "indices/"
```

Or if you want to contribute to an existing dataset you don't own:

1. Create a community dataset with indices: `username/original-dataset-indices`
2. Upload the JSON files there
3. Open a discussion on the original dataset suggesting they add the indices

### Creating New Indexed Datasets

If you're creating a new dataset, generate indices during creation:

```json
{
  "files": {
    "image_0001.webp": {"offset": 512, "length": 102400},
    "image_0002.webp": {"offset": 102912, "length": 98304},
    ...
  }
}
```

The JSON index should have the same name as the tar file (e.g., `shard_0000.tar` → `shard_0000.json`).

### Caption layouts and sidecars

Webshart recognizes both JSON metadata sidecars and plain-text caption sidecars:

```text
sample_0001.webp
sample_0001.json
sample_0002.webp
sample_0002.txt
```

Paired `.json` and `.txt` members are excluded from logical sample indexes. You
can inspect the layout from shard metadata without downloading tar members:

```python
layout = dataset.probe_caption_layout(max_shards=16)
print(layout["layout"])  # embedded, json_sidecar, txt_sidecar, mixed, or none
```

When metadata is extracted or loaded, sidecars are attached to their paired sample entries:

```json
{
  "files": {
    "sample_0001.webp": {
      "offset": 512,
      "length": 102400,
      "width": 1024,
      "height": 1024,
      "aspect": 1.0,
      "json_path": "sample_0001.json",
      "json_offset": 103424,
      "json_length": 128,
      "captions": "a product photo on a white background",
      "json_metadata": {
        "caption": "a product photo on a white background"
      }
    },
    "sample_0001.json": {
      "offset": 103424,
      "length": 128
    }
  }
}
```

Use file-oriented APIs when you want every archive member, including sidecars:

```python
dataset.list_files_in_shard(0)

reader = dataset.open_shard(0)
raw_file_bytes = reader.read_file(0)
```

Use sample-oriented APIs when you want training samples:

```python
dataset.list_samples_in_shard(0)
dataset.get_shard_sample_count(0)

reader = dataset.open_shard(0)
image_bytes = reader.read_sample(0)
json_bytes = reader.read_sample_json(0)

entry = loader.load_sample(0, 0)
print(entry.path)
print(entry.captions)
print(entry.json_data)

# Direct caption lookup also handles paired .txt sidecars.
caption = loader.load_caption(0, 0)
```

Captions are canonicalized to the plural `captions` metadata key. The value may be
a string, a native JSON object, a list containing either, or absent. Nested
objects, arrays, numbers, and booleans inside caption objects retain their JSON
types. `entry.captions` returns the complete value; `entry.caption` and
`loader.load_caption()` return the first string or object.

The optimizer and caption coalescer recognize JSON objects and mixed caption
lists in `.txt` sidecars. Plain text remains a string. JSON sidecars support
native values under their caption fields and standalone caption objects.

```python
webshart.write_captions_to_metadata(
    "shard_0000.json",
    {"sample_0001.webp": {"description": "a café", "elements": [{"bbox": [1, 2, 30, 40]}]}},
)
```

```python
webshart.write_captions_to_metadata(
    "shard_0000.json",
    {
        "sample_0001.webp": "a short caption",
        "sample_0002": ["caption one", "caption two"],
    },
)
```

The writer updates existing webshart metadata JSON in place, removes old singular `caption` keys from updated samples, and leaves paired `.json` sidecar entries untouched.

### Renaming metadata fields

Permanently rename a field across a dataset's JSON indexes:

```python
dataset = webshart.discover_dataset("/path/to/dataset")
updated = dataset.field_rename("captions", "v1_captions")

loader = webshart.TarDataLoader(dataset)
entry = loader.load_sample(0, 0)
print(entry.metadata["v1_captions"])
```

`field_rename(old_name, new_name, *, overwrite=False, destination=None)` returns
the number of file entries renamed. It moves top-level keys within each index's
`files` entries, preserving values, custom fields, and dict/list index layouts.
It processes one shard's metadata at a time without reading or rewriting tar
payloads. Nested `json_metadata` and sidecar contents are left intact.

Missing fields are skipped; renaming a field to itself returns zero. Existing
destination fields raise `ValueError` unless `overwrite=True`. Structural index
fields such as offsets, lengths, and paths cannot be renamed. All indexes are
validated before replacements begin, and each file replacement is atomic;
an I/O failure during replacement can still leave earlier shards updated.

Local indexes are rewritten in place and their loaded/disk metadata caches are
invalidated. Recreate existing loaders after renaming. Custom names are available
through `entry.metadata` and `loader.get_metadata()`; `entry.captions` continues
to represent the canonical caption field and may still read unchanged sidecars.

For remote indexes, pass `destination="./renamed-metadata"` to write a local
export, then upload those JSON files separately. This also works for local
datasets when an export is preferred. The dataset uses the exported indexes
afterward, and tar files stay at their original locations.

### Coalescing caption metadata

To avoid repeated `.txt` range reads, fold all sidecar captions into standard
webshart metadata files. If metadata caching is enabled, omitting the destination
persists the enriched indexes in webshart's cache:

```python
dataset.enable_metadata_cache("cache/metadata", init_shard_count=0)
loader = webshart.TarDataLoader(dataset, load_file_data=False)
loader.coalesce_caption_metadata()

# Or create a portable export tree for copying or upload.
loader.coalesce_caption_metadata("caption-metadata")
webshart.upload_caption_metadata(
    "caption-metadata",
    "organization/dataset-metadata",
    hf_token="hf_...",
)
```

The CLI provides the same operation. `--shard-cache-dir` lets coalescing reuse
full cached shards instead of issuing one range read per sidecar:

```bash
webshart optimize-captions \
  --source organization/dataset \
  --metadata organization/dataset-metadata \
  --destination caption-metadata \
  --shard-cache-dir cache/shards \
  --push-to-hub organization/dataset-metadata
```

`optimize-captions` expects existing `.tar` shards and webshart indexes. To
fully repackage a repository of loose media plus `.txt`/`.json` sidecars, or a
legacy SimpleTuner layout containing unindexed `.tar` archives whose member
filenames are captions, use the rolling `optimize-dataset` command instead:

```bash
webshart optimize-dataset \
  --source stablellama/Qwen-Image-2512_samples \
  --push-to-hub stablellama/Qwen-Image-2512_samples \
  --output-prefix webshart \
  --max-shard-size-gb 1
```

The target is always a Hugging Face **dataset** repository. It may be the same
repository as the source because generated files live under `--output-prefix`.
The input layout is detected automatically. Loose sidecars are coalesced into
metadata. Legacy tar members are repacked into bounded shards and their
filename stems become captions, matching SimpleTuner's filename strategy
(underscores become spaces). Remote legacy inputs use aligned HTTP ranges from
the saved member offset and retain only the current output shard locally.

After each shard is indexed, its sidecar captions are embedded in the JSON
index and the tar, index, and `.webshart-optimize-state.json` are uploaded in a
single commit. The state records relative positions and conversion settings,
never local absolute paths. For legacy tars this includes the source archive
index and tar-block member offset, so a rerun resumes within an archive after
the last committed output shard. Use `--max-shards N` to bound each worker
invocation.

For a local-only conversion, replace `--push-to-hub` with a local destination:

```bash
webshart optimize-dataset \
  --source /datasets/loose-pairs \
  --destination /datasets/indexed \
  --max-shards 10
```

Plain-text sidecars are omitted from the tar after their captions are embedded.
JSON sidecars are likewise coalesced: recognized caption fields become the
canonical `captions` value and the complete object is retained as
`json_metadata` in the index.

Hub reads accept `hf_token=` and also honor `HF_TOKEN`. This includes gated
datasets and separately hosted metadata. Local discovery recursively pairs tar
and JSON indexes, preserving their relative subdirectories.

### Aspect Bucketing Samples

`list_shard_aspect_buckets()` is file-oriented and buckets any indexed file that has `width` and `height`.

For training pipelines, prefer `list_shard_sample_aspect_buckets()`:

```python
loader = webshart.TarDataLoader(dataset)
buckets = loader.list_shard_sample_aspect_buckets(
    [0],
    key="geometry-tuple",
    target_pixel_area=1024**2,
)[0]["buckets"]

for bucket_key, entries in buckets.items():
    for item in entries:
        virtual_id = f"webshart://0/{item['sample_idx']}/{item['filename']}"
        image = loader.load_sample(0, item["sample_idx"])
```

This uses logical samples from `metadata.sample_range()` / `get_sample_by_index()` and excludes paired JSON sidecars before bucketing. Each bucket entry includes `sample_idx`, so callers can build stable IDs and load images directly with `loader.load_sample(shard_idx, sample_idx)`.

## Parquet Payloads With Custom Metadata

The Parquet API is available in this source checkout (not in the published 0.6.0
wheel). Existing tar APIs keep their behavior. Use `ParquetDataLoader`, not
`TarDataLoader` or `BucketDataLoader`, for Parquet sources:

```python
dataset = webshart.discover_dataset(
    "InternVL-U/ScaleEdit-12M",
    format="parquet",
    metadata="webshart/ScaleEdit-12M-metadata",
)
loader = webshart.ParquetDataLoader(
    dataset, image_columns=["source_image", "edited_image"]
)
metadata = loader.get_metadata(0, 0)
sample = loader.load_sample(0, 0)
edited_bytes = sample.data
source_bytes = sample.payloads["source_image"]
instruction = sample.captions
```

Local directories work too. Without separate metadata, specify the source column
names with `image_column`, `caption_column`, `width_column`, and `height_column`.
`discover_parquet_dataset(...)` is equivalent to discovery with `format="parquet"`.
Hub discovery requires `webshart[hub]`; payload decoding is native Rust and does
not require PyArrow, `datasets`, or `hf_xet`. Hub payload ranges reuse webshart's
native Xet transport, with HTTP Range for non-Xet files.

### Metadata manifest

Pass a local metadata directory or Hub dataset repository containing
`webshart-parquet.json`. The source files stay unchanged. For example:

```json
{
  "format": "webshart-parquet",
  "version": 1,
  "source": {"repo_id": "owner/source", "revision": "<40-character commit SHA>"},
  "columns": {"image": "image", "captions": "instruction", "width": "width", "height": "height"},
  "metadata_mode": "overlay",
  "shards": [{
    "path": "data/part-000.parquet",
    "num_rows": 2,
    "metadata_path": "custom/part-000.json"
  }]
}
```

The JSON metadata file is a list of row objects:

```json
[
  {"source_row": 1, "captions": [{"scene": "sunset"}, "warm lighting"], "rating": 5},
  {"source_row": 0, "captions": {"scene": "forest", "lighting": "soft"}}
]
```

Alternatively, `metadata_path` can name a Parquet file with a `source_row`
integer column and arbitrary custom columns. Row order and source IDs do not
matter: the join uses the **source shard path plus zero-based physical row**.
Optional `source_path` values must match the manifest. Duplicates, invalid row
positions, unsafe paths, mismatched file sizes/row counts, and incorrect optional
`metadata_sha256` checksums fail explicitly. `size`, when present, is the source
Parquet file size in bytes. Remote manifests must pin their source commit;
`metadata_revision` can additionally pin the metadata repository commit.

`overlay` merges custom values over source non-binary columns and permits sparse
overrides. Custom `captions`, `width`, and `height` override configured aliases.
`replace` reads metadata only from the companion file and requires every row to
be present. It avoids reading upstream metadata columns. Payload columns cannot
be overridden. Metadata is cached per shard with a bounded LRU (default: one
shard; adjust `metadata_cache_size`). Recreate the dataset after changing files.
Companion files up to 64 MiB are buffered while decoding so many small Parquet
pages do not cause repeated network round trips; image payload files are not.

Parquet structs, lists, and maps retain their native Python representations.
For mixed strings/objects that Parquet cannot represent in one homogeneous
column, store JSON text and declare `"json_columns": ["captions"]` in the
manifest (or discovery options). Only explicitly declared columns are parsed;
ordinary caption strings are not guessed or stringified. Decimal values remain
`decimal.Decimal`; date/time values use their Parquet integer units.

### Streaming and access costs

`loader.iter_shard(shard_index, start=0, stop=None)` streams selected image
columns page by page, reusing dictionaries. `load_sample(shard_index, row_index)`
skips preceding row groups but scans earlier rows within the selected group.
Unlike indexed tar, compressed Parquet does **not** provide constant-cost
per-image byte offsets. Prefer sequential iteration for large row groups.
Memory scales with selected columns' decoded pages/dictionaries, not an entire
shard; oversized dictionaries can still require hundreds of MiB. The
`max_page_bytes` discovery option limits individual compressed page/footer reads
(default 512 MiB), not total or decompressed memory.

Binary image columns and Hugging Face `{bytes, path}` image structs are supported.
Missing bytes produce `sample.data is None`; original structs remain in
`sample.payloads`. External URLs/path-only images are **never fetched
automatically**. Keep URL/hash metadata and apply your own retrieval policy.
The loader does not rewrite or download whole source shards, and does not expose
tar-only byte-offset, bucket, shard-cache, or field-rewrite APIs.

## Why is indexed tar fast?

**Problem**: Standard tar files require sequential reading. To get file #10,000, you must read through files #1-9,999 first.

**Solution**: The indexed format stores byte offsets and sample metadata in a separate JSON file, enabling:

- HTTP range requests for any file
- True random access over network
- Parallel reads from multiple shards
- Large scale, aspect-bucketed datasets
- No wasted bandwidth

The Rust implementation provides:

- Real parallelism (no Python GIL)
- Zero-copy operations where possible
- Efficient HTTP connection pooling
- Optimized tokio async runtime
- Optional local caching for metadata and shards
- Fast aspect bucketing for image data

## Datasets Using This Format

I discovered after creating this library that [cheesechaser](https://github.com/deepghs/cheesechaser) is the origin of the indexed tar format, which webshart has formalised and extended to include aspect bucketing support.

- `NebulaeWis/e621-2024-webp-4Mpixel`
- `picollect/danbooru2` (subfolder: `images`)
- [`webshart/OpenVid-1M-webshart-indices`](https://huggingface.co/datasets/webshart/OpenVid-1M-webshart-indices) (indices for [`Dev-Jahn/OpenVid-1M-wds`](https://huggingface.co/datasets/Dev-Jahn/OpenVid-1M-wds))
- Many picollect image datasets
- Your dataset could be next! See "Creating Indices" above

## Requirements

- Python 3.12+
- Linux/macOS/Windows

## Roadmap

- image decoding is currently not handled by this library, but it will be added with zero-copy.
- more informative API for caching and other Rust implementation details
- multi-gpu/multi-node friendly dataloader

## Projects using webshart

- [CaptionFlow](https://github.com/bghira/CaptionFlow) uses this library to solve memory use and seek performance issues typical to webdatasets

## License

MIT
