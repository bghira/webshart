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
- **Caption metadata**: Store captions in shard metadata under the plural `captions` key as either a string or a list of strings
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

Captions are canonicalized to the plural `captions` metadata key. The value may be a single string, a list of strings, or absent.

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
convert a repository of loose media plus `.txt`/`.json` sidecars, use the
rolling `optimize-dataset` command instead:

```bash
webshart optimize-dataset \
  --source stablellama/Qwen-Image-2512_samples \
  --push-to-hub stablellama/Qwen-Image-2512_samples \
  --output-prefix webshart \
  --max-shard-size-gb 1
```

The target is always a Hugging Face **dataset** repository. It may be the same
repository as the source because generated files live under `--output-prefix`.
The converter streams one payload at a time and retains only the current shard
locally when `--destination` is omitted.

After each shard is indexed, its sidecar captions are embedded in the JSON
index and the tar, index, and `.webshart-optimize-state.json` are uploaded in a
single commit. The state records relative positions and conversion settings,
never local absolute paths. Rerunning the same command resumes after the last
committed shard. Use `--max-shards N` to bound each worker invocation.

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

## Why is it fast?

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
