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
- **Aspect bucketing**: Optionally include image geometry hints `width`, `height` and `aspect` for the ability to bucket by shape
- **Custom DataLoader**: Includes state dict methods on the DataLoader so that you can resume training deterministically
- **Rate-limit friendly**: Local caching allows high-frequency random seeking without encountering storage provider rate limits
- **Instant start-up** with pre-sorted aspect buckets

**Growing ecosystem**: While not all datasets use this format yet, you can easily create indices for any tar-based dataset (see below).

## Quick Start

```python
import webshart

# Find your dataset
dataset = discover_dataset(
    source="laion/conceptual-captions-12m-webdataset",
    # we're able to upload metadata separately so that we reduce load on huggingface infra.
    metadata="webshart/conceptual-captions-12m-webdataset-metadata",
)
print(f"Found {dataset.num_shards} shards")
```

## Common Patterns

For real-world, working examples:

- [Use as a DataLoader](/examples/dataloader.py)
- [Retrieve data subset/range](/examples/retrieve_range.py)
- [Get dataset statistics without downloading](/examples/dataset_stats.py)

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
- Many picollect image datasets
- Your dataset could be next! See "Creating Indices" above

## Requirements

- Python 3.8+
- Linux/macOS/Windows

## Roadmap

- image decoding is currently not handled by this library, but it will be added with zero-copy.
- more informative API for caching and other Rust implementation details
- multi-gpu/multi-node friendly dataloader

## License

MIT
