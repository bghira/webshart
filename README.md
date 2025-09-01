<img width="1530" height="492" alt="image" src="https://github.com/user-attachments/assets/ebf0d101-eae7-4908-bb73-a264bf89a479" />

Fast parallel reader for webdataset tar shards. Rust core with Python bindings. Built for streaming large video and image datasets, but handles any byte data.

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

**Performance**: 10-20x faster for random access, 5-10x faster for batch reads compared to standard tar extraction.

**Growing ecosystem**: While not all datasets use this format yet, you can easily create indices for any tar-based dataset (see below).

## Quick Start

```python
import webshart

# Find your dataset
dataset = webshart.discover_dataset("NebulaeWis/e621-2024-webp-4Mpixel", subfolder="original")
print(f"Found {dataset.num_shards} shards")

# Read a single file
shard = dataset.open_shard(0)
data = shard.read_file(42)  # -> bytes

# Read many files at once (fast)
byte_list = webshart.read_files_batch(dataset, [
    (0, 0),   # shard 0, file 0
    (0, 1),   # shard 0, file 1  
    (1, 0),   # shard 1, file 0
    (10, 5),  # shard 10, file 5
])

# Save the files
for i, data in enumerate(byte_list):
    if data:  # skip failed reads
        with open(f"image_{i}.webp", "wb") as f:
            f.write(data)
```

## Common Patterns

Stream a subset efficiently:

```python
# Read files 0-100 from each of the first 10 shards
requests = []
for shard_idx in range(10):
    for file_idx in range(100):
        requests.append((shard_idx, file_idx))

# Batch read in chunks of 500 files
for chunk_idx, i in enumerate(range(0, len(requests), 500)):
    byte_list = webshart.read_files_batch(dataset, requests[i:i+500])
    for j, data in enumerate(byte_list):
        if data:  # process successful reads
            # Save with meaningful names
            shard, file = requests[i+j]
            with open(f"shard_{shard:04d}_file_{file:04d}.webp", "wb") as f:
                f.write(data)
```

Get dataset statistics without downloading:

```python
# Quick stats (instant, uses cached values if available)
stats = dataset.get_stats()
print(f"Total shards: {stats['total_shards']}")
print(f"Estimated total files: {stats.get('total_files', 'Unknown')}")

# Detailed stats (loads all metadata)
detailed = dataset.get_detailed_stats()
print(f"Exact total files: {detailed['total_files']:,}")
print(f"Average files per shard: {detailed['average_files_per_shard']:.1f}")

# Pretty print summary
dataset.print_summary(detailed=True)

# Get info for specific shard
file_count = dataset.get_shard_file_count(0)
shard_info = dataset.get_shard_by_name('shard-0042')
```

## Creating Indices for Existing Datasets

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

Or, if you prefer/require direct-integration to an existing Python application, use the API:

```python
from webshart import MetadataExtractor

# Create an extractor (optionally with HF token for private datasets)
extractor = MetadataExtractor(hf_token="hf_...")

# Generate indices for a dataset
extractor.extract_metadata(
    source="username/dataset-name",  # HF dataset or local path
    destination="./indices/",        # Where to save JSON files
    max_workers=4,                   # Parallel processing
    include_image_geometry=True,     # Not much slower, but far more useful
)
```

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

## Batch Operations

```python
# Discover multiple datasets in parallel
datasets = webshart.discover_datasets_batch([
    "NebulaeWis/e621-2024-webp-4Mpixel",
    "picollect/danbooru2",
    "/local/path/to/dataset"
], subfolders=["original", "images", None])

# Process large dataset in chunks
processor = webshart.BatchProcessor()
results = processor.process_dataset(
    "NebulaeWis/e621-2024-webp-4Mpixel",
    batch_size=100,
    callback=lambda data: len(data)  # process each file
)
```

## Advanced

Local dataset:

```python
dataset = webshart.discover_dataset("/path/to/shards/")
```

Custom auth:

```python
# Pass token directly
dataset = webshart.discover_dataset("private/dataset", hf_token="hf_...")

# Or use your existing HF token from huggingface_hub
from huggingface_hub import get_token
token = get_token()
dataset = webshart.discover_dataset("private/dataset", hf_token=token)
```

Async interface (if you're already in async code):

```python
dataset = await webshart.discover_dataset_async("NebulaeWis/e621-2024-webp-4Mpixel")
```

## Why is it fast?

**Problem**: Standard tar files require sequential reading. To get file #10,000, you must read through files #1-9,999 first.

**Solution**: The indexed format stores byte offsets in a separate JSON file, enabling:

- HTTP range requests for any file
- True random access over network
- Parallel reads from multiple shards
- No wasted bandwidth

The Rust implementation provides:

- Real parallelism (no Python GIL)
- Zero-copy operations where possible
- Efficient HTTP connection pooling
- Optimized tokio async runtime

## Datasets Using This Format

- `NebulaeWis/e621-2024-webp-4Mpixel`
- `picollect/danbooru2` (subfolder: `images`)
- Many picollect image datasets
- Your dataset could be next! See "Creating Indices" above

## Requirements

- Python 3.8+
- Linux/macOS/Windows

## License

MIT
