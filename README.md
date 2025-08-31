<img width="1530" height="492" alt="image" src="https://github.com/user-attachments/assets/1ff6cd93-1261-42ff-b120-16509fa3cc27" />


Fast parallel reader for webdataset tar shards. Rust core with Python bindings. Built for streaming large video and image datasets, but handles any byte data.

## Install

```bash
pip install webshart
```

## What is this?

Webshart is a fast reader for a specific webdataset format: tar files with separate JSON index files. This format enables random access to any file in the dataset without downloading the entire archive.

**The format is rare** but used by some large image datasets:
- `NebulaeWis/e621-2024-webp-4Mpixel`
- `picollect/danbooru2` (subfolder: `images`)
- Other picollect datasets

**Not a replacement** for HF datasets or the webdataset library - just a purpose-built tool for this indexed format.

**Performance**: 10-20x faster for random access, 5-10x faster for batch reads compared to standard tar extraction.

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

Quick dataset stats:
```python
# Without downloading anything
size, num_files = dataset.quick_stats()
print(f"Dataset size: {size / 1e9:.1f} GB")
```

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

## Creating indexed datasets

If you're making a new webdataset, consider using the indexed format:

```json
{
  "files": {
    "image_0001.webp": {"offset": 512, "length": 102400},
    "image_0002.webp": {"offset": 102912, "length": 98304},
    ...
  }
}
```

This enables random access over HTTP, making cloud-stored datasets as fast as local ones for many use cases.

## Requirements

- Python 3.8+
- Linux/macOS/Windows

## License

MIT
