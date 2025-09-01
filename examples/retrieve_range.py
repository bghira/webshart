"""
⚠️ This method uses several parallel connections to the source dataset, which will
result in **429 Too Many Requests** responses from providers like Hugging Face Hub.
It is intended for low-volume IO, with better solutions on the way for extended streaming.

Alternatively, if you host your own minIO or similar, feel free to use this API, it is fast.
"""

import webshart
from huggingface_hub import get_token

dataset = webshart.discover_dataset(
    "NebulaeWis/e621-2024-webp-4Mpixel", hf_token=get_token()
)

# Read files 0-100 from each of the first 10 shards
requests = []
for shard_idx in range(10):
    for file_idx in range(100):
        requests.append((shard_idx, file_idx))

# Batch read in chunks of 500 files
for chunk_idx, i in enumerate(range(0, len(requests), 500)):
    byte_list = webshart.read_files_batch(dataset, requests[i : i + 500])
    for j, data in enumerate(byte_list):
        if data:  # process successful reads
            # Save with meaningful names
            shard, file = requests[i + j]
            with open(f"shard_{shard:04d}_file_{file:04d}.webp", "wb") as f:
                f.write(data)
