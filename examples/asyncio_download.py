import asyncio
import time
import random
import webshart

from huggingface_hub import get_token

hf_token = get_token()
dataset = webshart.discover_dataset(
    "NebulaeWis/e621-2024-webp-4Mpixel", hf_token=hf_token
)

num_files = 20
requests = []
for _ in range(num_files):
    shard_idx = random.randint(0, min(10, dataset.num_shards - 1))
    file_idx = random.randint(0, 10)
    requests.append((shard_idx, file_idx))


async def read_file_async(shard_idx, file_idx):
    """Read a single file asynchronously."""
    try:
        # Run blocking I/O in thread pool
        loop = asyncio.get_event_loop()
        reader = await loop.run_in_executor(None, dataset.open_shard, shard_idx)
        data = await loop.run_in_executor(None, reader.read_file, file_idx)
        return data
    except Exception:
        return None


async def read_all_async():
    """Read all files concurrently."""
    tasks = [read_file_async(s, f) for s, f in requests]
    return await asyncio.gather(*tasks)


start = time.time()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
files = loop.run_until_complete(read_all_async())
elapsed = time.time() - start

# Calculate stats
successful = sum(1 for f in files if f is not None)
total_size = sum(len(f) for f in files if f is not None)

print(f"   ✓ Read {successful}/{num_files} files in {elapsed:.3f}s")
print(f"   - Average per file: {elapsed/num_files:.3f}s")
