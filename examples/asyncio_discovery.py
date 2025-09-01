import asyncio
import time
import webshart

sources = [
    "webshart/conceptual-captions-12m-webdataset-indexed",
    "NebulaeWis/e621-2024-webp-4Mpixel",
    "picollect/danbooru2",
]

subfolders = ["original", None, None, None, None]


async def discover_all_async():
    """Discover all datasets concurrently using asyncio."""
    tasks = []
    for source, subfolder in zip(sources, subfolders):
        task = webshart.discover_dataset_async(source, subfolder=subfolder)
        tasks.append(task)

    # Run all discoveries in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to None
    return [None if isinstance(r, Exception) else r for r in results]


print(f"   Discovering {len(sources)} datasets with asyncio...")
start = time.time()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
datasets = loop.run_until_complete(discover_all_async())
elapsed = time.time() - start

# Report results
successful = sum(1 for d in datasets if d is not None)
print(f"   ✓ Asyncio discovery completed in {elapsed:.3f}s")
print(f"   - Successful: {successful}/{len(sources)}")
print(f"   - Average per dataset: {elapsed/len(sources):.3f}s")
