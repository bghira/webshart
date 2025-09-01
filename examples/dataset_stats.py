import webshart
from huggingface_hub import get_token

dataset = webshart.discover_dataset(
    "NebulaeWis/e621-2024-webp-4Mpixel", hf_token=get_token()
)

# Quick stats (instant, uses cached values if available)
stats = dataset.get_stats()
print(f"Total shards: {stats['total_shards']}")
print(f"Estimated total files: {stats.get('total_files', 'Unknown')}")

# Detailed stats (loads all metadata)
# detailed = dataset.get_detailed_stats()
# print(f"Exact total files: {detailed['total_files']:,}")
# print(f"Average files per shard: {detailed['average_files_per_shard']:.1f}")

# Pretty print summary
dataset.print_summary(detailed=False)

# Get info for specific shard
file_count = dataset.get_shard_file_count(0)
shard_info = dataset.get_shard_by_name("data-0042")
print(f"Shard 'data-0042' has {shard_info['num_files']} files")
