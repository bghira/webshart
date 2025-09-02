from webshart import TarDataLoader, discover_dataset
from huggingface_hub import get_token
import os
import pickle

hf_token = get_token()
dataset = discover_dataset(
    source="laion/conceptual-captions-12m-webdataset",
    metadata="webshart/conceptual-captions-12m-webdataset-metadata",
    # subfolder="data",
    hf_token=hf_token,
)
loader = TarDataLoader(dataset)

shards_to_bucket = [0]
for shard in shards_to_bucket:
    buckets_info = loader.list_shard_aspect_buckets(
        [shard],
        key="geometry-tuple",
        target_pixel_area=1024**2,
    )[0]["buckets"]
    for bucket_key in buckets_info.keys():
        print(f"Bucket {bucket_key}: {len(buckets_info[bucket_key])} items")
