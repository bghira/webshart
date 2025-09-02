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
    """
    `key` can be `aspect`, `geometry-tuple` or `geometry-list` depending on how you require aspect buckets to be indexed.
    - aspect: use the float value of w / h
    - geometry-tuple: use the tuple (w, h)
    - geometry-list: use the list [w, h]
    """
    buckets_info = loader.list_shard_aspect_buckets(
        [shard],
        key="geometry-tuple",
        target_pixel_area=1024
        ** 2,  # you can modify this to 512**2 or 768**2 etc for different target resolutions
    )[0]["buckets"]
    # for bucket_key in buckets_info.keys():
    #     # each of the items has the original_size element, if the original was transformed.
    #     print(f"Bucket {bucket_key}")
    #     print(f"- {len(buckets_info[bucket_key])} items")
    #     print(f"- Original size: {buckets_info[bucket_key][0]['original_size']}")
