from webshart import BucketDataLoader, discover_dataset
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
print("Enabling cache.")
dataset.enable_metadata_cache(location=os.path.join(os.getcwd(), "metadata_cache"))
print("Cache enabled. Creating loader.")
loader = BucketDataLoader(dataset, batch_size=4)
print("Loader OK.")

processed = 0

print("- Starting to iterate over dataloader. -")
for batch in loader.iter_batches():
    print(f"Processing batch: {batch}")
    processed += 1
    if processed >= 100:
        break

print(f"✅ Processed {processed} batches.")
