from webshart import TarDataLoader, discover_dataset
from huggingface_hub import get_token
import os
import pickle

hf_token = get_token()
dataset = discover_dataset("NebulaeWis/e621-2024-webp-4Mpixel", hf_token=hf_token)
loader = TarDataLoader(dataset)

# Resume from checkpoint if it exists
checkpoint_file = "dataloader_state.pkl"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "rb") as f:
        state = pickle.load(f)
    loader.load_state_dict(state)
    print(f"📂 Resumed from checkpoint: {loader.state_dict()}")

# Or manually set position:
# loader.shard(shard_idx=0)                    # Jump to a specific shard
# loader.shard(shard_idx=0, cursor_idx=100)    # Jump to shard 0, file 100
# loader.skip(1000)                            # Skip to global file index 1000

processed = 0

for entry in loader:
    data = entry.data
    processed += 1

    # Save checkpoint every 50 files
    if processed % 50 == 0:
        with open(checkpoint_file, "wb") as f:
            pickle.dump(loader.state_dict(), f)

    if processed >= 100:
        break

print(f"✅ Processed {processed} files.")
