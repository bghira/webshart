from webshart import MetadataExtractor
from huggingface_hub import get_token

# Create an extractor (optionally with HF token for private datasets)
extractor = MetadataExtractor(hf_token=get_token())

# Generate indices for a dataset
extractor.extract_metadata(
    source="username/dataset-name",  # HF dataset or local path
    destination="./indices/",  # Where to save JSON files
    max_workers=4,  # Parallel processing
    include_image_geometry=True,  # Not much slower, but far more useful
)
