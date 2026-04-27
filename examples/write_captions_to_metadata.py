import webshart


updated = webshart.write_captions_to_metadata(
    "shard-0000.json",
    {
        "image_0001.webp": "a concise caption",
        "image_0002": ["first caption", "alternate caption"],
    },
)

print(f"Updated {updated} sample metadata entries.")
