import json
from pathlib import Path

Path("data/manifests").mkdir(parents=True, exist_ok=True)
manifest = []
for i in range(20):
    manifest.append({
        "audio_path": "data/audio/original_segment.wav",
        "language_id": i % 2 
    })
with open("data/manifests/train.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Created mock manifest")
