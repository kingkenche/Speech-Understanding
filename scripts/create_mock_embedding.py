#!/usr/bin/env python3
"""
Create a mock speaker embedding for testing Phase 1-2 implementation.
You can replace this with your actual voice recording later.
"""

import json
import numpy as np
from pathlib import Path

# Create mock speaker embedding (256-dimensional vector)
speaker_embedding = np.random.randn(256).astype(np.float32)

# Save as PyTorch tensor format
import torch
embedding_tensor = torch.from_numpy(speaker_embedding)
torch.save(embedding_tensor, 'data/processed/embeddings/speaker_embedding.pt')

# Also save as JSON for reference
embedding_dict = {
    "shape": [256],
    "dtype": "float32",
    "note": "Mock embedding - replace with actual speaker embedding from voice recording",
    "values": speaker_embedding.tolist()[:10]  # First 10 values for reference
}

Path('data/processed/embeddings').mkdir(parents=True, exist_ok=True)
with open('data/processed/embeddings/speaker_embedding.json', 'w') as f:
    json.dump(embedding_dict, f, indent=2)

print("✅ Mock speaker embedding created:")
print(f"   Shape: {speaker_embedding.shape}")
print(f"   Saved to: data/processed/embeddings/speaker_embedding.pt")
print(f"\n📝 Note: This is a random placeholder.")
print(f"   Once you can record your voice, replace it with actual embedding:")
print(f"   python scripts/record_voice.py --output data/audio/student_voice_ref.wav")
print(f"   Then extract embedding in Part III")
