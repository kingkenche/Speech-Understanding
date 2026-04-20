#!/usr/bin/env python3
"""Quick audio segment extraction tool"""
import sys
sys.path.insert(0, '/home/m25csa028/A-2')

import librosa
import soundfile as sf
from pathlib import Path

# Load the downloaded segment
print("Loading downloaded audio...")
y, sr = librosa.load('data/audio/original_segment.wav', sr=None)
duration_sec = len(y) / sr
duration_min = duration_sec / 60

print(f"Current segment: {duration_sec:.1f} seconds ({duration_min:.1f} minutes) at {sr}Hz")
print(f"File size: {Path('data/audio/original_segment.wav').stat().st_size / 1e6:.1f} MB")

# If longer than 10 minutes, trim to 10 minutes
if duration_sec > 600:  # 10 minutes = 600 seconds
    print("\nSegment is longer than 10 minutes. Trimming to 10 minutes (600 seconds)...")
    y_trimmed = y[:600 * sr]
    sf.write('data/audio/original_segment.wav', y_trimmed, sr, subtype='PCM_16')
    print(f"✅ Trimmed to 10 minutes and saved")
    print(f"New size: {Path('data/audio/original_segment.wav').stat().st_size / 1e6:.1f} MB")
