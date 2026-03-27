#!/bin/bash
# batch_manifest_and_slurm.sh
# 1. Update data/manifest.txt for all .wav files
# 2. Create and submit a SLURM job for each .wav file

set -e

DATA_DIR="data"
RESULTS_DIR="results"
MANIFEST="$DATA_DIR/manifest.txt"

# 1. Update manifest with all .wav files
echo "# Auto-generated manifest for LibriSpeech test-clean" > "$MANIFEST"
echo "# filename | source | duration_s | sr_hz | description" >> "$MANIFEST"

total=$(ls $DATA_DIR/*.wav | wc -l)
count=0
for wav in $DATA_DIR/*.wav; do
    # Get duration in seconds
    dur=$(soxi -D "$wav" 2>/dev/null || ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$wav")
    echo "$wav | LibriSpeech test-clean | $dur | 16000 | Converted from .flac" >> "$MANIFEST"
    count=$((count+1))
    echo "[$count/$total] Added $wav to manifest."
done
