#!/bin/bash
# Quick setup script for LibriSpeech training

set -e

echo "================================"
echo "LibriSpeech Setup for Q2"
echo "================================"
echo ""

# Check if tar file exists
if [ ! -f "train-clean-100.tar.gz" ]; then
    echo "❌ Error: train-clean-100.tar.gz not found in current directory"
    echo "   Expected: $(pwd)/train-clean-100.tar.gz"
    exit 1
fi

echo "✓ Found train-clean-100.tar.gz ($(du -h train-clean-100.tar.gz | cut -f1))"
echo ""

# Extract
echo "📦 Extracting LibriSpeech (this may take a few minutes)..."
tar -xzf train-clean-100.tar.gz
echo "✓ Extraction complete"
echo ""

# Verify
if [ -d "LibriSpeech/train-clean-100" ]; then
    SPEAKER_COUNT=$(ls LibriSpeech/train-clean-100 | wc -l)
    echo "✓ LibriSpeech ready!"
    echo "  - Speakers: $SPEAKER_COUNT"
    echo "  - Location: $(pwd)/LibriSpeech/train-clean-100"
else
    echo "❌ Extraction failed - LibriSpeech directory not found"
    exit 1
fi

echo ""
echo "================================"
echo "Next Steps:"
echo "================================"
echo ""
echo "1. Navigate to q2 folder:"
echo "   cd q2"
echo ""
echo "2. Install dependencies:"
echo "   pip install torch==2.1.0 torchaudio==2.1.0 pyyaml scipy numpy"
echo ""
echo "3. Train baseline:"
echo "   python train.py --config configs/librispeech_baseline.yaml"
echo ""
echo "4. Train AE-Disentangler:"
echo "   python train.py --config configs/librispeech_disentangler.yaml --encoder_ckpt checkpoints/librispeech_baseline/best.pt"
echo ""
echo "5. Train VAE-Disentangler:"
echo "   python train.py --config configs/librispeech_vae.yaml --encoder_ckpt checkpoints/librispeech_baseline/best.pt"
echo ""
echo "For more info, see: q2/LIBRISPEECH_SETUP.md"
echo ""
