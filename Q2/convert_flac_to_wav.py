#!/usr/bin/env python3
"""
Convert LibriSpeech test-clean FLAC files to WAV format.
Preserves directory structure and metadata.
"""

import logging
from pathlib import Path
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_flac_to_wav(flac_dir: str, output_dir: str = None):
    """Convert all FLAC files in directory to WAV format."""

    flac_path = Path(flac_dir)

    # If no output dir specified, convert in place
    if output_dir is None:
        output_dir = flac_dir
    else:
        output_dir = Path(output_dir)

    # Find all FLAC files
    flac_files = list(flac_path.rglob("*.flac"))
    logger.info(f"Found {len(flac_files)} FLAC files")

    if len(flac_files) == 0:
        logger.warning("No FLAC files found!")
        return

    converted = 0
    failed = 0

    for i, flac_file in enumerate(flac_files, 1):
        logger.info(f"[{i}/{len(flac_files)}] Converting {flac_file.name}...")
        try:
            # Get relative path from root
            relative_path = flac_file.relative_to(flac_path)

            # Create output path (replace .flac with .wav)
            wav_file = output_dir / relative_path.parent / relative_path.stem
            wav_file = wav_file.with_suffix('.wav')

            # Create parent directories if needed
            wav_file.parent.mkdir(parents=True, exist_ok=True)

            # Load FLAC and save as WAV
            waveform, sample_rate = torchaudio.load(str(flac_file))
            torchaudio.save(str(wav_file), waveform, sample_rate)

            converted += 1

        except Exception as e:
            logger.error(f"Failed to convert {flac_file}: {e}")
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Conversion Complete!")
    logger.info(f"Successfully converted: {converted}/{len(flac_files)}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert LibriSpeech FLAC to WAV')
    parser.add_argument('--input_dir', default='./LibriSpeech', help='Input directory with FLAC files')
    parser.add_argument('--output_dir', default=None, help='Output directory for WAV files (default: same as input)')
    parser.add_argument('--keep_flac', action='store_true', help='Keep original FLAC files')
    args = parser.parse_args()

    input_path = Path(args.input_dir)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return

    # Convert files
    convert_flac_to_wav(str(input_path), args.output_dir)

    # Optionally delete FLAC files
    if not args.keep_flac:
        logger.info("Deleting original FLAC files...")
        flac_files = list(input_path.rglob("*.flac"))
        for i, flac_file in enumerate(flac_files, 1):
            try:
                logger.info(f"[{i}/{len(flac_files)}] Deleting {flac_file.name}...")
                flac_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete {flac_file}: {e}")
        logger.info(f"Deleted {len(flac_files)} FLAC files")


if __name__ == '__main__':
    main()
