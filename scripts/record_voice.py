#!/usr/bin/env python3
"""
Interactive Voice Recording Script

Records exactly 60 seconds of your voice for speaker embedding extraction.
Used for Phase 0 data acquisition (Task 3.1).

Usage:
    python scripts/record_voice.py --output data/audio/student_voice_ref.wav \\
                                   --duration 60 \\
                                   --sample-rate 22050
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def record_audio(
    duration: int = 60,
    sample_rate: int = 22050,
    channels: int = 1,
    dtype: str = "float32",
    verbose: bool = True
) -> np.ndarray:
    """
    Record audio from microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels (1 for mono)
        dtype: Data type for recording
        verbose: Print progress information

    Returns:
        Numpy array of recorded audio
    """
    if verbose:
        logger.info(f"Recording {duration}s of audio at {sample_rate}Hz...")
        logger.info("🎤 Ready to record. Click START to begin (you will see real-time level meters).")

    # Record with progress indication
    num_frames = duration * sample_rate
    frames_per_chunk = sample_rate // 10  # 100ms chunks for progress updates

    recording = []

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            blocksize=frames_per_chunk
        ) as stream:
            logger.info("📍 Recording started... ▶️")
            for i in range(num_frames // frames_per_chunk):
                chunk, overflowed = stream.read(frames_per_chunk)
                recording.append(chunk)
                if overflowed:
                    logger.warning("Audio buffer overflow detected")

                # Simple level meter
                if verbose and (i + 1) % 5 == 0:
                    elapsed = (i + 1) * frames_per_chunk / sample_rate
                    progress = "█" * int(elapsed / 2) + "░" * (30 - int(elapsed / 2))
                    max_level = np.max(np.abs(chunk))
                    logger.info(f"  {progress} [{elapsed:.0f}s/{duration}s] Level: {max_level:.3f}")

    except Exception as e:
        logger.error(f"Recording error: {e}")
        raise

    audio = np.concatenate(recording, axis=0)
    if verbose:
        logger.info(f"✅ Recording complete! Captured {len(audio) / sample_rate:.2f}s of audio.")

    return audio


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 22050,
    normalize: bool = True
) -> None:
    """
    Save recorded audio to file.

    Args:
        audio: Audio array
        output_path: Output file path
        sample_rate: Sample rate in Hz
        normalize: Normalize audio to [-1, 1] range
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    logger.info(f"💾 Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Record 60 seconds of your voice for speaker embedding"
    )
    parser.add_argument(
        "--output",
        default="data/audio/student_voice_ref.wav",
        help="Output WAV file path"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Recording duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate in Hz (default: 22050)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index (leave None for default)"
    )

    args = parser.parse_args()

    # List available audio devices if needed
    try:
        logger.info("\n🎵 Available Audio Devices:")
        logger.info(sd.query_devices())
        logger.info("\n")

        # Set device if specified
        if args.device is not None:
            sd.default.device = args.device

        # Record audio
        audio = record_audio(
            duration=args.duration,
            sample_rate=args.sample_rate,
            verbose=True
        )

        # Save audio
        save_audio(
            audio=audio,
            output_path=args.output,
            sample_rate=args.sample_rate,
            normalize=True
        )

        # Verification
        loaded_audio, sr = sf.read(args.output)
        duration_loaded = len(loaded_audio) / sr
        logger.info(f"\n✨ Verification: Loaded {duration_loaded:.2f}s at {sr}Hz")
        logger.info(f"📊 Audio shape: {loaded_audio.shape}, dtype: {loaded_audio.dtype}")

    except Exception as e:
        logger.error(f"Error during recording: {e}")
        raise


if __name__ == "__main__":
    main()
