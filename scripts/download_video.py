#!/usr/bin/env python3
"""
Download & Extract Audio Segment from YouTube

This script downloads a YouTube video and extracts a specific time segment
as audio. Used for Phase 0 data acquisition.

Usage:
    python scripts/download_video.py --url <youtube_url> \\
                                      --start-time "2:20:00" \\
                                      --end-time "2:54:00" \\
                                      --output-path data/audio/original_segment.wav
"""

import argparse
import subprocess
import os
from pathlib import Path
import sys
import logging

import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seconds_from_timestamp(timestamp: str) -> int:
    """Convert HH:MM:SS or MM:SS format to seconds."""
    parts = timestamp.split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def download_video(youtube_url: str, output_dir: str = "data/raw/lecture_video/") -> str:
    """
    Download audio from YouTube video.

    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save downloaded audio

    Returns:
        Path to downloaded audio file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    logger.info(f"Downloading video from: {youtube_url}")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x",
        "--audio-format", "wav",
        "-o", output_template,
        youtube_url
    ]

    try:
        subprocess.run(cmd, check=True)
        # Find the downloaded file
        files = list(Path(output_dir).glob("*.wav"))
        if files:
            return str(files[0])
        else:
            raise RuntimeError("No audio file found after download")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download video: {e}")
        sys.exit(1)


def extract_segment(
    input_audio: str,
    start_time: str,
    end_time: str,
    output_audio: str,
    target_sr: int = 22050
) -> None:
    """
    Extract audio segment and resample to target sample rate.

    Args:
        input_audio: Path to input audio file
        start_time: Start timestamp (HH:MM:SS or MM:SS)
        end_time: End timestamp (HH:MM:SS or MM:SS)
        output_audio: Path to save extracted segment
        target_sr: Target sample rate (default 22.05 kHz)
    """
    start_sec = seconds_from_timestamp(start_time)
    end_sec = seconds_from_timestamp(end_time)

    logger.info(f"Loading audio from {input_audio}")
    y, sr = librosa.load(input_audio, sr=None)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    logger.info(f"Extracting segment: {start_time} ({start_sec}s) to {end_time} ({end_sec}s)")
    segment = y[start_sample:end_sample]

    logger.info(f"Resampling from {sr}Hz to {target_sr}Hz")
    segment_resampled = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)

    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio, segment_resampled, target_sr)

    duration = len(segment_resampled) / target_sr
    logger.info(f"Saved {duration:.2f}s segment to {output_audio}")


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video and extract audio segment"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="YouTube video URL"
    )
    parser.add_argument(
        "--start-time",
        default="2:20:00",
        help="Start time in HH:MM:SS format (default: 2:20:00)"
    )
    parser.add_argument(
        "--end-time",
        default="2:54:00",
        help="End time in HH:MM:SS format (default: 2:54:00)"
    )
    parser.add_argument(
        "--output-path",
        default="data/audio/original_segment.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=22050,
        help="Target sample rate in Hz (default: 22050)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download and use existing audio file"
    )
    parser.add_argument(
        "--input-audio",
        help="Path to existing audio file (used with --no-download)"
    )

    args = parser.parse_args()

    if args.no_download:
        if not args.input_audio:
            logger.error("--input-audio required when using --no-download")
            sys.exit(1)
        input_audio = args.input_audio
    else:
        input_audio = download_video(args.url)

    extract_segment(
        input_audio=input_audio,
        start_time=args.start_time,
        end_time=args.end_time,
        output_audio=args.output_path,
        target_sr=args.target_sr
    )


if __name__ == "__main__":
    main()
