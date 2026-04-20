#!/usr/bin/env python3
"""
Alternative Video Download Script (without yt-dlp command-line requirement)

Uses yt-dlp Python library directly instead of subprocess call.
"""

import argparse
import logging
from pathlib import Path
import sys

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
    Download audio from YouTube video using yt-dlp library.
    """
    import yt_dlp

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading video from: {youtube_url}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(Path(output_dir) / '%(title)s'),
        'quiet': False,
        'no_warnings': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
            # Replace mp4/webm with wav
            wav_filename = filename.rsplit('.', 1)[0] + '.wav'
            if Path(wav_filename).exists():
                return wav_filename
            # Try to find any wav file in directory
            wav_files = list(Path(output_dir).glob("*.wav"))
            if wav_files:
                return str(wav_files[0])
            else:
                raise RuntimeError("No audio file found after download")
    except Exception as e:
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
    sf.write(output_audio, segment_resampled, target_sr, subtype="PCM_16")

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
        help="Start time in HH:MM:SS format"
    )
    parser.add_argument(
        "--end-time",
        default="2:54:00",
        help="End time in HH:MM:SS format"
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
        help="Target sample rate in Hz"
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
