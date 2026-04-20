"""Utilities for downloading and slicing lecture audio."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import soundfile as sf
import yt_dlp


def parse_timestamp(value: str) -> int:
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Invalid timestamp format: {value}")


def download_best_audio(url: str, out_dir: str = "data/raw/lecture_video") -> str:
    """Download best available audio from a video URL and return wav path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": str(out_path / "%(title)s.%(ext)s"),
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        maybe = Path(ydl.prepare_filename(info)).with_suffix(".wav")

    if maybe.exists():
        return str(maybe)

    matches = list(out_path.glob("*.wav"))
    if not matches:
        raise RuntimeError("No wav audio found after download")
    return str(matches[0])


def extract_audio_segment(
    input_audio: str,
    start_time: str,
    end_time: str,
    output_audio: str,
    target_sr: int = 22050,
) -> Tuple[str, float]:
    """Extract a segment from an audio file and resample it."""
    start_s = parse_timestamp(start_time)
    end_s = parse_timestamp(end_time)
    if end_s <= start_s:
        raise ValueError("end_time must be greater than start_time")

    signal, sr = librosa.load(input_audio, sr=None, mono=True)
    clipped = signal[start_s * sr : end_s * sr]
    clipped = librosa.resample(clipped, orig_sr=sr, target_sr=target_sr)

    out_file = Path(output_audio)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_file), clipped, target_sr, subtype="PCM_16")
    duration = len(clipped) / float(target_sr)
    return str(out_file), duration