"""Extract F0 and energy contours from source and synthesized speech."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class ProsodyFeatures:
    f0: np.ndarray
    energy: np.ndarray
    times: np.ndarray


def extract_prosody(audio_path_or_y: str | np.ndarray, sr: int = 22050, hop_length: int = 256) -> ProsodyFeatures:
    if isinstance(audio_path_or_y, (str, Path)):
        y, _ = librosa.load(audio_path_or_y, sr=sr, mono=True)
    else:
        y = audio_path_or_y
    
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=450, sr=sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    rms = librosa.feature.rms(y=y, hop_length=hop_length).squeeze(0)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return ProsodyFeatures(f0=f0.astype(np.float32), energy=rms.astype(np.float32), times=times)


def save_prosody(path: str, features: ProsodyFeatures) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), f0=features.f0, energy=features.energy, times=features.times)
    return str(out)
