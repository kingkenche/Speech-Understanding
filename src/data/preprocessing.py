"""Audio preprocessing helpers for denoising and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


@dataclass
class SpectralSubtractionConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    noise_seconds: float = 0.75
    floor: float = 0.02


def normalize_audio(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < eps:
        return audio
    return audio / peak


def spectral_subtraction_denoise(
    audio: np.ndarray,
    sr: int,
    cfg: SpectralSubtractionConfig | None = None,
) -> np.ndarray:
    """Apply simple spectral subtraction based denoising."""
    cfg = cfg or SpectralSubtractionConfig()
    stft = librosa.stft(audio, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_frames = max(1, int(cfg.noise_seconds * sr / cfg.hop_length))
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    denoised_mag = np.maximum(magnitude - noise_profile, cfg.floor * magnitude)

    denoised = librosa.istft(denoised_mag * np.exp(1j * phase), hop_length=cfg.hop_length, win_length=cfg.win_length)
    return normalize_audio(denoised)


def load_audio(path: str, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def preprocess_audio_file(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    denoise: bool = True,
) -> str:
    audio, sr = load_audio(input_path, target_sr)
    if denoise:
        audio = spectral_subtraction_denoise(audio, sr)
    audio = normalize_audio(audio)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr, subtype="PCM_16")
    return str(out)
