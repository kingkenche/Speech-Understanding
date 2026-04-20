"""Denoising module for classroom noise and reverberation."""

from __future__ import annotations

from pathlib import Path

import librosa
import soundfile as sf

from src.data.preprocessing import spectral_subtraction_denoise


def deepfilternet_denoise(audio_path: str, output_path: str) -> str:
    """
    DeepFilterNet entrypoint.

    This function intentionally falls back to spectral subtraction if DeepFilterNet
    is not installed in the current environment.
    """
    try:
        from df.enhance import enhance, init_df  # type: ignore

        model, df_state, _ = init_df()
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        enhanced = enhance(model, df_state, audio)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, enhanced, 16000, subtype="PCM_16")
        return output_path
    except Exception:
        return spectral_subtraction_file(audio_path, output_path)


def spectral_subtraction_file(audio_path: str, output_path: str, target_sr: int = 16000) -> str:
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    clean = spectral_subtraction_denoise(audio, target_sr)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, clean, target_sr, subtype="PCM_16")
    return output_path


def denoise_audio(audio_path: str, output_path: str, use_deepfilter: bool = True) -> str:
    if use_deepfilter:
        return deepfilternet_denoise(audio_path, output_path)
    return spectral_subtraction_file(audio_path, output_path)
