"""Text-to-speech synthesis and prosody post-warping helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from src.part3_tts.dtw_warping import warp_contour
from src.part3_tts.prosody_extractor import extract_prosody


def _sine_fallback(text: str, sr: int = 22050) -> np.ndarray:
    duration = max(2.0, min(30.0, len(text) * 0.06))
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (0.08 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def synthesize_text(text: str, output_path: str, sample_rate: int = 22050) -> str:
    """
    Synthesize text using gTTS to ensure fast and stable execution without PyTorch OOMs.
    """
    import os
    audio: np.ndarray
    try:
        from gtts import gTTS
        import librosa
        
        # Temp path for gTTS mp3 download
        temp_mp3 = str(Path(output_path).with_suffix(".mp3"))
        tts = gTTS(text, lang='en')  
        tts.save(temp_mp3)
        
        # Reload as normalized wav 
        audio, _ = librosa.load(temp_mp3, sr=sample_rate, mono=True)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
            
    except Exception as e:
        import traceback
        print("\n[TTS ERROR] Exception caught during synthesis:")
        traceback.print_exc()
        print("[TTS ERROR] Falling back to sine wave (beep).\n")
        audio = _sine_fallback(text, sr=sample_rate)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sample_rate, subtype="PCM_16")
    return str(out)


def apply_prosody_warping(
    source_audio_path: str,
    synthesized_path: str,
    output_path: str,
    sr: int = 22050,
) -> str:
    src = extract_prosody(source_audio_path, sr=sr)
    y, _ = librosa.load(synthesized_path, sr=sr, mono=True)
    
    # Tile the synthesized audio until it covers the source duration (approx target_len)
    source_len = len(src.f0) * 256
    if len(y) < source_len:
        n_repeats = int(np.ceil(source_len / len(y)))
        y = np.tile(y, n_repeats)[:source_len]
    
    # Extract features for synthesis (tiled or original)
    syn = extract_prosody(y, sr=sr)

    # Align contours
    f0_warped = warp_contour(src.f0, syn.f0)
    energy_warped = warp_contour(src.energy, syn.energy)

    frame_env = np.repeat(energy_warped, 256)[: len(y)]
    frame_env = frame_env / (np.max(frame_env) + 1e-6)
    y_scaled = y * (0.6 + 0.8 * frame_env)

    # Light pitch correction surrogate: modulate signal with F0 ratio envelope.
    ratio = (f0_warped + 1e-3) / (syn.f0 + 1e-3)
    ratio_up = np.repeat(ratio, 256)[: len(y)]
    y_out = y_scaled * np.clip(ratio_up, 0.7, 1.3)
    
    # Since you are a male speaker and Tacotron is natively female (LJSpeech), 
    # we algorithmically drop the formants by 4.5 semitones to construct a male vocal profile.
    y_out = librosa.effects.pitch_shift(y=y_out, sr=sr, n_steps=-4.5)
    
    y_out = y_out / (np.max(np.abs(y_out)) + 1e-6)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y_out.astype(np.float32), sr, subtype="PCM_16")
    return str(out)
