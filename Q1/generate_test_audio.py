"""
generate_test_audio.py
======================
Generates a synthetic .wav file for testing the Q1 pipeline when
real speech data is unavailable.

Creates a 3-second signal with:
  - 0.0–1.0 s : voiced-like (periodic 150 Hz fundamental + harmonics)
  - 1.0–2.0 s : unvoiced-like (band-pass filtered noise)
  - 2.0–3.0 s : voiced-like again
"""

import numpy as np
import scipy.io.wavfile as wav
import os


def generate_voiced(duration: float, sr: int, f0: float = 150.0) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    sig = np.zeros_like(t)
    for k in range(1, 10):                    # harmonics
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    sig /= np.max(np.abs(sig) + 1e-10)
    return sig * 0.7


def generate_unvoiced(duration: float, sr: int,
                      lo: float = 2000.0, hi: float = 6000.0) -> np.ndarray:
    import scipy.signal as ss
    n   = int(duration * sr)
    noise = np.random.randn(n)
    sos  = ss.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    filt = ss.sosfilt(sos, noise)
    filt /= np.max(np.abs(filt) + 1e-10)
    return filt * 0.5


def main(out_path: str = "data/test_speech.wav", sr: int = 16000) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    v1 = generate_voiced(1.0, sr)
    u  = generate_unvoiced(1.0, sr)
    v2 = generate_voiced(1.0, sr, f0=180.0)
    signal = np.concatenate([v1, u, v2])
    # Normalise
    signal /= np.max(np.abs(signal) + 1e-10)
    int16  = (signal * 32767).astype(np.int16)
    wav.write(out_path, sr, int16)
    print(f"Saved synthetic test audio: {out_path}  ({len(signal)/sr:.1f} s @ {sr} Hz)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/test_speech.wav")
    p.add_argument("--sr",  type=int, default=16000)
    args = p.parse_args()
    main(args.out, args.sr)
