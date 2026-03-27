"""
mfcc_manual.py
==============
Manual MFCC / Cepstrum extraction engine.

Pipeline
--------
1. Load audio
2. Pre-emphasis filter
3. Framing + Windowing (Hamming / Hanning)
4. FFT  →  Power Spectrum
5. Mel Filterbank
6. Log compression
7. DCT  →  MFCCs
8. (Optional) Cepstrum via IFFT of log-magnitude spectrum

No librosa.feature.mfcc — every step is hand-rolled with NumPy / SciPy.
"""

import os
import argparse
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Audio I/O helpers
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a .wav file; return (mono float64 array, sample_rate)."""
    sr, data = wav.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float64:
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    return data, int(sr)


def save_wav(path: str, data: np.ndarray, sr: int) -> None:
    data_int16 = np.clip(data, -1.0, 1.0)
    data_int16 = (data_int16 * 32767).astype(np.int16)
    wav.write(path, sr, data_int16)


# ---------------------------------------------------------------------------
# 2. Pre-emphasis
# ---------------------------------------------------------------------------

def pre_emphasis(signal_: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Apply first-order high-pass pre-emphasis filter.
    y[n] = x[n] - coef * x[n-1]
    Boosts high-frequency components to compensate for spectral tilt.
    """
    return np.append(signal_[0], signal_[1:] - coef * signal_[:-1])


# ---------------------------------------------------------------------------
# 3. Framing
# ---------------------------------------------------------------------------

def frame_signal(
    sig: np.ndarray,
    sr: int,
    frame_len_ms: float = 25.0,
    frame_step_ms: float = 10.0,
) -> tuple[np.ndarray, int, int]:
    """
    Split signal into overlapping frames.

    Returns
    -------
    frames     : (num_frames, frame_length) ndarray
    frame_len  : frame length in samples
    frame_step : hop size in samples
    """
    frame_len  = int(round(frame_len_ms  * sr / 1000))
    frame_step = int(round(frame_step_ms * sr / 1000))

    sig_len    = len(sig)
    num_frames = 1 + (sig_len - frame_len) // frame_step
    # zero-pad to fit last frame
    pad_len    = max(0, (num_frames - 1) * frame_step + frame_len - sig_len)
    sig_padded = np.append(sig, np.zeros(pad_len))

    indices = (
        np.tile(np.arange(frame_len), (num_frames, 1))
        + np.tile(np.arange(num_frames) * frame_step, (frame_len, 1)).T
    )
    return sig_padded[indices], frame_len, frame_step


# ---------------------------------------------------------------------------
# 4. Windowing
# ---------------------------------------------------------------------------

def apply_window(frames: np.ndarray, window_type: str = "hamming") -> np.ndarray:
    """
    Multiply each frame by a window function.

    Parameters
    ----------
    frames      : (num_frames, frame_length)
    window_type : 'hamming' | 'hanning' | 'rectangular'
    """
    n = frames.shape[1]
    wt = window_type.lower()
    if wt == "hamming":
        win = np.hamming(n)
    elif wt in ("hanning", "hann"):
        win = np.hanning(n)
    elif wt in ("rectangular", "rect", "boxcar"):
        win = np.ones(n)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    return frames * win


# ---------------------------------------------------------------------------
# 5. FFT → Power Spectrum
# ---------------------------------------------------------------------------

def compute_power_spectrum(windowed: np.ndarray, NFFT: int = 512) -> np.ndarray:
    """
    Compute one-sided power spectrum for each frame.

    Returns shape: (num_frames, NFFT // 2 + 1)
    """
    mag_frames = np.absolute(np.fft.rfft(windowed, n=NFFT))  # (F, NFFT//2+1)
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    return pow_frames


# ---------------------------------------------------------------------------
# 6. Mel Filterbank
# ---------------------------------------------------------------------------

def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    num_filters: int,
    NFFT: int,
    sr: int,
    low_freq_hz: float = 0.0,
    high_freq_hz: float | None = None,
) -> np.ndarray:
    """
    Build a triangular Mel filterbank.

    Returns
    -------
    fbank : (num_filters, NFFT // 2 + 1)  float64
    """
    if high_freq_hz is None:
        high_freq_hz = sr / 2.0

    low_mel  = hz_to_mel(low_freq_hz)
    high_mel = hz_to_mel(high_freq_hz)

    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points  = mel_to_hz(mel_points)

    # Map Hz to FFT bin indices
    bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((num_filters, NFFT // 2 + 1))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m       = bin_points[m]
        f_m_plus  = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-10)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-10)

    return fbank


# ---------------------------------------------------------------------------
# 7. Log compression
# ---------------------------------------------------------------------------

def log_compress(filter_energies: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Natural log compression (avoids log(0))."""
    return np.log(np.maximum(filter_energies, floor))


# ---------------------------------------------------------------------------
# 8. DCT
# ---------------------------------------------------------------------------

def dct_manual(x: np.ndarray) -> np.ndarray:
    """
    Type-II DCT implemented from scratch using the definition:
    X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1)/(2N))

    Works on the last axis of x.
    """
    N = x.shape[-1]
    n = np.arange(N)
    k = np.arange(N)[:, None]
    D = np.cos(np.pi * k * (2 * n + 1) / (2 * N))  # (N, N)
    return 2.0 * (x @ D.T)


# ---------------------------------------------------------------------------
# 9. MFCC pipeline
# ---------------------------------------------------------------------------

def extract_mfcc(
    audio_path: str,
    num_ceps: int        = 13,
    num_filters: int     = 40,
    frame_len_ms: float  = 25.0,
    frame_step_ms: float = 10.0,
    pre_emph_coef: float = 0.97,
    NFFT: int            = 512,
    window_type: str     = "hamming",
    low_freq_hz: float   = 0.0,
    high_freq_hz: float | None = None,
    apply_liftering: bool = True,
    lifter_L: int         = 22,
) -> dict:
    """
    Full manual MFCC extraction.

    Returns a dict with intermediate results for inspection.
    """
    sig, sr = load_wav(audio_path)

    # Step 1 — Pre-emphasis
    sig_pe = pre_emphasis(sig, pre_emph_coef)

    # Step 2 — Framing
    frames, frame_len, frame_step = frame_signal(sig_pe, sr, frame_len_ms, frame_step_ms)

    # Step 3 — Windowing
    windowed = apply_window(frames, window_type)

    # Step 4 — Power spectrum
    pow_spec = compute_power_spectrum(windowed, NFFT)

    # Step 5 — Mel filterbank
    fbank = mel_filterbank(num_filters, NFFT, sr, low_freq_hz, high_freq_hz)
    filter_energies = pow_spec @ fbank.T                      # (F, num_filters)

    # Step 6 — Log
    log_energies = log_compress(filter_energies)

    # Step 7 — DCT
    mfccs = dct_manual(log_energies)[:, :num_ceps]           # (F, num_ceps)

    # Optional: sinusoidal liftering
    if apply_liftering:
        cep_idx   = np.arange(1, num_ceps + 1)
        lift      = 1 + (lifter_L / 2) * np.sin(np.pi * cep_idx / lifter_L)
        mfccs     = mfccs * lift

    return {
        "mfccs":           mfccs,
        "log_energies":    log_energies,
        "filter_energies": filter_energies,
        "pow_spec":        pow_spec,
        "fbank":           fbank,
        "windowed":        windowed,
        "frames":          frames,
        "sig":             sig,
        "sig_pe":          sig_pe,
        "sr":              sr,
        "frame_len":       frame_len,
        "frame_step":      frame_step,
        "NFFT":            NFFT,
        "num_filters":     num_filters,
        "num_ceps":        num_ceps,
    }


# ---------------------------------------------------------------------------
# 10. Cepstrum (full IFFT-based)
# ---------------------------------------------------------------------------

def compute_cepstrum(sig: np.ndarray, sr: int, NFFT: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """
    Real cepstrum:  c[n] = IFFT( log |FFT(x)| )

    Returns (cepstrum, quefrency_axis_in_seconds)
    """
    spectrum   = np.fft.fft(sig, n=NFFT)
    log_mag    = np.log(np.abs(spectrum) + 1e-10)
    cepstrum   = np.real(np.fft.ifft(log_mag))
    quefrency  = np.arange(NFFT) / sr
    return cepstrum, quefrency


# ---------------------------------------------------------------------------
# 11. Visualisation helpers
# ---------------------------------------------------------------------------

def plot_mfcc(result: dict, save_path: str = "mfcc_output.png") -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    # Waveform
    sr   = result["sr"]
    t    = np.arange(len(result["sig"])) / sr
    axes[0].plot(t, result["sig"], linewidth=0.6)
    axes[0].set_title("Waveform (original)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # Mel filterbank
    freqs = np.linspace(0, sr / 2, result["NFFT"] // 2 + 1)
    for filt in result["fbank"]:
        axes[1].plot(freqs, filt, linewidth=0.8)
    axes[1].set_title(f"Mel Filterbank ({result['num_filters']} filters)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Gain")

    # Log-filterbank energies
    im2 = axes[2].imshow(
        result["log_energies"].T,
        aspect="auto",
        origin="lower",
        cmap="inferno",
    )
    axes[2].set_title("Log Mel-Filterbank Energies")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Filter")
    plt.colorbar(im2, ax=axes[2])

    # MFCCs
    im3 = axes[3].imshow(
        result["mfccs"].T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
    )
    axes[3].set_title(f"MFCCs (first {result['num_ceps']} coefficients)")
    axes[3].set_xlabel("Frame")
    axes[3].set_ylabel("Coefficient")
    plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[mfcc_manual] Saved: {save_path}")


def plot_cepstrum(sig: np.ndarray, sr: int, save_path: str = "cepstrum_output.png") -> None:
    cep, quefrency = compute_cepstrum(sig, sr)
    half = len(cep) // 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(quefrency[:half] * 1000, cep[:half], linewidth=0.8)
    axes[0].set_title("Real Cepstrum (low quefrency = vocal-tract envelope)")
    axes[0].set_xlabel("Quefrency (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].axvline(x=2.0,  color="r", linestyle="--", label="Low/High boundary (~2 ms)")
    axes[0].legend()

    # Zoomed — pitch region
    pitch_lo_ms, pitch_hi_ms = 2.0, 20.0
    mask = (quefrency[:half] * 1000 >= pitch_lo_ms) & (quefrency[:half] * 1000 <= pitch_hi_ms)
    axes[1].plot(quefrency[:half][mask] * 1000, cep[:half][mask], linewidth=0.8, color="orange")
    axes[1].set_title("High-Quefrency Region (pitch / voicing)")
    axes[1].set_xlabel("Quefrency (ms)")
    axes[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[mfcc_manual] Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Manual MFCC / Cepstrum engine")
    p.add_argument("audio", help="Path to .wav file")
    p.add_argument("--num_ceps",    type=int,   default=13)
    p.add_argument("--num_filters", type=int,   default=40)
    p.add_argument("--frame_ms",    type=float, default=25.0)
    p.add_argument("--hop_ms",      type=float, default=10.0)
    p.add_argument("--window",      type=str,   default="hamming",
                   choices=["hamming", "hanning", "rectangular"])
    p.add_argument("--nfft",        type=int,   default=512)
    p.add_argument("--out_dir",     type=str,   default=".")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[mfcc_manual] Processing: {args.audio}")
    result = extract_mfcc(
        args.audio,
        num_ceps=args.num_ceps,
        num_filters=args.num_filters,
        frame_len_ms=args.frame_ms,
        frame_step_ms=args.hop_ms,
        window_type=args.window,
        NFFT=args.nfft,
    )

    print(f"  → MFCC matrix shape: {result['mfccs'].shape}")
    print(f"  → Sample rate      : {result['sr']} Hz")

    mfcc_png  = os.path.join(args.out_dir, "mfcc_output.png")
    cep_png   = os.path.join(args.out_dir, "cepstrum_output.png")

    plot_mfcc(result, mfcc_png)
    plot_cepstrum(result["sig"], result["sr"], cep_png)

    npy_path = os.path.join(args.out_dir, "mfccs.npy")
    np.save(npy_path, result["mfccs"])
    print(f"[mfcc_manual] MFCCs saved to: {npy_path}")


if __name__ == "__main__":
    main()
