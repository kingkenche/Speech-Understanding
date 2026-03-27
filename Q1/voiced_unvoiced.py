"""
voiced_unvoiced.py
==================
Voiced / Unvoiced boundary detection using the Cepstrum.

Strategy
--------
For each short-time frame:
  1. Compute the real cepstrum.
  2. Measure the energy in the HIGH-quefrency region (≈ 2–20 ms) —
     strong energy here indicates periodicity (voiced).
  3. Measure the energy in the LOW-quefrency region (0–2 ms) —
     represents the smooth vocal-tract envelope.
  4. Classify the frame as VOICED if high-quefrency energy exceeds a
     threshold, UNVOICED otherwise.
  5. Post-process with a median filter to smooth rapid transitions.

Energy features are normalised so the threshold is dataset-agnostic.
"""

import os
import argparse
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wav.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    dtype = data.dtype
    data  = data.astype(np.float64)
    if np.issubdtype(dtype, np.integer):
        data /= np.iinfo(dtype).max
    return data, int(sr)


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def frame_signal(
    sig: np.ndarray,
    sr: int,
    frame_len_ms: float = 25.0,
    hop_ms: float       = 10.0,
) -> tuple[np.ndarray, int, int]:
    fl  = int(round(frame_len_ms * sr / 1000))
    hs  = int(round(hop_ms       * sr / 1000))
    nf  = 1 + (len(sig) - fl) // hs
    pad = (nf - 1) * hs + fl - len(sig)
    sig = np.append(sig, np.zeros(max(pad, 0)))
    idx = (
        np.tile(np.arange(fl), (nf, 1))
        + np.tile(np.arange(nf) * hs, (fl, 1)).T
    )
    return sig[idx], fl, hs


# ---------------------------------------------------------------------------
# Cepstrum helpers
# ---------------------------------------------------------------------------

def real_cepstrum(frame: np.ndarray, NFFT: int = 512) -> np.ndarray:
    """Real cepstrum of a single frame."""
    spectrum = np.fft.fft(frame, n=NFFT)
    log_mag  = np.log(np.abs(spectrum) + 1e-10)
    return np.real(np.fft.ifft(log_mag))


def cepstrum_energy(cep: np.ndarray, sr: int, lo_ms: float, hi_ms: float, NFFT: int) -> float:
    """Energy of cepstrum in [lo_ms, hi_ms] quefrency band."""
    lo_samp = int(lo_ms * 1e-3 * sr)
    hi_samp = int(hi_ms * 1e-3 * sr)
    lo_samp = max(1, min(lo_samp, NFFT // 2))
    hi_samp = max(lo_samp + 1, min(hi_samp, NFFT // 2))
    return float(np.sum(cep[lo_samp:hi_samp] ** 2))


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

@dataclass
class BoundaryResult:
    labels:          np.ndarray   # 0=unvoiced, 1=voiced per frame
    voiced_prob:     np.ndarray   # combined voicing score per frame
    boundaries:      list[dict]   # list of {start_s, end_s, label}
    high_q_energy:   np.ndarray   # raw high-quefrency energy
    low_q_energy:    np.ndarray   # raw low-quefrency energy
    zcr:             np.ndarray   # zero-crossing rate per frame
    frame_times:     np.ndarray   # centre time of each frame (seconds)
    sr:              int
    sig:             np.ndarray


def detect_voiced_unvoiced(
    audio_path:      str,
    frame_len_ms:    float = 25.0,
    hop_ms:          float = 10.0,
    NFFT:            int   = 512,
    low_q_lo_ms:     float = 0.0,
    low_q_hi_ms:     float = 2.0,
    high_q_lo_ms:    float = 2.0,
    high_q_hi_ms:    float = 20.0,
    threshold:       float = 0.35,
    median_filt_k:   int   = 5,
    pre_emph_coef:   float = 0.97,
) -> BoundaryResult:
    """
    Detect voiced/unvoiced boundaries in an audio file.

    Parameters
    ----------
    threshold     : voiced_prob threshold (0–1). Frames above are VOICED.
    median_filt_k : length of median filter for label smoothing.
    """
    sig, sr = load_wav(audio_path)

    # Pre-emphasis
    sig_pe = np.append(sig[0], sig[1:] - pre_emph_coef * sig[:-1])

    frames, fl, hs = frame_signal(sig_pe, sr, frame_len_ms, hop_ms)
    nf = frames.shape[0]

    win = np.hamming(fl)
    high_q_e = np.zeros(nf)
    low_q_e  = np.zeros(nf)

    zcr          = np.zeros(nf)
    frame_energy = np.zeros(nf)

    for i, frame in enumerate(frames):
        f_win          = frame * win
        cep            = real_cepstrum(f_win, NFFT)
        high_q_e[i]    = cepstrum_energy(cep, sr, high_q_lo_ms, high_q_hi_ms, NFFT)
        low_q_e[i]     = cepstrum_energy(cep, sr, low_q_lo_ms,  low_q_hi_ms,  NFFT)
        # Zero-crossing rate: low ZCR ↔ voiced, high ZCR ↔ unvoiced
        zcr[i]         = float(np.sum(np.abs(np.diff(np.sign(f_win)))) / (2 * len(f_win) + 1e-10))
        frame_energy[i] = float(np.mean(f_win ** 2))

    # ── Voiced probability: weighted combination ──────────────────────────
    # 1) Cepstrum ratio (high-q / total)
    total_e   = high_q_e + low_q_e + 1e-30
    cep_ratio = high_q_e / total_e

    def _norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-10)

    # 2) ZCR score: low ZCR → more voiced → invert after normalisation
    zcr_voiced = 1.0 - _norm(zcr)

    # 3) Energy: louder frames more likely to be voiced
    energy_score = _norm(frame_energy)

    # Weighted average (cepstrum dominates)
    voiced_prob = 0.5 * _norm(cep_ratio) + 0.3 * zcr_voiced + 0.2 * energy_score

    # Normalise to [0, 1]
    vp_min, vp_max = voiced_prob.min(), voiced_prob.max()
    if vp_max > vp_min:
        voiced_prob = (voiced_prob - vp_min) / (vp_max - vp_min)

    # Hard classification + median smoothing
    labels_raw = (voiced_prob >= threshold).astype(int)
    if median_filt_k > 1:
        labels = signal.medfilt(labels_raw.astype(float), kernel_size=median_filt_k).astype(int)
    else:
        labels = labels_raw

    # Frame centre times
    frame_times = np.arange(nf) * (hs / sr) + (fl / 2 / sr)

    # Extract segment boundaries
    boundaries = _extract_boundaries(labels, frame_times)

    return BoundaryResult(
        labels=labels,
        voiced_prob=voiced_prob,
        boundaries=boundaries,
        high_q_energy=high_q_e,
        low_q_energy=low_q_e,
        zcr=zcr,
        frame_times=frame_times,
        sr=sr,
        sig=sig,
    )


def _extract_boundaries(labels: np.ndarray, times: np.ndarray) -> list[dict]:
    """Convert a per-frame label array to a list of (start, end, label) segments."""
    segments = []
    if len(labels) == 0:
        return segments
    curr_label = labels[0]
    start_time = times[0]
    for i in range(1, len(labels)):
        if labels[i] != curr_label:
            segments.append({
                "start_s": float(start_time),
                "end_s":   float(times[i - 1]),
                "label":   "voiced" if curr_label == 1 else "unvoiced",
            })
            curr_label = labels[i]
            start_time = times[i]
    segments.append({
        "start_s": float(start_time),
        "end_s":   float(times[-1]),
        "label":   "voiced" if curr_label == 1 else "unvoiced",
    })
    return segments


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    result: BoundaryResult,
    save_path: str = "voiced_unvoiced.png",
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 17))

    sr  = result.sr
    sig = result.sig
    t   = np.arange(len(sig)) / sr

    # — Waveform + boundary shading
    ax = axes[0]
    ax.plot(t, sig, linewidth=0.5, color="steelblue", label="Waveform")
    for seg in result.boundaries:
        color = "#a8e6cf" if seg["label"] == "voiced" else "#ffaaa5"
        ax.axvspan(seg["start_s"], seg["end_s"], alpha=0.35, color=color,
                   label=seg["label"])
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.set_title("Waveform with Voiced/Unvoiced Boundaries")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")

    # — Combined voiced probability
    ax = axes[1]
    ax.plot(result.frame_times, result.voiced_prob, color="darkorange", linewidth=0.9)
    ax.axhline(y=0.35, color="red", linestyle="--", linewidth=1.0, label="Threshold (0.35)")
    ax.fill_between(result.frame_times, result.voiced_prob, alpha=0.2, color="darkorange")
    ax.set_title("Combined Voicing Score (Cepstrum + ZCR + Energy)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Score"); ax.set_ylim([-0.05, 1.1])
    ax.legend()

    # — Cepstral energy
    ax = axes[2]
    ax.plot(result.frame_times, 10 * np.log10(result.high_q_energy + 1e-30),
            color="purple", linewidth=0.8, label="High-Q (pitch, 2–20 ms)")
    ax.plot(result.frame_times, 10 * np.log10(result.low_q_energy  + 1e-30),
            color="green",  linewidth=0.8, label="Low-Q (vocal tract, 0–2 ms)")
    ax.set_title("Cepstral Energy: High-Quefrency vs Low-Quefrency (dB)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (dB)"); ax.legend()

    # — ZCR
    ax = axes[3]
    ax.plot(result.frame_times, result.zcr, color="teal", linewidth=0.8)
    ax.set_title("Zero-Crossing Rate (low = voiced, high = unvoiced)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ZCR")
    ax.fill_between(result.frame_times, result.zcr, alpha=0.2, color="teal")

    # — Binary label sequence
    ax = axes[4]
    ax.step(result.frame_times, result.labels, where="post", linewidth=1.2, color="black")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Unvoiced", "Voiced"])
    ax.set_title("Final Voiced / Unvoiced Labels (after median filtering)")
    ax.set_xlabel("Time (s)"); ax.set_ylim([-0.1, 1.2])
    ax.fill_between(result.frame_times, result.labels, step="post", alpha=0.3, color="teal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[voiced_unvoiced] Saved: {save_path}")


def print_boundaries(boundaries: list[dict]) -> None:
    print("\n{:<8} {:<10} {:<10} {:<10}".format("#", "Start (s)", "End (s)", "Label"))
    print("-" * 40)
    for i, seg in enumerate(boundaries):
        print("{:<8} {:<10.3f} {:<10.3f} {:<10}".format(
            i + 1, seg["start_s"], seg["end_s"], seg["label"]
        ))
    print(f"\nTotal segments: {len(boundaries)}")
    voiced_dur   = sum(s["end_s"] - s["start_s"] for s in boundaries if s["label"] == "voiced")
    unvoiced_dur = sum(s["end_s"] - s["start_s"] for s in boundaries if s["label"] == "unvoiced")
    print(f"Voiced duration  : {voiced_dur:.3f} s")
    print(f"Unvoiced duration: {unvoiced_dur:.3f} s\n")


def save_boundaries_csv(boundaries: list[dict], path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_s", "end_s", "label"])
        writer.writeheader()
        writer.writerows(boundaries)
    print(f"[voiced_unvoiced] Boundaries saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Voiced/Unvoiced boundary detection via Cepstrum")
    p.add_argument("audio",         help="Path to .wav file")
    p.add_argument("--frame_ms",    type=float, default=25.0)
    p.add_argument("--hop_ms",      type=float, default=10.0)
    p.add_argument("--nfft",        type=int,   default=512)
    p.add_argument("--threshold",   type=float, default=0.35,
                   help="Voiced probability threshold (0–1)")
    p.add_argument("--median_k",    type=int,   default=5,
                   help="Median filter kernel size for label smoothing")
    p.add_argument("--out_dir",     type=str,   default=".")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[voiced_unvoiced] Processing: {args.audio}")
    result = detect_voiced_unvoiced(
        args.audio,
        frame_len_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        NFFT=args.nfft,
        threshold=args.threshold,
        median_filt_k=args.median_k,
    )

    print_boundaries(result.boundaries)

    plot_path = os.path.join(args.out_dir, "voiced_unvoiced.png")
    plot_results(result, plot_path)

    csv_path = os.path.join(args.out_dir, "boundaries.csv")
    save_boundaries_csv(result.boundaries, csv_path)

    npy_path = os.path.join(args.out_dir, "voiced_labels.npy")
    np.save(npy_path, result.labels)
    print(f"[voiced_unvoiced] Labels saved: {npy_path}")


if __name__ == "__main__":
    main()
