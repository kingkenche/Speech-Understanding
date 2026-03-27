"""
pp_demo.py
==========
Interactive demo for the Privacy-Preserving Voice Transformation module.

Works with real SPS Corpus clips or generates synthetic audio.

Usage:
    # Synthetic demo (no files needed):
    python pp_demo.py

    # Real audio from SPS corpus:
    python pp_demo.py \
        --input_wav data/sps-corpus-3.0-2026-03-09-en/clips/your_file.mp3 \
        --source_gender female_feminine --source_age twenties \
        --target_gender male_masculine  --target_age old \
        --output_dir examples/

    # Convert mp3 first if torchaudio can't load it directly:
    #   ffmpeg -i your_file.mp3 -ar 16000 -ac 1 your_file.wav
    #   python pp_demo.py --input_wav your_file.wav ...

    # All gender/age pair combinations (for examples/ folder):
    python pp_demo.py --all_pairs --output_dir examples/

Supported --source_gender / --target_gender values (SPS Corpus):
    female_feminine  male_masculine  intersex
    transgender      non-binary      do_not_wish_to_say  unknown

Supported --source_age / --target_age values (raw SPS or coarse bucket):
    teens  twenties  thirties  fourties  fifties
    sixties  seventies  eighties  nineties
    young  middle  old  elderly  child  unknown
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
    """Load WAV or MP3 to float32 mono numpy array at target_sr."""
    path = str(path)
    # Try torchaudio first (handles mp3 if ffmpeg/soundfile backend available)
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)                      # stereo → mono
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.numpy().astype(np.float32)
    except Exception:
        pass
    # Fallback: scipy (wav only)
    try:
        import scipy.io.wavfile as wf
        sr, data = wf.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2_147_483_648.0
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Simple linear resampling if needed
        if sr != target_sr:
            n_new = int(len(data) * target_sr / sr)
            data  = np.interp(
                np.linspace(0, len(data) - 1, n_new),
                np.arange(len(data)), data,
            ).astype(np.float32)
        return data
    except Exception as e:
        raise RuntimeError(
            f"Cannot load {path}.\n"
            "  → Install torchaudio + ffmpeg for mp3 support, "
            "or convert to wav first:\n"
            f"      ffmpeg -i {path} -ar 16000 -ac 1 output.wav"
        ) from e


def save_wav(path: str, waveform: np.ndarray, sr: int = 16_000):
    """Save float32 waveform as 16-bit PCM WAV."""
    import scipy.io.wavfile as wf
    clipped = np.clip(waveform, -1.0, 1.0)
    wf.write(path, sr, (clipped * 32767).astype(np.int16))
    print(f"  Saved  → {path}  ({len(waveform)/sr:.2f}s)")


def synthesise_signal(
    duration: float = 3.0,
    sr:       int   = 16_000,
    gender:   str   = "male_masculine",
    age:      str   = "old",
    seed:     int   = 42,
) -> np.ndarray:
    """
    Generate a synthetic speech-like harmonic signal.
    Pitch varies by gender and age to make demos audibly distinct.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    f0_table = {
        "female_feminine_young":   220, "female_feminine_middle": 200,
        "female_feminine_old":     185, "female_feminine_elderly":175,
        "male_masculine_young":    130, "male_masculine_middle":  115,
        "male_masculine_old":       95, "male_masculine_elderly":  85,
        "intersex_young":          165, "non-binary_young":        160,
        "transgender_young":       175,
    }
    # Coarse age bucket for lookup
    from privacymodule import AGE_BUCKET_MAP
    age_bucket = AGE_BUCKET_MAP.get(age.lower(), age.lower())
    key = f"{gender.lower()}_{age_bucket}"
    f0  = f0_table.get(key, 140)

    # Harmonic stack
    signal = np.zeros_like(t)
    for h in range(1, 12):
        amp    = (1.0 / h) * rng.uniform(0.8, 1.2)
        phase  = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * h * t + phase)

    # Syllable-shaped amplitude envelope
    n_syls = max(1, int(duration * 4))
    env    = np.zeros_like(t)
    for i in range(n_syls):
        c = int((i + 0.5) * len(t) / n_syls)
        w = int(sr * 0.09)
        s, e = max(0, c - w), min(len(t), c + w)
        env[s:e] += np.hanning(e - s)
    signal *= env
    signal += rng.standard_normal(signal.shape).astype(np.float32) * 0.008
    signal /= np.abs(signal).max() + 1e-8
    return (signal * 0.70).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    src_wav: np.ndarray, tgt_wav: np.ndarray,
    src_attrs: dict, tgt_attrs: dict,
    metrics: dict, sr: int, save_path: str,
):
    if not MPL_OK:
        print("  matplotlib not available – skipping plot.")
        return

    from scipy.signal import stft as scipy_stft

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Privacy-Preserving Voice Transformation\n"
        f"{src_attrs['gender']} / {src_attrs['age']}  →  "
        f"{tgt_attrs['gender']} / {tgt_attrs['age']}",
        fontsize=13, fontweight="bold",
    )

    t_src = np.arange(len(src_wav)) / sr
    t_tgt = np.arange(len(tgt_wav)) / sr

    # Waveforms
    axes[0, 0].plot(t_src, src_wav, color="#E07B54", lw=0.6, alpha=0.85)
    axes[0, 0].set_title("Source waveform", fontsize=11)
    axes[0, 0].set_xlabel("Time (s)"); axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(t_tgt, tgt_wav, color="#3B8BD4", lw=0.6, alpha=0.85)
    axes[0, 1].set_title("Transformed waveform", fontsize=11)
    axes[0, 1].set_xlabel("Time (s)"); axes[0, 1].set_ylabel("Amplitude")

    # Approximate spectrograms
    def stft_log(wav):
        _, _, Z = scipy_stft(wav, fs=sr, nperseg=512, noverlap=384)
        return np.log1p(np.abs(Z[:80, :]))

    for ax, wav, title, cmap in [
        (axes[1, 0], src_wav, "Source spectrogram",      "Oranges"),
        (axes[1, 1], tgt_wav, "Transformed spectrogram", "Blues"),
    ]:
        im = ax.imshow(
            stft_log(wav), aspect="auto", origin="lower",
            cmap=cmap, interpolation="nearest",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time frames"); ax.set_ylabel("Freq bin")
        fig.colorbar(im, ax=ax, shrink=0.9)

    metrics_txt = "\n".join([
        f"MEL MAE      : {metrics.get('mel_mae', 'N/A')}",
        f"Approx SNR   : {metrics.get('approx_snr_db', 'N/A')} dB",
    ])
    fig.text(0.01, 0.01, metrics_txt, fontsize=8, family="monospace",
             va="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot   → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Core demo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(args):
    os.makedirs(args.output_dir, exist_ok=True)
    SR = 16_000

    tgt_attrs = {"gender": args.target_gender, "age": args.target_age}

    # ── Load / generate source audio ─────────────────────────────────────
    # Resolve input: explicit --input_wav OR auto-pick from corpus TSV
    wav_path = args.input_wav
    if args.tsv and not wav_path:
        import pandas as pd
        df = pd.read_csv(args.tsv, sep="\t", low_memory=False)
        base_dir  = os.path.dirname(os.path.abspath(args.tsv))
        audio_dir = os.path.join(base_dir, "audios")
        col = "audio_file" if "audio_file" in df.columns else "path"
        labelled = df.dropna(subset=["gender", "age"])
        if len(labelled):
            row = labelled.iloc[0]
            wav_path = os.path.join(audio_dir, str(row[col]))
            args.source_gender = str(row["gender"]).strip()
            args.source_age    = str(row["age"]).strip()
            print(f"\n  Auto-selected clip : {wav_path}")
            print(f"  Source labels      : gender={args.source_gender}, age={args.source_age}")
    # Build src_attrs AFTER auto-pick may have updated source_gender/age
    src_attrs = {"gender": args.source_gender, "age": args.source_age}

    if wav_path and Path(wav_path).exists():
        print(f"\n  Loading {wav_path} …")
        try:
            source_wav = load_audio(wav_path, target_sr=SR)
            print(f"  Duration : {len(source_wav)/SR:.2f}s")
        except RuntimeError as e:
            print(f"  [WARNING] {e}")
            print("  Falling back to synthetic audio.")
            source_wav = synthesise_signal(
                3.0, SR, args.source_gender, args.source_age
            )
    else:
        if wav_path:
            print(f"  [WARNING] {wav_path} not found – using synthetic audio.")
        else:
            print("\n  Generating synthetic speech-like signal …")
        source_wav = synthesise_signal(3.0, SR, args.source_gender, args.source_age)

    # ── Build / load model ───────────────────────────────────────────────
    if not TORCH_OK:
        print("\n  [WARNING] PyTorch not installed. Running pitch-shift mock.")
        factor    = 1.25 if "female" in tgt_attrs["gender"] else 0.80
        n_new     = int(len(source_wav) * factor)
        resampled = np.interp(
            np.linspace(0, len(source_wav) - 1, n_new),
            np.arange(len(source_wav)), source_wav,
        ).astype(np.float32)
        output_wav = resampled[:len(source_wav)] if len(resampled) >= len(source_wav) \
                     else np.pad(resampled, (0, len(source_wav) - len(resampled)))
        metrics = {"mel_mae": "N/A (mock)", "approx_snr_db": "N/A (mock)",
                   "source_attrs": src_attrs, "target_attrs": tgt_attrs}
    else:
        import torch
        from privacymodule import PrivacyModule, VoiceTransformConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device : {device}")

        if args.checkpoint and Path(args.checkpoint).exists():
            print(f"  Loading checkpoint {args.checkpoint} …")
            model = PrivacyModule.load(args.checkpoint, device=device)
        else:
            if args.checkpoint:
                print(f"  [WARNING] Checkpoint not found – using random weights.")
            cfg   = VoiceTransformConfig()
            model = PrivacyModule(cfg).to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f"  Params : {params:,}")

        t0 = time.time()
        output_wav, metrics = model.transform(
            source_wav, SR, src_attrs, tgt_attrs
        )
        print(f"  Transform time : {(time.time()-t0)*1000:.1f} ms")

    # ── Save audio ───────────────────────────────────────────────────────
    stem = (Path(args.input_wav).stem + "_") if (args.input_wav and
             Path(args.input_wav).exists()) else "synthetic_"
    src_path = os.path.join(args.output_dir, f"{stem}source.wav")
    tgt_path = os.path.join(
        args.output_dir,
        f"{stem}{src_attrs['gender']}_{src_attrs['age']}"
        f"_to_{tgt_attrs['gender']}_{tgt_attrs['age']}.wav",
    )
    save_wav(src_path,  source_wav, SR)
    save_wav(tgt_path,  output_wav, SR)

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_path = tgt_path.replace(".wav", ".png")
    plot_comparison(source_wav, output_wav, src_attrs, tgt_attrs,
                    metrics, SR, plot_path)

    # ── Metrics JSON ─────────────────────────────────────────────────────
    mpath = os.path.join(args.output_dir, "transform_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics → {mpath}")

    print("\n── Results ──────────────────────────────────────────────────")
    for k, v in metrics.items():
        if not isinstance(v, dict):
            print(f"  {k:22s}: {v}")
    print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# All meaningful transformation pairs using SPS gender categories
ALL_PAIRS = [
    ("female_feminine", "twenties", "male_masculine",  "old"),
    ("male_masculine",  "old",      "female_feminine", "young"),
    ("female_feminine", "old",      "male_masculine",  "young"),
    ("male_masculine",  "young",    "female_feminine", "middle"),
    ("female_feminine", "young",    "non-binary",      "middle"),
    ("male_masculine",  "middle",   "female_feminine", "elderly"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Voice Transformation Demo (SPS Corpus)"
    )
    parser.add_argument("--tsv", type=str, default=None,
                        help="SPS corpus TSV – auto-picks a labelled clip")
    parser.add_argument("--input_wav", type=str, default=None,
                        help="Input WAV/MP3 file (optional; synthetic if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Trained model checkpoint (.pt)")
    parser.add_argument("--source_gender", type=str, default="male_masculine",
                        choices=[
                            "female_feminine", "male_masculine", "intersex",
                            "transgender", "non-binary", "do_not_wish_to_say", "unknown",
                        ])
    parser.add_argument("--source_age", type=str, default="old",
                        choices=[
                            "teens", "twenties", "thirties", "fourties",
                            "fifties", "sixties", "seventies", "eighties",
                            "nineties", "young", "middle", "old", "elderly",
                            "child", "unknown",
                        ])
    parser.add_argument("--target_gender", type=str, default="female_feminine",
                        choices=[
                            "female_feminine", "male_masculine", "intersex",
                            "transgender", "non-binary", "do_not_wish_to_say", "unknown",
                        ])
    parser.add_argument("--target_age", type=str, default="young",
                        choices=[
                            "teens", "twenties", "thirties", "fourties",
                            "fifties", "sixties", "seventies", "eighties",
                            "nineties", "young", "middle", "old", "elderly",
                            "child", "unknown",
                        ])
    parser.add_argument("--output_dir", type=str, default="examples")
    parser.add_argument("--all_pairs", action="store_true",
                        help="Run all predefined gender×age transformation pairs")
    args = parser.parse_args()

    if args.all_pairs:
        for sg, sa, tg, ta in ALL_PAIRS:
            args.source_gender = sg
            args.source_age    = sa
            args.target_gender = tg
            args.target_age    = ta
            run_demo(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()