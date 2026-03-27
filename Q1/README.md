# Q1 – Multi-Stage Cepstral Feature Extraction & Phoneme Boundary Detection

## Overview

This repository implements a complete, library-agnostic MFCC / cepstrum pipeline
for speech processing, along with voiced/unvoiced boundary detection and Wav2Vec2-based
phonetic alignment.

---

## Repository Structure

```
q1_solution/
├── mfcc_manual.py          # Part 1 – Handcrafted MFCC / cepstrum engine
├── leakage_snr.py          # Part 2 – Spectral leakage & SNR analysis
├── voiced_unvoiced.py      # Part 3 – Voiced/unvoiced boundary detection
├── phonetic_mapping.py     # Part 4 – Wav2Vec2 forced alignment + RMSE
├── generate_test_audio.py  # Helper – create synthetic test .wav
├── run_all.py              # Convenience runner for all parts
├── requirements.txt
├── data/
│   ├── manifest.txt        # List of audio files used
│   └── test_speech.wav     # (generated) Synthetic speech file
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `torchaudio` must match your CUDA version if using GPU.
> For CPU-only: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

---


## Quick Start (LibriSpeech Data)

### 1. Download a LibriSpeech sample (e.g., test-clean/61/70968/61-70968-0000.flac)

```bash
# Download the test-clean subset (or just the file you want)
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
# Example file: LibriSpeech/test-clean/61/70968/61-70968-0000.flac
```

### 2. Convert to 16 kHz mono WAV

```bash
ffmpeg -i LibriSpeech/test-clean/61/70968/61-70968-0000.flac -ar 16000 -ac 1 data/61-70968-0000.wav
```

### 3. Update the manifest

Add this line to `data/manifest.txt`:
```
data/61-70968-0000.wav | LibriSpeech test-clean | 12.0 | 16000 | Male speaker, clearly voiced/unvoiced transitions
```

### 4. Run the full pipeline

```bash
python run_all.py --audio data/61-70968-0000.wav --out_dir results/
```

All plots, CSVs, and JSON summaries will appear under `results/`.

---

## Quick Start (Synthetic Test Audio)

Generate the synthetic test file and run all four parts:

```bash
# 1. Generate synthetic audio (voiced + unvoiced + voiced, 3 s @ 16 kHz)
python generate_test_audio.py --out data/test_speech.wav

# 2. Run everything at once
python run_all.py --audio data/test_speech.wav --out_dir results/
```

All plots, CSVs, and JSON summaries will appear under `results/`.

---

## Running Each Part Individually

### Part 1 – MFCC Engine (`mfcc_manual.py`)

```bash
python mfcc_manual.py data/test_speech.wav \
    --window hamming \
    --num_ceps 13 \
    --num_filters 40 \
    --nfft 512 \
    --out_dir results/
```

**Outputs:**
- `results/mfcc_output.png`    – MFCC heatmap + filterbank + waveform
- `results/cepstrum_output.png` – Real cepstrum with quefrency regions labelled
- `results/mfccs.npy`          – MFCC matrix as NumPy array

---

### Part 2 – Spectral Leakage & SNR (`leakage_snr.py`)

```bash
python leakage_snr.py data/test_speech.wav \
    --nfft 2048 \
    --out_dir results/
```

**Outputs:**
- `results/leakage_comparison.png` – Power spectra for three windows
- `results/snr_comparison.png`     – Bar chart of SNR across windows
- `results/window_functions.png`   – Window shape visualisation
- `results/leakage_snr_table.csv`  – Comparison table (leakage ratio, sidelobe dB, SNR)

---

### Part 3 – Voiced/Unvoiced Detection (`voiced_unvoiced.py`)

```bash
python voiced_unvoiced.py data/test_speech.wav \
    --threshold 0.35 \
    --median_k 5 \
    --out_dir results/
```

**Outputs:**
- `results/voiced_unvoiced.png` – 4-panel visualisation (waveform, prob, energies, labels)
- `results/boundaries.csv`      – Segment table with start_s, end_s, label
- `results/voiced_labels.npy`   – Per-frame binary labels

---

### Part 4 – Phonetic Mapping + RMSE (`phonetic_mapping.py`)

```bash
python phonetic_mapping.py data/test_speech.wav \
    --threshold 0.35 \
    --device cpu \
    --out_dir results/
```

> First run downloads `facebook/wav2vec2-base-960h` (~360 MB) from Hugging Face.

**Outputs:**
- `results/phonetic_alignment.png`  – Manual vs model boundary comparison
- `results/phonetic_summary.json`   – Full transcript, token spans, RMSE
- `results/rmse_table.csv`          – RMSE/MAE table

---

## Using Real Speech Data

1. Download any 16 kHz mono `.wav` from LibriSpeech test-clean:
   ```
   https://www.openslr.org/12
   ```
2. Convert if needed:
   ```bash
   ffmpeg -i input.flac -ar 16000 -ac 1 data/librispeech_sample.wav
   ```
3. Run with the real file:
   ```bash
   python run_all.py --audio data/librispeech_sample.wav --out_dir results_real/
   ```

---

## Key Hyperparameters

| Parameter          | Default | Description                               |
|--------------------|---------|-------------------------------------------|
| `--frame_ms`       | 25 ms   | Frame length for MFCC / cepstrum          |
| `--hop_ms`         | 10 ms   | Frame hop size                            |
| `--nfft`           | 512     | FFT size                                  |
| `--num_ceps`       | 13      | Number of MFCC coefficients               |
| `--num_filters`    | 40      | Mel filterbank channels                   |
| `--window`         | hamming | Windowing function (hamming/hanning/rect) |
| `--threshold`      | 0.35    | Voiced probability cutoff (0–1)           |
| `--median_k`       | 5       | Median filter kernel for label smoothing  |
| `--pre_emph_coef`  | 0.97    | Pre-emphasis filter coefficient           |

---

## Method Notes

### MFCC Pipeline
Implements the classic Davis & Mermelstein (1980) pipeline without `librosa`:
pre-emphasis → Hamming window → FFT → triangular Mel filterbank → log →
Type-II DCT. Sinusoidal liftering (L=22) is applied by default.

### Spectral Leakage
Leakage ratio is defined as the fraction of spectral energy outside ±2 % of
the dominant bin. A 1 kHz pure tone deliberately placed between FFT bins
provides ground-truth comparison across window types.

### Voiced/Unvoiced Detection
Uses the high-quefrency region (2–20 ms, corresponding to F0 = 50–500 Hz)
of the real cepstrum as a voicing indicator. A median filter removes
isolated mis-classified frames.

### Forced Alignment
Uses `torchaudio.functional.forced_align` when available (torchaudio ≥ 2.1),
falling back to greedy best-path CTC decoding. RMSE is computed by matching
each manual boundary to the nearest model boundary.

---

## Data Manifest

See `data/manifest.txt` for the list of audio files used.

---

## License

Academic use only. Audio datasets are subject to their own licenses
(LibriSpeech: CC BY 4.0, Common Voice: CC0).
