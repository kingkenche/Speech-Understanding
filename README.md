# Speech Understanding Assignment 2: Code-Switched ASR to Low-Resource Language Voice Cloning

## 📋 Project Overview

This repository implements a comprehensive speech processing pipeline that:

1. **Transcribes code-switched (Hinglish) lectures** with frame-level language identification
2. **Translates to a target Low-Resource Language (Gondi)** with phonetic mapping
3. **Synthesizes speech in Gondi** using zero-shot voice cloning with prosody warping
4. **Detects spoofing and adversarial attacks** on the LID system

This is an academic assignment fulfilling the Speech Understanding course requirements.

---

## 🎯 Assignment Goals

### Part I: Robust Code-Switched Transcription
- ✅ Multi-head language identification (F1 ≥ 0.85)
- ✅ Constrained decoding with N-gram logit biasing
- ✅ DeepFilterNet-based audio denoising

### Part II: Phonetic Mapping & Translation
- ✅ Hinglish → IPA conversion (custom G2P mapper)
- ✅ Semantic translation to Gondi
- ✅ 500-word technical dictionary curation

### Part III: Zero-Shot Voice Cloning
- ✅ Speaker embedding extraction from 60s reference
- ✅ Prosody extraction (F0 + energy contours)
- ✅ Dynamic Time Warping (DTW) for prosody alignment
- ✅ VITS-based synthesis with speaker cloning

### Part IV: Adversarial Robustness
- ✅ LFCC-based anti-spoofing classifier (EER < 10%)
- ✅ FGSM adversarial perturbation robustness analysis

---

## 📦 Requirements

### Hardware
- **GPU**: NVIDIA GPU with 24GB+ VRAM (tested on L40S)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ for models and data

### Software
- Python 3.9+
- CUDA 12.1+ (for GPU acceleration)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository (or working directory)
cd A-2

# Create conda environment (recommended)
conda create -n speech-a2 python=3.10
conda activate speech-a2

# Install dependencies
pip install -r requirements.txt

# Optional: Install package in development mode
pip install -e .
```

### 2. Phase 0: Acquire Data

```bash
# Download YouTube video and extract 10-minute segment
# (2h 20m to 2h 54m from lecture video)
python scripts/download_video.py \
    --url "https://youtu.be/ZPUtA3W-7_I?si=wCClM6UD1HmuYHTa" \
    --start-time "2:20:00" \
    --end-time "2:54:00" \
    --output-path data/audio/original_segment.wav \
    --target-sr 22050

# Record your 60-second reference voice
python scripts/record_voice.py \
    --output data/audio/student_voice_ref.wav \
    --duration 60 \
    --sample-rate 22050
```

### 3. Configure Project

Edit `config/default.yaml` to set:
- Data paths
- Model paths (pretrained models will auto-download)
- Training hyperparameters
- Evaluation thresholds

---

## 📁 Project Structure

```
speech-assignment-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── config/
│   └── default.yaml                  # Configuration file
├── data/
│   ├── raw/lecture_video/           # Downloaded YouTube audio
│   ├── audio/
│   │   ├── original_segment.wav     # 10-min lecture segment
│   │   ├── student_voice_ref.wav    # 60s reference voice
│   │   └── output_LRL_cloned.wav    # Final synthesized output
│   ├── processed/
│   │   ├── denoised.wav
│   │   ├── transcripts/             # Intermediate outputs
│   │   └── embeddings/
│   └── corpus/
│       ├── gondi_technical_dict.json # Gondi translation dictionary
│       └── ngram_lm.arpa            # N-gram language model
├── models/
│   ├── pretrained/                  # Auto-downloaded pre-trained models
│   └── custom/
│       ├── lid_classifier.pt        # Fine-tuned LID model
│       ├── lfcc_antispoof_cm.pt    # Anti-spoofing classifier
│       └── prosody_mapper_dtw.pkl   # DTW prosody alignment
├── src/
│   ├── data/
│   │   ├── download.py              # Video download utilities
│   │   ├── preprocessing.py         # Audio preprocessing
│   │   └── dataset.py               # PyTorch DataLoaders
│   ├── part1_transcription/
│   │   ├── lid_system.py            # Language identification
│   │   ├── decoding.py              # Constrained beam search
│   │   └── denoising.py             # Audio denoising
│   ├── part2_phonetics/
│   │   ├── ipa_converter.py         # G2P to IPA mapping
│   │   ├── translator.py            # Hinglish to Gondi translation
│   │   └── corpus_builder.py        # Dictionary construction
│   ├── part3_tts/
│   │   ├── speaker_embedding.py     # Speaker embedding extraction
│   │   ├── prosody_extractor.py     # F0 & energy extraction
│   │   ├── dtw_warping.py           # DTW prosody alignment
│   │   └── synthesizer.py           # VITS synthesis
│   ├── part4_robustness/
│   │   ├── antispoof_classifier.py  # Spoofing detection
│   │   ├── adversarial_attack.py    # FGSM perturbations
│   │   └── evaluation.py            # EER & epsilon computation
│   ├── evaluation/
│   │   ├── metrics.py               # WER, MCD, LID accuracy
│   │   └── confusion_matrix.py      # Code-switching analysis
│   └── pipeline.py                  # Main orchestrator
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lid_ablation.ipynb
│   ├── 03_prosody_warping_viz.ipynb
│   └── 04_adversarial_robustness.ipynb
├── scripts/
│   ├── download_video.py            # YouTube downloader
│   ├── record_voice.py              # Voice recorder
│   ├── train_lid.py                 # Train LID classifier
│   ├── train_antispoof.py           # Train anti-spoofing model
│   ├── train_ngram_lm.py            # Build N-gram LM
│   ├── run_full_pipeline.py         # End-to-end execution
│   └── evaluate.py                  # Compute all metrics
├── tests/
│   ├── test_preprocessing.py
│   ├── test_lid_system.py
│   ├── test_tts_output.py
│   └── test_robustness.py
├── report/
│   ├── main.tex                     # IEEE/CVPR format report
│   ├── figures/                     # Report figures (PDF)
│   ├── tables/                      # Report tables (PDF)
│   └── implementation_notes.md      # Non-obvious design choices
└── .gitignore
```

---

## 🔄 Workflow

### Phase 1: Code-Switched Transcription
Train and evaluate LID system, implement constrained Whisper decoding, and audio denoising.

```bash
python scripts/train_lid.py --config config/default.yaml
python scripts/train_ngram_lm.py
```

### Phase 2: Phonetic Mapping & Translation
Build Hinglish G2P converter and Gondi translation pipeline.

### Phase 3: Voice Cloning
Extract speaker embeddings, synthesize, and apply prosody warping.

### Phase 4: Adversarial Robustness
Train anti-spoofing classifier and test adversarial perturbation robustness.

```bash
python scripts/train_antispoof.py
```

### Phase 5: Evaluation & Reporting
Compute metrics and generate report.

```bash
python scripts/evaluate.py --output report/
python scripts/run_full_pipeline.py --config config/default.yaml
```

---

## 📊 Evaluation Metrics

| Metric | Target | Component |
|--------|--------|-----------|
| **WER (English)** | < 15% | Part I (ASR) |
| **WER (Hindi)** | < 25% | Part I (ASR) |
| **MCD** | < 8.0 | Part III (TTS) |
| **LID Switching Accuracy** | ±200ms | Part I (LID) |
| **Anti-Spoofing EER** | < 10% | Part IV |
| **F1 Score (LID)** | ≥ 0.85 | Part I (LID) |

---

## 🎛️ Configuration

Edit `config/default.yaml` to customize:

```yaml
# Audio Processing
audio:
  sample_rate: 16000
  target_sr: 22050
  segment_duration: 600  # 10 minutes

# Model Selection
models:
  whisper_model: "openai/whisper-large-v3"
  vits_model: "facebook/mms-tts-eng"

# Training
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 1e-4
  num_gpus: 2
```

---

## 🧠 Key Components

### Language Identification (Task 1.1)
- Frame-level classification using Wav2Vec2 encoder + classification head
- Target: F1 ≥ 0.85 on English/Hindi distinction
- Trained on Common Voice + Mozilla Hindi corpora

### Constrained Decoding (Task 1.2)
- Whisper-v3 with logit biasing
- N-gram LM trained on speech syllabus
- Prioritizes technical terminology (stochastic, cepstrum, etc.)

### Audio Denoising (Task 1.3)
- DeepFilterNet v3 for classroom noise removal
- Fallback: Spectral subtraction

### IPA Converter (Task 2.1)
- Custom mapper for Hinglish phonology
- Handles code-switching boundaries

### Gondi Translation (Task 2.2)
- M2M-100 / mBART as backbone
- 500-word manually-curated technical dictionary
- ITRANS romanization format

### Voice Cloning (Task 3)
- Speaker embedding: ECAPA-TDNN (256-dim)
- TTS: VITS with speaker conditioning
- Prosody warping: Dynamic Time Warping (DTW)

### Anti-Spoofing (Task 4.1)
- LFCC feature extraction
- SVM/MLP classifier
- Target EER < 10%

### Adversarial Robustness (Task 4.2)
- FGSM perturbations
- SNR constraint (>40dB for inaudibility)
- Epsilon sweep to find misclassification threshold

---

## 🔗 Dependencies & References

### Core Libraries
- **PyTorch & TorchAudio**: Deep learning framework
- **Transformers** (HuggingFace): Pretrained models (Whisper, VITS)
- **Librosa**: Audio processing
- **SpeechBrain**: Speaker verification models
- **DeepFilterNet**: Audio denoising

### Datasets
- **Common Voice**: Multilingual speech corpus (English)
- **Mozilla Hindi**: Hindi speech data
- **Speech Course Syllabus**: Domain-specific text for N-gram LM

### References
- Whisper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- VITS: [Conditional Variational Autoencoder with Adversarial Learning](https://arxiv.org/abs/2106.06103)
- DTW: [Dynamic Time Warping for Speech Recognition](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- LFCC: [Linear Frequency Cepstral Coefficients for Voice Anti-Spoofing](https://www.aclweb.org/anthology/E17-1010/)

---

## 📝 Citation & Licensing

This project is developed as part of a graduate-level Speech Processing course assignment. 

**MIT License** - See LICENSE file for details.

---

## ✨ Implementation Notes

Key non-obvious design choices explained in `report/implementation_notes.md`:

1. **Why DTW over linear prosody warping?** - Preserves relative timing patterns crucial for teaching style
2. **Why frame-level LID?** - Code-switching happens mid-utterance; frame-level captures fine-grained boundaries
3. **Why logit biasing over vocabulary constraints?** - Allows model flexibility while probabilistically steering toward technical terms
4. **Why custom Hinglish G2P?** - No pretrained model handles mixed-language phonology accurately

---

## 🐛 Troubleshooting

### CUDA out of memory
```bash
# Reduce batch size in config/default.yaml
training:
  batch_size: 8
```

### yt-dlp download fails
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

### Missing pretrained models
```bash
# Models auto-download on first use, or pre-fetch manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('openai/whisper-large-v3')"
```

---

## 📧 Contact & Support

For questions or issues:
- Check the [plan file](/home/m25csa028/.claude/plans/lively-cooking-owl.md) for detailed architecture
- Review assignment requirements in the PDF
- See implementation notes for design rationale

---

## 📅 Timeline

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | Setup | GitHub, data download, voice recording |
| 2-3 | Part I | LID training, Whisper integration, denoising |
| 3-4 | Part II | IPA converter, Gondi translation |
| 4-5 | Part III | Speaker embedding, prosody extraction, TTS |
| 5-6 | Part IV | Anti-spoofing, adversarial robustness |
| 6-7 | Evaluation | Report writing, metrics computation, submission |

---

## ✅ Submission Checklist

- [ ] GitHub repository public with all code
- [ ] All audio files in `data/audio/`
- [ ] Report: `report/main.pdf` (IEEE/CVPR format)
- [ ] Implementation notes: `report/implementation_notes.md`
- [ ] Gondi dictionary: `data/corpus/gondi_technical_dict.json`
- [ ] WER < 15% (English), < 25% (Hindi)
- [ ] MCD < 8.0
- [ ] LID switching accuracy ± 200ms
- [ ] Anti-spoofing EER < 10%
- [ ] ZIP file: `RollNo_PA2.zip`

---

**Last Updated**: 2026-04-12  
**Status**: Phase 0 - Initial Setup Complete ✅
