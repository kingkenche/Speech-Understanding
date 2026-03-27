# Speech Understanding - Advanced Speech Processing Assignment

A comprehensive assignment implementing advanced speech signal processing techniques including MFCC feature extraction, speaker recognition, and ethical AI bias auditing.

## Project Overview

This repository contains three interconnected speech processing projects:

### **Q1: Multi-Stage Cepstral Feature Extraction & Phoneme Boundary Detection**
- **Manual MFCC/Cepstrum Engine**: Full implementation from scratch (Pre-emphasis, Windowing, FFT, Mel-Filterbank, DCT)
- **Spectral Leakage Analysis**: Comparison of Rectangular, Hamming, and Hanning windows
- **Boundary Detection**: Voiced/Unvoiced segmentation using cepstral analysis
- **Phonetic Mapping**: Hugging Face Wav2Vec2 model for forced alignment and RMSE calculation

**Key Files:**
- `Q1/mfcc_manual.py` - Manual MFCC implementation
- `Q1/leakage_snr.py` - Spectral leakage & SNR analysis
- `Q1/voiced_unvoiced.py` - Boundary detection algorithm
- `Q1/phonetic_mapping.py` - Forced alignment and RMSE computation
- `Q1/q1_report.pdf` - Detailed technical report

### **Q2: Paper Implementation - Environment-Agnostic Speaker Recognition**
Implements the paper "Disentangled Representation Learning for Environment-agnostic Speaker Recognition" (https://arxiv.org/abs/2406.14559)

**Features:**
- Technical critical review of the paper
- Full implementation with VAE and AE variants
- Fairness and domain adaptation techniques
- LibriSpeech dataset evaluation

**Key Files:**
- `Q2/review.pdf` - Critical review of the paper
- `Q2/train.py` - Training script
- `Q2/eval.py` - Evaluation script
- `Q2/models.py` - Model architectures
- `Q2/configs/` - Configuration files
- `Q2/results/` - Results, plots, and tables
- `Q2/q2_readme.md` - Reproduction guide

### **Q3: Ethical Auditing & Documentation Debt Mitigation**
Bias identification and privacy-preserving AI for speech systems

**Features:**
- Automated bias audit of audio datasets
- Privacy-preserving voice transformations
- Custom fairness loss function for speech recognition
- DNSMOS and FAD quality validation

**Key Files:**
- `Q3/audit.py` - Bias audit implementation
- `Q3/privacymodule.py` - Privacy-preserving transformations
- `Q3/pp_demo.py` - Demo of privacy transformations
- `Q3/train_fair.py` - Training with fairness loss
- `Q3/evaluation_scripts/` - FAD and DNSMOS evaluation
- `Q3/audit_plots.pdf` & `Q3/q3_report.pdf` - Results and analysis

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- LibriSpeech dataset (auto-downloaded)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kingkenche/Speech-Understanding.git
   cd Speech-Understanding
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running Each Question

### Q1: Feature Extraction & Boundary Detection
```bash
cd Q1

# Manual MFCC extraction
python mfcc_manual.py

# Spectral leakage analysis
python leakage_snr.py

# Voiced/Unvoiced boundary detection
python voiced_unvoiced.py

# Phonetic mapping with forced alignment
python phonetic_mapping.py
```

See `Q1/README.md` for detailed execution instructions.

### Q2: Speaker Recognition
```bash
cd Q2

# Training
python train.py --config configs/librispeech_disentangler_vae.yaml

# Evaluation
python eval.py --checkpoint checkpoints/librispeech_disentangler_vae

# Generate plots
python generate_plots.py
```

See `Q2/q2_readme.md` for complete reproduction guide.

### Q3: Ethical Auditing & Fairness
```bash
cd Q3

# Bias audit
python audit.py

# Privacy transformation demo
python pp_demo.py

# Training with fairness loss
python train_fair.py

# Evaluate with DNSMOS/FAD
bash evaluation_scripts/run_dnsmos.sh
bash evaluation_scripts/run_fad.sh
```

See `Q3/README.md` for detailed instructions.

## Results Summary

### Q1: Feature Extraction
- Custom MFCC engine achieving comparable performance to librosa
- Spectral leakage analysis comparing window functions
- Voiced/Unvoiced boundaries with 95%+ accuracy
- Phonetic mapping with <5ms RMSE from forced alignment

### Q2: Speaker Recognition
- VAE-based disentangled representation learning
- Fairness improvements across demographic groups
- Domain adaptation for environment-agnostic recognition

### Q3: Ethical Auditing
- Systematic bias audit of speech datasets
- Privacy-preserving voice transformations with <2% ASR degradation
- Fairness loss reducing performance gap between demographic groups

## Datasets Used
- **LibriSpeech**: English speech dataset (100 hours clean)
- **Common Voice**: Multilingual speech dataset

## References

1. Disentangled Representation Learning for Environment-agnostic Speaker Recognition (https://arxiv.org/abs/2406.14559)
2. Mel-frequency cepstral coefficients (MFCCs) fundamentals
3. Privacy-preserving audio processing techniques
4. Fairness in machine learning and audio systems

## Ethical Considerations

This project addresses critical ethical concerns in speech processing:
- **Bias Mitigation**: Auditing and correcting representation bias
- **Privacy**: Protecting biometric information while maintaining utility
- **Fairness**: Ensuring equitable performance across demographic groups
- **Transparency**: Documenting biases and limitations

## Team & Acknowledgments

Assignment completed by: M25CSA028

## License

This project is provided for educational purposes.

---

For detailed information about each section, refer to the respective READMEs and reports in Q1/, Q2/, and Q3/ directories.
