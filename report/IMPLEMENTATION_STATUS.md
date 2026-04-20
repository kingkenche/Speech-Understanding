# Assignment 2 Implementation Status (April 15, 2026)

## Completed in Codebase

### Part I: Robust Code-Switched Transcription
- Frame-level LID model + training loop:
  - `src/part1_transcription/lid_system.py`
- Constrained decoding with N-gram logit biasing:
  - `src/part1_transcription/decoding.py`
  - `scripts/train_ngram_lm.py`
- Denoising and normalization (DeepFilter hook + spectral subtraction fallback):
  - `src/part1_transcription/denoising.py`
  - `src/data/preprocessing.py`

### Part II: Phonetic Mapping & Translation
- Custom Hinglish-to-IPA mapper:
  - `src/part2_phonetics/ipa_converter.py`
- Dictionary-based semantic translation to Gondi:
  - `src/part2_phonetics/translator.py`
- Corpus validation helpers:
  - `src/part2_phonetics/corpus_builder.py`
- Existing dictionary already has >500 entries:
  - `data/corpus/gondi_technical_dict.json`

### Part III: Zero-Shot Cross-Lingual Voice Cloning
- Speaker embedding extraction (ECAPA with fallback):
  - `src/part3_tts/speaker_embedding.py`
- Prosody extraction (F0 + energy):
  - `src/part3_tts/prosody_extractor.py`
- DTW-based prosody warping:
  - `src/part3_tts/dtw_warping.py`
- Synthesis pipeline + fallback:
  - `src/part3_tts/synthesizer.py`

### Part IV: Adversarial Robustness & Spoofing
- LFCC feature extraction + anti-spoof classifier + EER utilities:
  - `src/part4_robustness/antispoof_classifier.py`
- FGSM attack + minimum epsilon search under SNR constraint:
  - `src/part4_robustness/adversarial_attack.py`
- Unified robustness evaluators:
  - `src/part4_robustness/evaluation.py`

### Evaluation and Orchestration
- Metrics (WER, MCD, switching accuracy, pass/fail):
  - `src/evaluation/metrics.py`
- Boundary confusion matrix:
  - `src/evaluation/confusion_matrix.py`
- End-to-end orchestration:
  - `src/pipeline.py`
  - `scripts/run_full_pipeline.py`
  - `scripts/evaluate.py`

### Training / Execution Scripts Added
- `scripts/train_lid.py`
- `scripts/train_ngram_lm.py`
- `scripts/train_antispoof.py`
- `scripts/run_full_pipeline.py`
- `scripts/evaluate.py`

## Verified
- All Python files compile successfully:
  - `python3 -m compileall src scripts`

## Still Required to Fully Satisfy Graded Submission

1. Collect required audio artifacts:
- `data/audio/student_voice_ref.wav` (60s own voice) is missing.
- Generate final `data/audio/output_LRL_cloned.wav` using full run.

2. Train models on real labeled data:
- LID model checkpoint target: F1 >= 0.85.
- Anti-spoof classifier checkpoint target: EER < 10%.

3. Produce final quantitative results on your real test split:
- WER English < 15%
- WER Hindi < 25%
- MCD < 8
- LID switch precision within 200 ms
- Adversarial epsilon at SNR > 40 dB

4. Complete report deliverables:
- 10-page IEEE/CVPR two-column report
- 1-page implementation note (non-obvious design choices)
- Confusion matrix + ablation (with/without prosody warping)

5. Package and submission:
- Push final code/report/readme to GitHub
- Submit zip with required naming format
- Include GitHub link in report + private classroom comment
