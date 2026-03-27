# Complete Evaluation Commands Guide

## Quick Reference

### ⚡ **Fastest (2 minutes)** - Single Model on WAV Files
```bash
python evaluate_librispeech.py \
    --config configs/vae_improved.yaml \
    --checkpoint checkpoints/librispeech_disentangler_vae/best.pt \
    --eval_dir ./LibriSpeech \
    --output results/librispeech_eval.json
```

### 📊 **Batch Evaluation** - All Available Models
```bash
python evaluate_all_models.py
```

---

## Detailed Commands by Use Case

### 1️⃣ Evaluate VAE-Disentangler on LibriSpeech (Recommended)

**Single evaluation:**
```bash
python evaluate_librispeech.py \
    --config configs/vae_improved.yaml \
    --checkpoint checkpoints/librispeech_disentangler_vae/best.pt \
    --eval_dir ./LibriSpeech \
    --output results/librispeech_eval.json

# View results
cat results/librispeech_eval.json
```

**Expected output:**
```json
{
  "model": "librispeech_disentangler_vae",
  "EER_%": 32.54,
  "minDCF": 0.0,
  "num_pairs": 10000,
  "num_speakers": 40,
  "num_utterances": 2620
}
```

---

### 2️⃣ Batch Evaluate All Models

**Run all available models:**
```bash
python evaluate_all_models.py
```

**Output:**
- `results/librispeech_baseline_eval.json` (if baseline checkpoint exists)
- `results/librispeech_disentangler_eval.json` (if AE checkpoint exists)
- `results/librispeech_vae_eval.json` (VAE-Disentangler)
- `results/librispeech_eval_summary.json` (summary table)

---

### 3️⃣ Convert FLAC Files to WAV (If Needed)

**Convert and keep both formats:**
```bash
python convert_flac_to_wav.py --input_dir ./LibriSpeech --keep_flac
```

**Convert and delete FLAC files (saves ~330 MB):**
```bash
python convert_flac_to_wav.py --input_dir ./LibriSpeech
```

---

### 4️⃣ Evaluate on VoxCeleb (Legacy - Requires VoxCeleb Data)

**Single model on VoxCeleb test:**
```bash
python eval.py \
    --config configs/vae_improved.yaml \
    --checkpoint checkpoints/librispeech_disentangler_vae/best.pt \
    --eval_dir /path/to/vox1_test_wav \
    --trial_file /path/to/veri_test2.txt \
    --output results/voxceleb_eval.json
```

**Auto-format detection notes:**
- If both WAV and FLAC exist: **prefers WAV** (faster)
- LibriSpeech: No trial file needed (auto-generates)
- VoxCeleb: Requires explicit trial file

---

### 5️⃣ Training Commands (Optional)

**Train Baseline on LibriSpeech:**
```bash
python train.py --config configs/librispeech_baseline.yaml
```

**Train AE-Disentangler:**
```bash
python train.py --config configs/librispeech_disentangler.yaml \
    --encoder_ckpt checkpoints/librispeech_baseline/best.pt
```

**Train VAE-Disentangler:**
```bash
python train.py --config configs/librispeech_vae_improved.yaml \
    --encoder_ckpt checkpoints/librispeech_baseline/best.pt
```

---

## Command Options Reference

### `evaluate_librispeech.py` Options
```bash
python evaluate_librispeech.py \
    --config <path>              # Required: Config file
    --checkpoint <path>          # Required: Model checkpoint
    --eval_dir <path>            # Default: ./LibriSpeech
    --output <path>              # Required: Output JSON file
```

### `evaluate_all_models.py` Options
```bash
python evaluate_all_models.py
# Automatically detects available checkpoints and evaluates them
```

### `convert_flac_to_wav.py` Options
```bash
python convert_flac_to_wav.py \
    --input_dir <path>          # Default: ./LibriSpeech
    --output_dir <path>         # Default: same as input
    --keep_flac                 # Keep original FLAC files
```

### `eval.py` Options (VoxCeleb)
```bash
python eval.py \
    --config <path>              # Required: Config file
    --checkpoint <path>          # Required: Model checkpoint
    --eval_dir <path>            # Required: Evaluation data directory
    --trial_file <path>          # Required: Trial file
    --output <path>              # Required: Output JSON file
```

---

## Data Format Support

| Format | Command | Trial File | Auto-Detect |
|--------|---------|-----------|-------------|
| **LibriSpeech WAV** | `evaluate_librispeech.py` | ❌ No (auto-generated) | ✅ Yes |
| **LibriSpeech FLAC** | `evaluate_librispeech.py` | ❌ No (auto-generated) | ✅ Yes |
| **VoxCeleb WAV** | `eval.py` | ✅ Yes (required) | ✅ Yes |
| **Synthetic** | `train.py` | ❌ N/A | ✅ Auto-created |

---

## Expected Runtimes

| Command | Time | GPU |
|---------|------|-----|
| Single model eval on LibriSpeech | 5-10 min | NVIDIA GPU (recommended) |
| Batch eval (3 models) | 10-15 min | NVIDIA GPU |
| FLAC→WAV conversion (2,620 files) | 5-10 min | CPU/GPU (I/O bound) |
| Train Baseline | ~4 hours | A100 40GB |
| Train AE-Disentangler | ~6 hours | A100 40GB |
| Train VAE-Disentangler | ~6.5 hours | A100 40GB |

---

## Troubleshooting Evaluation

### Error: "Loaded 0 evaluation utterances"
```bash
# Check if eval_dir exists and has audio files
find ./LibriSpeech -name "*.wav" -o -name "*.flac" | head -5
# Should show audio files
```

### Error: "No trial file provided"
```bash
# LibriSpeech auto-generates trials - no file needed
# For VoxCeleb, provide --trial_file explicitly
python eval.py --eval_dir ... --trial_file /path/to/trials.txt
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size in config file
# Edit configs/vae_improved.yaml:
# data:
#   batch_size: 32  # Reduce from default
```

### Error: Module not found (torchaudio, torch, etc.)
```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Verification Checklist

After running evaluation, verify:

- ✅ Results JSON created: `results/librispeech_eval.json`
- ✅ Contains EER and minDCF scores
- ✅ EER ~32% range (VAE-Disentangler on LibriSpeech)
- ✅ minDCF ~0.0 (good separation)
- ✅ Utterances loaded: 2,620
- ✅ Speakers loaded: 40
- ✅ Trial pairs generated: 10,000

---

## File Locations

| Component | Path |
|-----------|------|
| Test data (WAV) | `./LibriSpeech/test-clean/` |
| Pre-trained model | `checkpoints/librispeech_disentangler_vae/best.pt` |
| Results | `results/librispeech_eval.json` |
| Config | `configs/vae_improved.yaml` |
| This guide | `EVALUATION_COMMANDS.md` |

---

## Quick Copy-Paste Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Evaluate VAE on LibriSpeech (included data)
python evaluate_librispeech.py \
    --config configs/vae_improved.yaml \
    --checkpoint checkpoints/librispeech_disentangler_vae/best.pt \
    --eval_dir ./LibriSpeech \
    --output results/librispeech_eval.json

# 3. View results
cat results/librispeech_eval.json

# 4. (Optional) Batch evaluate all models
python evaluate_all_models.py

# 5. (Optional) Convert FLAC to WAV
python convert_flac_to_wav.py --input_dir ./LibriSpeech --keep_flac
```

---

## See Also

- `README.md` - Overview and project structure
- `LIBRISPEECH_EVAL.md` - LibriSpeech evaluation guide
- `LIBRISPEECH_WAV_CONVERSION.md` - WAV conversion details
- `requirements.txt` - Package dependencies
