# Q2 Implementation: Critical Issues & Execution Guide

**Purpose**: Identify potential issues when running experiments and provide solutions

---

## CRITICAL ISSUES TO ADDRESS

### 1. ⚠️ LibriSpeech Path Configuration

**Issue**: Config files hardcode path to user home directory
```yaml
# librispeech_baseline.yaml (Line 14)
data:
  train_root: "/csehome/m25csa028/Q2"  # ← This is YOUR home path
```

**Problem**:
- Path is specific to one machine (m25csa028)
- Will fail on any other system
- Grading system may use different path

**Solution**:
```bash
# Before running, update ALL config files:

# Option 1: Use environment variable approach
export Q2_DATA_ROOT="/path/to/your/data"
# Then create symlink in q2/ folder
cd q2
ln -s $Q2_DATA_ROOT data_root

# Option 2: Update configs before running
sed -i 's|/csehome/m25csa028/Q2|/your/actual/path|g' configs/*.yaml

# Option 3: Create machine-specific config
cp configs/librispeech_baseline.yaml configs/librispeech_baseline_local.yaml
# Edit configs/librispeech_baseline_local.yaml with your actual paths
```

**Action Required**:
- [ ] Update data paths in ALL yaml files before running
- [ ] Test data path exists: `ls /path/to/LibriSpeech/train-clean-100/`

---

### 2. ⚠️ Dataset Download & Extraction

**Issue**: LibriSpeech must be downloaded and extracted properly

**Expected Structure**:
```
/path/to/Q2/
└── LibriSpeech/
    ├── train-clean-100/
    │   ├── 121/              # Speaker ID
    │   │   ├── 123859/       # Chapter ID
    │   │   │   ├── 121-123859-0001.flac
    │   │   │   ├── 121-123859-0002.flac
    │   │   │   └── ...
    │   │   └── 123860/
    │   ├── 122/
    │   └── ...
    └── README (optional)
```

**Download Instructions**:
```bash
# Download (1 GB, ~10 minutes)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz

# Extract (creates LibriSpeech/ folder)
tar -xzf train-clean-100.tar.gz

# Verify structure
ls -la LibriSpeech/train-clean-100/ | head -5
# Should show speaker ID folders: 121, 122, 126, etc.
```

**Verification Script**:
```bash
#!/bin/bash
CHECK_PATH="${1:-/csehome/m25csa028/Q2}"

echo "Checking LibriSpeech structure at: $CHECK_PATH"

# Check root exists
if [ ! -d "$CHECK_PATH/LibriSpeech" ]; then
    echo "❌ MISSING: $CHECK_PATH/LibriSpeech"
    exit 1
fi

# Check subset exists
if [ ! -d "$CHECK_PATH/LibriSpeech/train-clean-100" ]; then
    echo "❌ MISSING: train-clean-100 subset"
    exit 1
fi

# Count speakers and utterances
SPEAKERS=$(find "$CHECK_PATH/LibriSpeech/train-clean-100" -mindepth 1 -maxdepth 1 -type d | wc -l)
UTTERANCES=$(find "$CHECK_PATH/LibriSpeech/train-clean-100" -name "*.flac" | wc -l)

echo "✅ Found $SPEAKERS speakers"
echo "✅ Found $UTTERANCES utterances"

if [ "$SPEAKERS" -ge 100 ] && [ "$UTTERANCES" -ge 28000 ]; then
    echo "✅ Dataset structure VALID"
else
    echo "⚠️  WARNING: Smaller dataset than expected"
fi
```

**Action Required**:
- [ ] Download and extract LibriSpeech train-clean-100
- [ ] Verify folder structure matches expected format
- [ ] Update config file paths

---

### 3. ⚠️ GPU Memory Requirements

**Issue**: Different models have different GPU memory footprints

**Memory Estimates** (batch_size=128):
```
TDNNEncoder alone:           ~2 GB
+ Baseline framework:        ~3 GB
+ AE-Disentangler:          ~4 GB
+ VAE-Disentangler:         ~4.5 GB
```

**Problem**:
- Most grad cluster GPUs are 24-32 GB (sufficient)
- Some may have older 16 GB GPUs (tight for VAE)
- Out-of-memory errors will silently kill training

**Solution**:
```python
# In train.py, before training loop:
import torch

# Check available memory
device = torch.device('cuda:0')
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {device_name}, Memory: {total_mem:.1f} GB")
else:
    print("WARNING: CUDA not available, using CPU (very slow!)")

# Monitor during training
torch.cuda.reset_peak_memory_stats()
# ... training loop ...
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory used: {peak_mem:.2f} GB")
```

**If Out of Memory**:
```yaml
# In config file, reduce batch_size
data:
  batch_size: 64   # Instead of 128
  # or
  batch_size: 32   # For 16GB GPUs
```

**Action Required**:
- [ ] Check GPU available memory: `nvidia-smi`
- [ ] If <16GB, reduce batch_size in config
- [ ] Monitor memory during first epoch

---

### 4. ⚠️ Environment Variable Setup

**Issue**: Dependencies need to be imported correctly

**Potential Problems**:
```python
# ImportError if libraries not installed
import torch
import torchaudio
import yaml
import scipy
import numpy
```

**Solution**:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install exact versions from requirements.txt
pip install --upgrade pip
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml==6.0 scipy==1.10.0 numpy==1.24.0 matplotlib=3.7.0 scikit-learn==1.2.0

# Verify imports
python3 -c "import torch, torchaudio, yaml, scipy; print('✅ All imports OK')"
```

**Create requirements.txt** (if not present):
```
torch==2.1.0
torchaudio==2.1.0
pyyaml==6.0
scipy==1.10.0
numpy==1.24.0
matplotlib==3.7.0
scikit-learn==1.2.0
```

**Action Required**:
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Test imports before running

---

### 5. ⚠️ Checkpoint Loading Issues

**Issue**: Pre-training uses encoder checkpoint that may not exist

```python
# train.py Line 210-211
if args.encoder_ckpt:
    logger.info(f"Loading pretrained encoder from {args.encoder_ckpt}")
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    # This code doesn't actually LOAD the checkpoint!
```

**Problem**:
- Code loads checkpoint but doesn't apply it to model
- Disentangler training gives random encoder (not pretrained)

**Fix Required** (if you want to use pretrained encoder):
```python
# In train.py, after loading checkpoint:
if args.encoder_ckpt:
    logger.info(f"Loading pretrained encoder from {args.encoder_ckpt}")
    ckpt = torch.load(args.encoder_ckpt, map_location=device)

    # Extract encoder state dict from baseline checkpoint
    encoder_state = {}
    for key, value in ckpt['model_state_dict'].items():
        if key.startswith('embedding_extractor.'):
            # Remove prefix 'embedding_extractor.'
            encoder_key = key.replace('embedding_extractor.', '')
            encoder_state[encoder_key] = value

    # Load into current model's encoder
    model.embedding_extractor.load_state_dict(encoder_state, strict=True)
    logger.info("Encoder pretrained weights loaded")
```

**Alternative** (Recommended):
- Skip using pretrained encoder
- Train disentangler/VAE from scratch alongside encoder
- This is simpler and sometimes works better anyway

**Action Required**:
- [ ] Decide if pre-training is needed
- [ ] If yes, implement checkpoint loading correctly
- [ ] If no, remove `--encoder_ckpt` argument

---

### 6. ⚠️ Trial File Format for Evaluation

**Issue**: eval.py expects trial file in specific format

**Required Format** (space-separated):
```
speaker_id1 utterance_id1 speaker_id2 utterance_id2 label

# Examples:
id10001 1zcIwhp2r_c 1 id10001 1zcIwhp2r_c 1 1     # Same speaker: label=1
id10001 1zcIwhp2r_c 1 id10002 1zcIwhp2r_c 2 0     # Different speakers: label=0

# Where:
# - speaker_id1, speaker_id2: Speaker identifiers
# - utterance_id1, utterance_id2: Utterance identifiers
# - label: 1 if same speaker, 0 if different
```

**For LibriSpeech** (if you have test set):
```bash
# Create trial file from test utterances
# Format: spk_id chapter_id utt_file spk_id chapter_id utt_file label

cat > trials.txt << 'EOF'
121 123859 121-123859-0001 121 123859 121-123859-0002 1
121 123859 121-123859-0001 122 123860 122-123860-0001 0
EOF
```

**If Trial File Missing**:
```python
# eval.py will generate synthetic trials

# Better: generate trials from test set
def generate_trials(eval_dataset, num_pairs=10000):
    """Generate synthetic verification trials"""
    import random

    utterances = eval_dataset.utterances
    trials = []

    for _ in range(num_pairs // 2):
        # Same speaker pair
        utt1 = random.choice(utterances)
        utt2 = random.choice(utterances)
        if utt1[0] == utt2[0]:  # Same speaker ID
            trials.append((utt1, utt2, 1))

    for _ in range(num_pairs // 2):
        # Different speaker pair
        spk1, spk2 = random.sample(set(u[0] for u in utterances), 2)
        utt1 = random.choice([u for u in utterances if u[0] == spk1])
        utt2 = random.choice([u for u in utterances if u[0] == spk2])
        trials.append((utt1, utt2, 0))

    return trials
```

**Action Required**:
- [ ] Prepare valid trial file for test set
- [ ] Or implement synthetic trial generation
- [ ] Test eval.py with small trial set first

---

### 7. ⚠️ Random Seed and Reproducibility

**Issue**: Different seeds produce different results

**Current Setup**:
```yaml
# config files
random_seed: 42
```

**Ensure Reproducibility**:
```python
# Add to train.py main():
import random
seed = config.get('random_seed', 42)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Note**: Setting `deterministic=True` may slow down training by ~10-20%

**Action Required**:
- [ ] Add seed initialization code to train.py
- [ ] Document expected random seed in README
- [ ] Report results using fixed seed=42

---

## EXECUTION CHECKLIST

Before running any training:

```bash
# 1. Setup environment
[ ] Create Python virtual environment
[ ] Install dependencies: torch, torchaudio, yaml, scipy, numpy
[ ] Verify CUDA/GPU available: nvidia-smi

# 2. Prepare data
[ ] Download LibriSpeech train-clean-100 (1 GB)
[ ] Extract tar.gz file
[ ] Verify directory structure
[ ] Update config file paths to match your system

# 3. Verify code
[ ] Compile all Python files: python3 -m py_compile q2/*.py
[ ] Check all imports: python3 -c "import torch, torchaudio, yaml"
[ ] Validate YAML: python3 -c "import yaml; yaml.safe_load(open('q2/configs/baseline.yaml'))"

# 4. Test with small dataset
[ ] Create tiny test dataset (10 speakers, 100 utterances)
[ ] Run training for 1 epoch: python train.py --config configs/baseline.yaml
[ ] Verify no errors, GPU memory < available
[ ] Check checkpoint created: ls checkpoints/*/

# 5. Run full training
[ ] Train baseline: ~4 hours
[ ] Train AE-Disentangler: ~6 hours
[ ] Train VAE-Disentangler: ~6.5 hours

# 6. Evaluate models
[ ] Prepare test set and trial file
[ ] Run eval.py for each model
[ ] Generate results JSON files
[ ] Create plots: python generate_plots.py

# 7. Documentation
[ ] Verify all results in q2/results/
[ ] Update README with actual performance numbers
[ ] Document exact commands run
[ ] Record training times
```

---

## Quick Debugging Guide

### Error: `FileNotFoundError: Dataset not found`

```python
# Check if data path in config is correct
import os
config_path = "q2/configs/librispeech_baseline.yaml"
import yaml
config = yaml.safe_load(open(config_path))
train_root = config['data']['train_root']
print(f"Looking for dataset at: {train_root}")
print(f"Exists: {os.path.exists(train_root)}")

# List what's there
os.listens(train_root)
```

### Error: `CUDA out of memory`

```bash
# Solution 1: Reduce batch size
sed -i 's/batch_size: 128/batch_size: 64/g' configs/librispeech_baseline.yaml

# Solution 2: Use CPU (SLOW!)
# Modify train.py: device = torch.device('cpu')

# Solution 3: Use smaller model
# Create new config with smaller embedding_dim: 128 instead of 256
```

### Error: `ImportError: No module named 'torch'`

```bash
# Install PyTorch
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Check installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Training is very slow (< 5% GPU utilization)

```python
# Add to train.py to diagnose:
import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Available Memory: {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB")

# Check data loading speed:
# Run Training → look for "loading data taking X seconds"
# If >1 second per batch, increase num_workers
```

---

## Summary Table

| Requirement | Status | Action |
|------------|--------|--------|
| Python environment | ⚠️ Setup needed | Install dependencies |
| Data paths | ⚠️ Hardcoded | Update config files |
| LibriSpeech dataset | ⚠️ Download needed | Get from openslr.org |
| GPU memory | ⚠️ Check | Verify 8+ GB available |
| Checkpoint loading | ⚠️ Incomplete | Decide if needed or fix |
| Random seed | ⚠️ Add setup | Ensure reproducibility |
| Trial file | ⚠️ Must prepare | Create or generate |

---

**Last Updated**: March 26, 2026
**Status**: Ready for execution with proper setup
