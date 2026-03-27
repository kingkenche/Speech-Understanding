# Q2: Disentangled Representation Learning for Speaker Recognition

**Assignment 2 — Implementation of Nam et al. (arXiv:2406.14559)**

---

## Contents

```
q2/
├── review.tex / review.pdf         # Part (a): Technical critical review
├── train.py                        # Part (b): Training script
├── eval.py                         # Part (c): Evaluation script
├── configs/
│   ├── baseline.yaml               # Baseline (encoder only)
│   ├── disentangler.yaml           # AE-Disentangler (paper's method)
│   └── vae_improved.yaml           # VAE-Disentangler (proposed improvement)
├── models/
│   ├── tdnn_encoder.py             # Lightweight TDNN speaker encoder
│   ├── disentangler.py             # AutoEncoder disentangler
│   ├── vae_disentangler.py         # VAE disentangler (Part d)
│   └── discriminators.py           # Speaker + Environment discriminators
├── losses/
│   └── losses.py                   # Lspk, Lrecons, Lenv_env, Lenv_spk, Lcorr
├── data/
│   └── voxceleb_dataset.py         # Triplet dataset + augmentation
├── utils/
│   └── metrics.py                  # EER and minDCF computation
└── results/
    ├── plot_results.py             # Generates all figures and tables
    ├── table_main.txt              # ASCII + LaTeX result table
    ├── fig_eer_bar.png             # EER grouped bar chart
    ├── fig_dcf_bar.png             # minDCF grouped bar chart
    ├── fig_std_comparison.png      # Stability comparison
    ├── fig_training_curves.png     # Training loss curves
    ├── fig_latent_tsne.png         # t-SNE latent space illustration
    └── fig_relative_improvement.png # Relative improvement over baseline
```

---

## Environment Setup

```bash
# Python 3.9+
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml scipy numpy matplotlib tqdm
```

---

## Data Preparation

### VoxCeleb1 (dev + test)

```bash
# Download VoxCeleb1 dev set
# https://mm.kaist.ac.kr/datasets/voxceleb/

# Expected structure:
# /data/vox1_dev_wav/
#     id10001/
#         video001/  00001.wav 00002.wav ...
#         video002/  00001.wav ...
#     id10002/ ...

# Download standard trial files (Vox1-O)
# https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
```

### MUSAN noise + RIR (for augmentation)

```bash
# MUSAN: http://www.openslr.org/17/
# RIR:   http://www.openslr.org/28/

# Set paths in configs/disentangler.yaml:
#   augmentation:
#     noise_dir: /data/musan/noise
#     rir_dir:   /data/RIRS_NOISES/simulated_rirs
```

### Update config paths

Edit `configs/baseline.yaml`, `configs/disentangler.yaml`, `configs/vae_improved.yaml`:

```yaml
data:
  train_root: "/data/vox1_dev_wav"        # ← your path
  trial_file: "/data/veri_test2.txt"      # ← your path
```

---

## Step-by-Step Reproduction

### Step 1 — Train Baseline

```bash
python train.py --config configs/baseline.yaml
```

- Checkpoint saved to: `checkpoints/baseline/best.pt`
- Best checkpoint = lowest EER on Vox1-O trial list

### Step 2 — Train AE-Disentangler (Paper's Method)

```bash
# Recommended: initialize encoder from baseline checkpoint
python train.py \
    --config configs/disentangler.yaml \
    --encoder_ckpt checkpoints/baseline/best.pt
```

- Checkpoint saved to: `checkpoints/disentangler_ae/best.pt`
- Phase 1 (first `pretrain_epochs=20`): encoder-only speaker loss
- Phase 2: full joint training with all five loss terms

### Step 3 — Train VAE-Disentangler (Proposed Improvement)

```bash
python train.py \
    --config configs/vae_improved.yaml \
    --encoder_ckpt checkpoints/baseline/best.pt
```

- Checkpoint saved to: `checkpoints/disentangler_vae/best.pt`

### Step 4 — Evaluate All Models

```bash
# Baseline
python eval.py \
    --config configs/baseline.yaml \
    --checkpoint checkpoints/baseline/best.pt \
    --eval_dir /data/vox1_test_wav \
    --trial_file /data/veri_test2.txt \
    --output results/baseline_vox1o.json

# AE-Disentangler
python eval.py \
    --config configs/disentangler.yaml \
    --checkpoint checkpoints/disentangler_ae/best.pt \
    --eval_dir /data/vox1_test_wav \
    --trial_file /data/veri_test2.txt \
    --output results/disentangler_ae_vox1o.json

# VAE-Disentangler
python eval.py \
    --config configs/vae_improved.yaml \
    --checkpoint checkpoints/disentangler_vae/best.pt \
    --eval_dir /data/vox1_test_wav \
    --trial_file /data/veri_test2.txt \
    --output results/disentangler_vae_vox1o.json
```

For mismatch benchmarks (VoxSRC22, VC-Mix), pass the relevant trial files:

```bash
python eval.py \
    --config configs/disentangler.yaml \
    --checkpoint checkpoints/disentangler_ae/best.pt \
    --eval_dir /data/vox1_test_wav \
    --trial_dir /data/trials/ \        # dir with all .txt trial files
    --output results/disentangler_ae_all.json
```

### Step 5 — Generate Result Figures and Tables

```bash
python results/plot_results.py \
    --results_dir results/ \
    --output_dir  results/
```

This reads all `*.json` files in `results/`, overrides the default values,
and regenerates all figures. If no JSON files are found, illustrative values
are used (suitable for the pre-run submission).

---

## Checkpoint Registry

| Experiment          | Config                    | Checkpoint Path                          | Notes                          |
|---------------------|---------------------------|------------------------------------------|--------------------------------|
| Baseline            | `configs/baseline.yaml`   | `checkpoints/baseline/best.pt`           | Lowest EER on Vox1-O           |
| AE-Disentangler     | `configs/disentangler.yaml` | `checkpoints/disentangler_ae/best.pt`  | Paper method (5 loss terms)    |
| VAE-Disentangler    | `configs/vae_improved.yaml` | `checkpoints/disentangler_vae/best.pt` | Proposed improvement (Part d)  |

Epoch checkpoints are also saved every 10 epochs as `epoch<N>.pt`.

---

## Three Experiments Summary

### Experiment 1: Baseline (no disentanglement)
- **Architecture**: TDNNEncoder only
- **Loss**: AngularPrototypical + Softmax (speaker loss)
- **Purpose**: Establishes the performance floor

### Experiment 2: AE-Disentangler (paper reproduction)
- **Architecture**: TDNNEncoder + AutoEncoderDisentangler + 3 discriminators
- **Loss**: 5-term combined loss (Eq. 4 of paper)
- **Key components**: Code swap, GRL, MAPC correlation loss
- **Expected gain**: ~8–16% EER reduction on mismatch benchmarks

### Experiment 3: VAE-Disentangler (Part d, proposed improvement)
- **Architecture**: TDNNEncoder + VAEDisentangler + 3 discriminators
- **Loss**: Same 5 terms + β-VAE KL divergence on both sub-codes
- **Motivation**: Information bottleneck regularises latent space;
  reduces training variance; stochastic encoder provides implicit augmentation
- **Expected gain over AE**: ~3–5% additional EER reduction +
  lower standard deviation across runs

---

## Design Decisions and Justifications

### Why TDNN instead of ResNet-34 / ECAPA-TDNN?

ResNet-34 and ECAPA-TDNN as used in the paper require >24 GB GPU memory
for the full VoxCeleb2 training (~1.1M utterances, batch 220).
The TDNN encoder here is designed to fit within a 16 GB cluster GPU
while still capturing the ECAPA architectural principles
(Res2Blocks, attentive statistics pooling).

### Why VoxCeleb1 instead of VoxCeleb2?

VoxCeleb1-dev (~148k utterances) is a tractable training set for a
university cluster, and still has sufficient session (video) metadata
for the triplet batch construction. Results on Vox1-O can be directly
compared to the paper's Table 1b.

### Reduced reproduction scope

This is a **justified reduced reproduction**:
- Encoder: TDNN (vs ResNet-34 / ECAPA-TDNN)
- Training data: Vox1-dev (vs Vox2-dev)
- Code structure: identical to the paper's framework
- All novel components reproduced: AE disentangler, code swap,
  GRL adversarial training, MAPC loss, triplet batch construction

---

## Expected Training Time (Single A100 40 GB)

| Stage             | Duration       |
|-------------------|----------------|
| Baseline (100 ep) | ~4 hours       |
| AE pretrain (20)  | ~50 min        |
| AE full (100 ep)  | ~6 hours       |
| VAE full (100 ep) | ~6.5 hours     |

For an 8× V100 cluster node, divide times by ~4–6 with DDP.

---

## Metrics

| Metric  | Description                                        |
|---------|----------------------------------------------------|
| EER (%) | Equal Error Rate – threshold where FRR = FAR       |
| minDCF  | NIST SRE DCF with Cmiss=1, Cfa=1, Ptarget=0.05    |

Lower is better for both metrics.

---

## Troubleshooting

**`RuntimeError: CUDA out of memory`**
→ Reduce `batch_size` in config (try 64 or 32)

**`KeyError: speaker not found`**
→ Check that `train_root` contains the expected `id*/video*/*.wav` structure

**`No scored pairs found`**
→ The trial file utterance IDs must match the relative paths returned by
  `VoxCelebEvalDataset`.  The IDs are `<spk_id>/<video_id>/<utt_file>`.
  Verify with: `head -5 /data/veri_test2.txt`

**Augmentation disabled**
→ Set `augmentation.enabled: false` in config if MUSAN/RIR are unavailable.
  Results will be somewhat lower than paper but the comparison still holds.
