"""
train_fair.py
=============
Train a CTC-based ASR model with a custom Fairness Loss that minimises
the performance gap across demographic groups from the SPS Corpus audit.

Fairness Loss (Group-DRO variant):
    L_total = L_CTC_mean
            + λ_fair × (worst_group_loss − best_group_loss)
            + λ_reg  × Σ_g max(0, L_g − τ)²

Usage:
    # Quick demo – no data needed:
    python train_fair.py --demo --epochs 3

    # Single-TSV corpus (SPS format):
    python train_fair.py \
        --tsv data/sps-corpus-3.0-2026-03-09-en/ss-corpus-en.tsv \
        --clips_dir data/sps-corpus-3.0-2026-03-09-en/clips \
        --output_dir checkpoints/ \
        --epochs 20 --lambda_fair 0.5

    # Common Voice directory with train/dev splits already present:
    python train_fair.py \
        --data_dir /path/to/cv-corpus-en \
        --output_dir checkpoints/ \
        --epochs 20

    # Baseline run (no fairness loss, for comparison):
    python train_fair.py --demo --no_fair --epochs 3 \
        --output_dir checkpoints_baseline/
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    raise SystemExit("[ERROR] PyTorch not installed: pip install torch torchaudio")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Audio
    sample_rate:   int   = 16_000
    max_duration:  float = 10.0
    n_mels:        int   = 80
    n_fft:         int   = 1024
    hop_length:    int   = 256

    # Model
    vocab_size:    int   = 29        # a-z + space + <blank>
    encoder_dim:   int   = 256
    num_layers:    int   = 6
    num_heads:     int   = 4
    ffn_dim:       int   = 512
    dropout:       float = 0.1

    # Training
    batch_size:    int   = 8
    learning_rate: float = 3e-4
    epochs:        int   = 20
    clip_grad:     float = 5.0

    # Fairness
    lambda_fair:   float = 0.5
    lambda_reg:    float = 0.1
    fair_threshold:float = 0.05     # τ: acceptable per-group CTC loss ceiling

    # Demographic groups (gender × coarse age) – derived from audit
    # Uses full SPS 6-category gender set
    demographic_groups: List[str] = field(default_factory=lambda: [
        # female_feminine
        "female_feminine_young",   "female_feminine_middle",
        "female_feminine_old",     "female_feminine_elderly",
        # male_masculine
        "male_masculine_young",    "male_masculine_middle",
        "male_masculine_old",      "male_masculine_elderly",
        # other gender categories (rarer – grouped as singletons)
        "intersex",
        "transgender",
        "non-binary",
        "do_not_wish_to_say",
        # unlabelled / unknown
        "unknown",
    ])

    # I/O
    output_dir:    str   = "checkpoints"
    log_interval:  int   = 10
    val_split:     float = 0.10      # used when only a single TSV is provided


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

VOCAB   = list("abcdefghijklmnopqrstuvwxyz ") + ["<blank>"]
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
BLANK   = CHAR2ID["<blank>"]


def text_to_ids(text: str) -> List[int]:
    return [CHAR2ID[c] for c in text.lower() if c in CHAR2ID]


# ─────────────────────────────────────────────────────────────────────────────
# Age bucketing (mirrors privacymodule.py)
# ─────────────────────────────────────────────────────────────────────────────

AGE_BUCKET: Dict[str, str] = {
    "teens":     "young",  "twenties": "young",
    "thirties":  "middle", "fourties": "middle",
    "fifties":   "old",    "sixties":  "old",
    "seventies": "elderly","eighties": "elderly",
    "nineties":  "elderly",
}


def get_group(gender: str, age: str) -> str:
    """Map raw SPS gender/age strings to a fairness group label."""
    g = str(gender).strip().lower()
    a = str(age).strip().lower()

    age_bucket = AGE_BUCKET.get(a, None)

    if g in {"female_feminine", "male_masculine"} and age_bucket:
        return f"{g}_{age_bucket}"
    elif g in {"intersex", "transgender", "non-binary", "do_not_wish_to_say"}:
        return g
    else:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpeechDataset(Dataset):
    def __init__(self, manifest: list, clips_dir: Optional[str],
                 cfg: TrainConfig, augment: bool = False):
        self.samples   = manifest
        self.clips_dir = clips_dir
        self.cfg       = cfg
        self.augment   = augment
        self.max_len   = int(cfg.sample_rate * cfg.max_duration)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item  = self.samples[idx]
        # Support both SPS ('audio_file') and Common Voice ('path') column names
        audio_path = item.get("path", "") or item.get("audio_file", "")
        wav   = self._load(str(audio_path))
        # Support both SPS ('transcription') and CV ('sentence') column names
        text  = item.get("sentence", "") or item.get("transcription", "") or "hello world"
        label = text_to_ids(str(text))
        group = get_group(
            item.get("gender", ""), item.get("age", "")
        )
        return {
            "wav":   torch.from_numpy(wav).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "group": group,
        }

    def _load(self, rel_path: str) -> np.ndarray:
        # Try to locate the actual audio file
        candidates = []
        if rel_path:
            candidates.append(rel_path)                          # absolute
            if self.clips_dir:
                candidates.append(
                    os.path.join(self.clips_dir, os.path.basename(rel_path))
                )
        for p in candidates:
            if os.path.exists(p):
                try:
                    import torchaudio
                    wav, sr = torchaudio.load(p)
                    wav = wav.mean(0).numpy()
                    if sr != self.cfg.sample_rate:
                        import librosa
                        wav = librosa.resample(
                            wav, orig_sr=sr, target_sr=self.cfg.sample_rate
                        )
                    return wav[:self.max_len].astype(np.float32)
                except Exception:
                    pass
        # Fallback synthetic signal
        n    = self.cfg.sample_rate
        rng  = np.random.default_rng(hash(rel_path) % (2**31))
        t    = np.linspace(0, 1, n, dtype=np.float32)
        return (np.sin(2 * np.pi * 200 * t) * 0.3 +
                rng.standard_normal(n).astype(np.float32) * 0.05)

    @staticmethod
    def collate(batch):
        wavs   = [b["wav"]   for b in batch]
        labels = [b["label"] for b in batch]
        groups = [b["group"] for b in batch]

        wav_pad   = nn.utils.rnn.pad_sequence(wavs,   batch_first=True)
        label_pad = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return {
            "wav":       wav_pad,
            "wav_len":   torch.tensor([len(w) for w in wavs]),
            "label":     label_pad,
            "label_len": torch.tensor([len(l) for l in labels]),
            "group":     groups,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_single_tsv(tsv_path: str, val_split: float = 0.10):
    """
    Load SPS corpus TSV.
    Handles SPS-specific column names:
      - audio_file   -> normalised to 'path' (full path under audios/)
      - transcription -> normalised to 'sentence'
    Uses the existing split column when present (train/dev/test).
    """
    import pandas as pd

    df        = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    base_dir  = os.path.dirname(os.path.abspath(tsv_path))
    audio_dir = os.path.join(base_dir, "audios")

    # Normalise audio path: SPS uses 'audio_file', CV uses 'path'
    if "audio_file" in df.columns and "path" not in df.columns:
        df["path"] = df["audio_file"].apply(
            lambda f: os.path.join(audio_dir, str(f))
        )

    # Normalise text: SPS uses 'transcription', CV uses 'sentence'
    if "transcription" in df.columns and "sentence" not in df.columns:
        df["sentence"] = df["transcription"]

    # Use existing split column when present
    if "split" in df.columns and df["split"].notna().any():
        train_df = df[df["split"] == "train"]
        val_df   = df[df["split"].isin(["dev", "validation", "test"])]
        if len(val_df) == 0:
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                train_df, test_size=val_split, random_state=42
            )
        return train_df.to_dict("records"), val_df.to_dict("records")

    # Fallback: random split
    from sklearn.model_selection import train_test_split
    t, v = train_test_split(df, test_size=val_split, random_state=42)
    return t.to_dict("records"), v.to_dict("records")

def load_cv_directory(data_dir: str):
    """Load Common Voice train.tsv + dev.tsv."""
    import pandas as pd
    train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
    val_df   = pd.read_csv(os.path.join(data_dir, "dev.tsv"),   sep="\t")
    return train_df.to_dict("records"), val_df.to_dict("records")


def generate_synthetic_manifest(n: int = 200, seed: int = 0) -> list:
    rng   = np.random.default_rng(seed)
    words = [
        "hello world", "the cat sat on the mat",
        "how are you today", "machine learning is fascinating",
        "open source audio data", "speech recognition systems",
        "privacy preserving ai", "fairness in machine learning",
    ]
    genders = rng.choice(
        ["female_feminine", "male_masculine", "intersex",
         "transgender", "non-binary", "do_not_wish_to_say"],
        p=[0.38, 0.08, 0.026, 0.006, 0.002, 0.029],
        size=n,
    )
    # Normalise to sum to 1.0 for remaining
    genders_mask = rng.random(n) > 0.527   # ~47.3% will be overridden to unknown
    genders      = np.where(genders_mask, genders, "unknown")

    ages = rng.choice(
        ["twenties", "teens", "thirties", "sixties",
         "fifties", "fourties", "seventies"],
        p=[0.510, 0.124, 0.102, 0.078, 0.063, 0.021, 0.022],
        size=n,
    )
    return [
        {
            "path":     "",
            "sentence": words[i % len(words)],
            "gender":   genders[i],
            "age":      ages[i],
        }
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ConvFeatureExtractor(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1,   512, 10, stride=5,  bias=False), nn.GELU(),
            nn.Conv1d(512, 512,  3, stride=2,  bias=False), nn.GELU(),
            nn.Conv1d(512, 512,  3, stride=2,  bias=False), nn.GELU(),
            nn.Conv1d(512, 512,  3, stride=2,  bias=False), nn.GELU(),
            nn.Conv1d(512, out_dim, 2, stride=2, bias=False),
        )

    def forward(self, wav):          # (B, T) → (B, out_dim, T')
        return self.layers(wav.unsqueeze(1))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, nhead: int, ffn: int, dropout: float = 0.1):
        super().__init__()
        self.ff1   = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ffn), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn, d), nn.Dropout(dropout),
        )
        self.attn  = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                            batch_first=True)
        self.ln_a  = nn.LayerNorm(d)
        self.conv  = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GLU(dim=-1))
        self.dw    = nn.Conv1d(d, d, 31, padding=15, groups=d)
        self.ln_c  = nn.LayerNorm(d)
        self.ff2   = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ffn), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn, d), nn.Dropout(dropout),
        )
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ff1(x)
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln_a(x + h)
        h = self.dw(self.conv(x).transpose(1, 2)).transpose(1, 2)
        x = self.ln_c(x + h)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class FairASRModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.feat_ext = ConvFeatureExtractor(cfg.encoder_dim)
        self.proj_in  = nn.Linear(cfg.encoder_dim, cfg.encoder_dim)
        self.encoder  = nn.ModuleList([
            ConformerBlock(cfg.encoder_dim, cfg.num_heads,
                           cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.ctc_head = nn.Linear(cfg.encoder_dim, cfg.vocab_size)

    def forward(self, wav: torch.Tensor, wav_len: Optional[torch.Tensor] = None):
        feats = self.feat_ext(wav).transpose(1, 2)   # (B, T', D)
        x     = self.proj_in(feats)

        mask = None
        if wav_len is not None:
            T = x.size(1)
            feat_len = (wav_len / wav.size(1) * T).long().clamp(max=T)
            mask = torch.zeros(x.size(0), T, dtype=torch.bool, device=x.device)
            for i, l in enumerate(feat_len):
                if l < T:
                    mask[i, l:] = True

        for blk in self.encoder:
            x = blk(x, key_padding_mask=mask)

        log_probs = F.log_softmax(self.ctc_head(x), dim=-1)
        if wav_len is not None:
            feat_len = (
                wav_len / wav.size(1) * log_probs.size(1)
            ).long().clamp(min=1, max=log_probs.size(1))
            return log_probs, feat_len
        return log_probs, None


# ─────────────────────────────────────────────────────────────────────────────
# Fairness Loss
# ─────────────────────────────────────────────────────────────────────────────

class FairnessLoss(nn.Module):
    """
    Group-DRO inspired fairness penalty.

    L_fair = max_g(CTC_g) − min_g(CTC_g)            ← worst-best gap
           + λ_reg × Σ_g max(0, CTC_g − τ)²          ← threshold penalty

    Demographic groups use the full SPS Corpus 6-category gender taxonomy
    crossed with coarse age buckets.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.lambda_fair  = cfg.lambda_fair
        self.lambda_reg   = cfg.lambda_reg
        self.threshold    = cfg.fair_threshold
        self.group_names  = cfg.demographic_groups
        self.ema_decay    = 0.9
        self.register_buffer(
            "group_ema",
            torch.ones(len(cfg.demographic_groups)) / len(cfg.demographic_groups),
        )

    def forward(self, per_sample_loss: torch.Tensor,
                groups: List[str]):
        group_buckets: Dict[str, List[torch.Tensor]] = {}
        for loss, grp in zip(per_sample_loss, groups):
            group_buckets.setdefault(grp, []).append(loss)

        group_means = {g: torch.stack(v).mean()
                       for g, v in group_buckets.items()}

        if len(group_means) < 2:
            base = per_sample_loss.mean()
            return base, {"ctc_mean": base.item(), "fair_penalty": 0.0,
                          "gap": 0.0, "group_losses": {
                              k: v.item() for k, v in group_means.items()
                          }}

        vals  = torch.stack(list(group_means.values()))
        worst = vals.max()
        best  = vals.min()
        gap   = worst - best

        reg = torch.stack([
            F.relu(v - self.threshold) ** 2
            for v in group_means.values()
        ]).sum()

        ctc_mean = per_sample_loss.mean()

        # Update EMA
        with torch.no_grad():
            for i, gname in enumerate(self.group_names):
                if gname in group_means:
                    self.group_ema[i] = (
                        self.ema_decay * self.group_ema[i]
                        + (1 - self.ema_decay) * group_means[gname]
                    )

        total = (ctc_mean
                 + self.lambda_fair * gap
                 + self.lambda_reg  * reg)

        return total, {
            "ctc_mean":         ctc_mean.item(),
            "fair_penalty":     (self.lambda_fair * gap).item(),
            "reg_penalty":      (self.lambda_reg  * reg).item(),
            "worst_group_loss": worst.item(),
            "best_group_loss":  best.item(),
            "gap":              gap.item(),
            "group_losses":     {k: v.item() for k, v in group_means.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample CTC loss
# ─────────────────────────────────────────────────────────────────────────────

def ctc_per_sample(log_probs, targets, input_lengths, target_lengths):
    ctc_fn = nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=True)
    losses = []
    for i in range(log_probs.size(0)):
        T  = input_lengths[i].item()
        lp = log_probs[i, :T].unsqueeze(1)
        tg = targets[i, :target_lengths[i]].unsqueeze(0)
        il = input_lengths[i:i+1]
        tl = target_lengths[i:i+1]
        losses.append(ctc_fn(lp, tg, il, tl).squeeze())
    return torch.stack(losses)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class FairTrainer:
    def __init__(self, model: FairASRModel, fair_loss: FairnessLoss,
                 cfg: TrainConfig, device: str):
        self.model   = model.to(device)
        self.fair_fn = fair_loss.to(device)
        self.cfg     = cfg
        self.device  = device
        self.opt     = torch.optim.AdamW(
            list(model.parameters()) + list(fair_loss.parameters()),
            lr=cfg.learning_rate, weight_decay=1e-4,
        )
        self.sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.epochs
        )
        self.history = {"train_loss": [], "train_gap": [],
                        "val_loss":   [], "val_gap":   []}
        os.makedirs(cfg.output_dir, exist_ok=True)

    def _feat_len(self, wav, log_probs, wav_len):
        if wav_len is not None:
            return (
                wav_len / wav.size(1) * log_probs.size(1)
            ).long().clamp(min=1, max=log_probs.size(1))
        return torch.full(
            (wav.size(0),), log_probs.size(1), dtype=torch.long
        )

    def train_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        tot_loss = tot_gap = n = 0
        for batch in loader:
            wav  = batch["wav"].to(self.device)
            lbl  = batch["label"].to(self.device)
            wlen = batch["wav_len"].to(self.device)
            llen = batch["label_len"].to(self.device)
            grps = batch["group"]

            lp, fl = self.model(wav, wlen)
            fl     = self._feat_len(wav, lp, wlen if fl is None else None) if fl is None else fl
            ps     = ctc_per_sample(lp, lbl, fl, llen)
            loss, metrics = self.fair_fn(ps, grps)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.opt.step()

            tot_loss += metrics["ctc_mean"]
            tot_gap  += metrics["gap"]
            n        += 1

        return {"loss": tot_loss / max(n, 1), "gap": tot_gap / max(n, 1)}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        tot_loss = tot_gap = n = 0
        all_group_losses: Dict[str, List] = {}

        for batch in loader:
            wav  = batch["wav"].to(self.device)
            lbl  = batch["label"].to(self.device)
            wlen = batch["wav_len"].to(self.device)
            llen = batch["label_len"].to(self.device)
            grps = batch["group"]

            lp, fl = self.model(wav, wlen)
            fl     = self._feat_len(wav, lp, wlen) if fl is None else fl
            ps     = ctc_per_sample(lp, lbl, fl, llen)
            _, metrics = self.fair_fn(ps, grps)

            tot_loss += metrics["ctc_mean"]
            tot_gap  += metrics["gap"]
            n        += 1
            for g, v in metrics["group_losses"].items():
                all_group_losses.setdefault(g, []).append(v)

        return {
            "loss":         tot_loss / max(n, 1),
            "gap":          tot_gap  / max(n, 1),
            "group_losses": {g: float(np.mean(vs))
                             for g, vs in all_group_losses.items()},
        }

    def train(self, train_loader, val_loader):
        fair_str = (f"λ_fair={self.cfg.lambda_fair}" if self.cfg.lambda_fair > 0
                    else "NO FAIRNESS LOSS (baseline)")
        print(f"\n  Epochs={self.cfg.epochs}  device={self.device}  {fair_str}")
        print(f"  {'Epoch':>6} {'Train CTC':>11} {'Val CTC':>9} "
              f"{'Fair Gap':>9} {'LR':>10}")
        print("  " + "─" * 54)

        best_val = float("inf")
        for epoch in range(1, self.cfg.epochs + 1):
            t0  = time.time()
            tr  = self.train_epoch(train_loader)
            va  = self.eval_epoch(val_loader)
            self.sched.step()
            dt  = time.time() - t0

            self.history["train_loss"].append(tr["loss"])
            self.history["train_gap"].append(tr["gap"])
            self.history["val_loss"].append(va["loss"])
            self.history["val_gap"].append(va["gap"])

            lr = self.opt.param_groups[0]["lr"]
            print(f"  {epoch:>6}  {tr['loss']:>10.4f}  {va['loss']:>8.4f}  "
                  f"{va['gap']:>8.4f}  {lr:>10.2e}  [{dt:.1f}s]")

            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save({
                    "epoch":        epoch,
                    "model":        self.model.state_dict(),
                    "optimizer":    self.opt.state_dict(),
                    "val_loss":     best_val,
                    "group_losses": va.get("group_losses", {}),
                    "config":       self.cfg.__dict__,
                }, os.path.join(self.cfg.output_dir, "best_model.pt"))

            if epoch % 5 == 0 and va.get("group_losses"):
                print("    Per-group val CTC:")
                for g, l in sorted(va["group_losses"].items()):
                    bar = "▓" * int(l * 20)
                    print(f"      {g:30s}: {l:.4f}  {bar}")

        self._save_history()
        self._plot_training()
        print(f"\n  Best val loss → {best_val:.4f}")
        print(f"  Checkpoint    → {os.path.join(self.cfg.output_dir, 'best_model.pt')}")

    def _save_history(self):
        p = os.path.join(self.cfg.output_dir, "training_history.json")
        with open(p, "w") as f:
            json.dump(self.history, f, indent=2)

    def _plot_training(self):
        if not MPL_OK:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Fair ASR Training Curves", fontweight="bold")
        ep = range(1, len(self.history["train_loss"]) + 1)
        ax1.plot(ep, self.history["train_loss"], label="Train", color="#E07B54")
        ax1.plot(ep, self.history["val_loss"],   label="Val",   color="#3B8BD4")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("CTC Loss")
        ax1.set_title("CTC Loss"); ax1.legend()

        ax2.plot(ep, self.history["train_gap"], label="Train gap", color="#E07B54")
        ax2.plot(ep, self.history["val_gap"],   label="Val gap",   color="#3B8BD4")
        ax2.axhline(self.cfg.fair_threshold, color="green", linestyle="--",
                    label=f"τ = {self.cfg.fair_threshold}")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Fairness Gap")
        ax2.set_title("Fairness Gap (worst − best group)"); ax2.legend()

        plt.tight_layout()
        p = os.path.join(self.cfg.output_dir, "training_curves.png")
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Training curves → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fair ASR Training – SPS Corpus"
    )
    parser.add_argument("--tsv",         type=str,   default=None,
                        help="Single TSV file (SPS corpus format)")
    parser.add_argument("--data_dir",    type=str,   default=None,
                        help="Common Voice directory (needs train.tsv + dev.tsv)")
    parser.add_argument("--clips_dir",   type=str,   default=None,
                        help="Directory containing audio clips")
    parser.add_argument("--output_dir",  type=str,   default="checkpoints")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lambda_fair", type=float, default=0.5)
    parser.add_argument("--no_fair",     action="store_true",
                        help="Disable fairness loss (baseline comparison)")
    parser.add_argument("--demo",        action="store_true",
                        help="Run on 200 synthetic utterances (no files needed)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_fair=0.0 if args.no_fair else args.lambda_fair,
        output_dir=args.output_dir,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    if args.demo or (args.tsv is None and args.data_dir is None):
        print("  Using synthetic demo data (200 train / 50 val).")
        train_manifest = generate_synthetic_manifest(200, seed=0)
        val_manifest   = generate_synthetic_manifest(50,  seed=1)
        clips_dir      = None
    elif args.tsv:
        print(f"  Loading {args.tsv} and splitting {1-cfg.val_split:.0%}/{cfg.val_split:.0%} …")
        train_manifest, val_manifest = load_single_tsv(args.tsv, cfg.val_split)
        clips_dir = args.clips_dir
    else:
        print(f"  Loading Common Voice splits from {args.data_dir} …")
        train_manifest, val_manifest = load_cv_directory(args.data_dir)
        clips_dir = args.clips_dir or os.path.join(args.data_dir, "clips")

    print(f"  Train: {len(train_manifest):,}  |  Val: {len(val_manifest):,}")

    train_ds = SpeechDataset(train_manifest, clips_dir, cfg, augment=True)
    val_ds   = SpeechDataset(val_manifest,   clips_dir, cfg, augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=SpeechDataset.collate,
                          num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          collate_fn=SpeechDataset.collate, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = FairASRModel(cfg)
    fair_loss = FairnessLoss(cfg)
    params    = sum(p.numel() for p in model.parameters())
    print(f"  Model params : {params:,}")
    if args.no_fair:
        print("  [BASELINE] Fairness loss disabled.")
    else:
        print(f"  Fairness:  λ_fair={cfg.lambda_fair}  "
              f"λ_reg={cfg.lambda_reg}  τ={cfg.fair_threshold}")
        print(f"  Groups   : {len(cfg.demographic_groups)} "
              f"({', '.join(cfg.demographic_groups[:4])} …)")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = FairTrainer(model, fair_loss, cfg, device)
    trainer.train(train_dl, val_dl)


if __name__ == "__main__":
    main()