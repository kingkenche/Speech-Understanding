"""
privacymodule.py
================
Privacy-Preserving AI Module for Voice Biometric Obfuscation.

Architecture:
  Input Waveform → MelSpectrogram
        ↓
  ContentEncoder  →  z_content  (speaker-independent phonetic features)
  AttributeEncoder →  z_attr    (global speaker style)
        ↓
  Decoder(z_content, z_target_attr) → Transformed Mel
        ↓
  Griffin-Lim Vocoder (or HiFi-GAN in production) → Waveform

Gender vocabulary updated to match the SPS Corpus label set:
  female_feminine · male_masculine · intersex · transgender · non-binary
  do_not_wish_to_say · unknown

Usage:
    from privacymodule import PrivacyModule, VoiceTransformConfig
    cfg   = VoiceTransformConfig()
    model = PrivacyModule(cfg)

    # Inference (numpy in → numpy out):
    out_wav, metrics = model.transform(
        waveform, sample_rate=16000,
        source_attrs={"gender": "male_masculine",   "age": "old"},
        target_attrs={"gender": "female_feminine",  "age": "young"},
    )
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VoiceTransformConfig:
    sample_rate:    int   = 16_000
    n_fft:          int   = 1024
    hop_length:     int   = 256
    n_mels:         int   = 80

    content_dim:    int   = 128
    attr_dim:       int   = 64
    hidden_dim:     int   = 256
    num_res_blocks: int   = 4

    # ── SPS Corpus label sets ──────────────────────────────────────────
    gender_classes: List[str] = field(default_factory=lambda: [
        "female_feminine",
        "male_masculine",
        "intersex",
        "transgender",
        "non-binary",
        "do_not_wish_to_say",
        "unknown",
    ])
    age_classes: List[str] = field(default_factory=lambda: [
        "child",      # <13 (mapped from 'teens' if very young)
        "young",      # teens + twenties
        "middle",     # thirties + fourties
        "old",        # fifties + sixties
        "elderly",    # seventies, eighties, nineties
        "unknown",
    ])

    # Training hyperparameters
    lr:             float = 1e-4
    lambda_recon:   float = 10.0
    lambda_cycle:   float = 5.0
    lambda_attr:    float = 1.0


# Map raw SPS corpus age strings to coarse age_classes
AGE_BUCKET_MAP: Dict[str, str] = {
    "teens":     "young",
    "twenties":  "young",
    "thirties":  "middle",
    "fourties":  "middle",
    "fifties":   "old",
    "sixties":   "old",
    "seventies": "elderly",
    "eighties":  "elderly",
    "nineties":  "elderly",
}


def bucket_age(raw_age: str) -> str:
    """Convert a raw SPS age label to a coarse bucket."""
    return AGE_BUCKET_MAP.get(str(raw_age).strip().lower(), "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3,
                      padding=dilation, dilation=dilation),
            nn.InstanceNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

class ContentEncoder(nn.Module):
    """Removes speaker identity; keeps phonetic / linguistic information."""
    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        C = cfg.hidden_dim
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.n_mels, C, 5, padding=2),
            nn.InstanceNorm1d(C), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(C, C, 4, stride=2, padding=1),
            nn.InstanceNorm1d(C), nn.LeakyReLU(0.2, inplace=True),
        )
        self.res  = nn.Sequential(
            *[ResBlock1D(C, dilation=2 ** i)
              for i in range(cfg.num_res_blocks)]
        )
        self.proj = nn.Conv1d(C, cfg.content_dim, 1)

    def forward(self, mel):          # (B, n_mels, T) → (B, content_dim, T/2)
        return self.proj(self.res(self.stem(mel)))


class AttributeEncoder(nn.Module):
    """Extracts a global speaker attribute vector (gender, age, style)."""
    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        C = cfg.hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.n_mels, C, 5, padding=2),
            nn.InstanceNorm1d(C), nn.LeakyReLU(0.2),
            nn.Conv1d(C, C, 4, stride=2, padding=1),
            nn.InstanceNorm1d(C), nn.LeakyReLU(0.2),
            nn.Conv1d(C, C, 4, stride=2, padding=1),
            nn.InstanceNorm1d(C), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(C, cfg.attr_dim)

    def forward(self, mel):          # (B, n_mels, T) → (B, attr_dim)
        return self.fc(self.pool(self.conv(mel)).squeeze(-1))


class TargetAttributeEmbedder(nn.Module):
    """
    Converts discrete attribute labels (gender, age) to a continuous embedding.
    Supports the full SPS Corpus 6-category gender taxonomy.
    """
    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        self.gender_emb = nn.Embedding(
            len(cfg.gender_classes), cfg.attr_dim // 2
        )
        self.age_emb = nn.Embedding(
            len(cfg.age_classes), cfg.attr_dim // 2
        )
        self.gender2idx = {g: i for i, g in enumerate(cfg.gender_classes)}
        self.age2idx    = {a: i for i, a in enumerate(cfg.age_classes)}

    def forward(self, gender_ids: torch.Tensor,
                age_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.gender_emb(gender_ids), self.age_emb(age_ids)], dim=-1
        )

    def encode_attrs(
        self, attrs: Dict[str, str], device: torch.device
    ) -> torch.Tensor:
        """
        Convert a string dict → embedding tensor (batch=1).

        Accepted gender values (SPS corpus):
            female_feminine, male_masculine, intersex, transgender,
            non-binary, do_not_wish_to_say, unknown

        Accepted age values (coarse buckets):
            young, middle, old, elderly, child, unknown
        Raw SPS age values (teens, twenties, …) are auto-bucketed.
        """
        g = str(attrs.get("gender", "unknown")).strip().lower()
        a = str(attrs.get("age",    "unknown")).strip().lower()

        # Auto-bucket raw SPS age labels
        a = AGE_BUCKET_MAP.get(a, a)

        gidx = self.gender2idx.get(g, self.gender2idx["unknown"])
        aidx = self.age2idx.get(a,   self.age2idx["unknown"])

        gt = torch.tensor([gidx], dtype=torch.long, device=device)
        at = torch.tensor([aidx], dtype=torch.long, device=device)
        return self.forward(gt, at)   # (1, attr_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder  (AdaIN-based)
# ─────────────────────────────────────────────────────────────────────────────

class AdaIN(nn.Module):
    """Adaptive Instance Normalisation – injects style into content."""
    def __init__(self, content_dim: int, attr_dim: int):
        super().__init__()
        self.norm  = nn.InstanceNorm1d(content_dim, affine=False)
        self.scale = nn.Linear(attr_dim, content_dim)
        self.bias  = nn.Linear(attr_dim, content_dim)

    def forward(self, x, style):
        s = self.scale(style).unsqueeze(-1)
        b = self.bias(style).unsqueeze(-1)
        return s * self.norm(x) + b


class AdaINResBlock(nn.Module):
    def __init__(self, channels: int, attr_dim: int, dilation: int = 1):
        super().__init__()
        self.conv1  = nn.Conv1d(channels, channels, 3,
                                padding=dilation, dilation=dilation)
        self.adain1 = AdaIN(channels, attr_dim)
        self.act1   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2  = nn.Conv1d(channels, channels, 3, padding=1)
        self.adain2 = AdaIN(channels, attr_dim)
        self.act2   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        h = self.act1(self.adain1(self.conv1(x), style))
        h = self.adain2(self.conv2(h), style)
        return self.act2(x + h)


class Decoder(nn.Module):
    """Reconstructs mel spectrogram from content + target attribute embedding."""
    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        C = cfg.hidden_dim
        self.proj_in    = nn.Conv1d(cfg.content_dim, C, 1)
        self.res_blocks = nn.ModuleList([
            AdaINResBlock(C, cfg.attr_dim, dilation=2 ** i)
            for i in range(cfg.num_res_blocks)
        ])
        self.upsample   = nn.Sequential(
            nn.ConvTranspose1d(C, C, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.proj_out   = nn.Conv1d(C, cfg.n_mels, 7, padding=3)

    def forward(self, content, style):
        x = self.proj_in(content)
        for blk in self.res_blocks:
            x = blk(x, style)
        x = self.upsample(x)
        return self.proj_out(x)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminators
# ─────────────────────────────────────────────────────────────────────────────

class AttributeDiscriminator(nn.Module):
    """Predicts speaker attributes from mel – used as adversarial signal."""
    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        C = cfg.hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.n_mels, C, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(C, C, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(C, C, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.pool         = nn.AdaptiveAvgPool1d(1)
        self.gender_head  = nn.Linear(C, len(cfg.gender_classes))
        self.age_head     = nn.Linear(C, len(cfg.age_classes))

    def forward(self, mel):
        x = self.pool(self.conv(mel)).squeeze(-1)
        return self.gender_head(x), self.age_head(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN-style discriminator for mel realism."""
    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cfg.n_mels,  64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64,  128, 4, stride=2, padding=1),
            nn.InstanceNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256,   1, 3, padding=1),
        )

    def forward(self, mel):
        return self.net(mel)


# ─────────────────────────────────────────────────────────────────────────────
# Mel Spectrogram helper
# ─────────────────────────────────────────────────────────────────────────────

class MelSpectrogram(nn.Module):
    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        self.cfg = cfg
        try:
            import torchaudio
            self._mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
            )
        except ImportError:
            self._mel_fn = None

    def forward(self, wav):          # (B, T) → (B, n_mels, T')
        if self._mel_fn is not None:
            mel = self._mel_fn(wav)
            return torch.log(mel.clamp(min=1e-5))
        # Fallback (demo only – not for real training)
        T2 = wav.shape[-1] // self.cfg.hop_length
        return torch.randn(
            wav.shape[0], self.cfg.n_mels, T2,
            device=wav.device, dtype=wav.dtype,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full Privacy Module
# ─────────────────────────────────────────────────────────────────────────────

class PrivacyModule(nn.Module):
    """
    End-to-end Privacy-Preserving Voice Transformation Module.

    Supported gender classes (SPS Corpus):
        female_feminine, male_masculine, intersex,
        transgender, non-binary, do_not_wish_to_say, unknown

    Supported age classes (coarse buckets):
        young (teens/twenties), middle (thirties/fourties),
        old (fifties/sixties), elderly (seventies+), child, unknown
    """

    def __init__(self, cfg: VoiceTransformConfig):
        super().__init__()
        self.cfg          = cfg
        self.mel_fn       = MelSpectrogram(cfg)
        self.content_enc  = ContentEncoder(cfg)
        self.attr_enc     = AttributeEncoder(cfg)
        self.attr_embedder= TargetAttributeEmbedder(cfg)
        self.decoder      = Decoder(cfg)
        self.attr_disc    = AttributeDiscriminator(cfg)
        self.patch_disc   = PatchDiscriminator(cfg)

    # ── Core ──────────────────────────────────────────────────────────────
    def forward(
        self,
        mel_src:        torch.Tensor,
        gender_src_ids: torch.Tensor,
        age_src_ids:    torch.Tensor,
        gender_tgt_ids: torch.Tensor,
        age_tgt_ids:    torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_content    = self.content_enc(mel_src)
        z_attr_src   = self.attr_enc(mel_src)
        z_attr_tgt   = self.attr_embedder(gender_tgt_ids, age_tgt_ids)

        mel_fake     = self.decoder(z_content, z_attr_tgt)

        # Cycle
        z_content_cyc  = self.content_enc(mel_fake)
        z_attr_src_emb = self.attr_embedder(gender_src_ids, age_src_ids)
        mel_recon      = self.decoder(z_content_cyc, z_attr_src_emb)

        return {
            "mel_fake":      mel_fake,
            "mel_recon":     mel_recon,
            "z_content_src": z_content,
            "z_content_cyc": z_content_cyc,
            "z_attr_src":    z_attr_src,
            "z_attr_tgt":    z_attr_tgt,
        }

    # ── Losses ────────────────────────────────────────────────────────────
    def generator_losses(
        self, mel_src: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {
            "recon":   F.l1_loss(outputs["mel_recon"], mel_src),
            "content": F.mse_loss(
                outputs["z_content_cyc"].detach(), outputs["z_content_src"]
            ),
            "adv":     F.mse_loss(
                self.patch_disc(outputs["mel_fake"]),
                torch.ones_like(self.patch_disc(outputs["mel_fake"])),
            ),
        }
        losses["total_G"] = (
            self.cfg.lambda_recon * losses["recon"]
            + self.cfg.lambda_cycle * losses["content"]
            + self.cfg.lambda_attr  * losses["adv"]
        )
        return losses

    def discriminator_losses(
        self, mel_src: torch.Tensor, mel_fake: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        real = self.patch_disc(mel_src.detach())
        fake = self.patch_disc(mel_fake.detach())
        d_real = F.mse_loss(real, torch.ones_like(real))
        d_fake = F.mse_loss(fake, torch.zeros_like(fake))
        return {"d_real": d_real, "d_fake": d_fake,
                "total_D": (d_real + d_fake) * 0.5}

    # ── Inference ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def transform(
        self,
        waveform:     np.ndarray,
        sample_rate:  int,
        source_attrs: Dict[str, str],
        target_attrs: Dict[str, str],
    ) -> Tuple[np.ndarray, Dict]:
        """
        High-level API.  numpy waveform in → transformed numpy waveform out.

        Args:
            waveform     : (T,) float32 mono PCM in [-1, 1]
            sample_rate  : must match cfg.sample_rate (16 000 Hz)
            source_attrs : e.g. {"gender": "male_masculine", "age": "old"}
            target_attrs : e.g. {"gender": "female_feminine", "age": "young"}

        Returns:
            (output_waveform, metrics_dict)
        """
        device = next(self.parameters()).device
        self.eval()

        wav        = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
        mel        = self.mel_fn(wav)
        z_attr_tgt = self.attr_embedder.encode_attrs(target_attrs, device)
        z_content  = self.content_enc(mel)
        mel_fake   = self.decoder(z_content, z_attr_tgt)

        min_len = min(mel.size(-1), mel_fake.size(-1))
        mel_trunc = mel[..., :min_len]
        mel_fake_trunc = mel_fake[..., :min_len]

        mel_mae = (mel_fake_trunc - mel_trunc).abs().mean().item()
        snr     = self._approx_snr(mel_trunc.cpu().numpy()[0],
                                   mel_fake_trunc.cpu().numpy()[0])
        metrics = {
            "mel_mae":       round(float(mel_mae), 6),
            "approx_snr_db": round(float(snr), 2),
            "source_attrs":  source_attrs,
            "target_attrs":  target_attrs,
        }
        output_wav = self._mel_to_wav(mel_fake.squeeze(0).cpu().numpy())
        return output_wav, metrics

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _approx_snr(original: np.ndarray, transformed: np.ndarray) -> float:
        sig   = np.mean(original ** 2)
        noise = np.mean((original - transformed) ** 2)
        if noise < 1e-10:
            return float("inf")
        return float(10 * np.log10(sig / noise))

    def _mel_to_wav(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """Griffin-Lim mel inversion (production: replace with HiFi-GAN)."""
        mel_lin  = np.exp(mel)
        cfg      = self.cfg
        n_frames = mel_lin.shape[1]
        freq_bins = cfg.n_fft // 2 + 1
        spec     = np.zeros((freq_bins, n_frames))
        for t in range(n_frames):
            spec[:, t] = np.interp(
                np.linspace(0, cfg.n_mels - 1, freq_bins),
                np.arange(cfg.n_mels), mel_lin[:, t],
            )
        phase  = np.random.uniform(0, 2 * np.pi, spec.shape)
        angles = np.exp(1j * phase)
        for _ in range(n_iter):
            stft   = spec * angles
            signal = np.fft.irfft(stft, axis=0).flatten()
            signal = signal[:n_frames * cfg.hop_length]
            re_stft = np.array([
                np.fft.rfft(
                    signal[i * cfg.hop_length: i * cfg.hop_length + cfg.n_fft],
                    n=cfg.n_fft,
                )
                for i in range(n_frames)
            ]).T
            angles = np.exp(1j * np.angle(re_stft))
        stft   = spec * angles
        signal = np.fft.irfft(stft, axis=0).flatten()
        return signal[:n_frames * cfg.hop_length].astype(np.float32)

    # ── Serialisation ─────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "model_state": self.state_dict(),
            "config":      self.cfg.__dict__,
        }, path)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PrivacyModule":
        ckpt = torch.load(path, map_location=device)

        # Filter saved config to only keys VoiceTransformConfig accepts.
        # This prevents crashes when loading an ASR (train_fair.py) checkpoint
        # by mistake, or when config fields have changed between versions.
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(VoiceTransformConfig)}
        saved_cfg  = {k: v for k, v in ckpt.get("config", {}).items()
                      if k in valid_keys}

        if not saved_cfg:
            print("  [WARNING] Checkpoint has no compatible PrivacyModule config. "
                  "Using default VoiceTransformConfig.")

        cfg   = VoiceTransformConfig(**saved_cfg)
        model = cls(cfg)

        # Load weights – skip if the checkpoint is an ASR model (keys won't match)
        state = ckpt.get("model_state", ckpt.get("model", {}))
        pm_keys = set(model.state_dict().keys())
        ckpt_keys = set(state.keys())
        overlap = pm_keys & ckpt_keys
        if overlap:
            missing = model.load_state_dict(state, strict=False)
            print(f"  Loaded {len(overlap)} / {len(pm_keys)} weight tensors "
                  f"from checkpoint.")
        else:
            print("  [WARNING] Checkpoint weights don't match PrivacyModule "
                  "(likely an ASR checkpoint). Using random weights.")

        return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = VoiceTransformConfig()
    model = PrivacyModule(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"PrivacyModule  |  parameters: {total:,}")
    print(f"Gender classes : {cfg.gender_classes}")
    print(f"Age classes    : {cfg.age_classes}")

    B, T  = 2, 16_000
    wav   = torch.randn(B, T)
    mel   = model.mel_fn(wav)
    print(f"Input mel      : {mel.shape}")

    # female_feminine (idx 0), old → young
    g_src = torch.tensor([0, 0])
    a_src = torch.tensor([3, 3])   # "old"
    g_tgt = torch.tensor([1, 1])   # male_masculine
    a_tgt = torch.tensor([1, 1])   # "young"

    out    = model(mel, g_src, a_src, g_tgt, a_tgt)
    g_loss = model.generator_losses(mel, out)
    d_loss = model.discriminator_losses(mel, out["mel_fake"])
    print(f"mel_fake shape : {out['mel_fake'].shape}")
    print(f"G total loss   : {g_loss['total_G'].item():.4f}")
    print(f"D total loss   : {d_loss['total_D'].item():.4f}")
    print("privacymodule.py – sanity check passed ✓")