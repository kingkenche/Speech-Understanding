"""Frame-level language identification model and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class FrameLIDNet(nn.Module):
    """Compact CNN over log-mel spectrograms for frame-level EN/HI tagging."""

    def __init__(self, n_mels: int = 80, n_classes: int = 2):
        super().__init__()
        self.melspec = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels, n_fft=400, hop_length=160),
            torchaudio.transforms.AmplitudeToDB(),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(96, n_classes)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: [B, 1, T]
        x = waveforms.squeeze(1)
        x = self.melspec(x)
        x = (x - x.mean(dim=(-1, -2), keepdim=True)) / (x.std(dim=(-1, -2), keepdim=True) + 1e-5)
        x = x.unsqueeze(1)
        x = self.encoder(x).flatten(1)
        return self.head(x)


@dataclass
class LIDTrainConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: str):
    train = optimizer is not None
    model.train(train)
    losses = []
    preds_all = []
    labels_all = []

    for wav, labels in loader:
        wav = wav.to(device)
        labels = labels.to(device)
        logits = model(wav)
        loss = F.cross_entropy(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds_all.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    f1 = f1_score(labels_all, preds_all, average="macro") if labels_all else 0.0
    return float(sum(losses) / max(1, len(losses))), float(f1)


def train_lid_model(
    model: FrameLIDNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: LIDTrainConfig,
    save_path: str,
) -> Dict[str, float]:
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_f1 = -1.0
    best = {"train_f1": 0.0, "val_f1": 0.0, "val_loss": 0.0}

    for _ in range(cfg.num_epochs):
        _, train_f1 = _run_epoch(model, train_loader, optimizer, cfg.device)
        val_loss, val_f1 = _run_epoch(model, val_loader, None, cfg.device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = {"train_f1": train_f1, "val_f1": val_f1, "val_loss": val_loss}
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
    return best


def load_lid_model(weights_path: str, device: str | None = None) -> FrameLIDNet:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameLIDNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model



