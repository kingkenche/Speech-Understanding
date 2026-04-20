"""LFCC-based anti-spoofing classifier and EER evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, TensorDataset


def compute_lfcc(audio: np.ndarray, sr: int = 16000, n_lfcc: int = 20) -> np.ndarray:
    # Practical LFCC surrogate using linear-frequency filterbank via STFT bins.
    spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=160)) ** 2
    log_spec = np.log(spec + 1e-7)
    lfcc = librosa.feature.mfcc(S=log_spec, n_mfcc=n_lfcc)
    return lfcc.astype(np.float32)


class LFCCTDNN(nn.Module):
    def __init__(self, n_lfcc: int = 20, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_lfcc, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(96, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.net(x).squeeze(-1)
        return self.head(h)


@dataclass
class AntispoofResult:
    eer: float
    threshold: float


def prepare_features(paths: Sequence[str], sr: int = 16000) -> torch.Tensor:
    feats = []
    max_t = 0
    for path in paths:
        y, _ = librosa.load(path, sr=sr, mono=True)
        lfcc = compute_lfcc(y, sr=sr)
        feats.append(lfcc)
        max_t = max(max_t, lfcc.shape[1])

    padded = []
    for lfcc in feats:
        if lfcc.shape[1] < max_t:
            pad = np.zeros((lfcc.shape[0], max_t - lfcc.shape[1]), dtype=np.float32)
            lfcc = np.concatenate([lfcc, pad], axis=1)
        padded.append(lfcc)
    return torch.tensor(np.stack(padded), dtype=torch.float32)


def train_antispoof_model(
    real_paths: Sequence[str],
    spoof_paths: Sequence[str],
    save_path: str,
    epochs: int = 10,
    lr: float = 1e-3,
) -> str:
    all_paths = list(real_paths) + list(spoof_paths)
    X = prepare_features(all_paths)
    y = torch.tensor([0] * len(real_paths) + [1] * len(spoof_paths), dtype=torch.long)

    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    model = LFCCTDNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path


def compute_eer_from_scores(labels: Sequence[int], scores: Sequence[float]) -> AntispoofResult:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fnr[idx] + fpr[idx]) / 2.0)
    return AntispoofResult(eer=eer, threshold=float(thresholds[idx]))


def evaluate_antispoof(model: LFCCTDNN, paths: Sequence[str], labels: Sequence[int]) -> AntispoofResult:
    feats = prepare_features(paths)
    model.eval()
    with torch.inference_mode():
        logits = model(feats)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
    return compute_eer_from_scores(labels, probs)
