"""Adversarial perturbation generation for LID model robustness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AttackResult:
    epsilon: float
    snr_db: float
    flipped: bool


def snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    signal_p = float(np.mean(clean**2) + eps)
    noise_p = float(np.mean((noisy - clean) ** 2) + eps)
    return 10.0 * np.log10(signal_p / noise_p)


def fgsm_attack(model, waveform: torch.Tensor, label: torch.Tensor, epsilon: float) -> torch.Tensor:
    waveform = waveform.clone().detach().requires_grad_(True)
    logits = model(waveform)
    loss = F.cross_entropy(logits, label)
    model.zero_grad()
    loss.backward()
    perturb = epsilon * waveform.grad.sign()
    return torch.clamp(waveform + perturb, -1.0, 1.0).detach()


def find_min_epsilon(
    model,
    waveform: torch.Tensor,
    true_label: int,
    start: float = 0.001,
    stop: float = 0.1,
    step: float = 0.001,
    min_snr_db: float = 40.0,
) -> AttackResult:
    model.eval()
    label = torch.tensor([true_label], dtype=torch.long, device=waveform.device)
    clean = waveform.detach().cpu().numpy().squeeze()

    for eps in np.arange(start, stop + step / 2, step):
        adv = fgsm_attack(model, waveform, label, float(eps))
        pred = int(torch.argmax(model(adv), dim=1).item())
        adv_np = adv.detach().cpu().numpy().squeeze()
        curr_snr = snr_db(clean, adv_np)
        if pred != true_label and curr_snr >= min_snr_db:
            return AttackResult(epsilon=float(eps), snr_db=float(curr_snr), flipped=True)

    return AttackResult(epsilon=float(stop), snr_db=float(curr_snr), flipped=False)
