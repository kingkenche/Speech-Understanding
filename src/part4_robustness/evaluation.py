"""Evaluation wrappers for spoofing and adversarial robustness tasks."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from src.part4_robustness.adversarial_attack import find_min_epsilon
from src.part4_robustness.antispoof_classifier import LFCCTDNN, evaluate_antispoof


def evaluate_spoofing(model: LFCCTDNN, real_files: Sequence[str], spoof_files: Sequence[str]) -> Dict[str, float]:
    paths = list(real_files) + list(spoof_files)
    labels = [0] * len(real_files) + [1] * len(spoof_files)
    result = evaluate_antispoof(model, paths, labels)
    return {"eer": result.eer, "threshold": result.threshold}


def evaluate_adversarial_flip(model, waveform: torch.Tensor, true_label: int) -> Dict[str, float | bool]:
    result = find_min_epsilon(model, waveform, true_label=true_label)
    return {"epsilon": result.epsilon, "snr_db": result.snr_db, "flipped": result.flipped}
