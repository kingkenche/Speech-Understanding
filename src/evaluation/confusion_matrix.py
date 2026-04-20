"""Confusion matrix utilities for language boundary decisions."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


def boundary_confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[np.ndarray, dict]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return cm, {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
