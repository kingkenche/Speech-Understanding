"""Dynamic Time Warping utilities for prosody transfer."""

from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np


def dtw_align(reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return alignment indices between reference and target contours."""
    # librosa returns path in reverse order; flip to time-forward.
    _, wp = librosa.sequence.dtw(X=reference.reshape(1, -1), Y=target.reshape(1, -1), metric="euclidean")
    wp = np.array(wp[::-1])
    return wp[:, 0], wp[:, 1]


def warp_contour(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_idx, tgt_idx = dtw_align(reference, target)
    warped = np.zeros_like(target)
    for r, t in zip(ref_idx, tgt_idx):
        if t < warped.shape[0]:
            warped[t] = reference[r]
    # Fill unaligned values with nearest non-zero sample.
    nz = np.where(warped != 0)[0]
    if len(nz) == 0:
        return target
    for i in range(len(warped)):
        if warped[i] == 0:
            nearest = nz[np.argmin(np.abs(nz - i))]
            warped[i] = warped[nearest]
    return warped
