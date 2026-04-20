"""PyTorch datasets for frame-level LID and clip-level spoof detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class LIDExample:
    audio_path: str
    language_id: int


class FrameLIDDataset(Dataset):
    """Slice waveforms into fixed windows and assign language labels.

    Audio is loaded lazily per-item so __init__ is cheap even for large files.
    """

    def __init__(
        self,
        examples: Sequence[LIDExample],
        sample_rate: int = 16000,
        frame_ms: int = 20,
        window_frames: int = 40,
    ) -> None:
        import soundfile as _sf

        self.examples = list(examples)
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.window_len = self.frame_len * window_frames

        self.index: List[tuple[int, int]] = []
        for ex_i, ex in enumerate(self.examples):
            try:
                info = _sf.info(ex.audio_path)
                n = int(info.frames * sample_rate / info.samplerate)
            except Exception:
                # fallback: at least one window
                n = self.window_len + 1
            for start in range(0, max(1, n - self.window_len + 1), self.frame_len):
                self.index.append((ex_i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ex_i, start = self.index[idx]
        ex = self.examples[ex_i]
        wav, _ = librosa.load(ex.audio_path, sr=self.sample_rate, mono=True,
                              offset=start / self.sample_rate,
                              duration=self.window_len / self.sample_rate)
        if len(wav) < self.window_len:
            wav = np.pad(wav, (0, self.window_len - len(wav)))
        clip_tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        label_tensor = torch.tensor(ex.language_id, dtype=torch.long)
        return clip_tensor, label_tensor



class AudioClipDataset(Dataset):
    """Clip-level dataset for anti-spoof classifier."""

    def __init__(self, real_files: Sequence[str], spoof_files: Sequence[str], sample_rate: int = 16000):
        self.paths = list(real_files) + list(spoof_files)
        self.labels = [0] * len(real_files) + [1] * len(spoof_files)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return wav_t, label, Path(path).name
