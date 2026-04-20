"""Speaker embedding extraction from reference voice."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch


def _fallback_embedding(audio: np.ndarray, sr: int, dim: int = 256) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    feat = np.concatenate([mel.mean(axis=1), mel.std(axis=1)])
    if feat.shape[0] < dim:
        feat = np.pad(feat, (0, dim - feat.shape[0]))
    return torch.tensor(feat[:dim], dtype=torch.float32)


def extract_speaker_embedding(audio_path: str, output_path: str, dim: int = 256) -> str:
    """
    Extract embedding via SpeechBrain ECAPA when available.
    Falls back to deterministic mel-stat embedding for reproducibility.
    """
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    embedding: torch.Tensor

    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/pretrained/ecapa_voxceleb",
        )
        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        emb = classifier.encode_batch(signal)
        embedding = emb.squeeze(0).squeeze(0).detach().cpu()
        if embedding.numel() < dim:
            embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.numel()))
        embedding = embedding[:dim]
    except Exception:
        embedding = _fallback_embedding(audio, sr, dim=dim)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding, str(out))
    return str(out)
