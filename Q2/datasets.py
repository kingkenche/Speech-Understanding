"""LibriSpeech and VoxCeleb dataset loaders for speaker recognition."""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset for speaker recognition."""

    def __init__(self, root_dir: str, subset: str = "train-clean-100", n_mels: int = 80,
                 sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 160,
                 num_frames: int = 300):
        """
        Args:
            root_dir: Path to LibriSpeech root (containing LibriSpeech/ folder)
            subset: Subset name (train-clean-100, train-clean-360, etc.)
            n_mels, sample_rate, n_fft, hop_length, num_frames: Feature extraction params
        """
        self.root_dir = Path(root_dir) / "LibriSpeech" / subset
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames

        if not self.root_dir.exists():
            raise FileNotFoundError(f"LibriSpeech dataset not found at {self.root_dir}")

        # Build speaker -> utterances mapping
        # Structure: speaker_id/chapter_id/speaker-chapter-utterance.flac
        self.spk2utts = {}
        self.spk2chapters = {}  # Use chapter_id as "session" proxy
        self.spk_id_map = {}  # Map actual speaker IDs to 0-indexed IDs
        self.chapter_id_map = {}  # Map actual chapter IDs to session indices
        utterance_count = 0
        spk_idx = 0  # 0-indexed speaker ID for use in training
        unique_chapters = set()  # Collect all unique chapter IDs

        # First pass: collect all unique chapter IDs
        for spk_dir in sorted(self.root_dir.glob("*")):
            if not spk_dir.is_dir():
                continue
            for chapter_dir in sorted(spk_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                chapter_id = int(chapter_dir.name)
                unique_chapters.add(chapter_id)

        # Create mapping from chapter_id to session index (0 to num_unique_chapters-1)
        for session_idx, chapter_id in enumerate(sorted(unique_chapters)):
            self.chapter_id_map[chapter_id] = session_idx

        # Second pass: build utterance list
        spk_idx = 0
        for spk_dir in sorted(self.root_dir.glob("*")):
            if not spk_dir.is_dir():
                continue
            actual_spk_id = int(spk_dir.name)  # Actual speaker ID from directory name
            self.spk_id_map[actual_spk_id] = spk_idx  # Map to 0-indexed ID
            self.spk2utts[spk_idx] = []
            self.spk2chapters[spk_idx] = {}

            for chapter_dir in sorted(spk_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                chapter_id = int(chapter_dir.name)
                self.spk2chapters[spk_idx][chapter_id] = []

                # Look for both .flac and .wav files
                audio_files = sorted(list(chapter_dir.glob("*.flac")) + list(chapter_dir.glob("*.wav")))
                for audio_file in audio_files:
                    utt_path = str(audio_file.relative_to(self.root_dir))
                    self.spk2utts[spk_idx].append((utt_path, spk_idx, chapter_id))
                    self.spk2chapters[spk_idx][chapter_id].append(utt_path)
                    utterance_count += 1

            spk_idx += 1  # Increment 0-indexed speaker ID

        self.utterances = []
        for spk_id, utts in self.spk2utts.items():
            self.utterances.extend(utts)

        logger.info(f"LibriSpeech: Loaded {len(self.spk2utts)} speakers, {utterance_count} utterances from {subset}")

    def _load_and_process_audio(self, wav_path: str) -> torch.Tensor:
        """Load audio and convert to mel-spectrogram."""
        full_path = self.root_dir / wav_path
        waveform, sr = torchaudio.load(full_path)

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)

        # Mel-spectrogram using MelSpectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft
        )
        mel_spec = mel_transform(waveform)

        # Remove channel dimension if present [1, n_mels, time] -> [n_mels, time]
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)

        # Ensure 2D: [n_mels, time]
        if mel_spec.ndim != 2:
            mel_spec = mel_spec.view(self.n_mels, -1)

        # Pad or truncate to num_frames
        time_frames = mel_spec.shape[1]
        if time_frames < self.num_frames:
            pad_amount = self.num_frames - time_frames
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))
        else:
            mel_spec = mel_spec[:, :self.num_frames]

        return mel_spec

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict:
        utt_path, spk_id, chapter_id = self.utterances[idx]
        mel_spec = self._load_and_process_audio(utt_path)
        session_idx = self.chapter_id_map[chapter_id]  # Map chapter_id to session index
        return {
            'features': mel_spec,
            'speaker': spk_id,
            'session': session_idx,  # Use mapped session index (0 to num_unique_chapters-1)
            'utt_path': utt_path
        }


class VoxCelebDataset(Dataset):
    """VoxCeleb1 dataset for speaker recognition."""

    def __init__(self, root_dir: str, n_mels: int = 80,
                 sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 160,
                 num_frames: int = 300, augmentation: bool = False):
        self.root_dir = Path(root_dir)
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.augmentation = augmentation

        # Build speaker -> [utterances] mapping
        self.spk2utts = {}
        self.spk2vidid = {}  # speaker -> {video_id -> [utterances]}
        utterance_count = 0

        for spk_dir in sorted(self.root_dir.glob("id*")):
            spk_id = int(spk_dir.name[2:]) if spk_dir.name.startswith("id") else spk_dir.name
            self.spk2utts[spk_id] = []
            self.spk2vidid[spk_id] = {}

            for video_dir in sorted(spk_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                vid_id = video_dir.name
                self.spk2vidid[spk_id][vid_id] = []

                for wav_file in sorted(video_dir.glob("*.wav")):
                    utt_path = str(wav_file.relative_to(self.root_dir))
                    self.spk2utts[spk_id].append((utt_path, spk_id, vid_id))
                    self.spk2vidid[spk_id][vid_id].append(utt_path)
                    utterance_count += 1

        self.utterances = []
        for spk_id, utts in self.spk2utts.items():
            self.utterances.extend(utts)

        logger.info(f"VoxCeleb: Loaded {len(self.spk2utts)} speakers, {utterance_count} utterances")

    def _load_and_process_audio(self, wav_path: str) -> torch.Tensor:
        """Load audio and convert to mel-spectrogram."""
        full_path = self.root_dir / wav_path
        waveform, sr = torchaudio.load(full_path)

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)

        # Mel-spectrogram using MelSpectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft
        )
        mel_spec = mel_transform(waveform)

        # Remove channel dimension if present [1, n_mels, time] -> [n_mels, time]
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)

        # Ensure 2D: [n_mels, time]
        if mel_spec.ndim != 2:
            mel_spec = mel_spec.view(self.n_mels, -1)

        # Pad or truncate to num_frames
        time_frames = mel_spec.shape[1]
        if time_frames < self.num_frames:
            pad_amount = self.num_frames - time_frames
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))
        else:
            mel_spec = mel_spec[:, :self.num_frames]

        return mel_spec

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict:
        utt_path, spk_id, vid_id = self.utterances[idx]
        mel_spec = self._load_and_process_audio(utt_path)
        return {
            'features': mel_spec,
            'speaker': spk_id,
            'session': vid_id,
            'utt_path': utt_path
        }
