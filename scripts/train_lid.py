#!/usr/bin/env python3
"""Train frame-level EN/HI LID classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, random_split

from src.data.dataset import FrameLIDDataset, LIDExample
from src.part1_transcription.lid_system import FrameLIDNet, LIDTrainConfig, train_lid_model


def build_examples_from_manifest(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [LIDExample(audio_path=r["audio_path"], language_id=int(r["language_id"])) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--manifest", required=True, help="JSON list of {audio_path, language_id}")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    examples = build_examples_from_manifest(args.manifest)
    ds = FrameLIDDataset(examples, sample_rate=cfg["audio"].get("sample_rate", 16000))
    n_val = max(1, int(0.2 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=False)

    model = FrameLIDNet()
    result = train_lid_model(
        model,
        train_loader,
        val_loader,
        LIDTrainConfig(
            learning_rate=float(cfg["training"].get("learning_rate", 1e-3)),
            num_epochs=int(cfg["training"].get("num_epochs", 10)),
            device=cfg["training"].get("device", "cpu"),
        ),
        save_path=str(Path(cfg["models"]["custom_dir"]) / "lid_classifier.pt"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
