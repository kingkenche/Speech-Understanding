#!/usr/bin/env python3
"""Train LFCC anti-spoofing classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.part4_robustness.antispoof_classifier import train_antispoof_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--manifest", required=True, help="JSON with keys: real_files, spoof_files")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    save_path = str(Path(cfg["models"]["custom_dir"]) / "lfcc_antispoof_cm.pt")
    train_antispoof_model(
        real_paths=manifest["real_files"],
        spoof_paths=manifest["spoof_files"],
        save_path=save_path,
        epochs=int(cfg["training"].get("num_epochs", 10)),
    )
    print(f"Saved anti-spoofing model to {save_path}")


if __name__ == "__main__":
    main()
