#!/usr/bin/env python3
"""Train custom n-gram LM used for constrained ASR decoding."""

from __future__ import annotations

import argparse

import yaml

from src.part1_transcription.decoding import build_ngram_from_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    corpus_path = cfg["ngram_lm"]["syllabus_corpus"]
    output_path = cfg["ngram_lm"]["output_path"]
    order = int(cfg["ngram_lm"].get("order", 3))

    build_ngram_from_file(corpus_path, order=order, output_path=output_path)
    print(f"Saved n-gram model to {output_path}")


if __name__ == "__main__":
    main()
