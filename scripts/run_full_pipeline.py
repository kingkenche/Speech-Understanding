#!/usr/bin/env python3
"""Run the end-to-end assignment pipeline."""

from __future__ import annotations

import argparse
import json

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    result = run_pipeline(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
