"""Helpers to build and validate a 500+ term technical parallel dictionary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_dictionary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_terms(path: str) -> int:
    data = load_dictionary(path)
    return len(data.get("terms", {}))


def validate_min_entries(path: str, minimum: int = 500) -> bool:
    return count_terms(path) >= minimum


def merge_terms(base_path: str, new_terms: Dict[str, Dict], out_path: str | None = None) -> str:
    data = load_dictionary(base_path)
    terms = data.setdefault("terms", {})
    terms.update(new_terms)
    data.setdefault("metadata", {})["total_entries"] = len(terms)

    final = out_path or base_path
    Path(final).parent.mkdir(parents=True, exist_ok=True)
    with open(final, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return final
