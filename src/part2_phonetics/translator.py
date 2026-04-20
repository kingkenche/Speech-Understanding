"""Semantic translation from code-switched text to target LRL using curated dictionary."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


def normalize_token(token: str) -> str:
    # Preserve Hindi characters and alphanumeric, strip punctuation
    return re.sub(r"[^\w\u0900-\u097F]+", "", token).lower()


@dataclass
class TranslationResult:
    source: str
    translated: str
    oov_tokens: List[str]


class DictionaryTranslator:
    def __init__(self, dictionary_path: str):
        with open(dictionary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        terms = payload.get("terms", {})
        self.term_map: Dict[str, str] = {}
        
        # Build map from both English keys and Hindi values to Gondi ITRANS
        for k, v in terms.items():
            gondi = v.get("gondi_itrans", k.lstrip("*"))
            # Map English key
            eng_key = normalize_token(k.lstrip("*"))
            self.term_map[eng_key] = gondi
            # Map Hindi value if present
            hindi_val = normalize_token(v.get("hindi", ""))
            if hindi_val:
                self.term_map[hindi_val] = gondi
        
        # Heuristic common word mapping for Hinglish -> Gondi connectives
        self.common_map = {
            "hum": "mattu",
            "log": "mandas",
            "aur": "ni",
            "baare": "pache",
            "mein": "ne",
            "baat": "wari",
            "karenge": "keukika",
            "aaj": "nendu",
            "hai": "and",
            "ke": "na",
            "hoga": "andur",
            "karta": "tungar",
            "ka": "na",
            "se": "tal",
            "par": "paj"
        }

    def translate(self, text: str) -> TranslationResult:
        out = []
        oov = []
        for raw in text.split():
            norm = normalize_token(raw)
            if not norm:
                continue
            
            # 1. Check technical dictionary (English or Hindi matches)
            if norm in self.term_map:
                out.append(self.term_map[norm])
            # 2. Check common word heuristic map
            elif norm in self.common_map:
                out.append(self.common_map[norm])
            # 3. Fallback
            else:
                oov.append(norm)
                out.append(norm)
        return TranslationResult(source=text, translated=" ".join(out), oov_tokens=oov)


def save_translation(path: str, result: TranslationResult) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": result.source,
                "translated": result.translated,
                "oov_tokens": result.oov_tokens,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path
