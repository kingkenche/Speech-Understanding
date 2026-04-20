"""Custom Hinglish-to-IPA conversion for code-switched transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


EN_IPA_MAP: Dict[str, str] = {
    "a": "ə",
    "b": "b",
    "c": "k",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "h": "h",
    "i": "ɪ",
    "j": "dʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "k",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "ks",
    "y": "j",
    "z": "z",
}


HI_ROMAN_IPA_MAP: Dict[str, str] = {
    "aa": "aː",
    "ai": "ɛː",
    "au": "ɔː",
    "ch": "tʃ",
    "dh": "d̪ʱ",
    "gh": "gʱ",
    "kh": "kʰ",
    "ph": "pʰ",
    "th": "t̪ʰ",
    "bh": "bʱ",
    "jh": "dʒʱ",
    "sh": "ʃ",
    "ri": "rɪ",
}


@dataclass
class TokenMeta:
    token: str
    language: str


def _word_to_ipa(word: str, language: str) -> str:
    w = word.lower().strip()
    if not w:
        return ""

    if language == "hi":
        # Greedy digraph matching for romanized Hindi chunks.
        out: List[str] = []
        i = 0
        while i < len(w):
            pair = w[i : i + 2]
            if pair in HI_ROMAN_IPA_MAP:
                out.append(HI_ROMAN_IPA_MAP[pair])
                i += 2
                continue
            out.append(EN_IPA_MAP.get(w[i], w[i]))
            i += 1
        return "".join(out)

    return "".join(EN_IPA_MAP.get(ch, ch) for ch in w)


def hinglish_to_ipa(tokens: List[TokenMeta]) -> str:
    parts = [_word_to_ipa(tok.token, tok.language) for tok in tokens]
    return " ".join(p for p in parts if p)
