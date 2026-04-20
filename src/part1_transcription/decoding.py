"""Constrained decoding with custom n-gram language model logit bias."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def tokenize_words(text: str) -> List[str]:
    return [w.strip().lower() for w in text.replace("\n", " ").split() if w.strip()]


class NGramLM:
    """Tiny n-gram LM used to bias decoding toward syllabus terms."""

    def __init__(self, order: int = 3):
        self.order = order
        self.counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)

    def fit(self, corpus_lines: Sequence[str]) -> None:
        for line in corpus_lines:
            words = ["<s>"] * (self.order - 1) + tokenize_words(line) + ["</s>"]
            for i in range(self.order - 1, len(words)):
                ctx = tuple(words[i - self.order + 1 : i])
                self.counts[ctx][words[i]] += 1

    def next_word_score(self, context: Sequence[str], candidate: str, alpha: float = 0.1) -> float:
        ctx = tuple((["<s>"] * (self.order - 1) + list(context))[-(self.order - 1) :])
        dist = self.counts.get(ctx)
        if not dist:
            return 0.0
        total = sum(dist.values()) + alpha * len(dist)
        return float((dist.get(candidate, 0) + alpha) / total)

    def save(self, path: str) -> None:
        payload = {
            "order": self.order,
            "counts": {
                "\t".join(ctx): dict(counter) for ctx, counter in self.counts.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "NGramLM":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls(order=int(data["order"]))
        for ctx, counter in data["counts"].items():
            model.counts[tuple(ctx.split("\t"))] = Counter(counter)
        return model


@dataclass
class BiasConfig:
    bias_scale: float = 4.0
    top_k: int = 64


class NGramLogitBiasProcessor(LogitsProcessor):
    """Additive bias on next-token logits based on n-gram probabilities."""

    def __init__(self, lm: NGramLM, processor: WhisperProcessor, cfg: BiasConfig | None = None):
        self.lm = lm
        self.processor = processor
        self.cfg = cfg or BiasConfig()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_vals, top_ids = torch.topk(scores, k=min(self.cfg.top_k, scores.shape[-1]), dim=-1)
        del top_vals
        for b in range(scores.shape[0]):
            prefix = self.processor.tokenizer.decode(input_ids[b], skip_special_tokens=True).lower().split()
            for tok_id in top_ids[b].tolist():
                candidate = self.processor.tokenizer.decode([tok_id], skip_special_tokens=True).strip().lower()
                if not candidate:
                    continue
                prob = self.lm.next_word_score(prefix[-(self.lm.order - 1) :], candidate)
                if prob > 0:
                    scores[b, tok_id] += self.cfg.bias_scale * prob
        return scores


def build_ngram_from_file(text_file: str, order: int = 3, output_path: str | None = None) -> NGramLM:
    with open(text_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    lm = NGramLM(order=order)
    lm.fit(lines)
    if output_path:
        lm.save(output_path)
    return lm


def constrained_whisper_transcribe(
    audio_features: torch.Tensor,
    model_name: str,
    lm: NGramLM,
    language: str = "hi",
    task: str = "transcribe",
    max_new_tokens: int = 256,
) -> str:
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    logits_processor = LogitsProcessorList([NGramLogitBiasProcessor(lm, processor)])

    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    generated = model.generate(
        audio_features,
        forced_decoder_ids=forced_ids,
        logits_processor=logits_processor,
        max_new_tokens=max_new_tokens,
    )
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
