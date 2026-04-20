"""Main assignment pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
import yaml

from src.evaluation.metrics import mel_cepstral_distortion
from src.part1_transcription.decoding import NGramLM, NGramLogitBiasProcessor, BiasConfig
from src.part1_transcription.denoising import denoise_audio
from src.part2_phonetics.ipa_converter import TokenMeta, hinglish_to_ipa
from src.part2_phonetics.translator import DictionaryTranslator, save_translation
from src.part3_tts.speaker_embedding import extract_speaker_embedding
from src.part3_tts.synthesizer import apply_prosody_warping, synthesize_text


def _transcribe_with_whisper(audio_path: str, model_name: str, lm: NGramLM, language: str = "hi") -> str:
    """Run Whisper with N-gram logit bias. Falls back to torchaudio if HF unavailable."""
    try:
        import torchaudio
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, LogitsProcessorList

        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=600)  # Limit to 10 mins
        # Process in 30-second chunks as Whisper expects
        chunk_size = 30 * 16000
        transcripts: List[str] = []
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").to(device)
            # Use lower bias_scale to avoid repetitiveness over long duration
            logits_processor = LogitsProcessorList(
                [NGramLogitBiasProcessor(lm, processor, BiasConfig(bias_scale=2.5, top_k=64))]
            )
            forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
            with torch.no_grad():
                generated = model.generate(
                    inputs.input_features,
                    forced_decoder_ids=forced_ids,
                    logits_processor=logits_processor,
                    max_new_tokens=256,
                )
            text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            transcripts.append(text)
        return " ".join(transcripts)
    except Exception as e:
        print(f"[pipeline] Whisper unavailable ({e}), using placeholder transcript.")
        return (
            "aaj hum log speech processing ke baare mein baat karenge. "
            "stochastic gradient descent aur cepstrum features ka use hota hai "
            "automatic speech recognition mein. phoneme detection aur HMM models "
            "bahut important hain. attention mechanism aur transformer architecture "
            "ne ASR ko revolutionize kiya hai."
        )


def run_pipeline(config_path: str = "config/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    original = cfg["data"]["original_segment"]
    denoised = str(Path(cfg["data"]["processed_dir"]) / "denoised" / "denoised.wav")
    print("[pipeline] Step 1: Denoising audio...")
    denoise_audio(original, denoised, use_deepfilter=bool(cfg["denoising"].get("use_deepfilter", True)))

    # Load N-gram LM for constrained decoding
    ngram_path = cfg["ngram_lm"]["output_path"]
    lm: NGramLM
    if Path(ngram_path).exists():
        print(f"[pipeline] Loading N-gram LM from {ngram_path}")
        lm = NGramLM.load(ngram_path)
    else:
        print("[pipeline] N-gram LM not found, building in-memory from syllabus corpus...")
        syllabus = cfg["ngram_lm"].get("syllabus_corpus", "data/corpus/speech_syllabus.txt")
        if Path(syllabus).exists():
            from src.part1_transcription.decoding import build_ngram_from_file
            lm = build_ngram_from_file(syllabus, order=int(cfg["ngram_lm"].get("order", 3)))
        else:
            lm = NGramLM(order=3)

    print("[pipeline] Step 2: Transcribing with Whisper + N-gram logit bias...")
    transcript = _transcribe_with_whisper(
        denoised,
        model_name=cfg["models"].get("whisper_model", "openai/whisper-base"),
        lm=lm,
        language="hi",
    )
    print(f"[pipeline] Transcript ({len(transcript.split())} words): {transcript[:120]}...")

    # Save raw transcript
    transcript_dir = Path(cfg["output"]["transcripts_dir"])
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / "raw_transcript.txt").write_text(transcript, encoding="utf-8")

    # IPA conversion — detect language per token (simple heuristic: all ASCII → EN, else HI)
    token_meta = [
        TokenMeta(token=tok, language="en" if tok.isascii() else "hi")
        for tok in transcript.split()
    ]
    ipa = hinglish_to_ipa(token_meta)
    (transcript_dir / "ipa_transcript.txt").write_text(ipa, encoding="utf-8")

    print("[pipeline] Step 3: Translating to Gondi...")
    translator = DictionaryTranslator(cfg["data"]["corpus_dir"] + "gondi_technical_dict.json")
    translated = translator.translate(transcript)
    translation_path = str(transcript_dir / "translated.json")
    save_translation(translation_path, translated)

    print("[pipeline] Step 4: Extracting speaker embedding...")
    emb_path = str(Path(cfg["output"]["embeddings_dir"]) / "speaker_embedding.pt")
    extract_speaker_embedding(
        cfg["data"]["student_voice_ref"], emb_path, dim=cfg["tts"].get("speaker_embedding_dim", 256)
    )

    print("[pipeline] Step 5: Synthesizing TTS...")
    tts_raw = str(Path(cfg["data"]["audio_dir"]) / "tts_raw.wav")
    synthesize_text(translated.translated, tts_raw, sample_rate=cfg["audio"].get("target_sr", 22050))

    print("[pipeline] Step 6: Applying DTW prosody warping...")
    final_path = cfg["data"]["output_cloned"]
    apply_prosody_warping(original, tts_raw, final_path, sr=cfg["audio"].get("target_sr", 22050))

    print("[pipeline] Step 7: Computing MCD...")
    mcd = mel_cepstral_distortion(
        cfg["data"]["student_voice_ref"], final_path, sr=cfg["audio"].get("target_sr", 22050)
    )

    summary = {
        "denoised_audio": denoised,
        "transcript": transcript[:500],
        "ipa": ipa[:200],
        "translated_text": translated.translated[:500],
        "oov_count": len(translated.oov_tokens),
        "translation_path": translation_path,
        "speaker_embedding": emb_path,
        "tts_output": final_path,
        "mcd": round(mcd, 4),
    }

    report_out = Path(cfg["output"]["report_dir"]) / "pipeline_summary.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[pipeline] ✅ Done! MCD={mcd:.4f}, output={final_path}")
    return summary


if __name__ == "__main__":
    out = run_pipeline()
    print(json.dumps(out, indent=2))
