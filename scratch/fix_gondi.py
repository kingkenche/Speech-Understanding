import json
import yaml
from pathlib import Path
from src.part2_phonetics.translator import DictionaryTranslator, save_translation
from src.part3_tts.synthesizer import synthesize_text, apply_prosody_warping
from src.evaluation.metrics import mel_cepstral_distortion

def fix_gondi_audio():
    config_path = "config/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Load raw transcript
    transcript_dir = Path(cfg["output"]["transcripts_dir"])
    raw_transcript_path = transcript_dir / "raw_transcript.txt"
    if not raw_transcript_path.exists():
        print("Raw transcript not found. Run pipeline first.")
        return
    
    transcript = raw_transcript_path.read_text(encoding="utf-8")
    print(f"Original Transcript: {transcript[:100]}...")

    # 2. Re-translate with fixed DictionaryTranslator
    print("Translating to Gondi...")
    translator = DictionaryTranslator(cfg["data"]["corpus_dir"] + "gondi_technical_dict.json")
    translated = translator.translate(transcript)
    print(f"Translated (Gondi): {translated.translated[:150]}...")
    
    save_translation(str(transcript_dir / "translated.json"), translated)

    # 3. Re-synthesize
    print("Synthesizing Gondi TTS...")
    tts_raw = str(Path(cfg["data"]["audio_dir"]) / "tts_raw.wav")
    synthesize_text(translated.translated, tts_raw, sample_rate=cfg["audio"].get("target_sr", 22050))

    # 4. Apply Prosody (Warping)
    print("Applying Prosody Warping (Modi voice)...")
    original = cfg["data"]["original_segment"]
    final_path = cfg["data"]["output_cloned"]
    apply_prosody_warping(original, tts_raw, final_path, sr=cfg["audio"].get("target_sr", 22050))

    # 5. Verify MCD
    mcd = mel_cepstral_distortion(cfg["data"]["student_voice_ref"], final_path, sr=cfg["audio"].get("target_sr", 22050))
    print(f"DONE. Final MCD: {mcd:.4f}")
    print(f"Output saved to: {final_path}")

if __name__ == "__main__":
    fix_gondi_audio()
