"""Build anti-spoofing test manifest by slicing WAV files using stdlib only."""
import os
import json
import wave
import struct

def slice_wav(input_path, output_dir, prefix, duration_sec=2, max_clips=25):
    """Slice a WAV file into N-second clips, return list of paths."""
    clips = []
    with wave.open(input_path, 'rb') as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        frames_per_clip = duration_sec * sr

        for i in range(max_clips):
            start = i * frames_per_clip
            if start + frames_per_clip > total_frames:
                break
            wf.setpos(start)
            frames = wf.readframes(frames_per_clip)
            out_path = os.path.join(output_dir, f"{prefix}_{i:02d}.wav")
            with wave.open(out_path, 'wb') as ow:
                ow.setnchannels(channels)
                ow.setsampwidth(sampwidth)
                ow.setframerate(sr)
                ow.writeframes(frames)
            clips.append(out_path)
            print(f"  Written: {out_path}")
    return clips

os.makedirs('data/audio/mock/real', exist_ok=True)
os.makedirs('data/audio/mock/spoof', exist_ok=True)

print("Slicing student_voice_ref.wav into real clips...")
real_files = slice_wav(
    'data/audio/student_voice_ref.wav',
    'data/audio/mock/real',
    'real'
)

print("Slicing output_LRL_cloned.wav into spoof clips...")
spoof_files = slice_wav(
    'data/audio/output_LRL_cloned.wav',
    'data/audio/mock/spoof',
    'spoof'
)

manifest = {"real_files": real_files, "spoof_files": spoof_files}
with open("data/manifests/antispoof_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written: {len(real_files)} real clips, {len(spoof_files)} spoof clips.")
