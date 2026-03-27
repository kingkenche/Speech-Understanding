"""
run_all.py
==========
Convenience runner for all Q1 sub-tasks.

Usage
-----
  python run_all.py --audio path/to/file.wav --out_dir results/

Each sub-task runs independently; results are saved to out_dir.
"""

import os
import sys
import argparse
import subprocess


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 60)
    print(" ".join(cmd))
    print("=" * 60)
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"[run_all] WARNING: command returned code {ret.returncode}")


def parse_args():
    p = argparse.ArgumentParser(description="Run all Q1 scripts")
    p.add_argument("--audio",      required=True, help="Path to .wav file")
    p.add_argument("--out_dir",    default="results", help="Output directory")
    p.add_argument("--window",     default="hamming",
                   choices=["hamming", "hanning", "rectangular"])
    p.add_argument("--threshold",  type=float, default=0.35)
    p.add_argument("--nfft",       type=int,   default=512)
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    py = sys.executable
    base = os.path.dirname(os.path.abspath(__file__))

    # Part 1 – MFCC
    run([py, os.path.join(base, "mfcc_manual.py"),
         args.audio,
         "--window",  args.window,
         "--nfft",    str(args.nfft),
         "--out_dir", args.out_dir])

    # Part 2 – Leakage & SNR
    run([py, os.path.join(base, "leakage_snr.py"),
         args.audio,
         "--nfft",    str(args.nfft),
         "--out_dir", args.out_dir])

    # Part 3 – Voiced/Unvoiced
    run([py, os.path.join(base, "voiced_unvoiced.py"),
         args.audio,
         "--threshold", str(args.threshold),
         "--nfft",      str(args.nfft),
         "--out_dir",   args.out_dir])

    # Part 4 – Phonetic mapping
    run([py, os.path.join(base, "phonetic_mapping.py"),
         args.audio,
         "--threshold", str(args.threshold),
         "--device",    args.device,
         "--out_dir",   args.out_dir])

    print(f"\n[run_all] All tasks complete. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
