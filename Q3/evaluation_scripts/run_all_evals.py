"""
evaluation_scripts/run_all_evals.py
=====================================
Run all evaluation metrics (FAD + DNSMOS) in one command.

Usage:
    python evaluation_scripts/run_all_evals.py \\
        --reference_dir examples/reference/ \\
        --generated_dir examples/generated/ \\
        --output_dir    eval_results/

    python evaluation_scripts/run_all_evals.py --demo
"""

import argparse
import json
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_scripts.fad_eval   import evaluate_fad,      run_demo as fad_demo
from evaluation_scripts.dnsmos_eval import evaluate_directory, plot_scores

import numpy as np


def run_all(reference_dir: str, generated_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  AUDIO QUALITY EVALUATION SUITE")
    print("="*60)

    # ── FAD ──────────────────────────────────────────────────────────────
    print("\n[1/2] Fréchet Audio Distance …")
    fad_results = evaluate_fad(reference_dir, generated_dir)
    fad_path    = os.path.join(output_dir, "fad_results.json")
    with open(fad_path, "w") as f:
        json.dump(fad_results, f, indent=2)

    print(f"  FAD Score       : {fad_results['fad_score']}")
    print(f"  Interpretation  : {fad_results['interpretation']}")
    print(f"  Saved → {fad_path}")

    # ── DNSMOS ───────────────────────────────────────────────────────────
    print("\n[2/2] DNSMOS / Speech Quality …")
    dnsmos_results = evaluate_directory(generated_dir, reference_dir)
    dnsmos_path    = os.path.join(output_dir, "dnsmos_results.json")
    with open(dnsmos_path, "w") as f:
        json.dump(dnsmos_results, f, indent=2)

    plot_path = os.path.join(output_dir, "dnsmos_plot.png")
    plot_scores(dnsmos_results, plot_path)

    print(f"  OVR MOS         : {dnsmos_results['OVR_mean']:.3f}")
    print(f"  Saved → {dnsmos_path}")

    # ── Combined report ───────────────────────────────────────────────────
    combined = {
        "fad":    fad_results,
        "dnsmos": {k: v for k, v in dnsmos_results.items() if k != "per_file"},
        "pass_fail": {
            "fad_ok":     fad_results["fad_score"] < 30.0,
            "mos_ok":     dnsmos_results["OVR_mean"] >= 3.0,
            "no_toxicity_traps": (fad_results["fad_score"] < 30.0 and
                                   dnsmos_results["OVR_mean"] >= 3.0),
        }
    }
    combined_path = os.path.join(output_dir, "combined_eval.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("\n── Summary ─────────────────────────────────────────────────")
    pf = combined["pass_fail"]
    print(f"  FAD < 30  : {'✓ PASS' if pf['fad_ok']  else '✗ FAIL'}")
    print(f"  OVR ≥ 3.0 : {'✓ PASS' if pf['mos_ok']  else '✗ FAIL'}")
    print(f"  No Toxicity Traps : "
          f"{'✓ PASS' if pf['no_toxicity_traps'] else '✗ FAIL'}")
    print(f"\n  Full report → {combined_path}")
    print("="*60 + "\n")


def run_demo_all():
    """Demo mode – all synthetic."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_all(
            reference_dir=os.path.join(tmp, "ref"),
            generated_dir=os.path.join(tmp, "gen"),
            output_dir="eval_results_demo"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run all audio quality evaluations")
    parser.add_argument("--reference_dir", type=str, default="examples/reference")
    parser.add_argument("--generated_dir", type=str, default="examples/generated")
    parser.add_argument("--output_dir",    type=str, default="eval_results")
    parser.add_argument("--demo",          action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo_all()
    else:
        run_all(args.reference_dir, args.generated_dir, args.output_dir)


if __name__ == "__main__":
    main()
