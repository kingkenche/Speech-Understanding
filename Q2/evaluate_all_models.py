#!/usr/bin/env python3
"""
Batch evaluation script for all models on LibriSpeech test-clean data.
Evaluates: Baseline, AE-Disentangler, VAE-Disentangler
"""

import json
import logging
from pathlib import Path

from evaluate_librispeech import evaluate_librispeech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    eval_dir = './LibriSpeech'
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Define models to evaluate
    models = [
        {
            'name': 'VAE-Disentangler (Proposed)',
            'config': 'configs/vae_improved.yaml',
            'checkpoint': 'checkpoints/librispeech_disentangler_vae/best.pt',
            'output': 'results/librispeech_vae_eval.json'
        }
    ]

    # Try to add baseline if it exists
    if Path('checkpoints/librispeech_baseline/best.pt').exists():
        models.insert(0, {
            'name': 'Baseline (TDNN)',
            'config': 'configs/librispeech_baseline.yaml',
            'checkpoint': 'checkpoints/librispeech_baseline/best.pt',
            'output': 'results/librispeech_baseline_eval.json'
        })

    # Try to add AE-disentangler if it exists
    if Path('checkpoints/librispeech_disentangler/best.pt').exists():
        models.insert(1, {
            'name': 'AE-Disentangler',
            'config': 'configs/librispeech_disentangler.yaml',
            'checkpoint': 'checkpoints/librispeech_disentangler/best.pt',
            'output': 'results/librispeech_disentangler_eval.json'
        })

    all_results = []

    for model_info in models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating: {model_info['name']}")
        logger.info(f"{'='*70}")

        try:
            results = evaluate_librispeech(
                model_info['config'],
                model_info['checkpoint'],
                eval_dir,
                model_info['output']
            )
            all_results.append({
                'Model': model_info['name'],
                'EER (%)': f"{results['EER_%']:.2f}",
                'minDCF': f"{results['minDCF']:.4f}",
                'Pairs': results['num_pairs'],
                'Speakers': results['num_speakers']
            })

        except Exception as e:
            logger.error(f"Failed to evaluate {model_info['name']}: {e}")

    # Print comparison table
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info(f"{'='*80}\n")

        # Print header
        header = f"{'Model':<30} {'EER (%)':<12} {'minDCF':<12} {'Pairs':<10} {'Speakers':<10}"
        logger.info(header)
        logger.info("-" * 80)

        # Print rows
        for result in all_results:
            row = f"{result['Model']:<30} {result['EER (%)']:<12} {result['minDCF']:<12} {result['Pairs']:<10} {result['Speakers']:<10}"
            logger.info(row)

        logger.info(f"{'='*80}\n")

    # Save summary
    summary = {
        'eval_data': eval_dir,
        'results': all_results,
        'individual_results': [Path(m['output']).name for m in models]
    }

    summary_path = results_dir / 'librispeech_eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
