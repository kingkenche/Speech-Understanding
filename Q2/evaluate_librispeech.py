#!/usr/bin/env python3
"""
Simple evaluation script for LibriSpeech test-clean data.
Automatically generates trial pairs and evaluates models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval import LibriSpeechEvalDataset, load_config, load_model, extract_embeddings, compute_eer, compute_minDCF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_librispeech(config_path: str, checkpoint_path: str, eval_dir: str, output_path: str):
    """Evaluate a model on LibriSpeech test-clean data."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load config and model
    config = load_config(config_path)
    model = load_model(config, checkpoint_path, device)
    logger.info(f"Loaded model from: {checkpoint_path}")

    # Load LibriSpeech test-clean data
    dataset = LibriSpeechEvalDataset(
        eval_dir,
        n_mels=config['data']['n_mels'],
        sample_rate=config['data']['sample_rate'],
        num_frames=config['data']['num_frames']
    )
    logger.info(f"LibriSpeech test-clean: {len(dataset.utterances)} utterances from {len(dataset.spk_to_utts)} speakers")

    # Generate trial pairs
    logger.info("Generating trial pairs...")
    pairs, labels = dataset.generate_trials(num_pairs=10000)
    logger.info(f"Generated {len(pairs)} trial pairs")

    # Extract embeddings
    logger.info("Extracting speaker embeddings...")
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    embeddings, utt_ids = extract_embeddings(model, eval_loader, device)
    logger.info(f"Extracted {len(embeddings)} embeddings")

    # Create embedding lookup
    utt_id_to_emb = {utt_id: emb for utt_id, emb in zip(utt_ids, embeddings)}

    # Compute similarities
    logger.info("Computing similarity scores...")
    scores = []
    matched_labels = []

    for utt1, utt2 in pairs:
        if utt1 in utt_id_to_emb and utt2 in utt_id_to_emb:
            emb1 = utt_id_to_emb[utt1]
            emb2 = utt_id_to_emb[utt2]
            # Cosine similarity
            score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            scores.append(score)
            matched_labels.append(labels[len(matched_labels)])

    scores = np.array(scores)
    matched_labels = np.array(matched_labels)

    logger.info(f"Matched {len(scores)} trial pairs")

    # Compute metrics
    logger.info("Computing EER and minDCF...")
    eer, eer_threshold = compute_eer(scores, matched_labels)
    min_dcf = compute_minDCF(scores, matched_labels)

    # Compute statistics
    same_spk_scores = scores[matched_labels == 1]
    diff_spk_scores = scores[matched_labels == 0]

    results = {
        'model': Path(checkpoint_path).parent.name,
        'config': Path(config_path).name,
        'EER_%': eer * 100,
        'minDCF': min_dcf,
        'EER_threshold': float(eer_threshold),
        'num_pairs': len(scores),
        'num_speakers': len(dataset.spk_to_utts),
        'num_utterances': len(dataset.utterances),
        'same_speaker_scores': {
            'mean': float(np.mean(same_spk_scores)),
            'std': float(np.std(same_spk_scores)),
            'min': float(np.min(same_spk_scores)),
            'max': float(np.max(same_spk_scores))
        },
        'diff_speaker_scores': {
            'mean': float(np.mean(diff_spk_scores)),
            'std': float(np.std(diff_spk_scores)),
            'min': float(np.min(diff_spk_scores)),
            'max': float(np.max(diff_spk_scores))
        }
    }

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_file}")
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {results['model']}")
    logger.info(f"EER: {results['EER_%']:.2f}%")
    logger.info(f"minDCF: {results['minDCF']:.4f}")
    logger.info(f"Same-speaker pairs - Mean: {results['same_speaker_scores']['mean']:.4f}, Std: {results['same_speaker_scores']['std']:.4f}")
    logger.info(f"Diff-speaker pairs - Mean: {results['diff_speaker_scores']['mean']:.4f}, Std: {results['diff_speaker_scores']['std']:.4f}")
    logger.info(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on LibriSpeech test-clean')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--eval_dir', default='./LibriSpeech', help='LibriSpeech test-clean directory')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()

    evaluate_librispeech(args.config, args.checkpoint, args.eval_dir, args.output)


if __name__ == '__main__':
    main()
