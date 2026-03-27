"""Training script for disentangled speaker recognition."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.functional as AF
import yaml
from torch.utils.data import DataLoader, Dataset

from losses import AngularProtoLoss, ReconstructionLoss, CorrelationLoss, KLDivergenceLoss, GradientReversalLoss
from models import (TDNNEncoder, BaselineFramework, DisentanglementFramework,
                    VAEDisentanglementFramework)
from datasets import LibriSpeechDataset, VoxCelebDataset


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: Dict, num_speakers: int) -> nn.Module:
    """Create model based on config."""
    model_type = config['model']['type']
    encoder_cfg = config['model']['encoder']
    embedding_dim = encoder_cfg['embedding_dim']

    encoder = TDNNEncoder(embedding_dim, encoder_cfg['feat_dim'])

    if model_type == 'baseline':
        return BaselineFramework(encoder, embedding_dim, num_speakers)

    elif model_type == 'disentangler_ae':
        latent_dim = config['model']['disentangler']['latent_dim']
        num_envs = config['model']['discriminators']['num_envs']
        return DisentanglementFramework(encoder, embedding_dim, latent_dim, num_speakers, num_envs)

    elif model_type == 'disentangler_vae':
        latent_dim = config['model']['disentangler']['latent_dim']
        hidden_dim = config['model']['disentangler']['hidden_dim']
        num_envs = config['model']['discriminators']['num_envs']
        return VAEDisentanglementFramework(encoder, embedding_dim, latent_dim, num_speakers, num_envs, hidden_dim)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch_baseline(model, train_loader, criterion, optimizer, device, config):
    """Train one epoch for baseline model."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        features = batch['features'].to(device)
        speaker_labels = batch['speaker'].to(device)

        optimizer.zero_grad()

        outputs = model(features)
        # AngularProtoLoss expects embeddings, not logits
        embeddings = outputs.get('embedding', outputs.get('embeddings', outputs.get('speaker_embeddings')))
        loss = criterion(embeddings, speaker_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_epoch_disentangler(model, train_loader, losses_dict, optimizer, device, config, phase='full', grl_lambda=0.5):
    """Train one epoch for disentanglement model."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    L_spk_weight = config['training']['losses']['L_spk']
    L_recons_weight = config['training']['losses']['L_recons']
    L_env_env_weight = config['training']['losses']['L_env_env']
    L_env_spk_weight = config['training']['losses']['L_env_spk']
    L_corr_weight = config['training']['losses']['L_corr']

    for batch in train_loader:
        features = batch['features'].to(device)
        speaker_labels = batch['speaker'].to(device)
        env_labels = batch['session'].to(device)  # Use session ID as environment proxy

        optimizer.zero_grad()

        outputs = model(features)

        # Speaker loss - use embeddings, not logits
        embeddings = outputs.get('embedding', outputs.get('embeddings', outputs.get('speaker_embeddings')))
        loss_spk = losses_dict['spk'](embeddings, speaker_labels)

        # Reconstruction loss
        loss_recons = losses_dict['recons'](outputs['reconstructed'], outputs['embedding'])

        # Environment losses
        loss_env_env = torch.nn.functional.cross_entropy(outputs['env_logits_e'], env_labels)

        # Environment loss on speaker code (with GRL)
        env_logits_reversed = losses_dict['grl'].apply_grl(outputs['env_logits_s']) * grl_lambda
        loss_env_spk = torch.nn.functional.cross_entropy(env_logits_reversed, env_labels)

        # Correlation loss
        loss_corr = losses_dict['corr'](outputs['speaker_code'], outputs['environment_code'])

        # Combined loss
        total_batch_loss = (
            L_spk_weight * loss_spk +
            L_recons_weight * loss_recons +
            L_env_env_weight * loss_env_env +
            L_env_spk_weight * loss_env_spk +
            L_corr_weight * loss_corr
        )

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += total_batch_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_epoch_vae(model, train_loader, losses_dict, optimizer, device, config, grl_lambda=0.5):
    """Train one epoch for VAE disentanglement model."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    L_spk_weight = config['training']['losses']['L_spk']
    L_recons_weight = config['training']['losses']['L_recons']
    L_env_env_weight = config['training']['losses']['L_env_env']
    L_env_spk_weight = config['training']['losses']['L_env_spk']
    L_corr_weight = config['training']['losses']['L_corr']
    L_kl_speaker = config['training']['losses'].get('L_kl_speaker', 0.1)
    L_kl_environment = config['training']['losses'].get('L_kl_environment', 0.1)

    for batch in train_loader:
        features = batch['features'].to(device)
        speaker_labels = batch['speaker'].to(device)
        env_labels = batch['session'].to(device)

        optimizer.zero_grad()

        outputs = model(features)

        # Speaker loss - use embeddings, not logits
        embeddings = outputs.get('embedding', outputs.get('embeddings', outputs.get('speaker_embeddings')))
        loss_spk = losses_dict['spk'](embeddings, speaker_labels)

        # Reconstruction loss
        loss_recons = losses_dict['recons'](outputs['reconstructed'], outputs['embedding'])

        # Environment losses
        loss_env_env = torch.nn.functional.cross_entropy(outputs['env_logits_e'], env_labels)
        env_logits_reversed = losses_dict['grl'].apply_grl(outputs['env_logits_s']) * grl_lambda
        loss_env_spk = torch.nn.functional.cross_entropy(env_logits_reversed, env_labels)

        # Correlation loss
        loss_corr = losses_dict['corr'](outputs['speaker_code'], outputs['environment_code'])

        # KL losses for VAE
        kl_info = outputs['kl_info']
        loss_kl_spk = losses_dict['kl'](kl_info['spk_mean'], kl_info['spk_logvar'])
        loss_kl_env = losses_dict['kl'](kl_info['env_mean'], kl_info['env_logvar'])

        # Combined loss
        total_batch_loss = (
            L_spk_weight * loss_spk +
            L_recons_weight * loss_recons +
            L_env_env_weight * loss_env_env +
            L_env_spk_weight * loss_env_spk +
            L_corr_weight * loss_corr +
            L_kl_speaker * loss_kl_spk +
            L_kl_environment * loss_kl_env
        )

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += total_batch_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--encoder_ckpt', default=None, help='Path to pretrained encoder checkpoint')
    args = parser.parse_args()

    config = load_config(args.config)

    # ===== FIX #3: Initialize random seed for reproducibility =====
    random_seed = config.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {random_seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create checkpoint directory
    ckpt_dir = Path(config['training']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (VoxCeleb or LibriSpeech)
    dataset_type = config['data'].get('dataset', 'voxceleb').lower()

    try:
        if dataset_type == 'librispeech':
            dataset = LibriSpeechDataset(
                config['data']['train_root'],
                subset=config['data'].get('subset', 'train-clean-100'),
                n_mels=config['data']['n_mels'],
                sample_rate=config['data']['sample_rate'],
                num_frames=config['data']['num_frames']
            )
            logger.info(f"Using LibriSpeech {config['data'].get('subset', 'train-clean-100')}")
        else:  # VoxCeleb
            dataset = VoxCelebDataset(
                config['data']['train_root'],
                n_mels=config['data']['n_mels'],
                sample_rate=config['data']['sample_rate'],
                num_frames=config['data']['num_frames']
            )
            logger.info("Using VoxCeleb")

        num_speakers = len(dataset.spk2utts)
        logger.info(f"Loaded {num_speakers} speakers from {dataset_type}")

        # Check if no speakers found (dataset exists but is empty)
        if num_speakers == 0:
            raise FileNotFoundError(f"No speakers found in {dataset_type} dataset")

    except FileNotFoundError as e:
        logger.warning(f"Dataset not found: {e}")
        logger.info("Using synthetic data for demonstration")

        # Create a simple synthetic dataset that returns proper dict format
        class SyntheticDataset(Dataset):
            def __init__(self, n_samples, n_mels, num_frames, num_speakers):
                self.n_samples = n_samples
                self.n_mels = n_mels
                self.num_frames = num_frames
                self.num_speakers = num_speakers

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return {
                    'features': torch.randn(self.n_mels, self.num_frames),
                    'speaker': torch.randint(0, self.num_speakers, (1,)).item(),
                    'session': torch.randint(0, 100, (1,)).item(),
                }

        dataset = SyntheticDataset(
            n_samples=5000,
            n_mels=config['data']['n_mels'],
            num_frames=config['data']['num_frames'],
            num_speakers=1000
        )
        num_speakers = 1000
        logger.info("Using synthetic data")

    # Create data loader
    batch_size = config['data']['batch_size']
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    model = create_model(config, num_speakers).to(device)

    # ===== FIX #2: Load pretrained encoder if provided =====
    if args.encoder_ckpt:
        logger.info(f"Loading pretrained encoder from {args.encoder_ckpt}")
        try:
            ckpt = torch.load(args.encoder_ckpt, map_location=device)

            # Extract encoder state dict from checkpoint
            encoder_state_dict = {}
            if 'model_state_dict' in ckpt:
                checkpoint_state = ckpt['model_state_dict']
            else:
                checkpoint_state = ckpt

            # Handle different checkpoint formats
            for key, value in checkpoint_state.items():
                # Skip non-encoder keys (discriminators, disentanglers, etc.)
                if key.startswith('embedding_extractor.'):
                    # Remove prefix for direct loading
                    encoder_key = key.replace('embedding_extractor.', '')
                    encoder_state_dict[encoder_key] = value

            if encoder_state_dict:
                # Load encoder weights
                model.embedding_extractor.load_state_dict(encoder_state_dict, strict=True)
                logger.info(f"Successfully loaded {len(encoder_state_dict)} encoder parameters from checkpoint")
            else:
                logger.warning("No encoder weights found in checkpoint. Training from scratch.")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Training model from scratch")
    else:
        logger.info("Training model from scratch (no pretrained encoder)")

    # Create losses
    losses_dict = {
        'spk': AngularProtoLoss(num_speakers, config['model']['encoder']['embedding_dim']),
        'recons': ReconstructionLoss(),
        'corr': CorrelationLoss(),
        'grl': GradientReversalLoss(config['training'].get('grl_lambda', 0.5)),
    }

    if 'vae' in config['model']['type']:
        losses_dict['kl'] = KLDivergenceLoss()

    # Move losses to device
    for loss_name, loss_fn in losses_dict.items():
        if hasattr(loss_fn, 'to'):
            losses_dict[loss_name] = loss_fn.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['optimizer']['lr'],
                            weight_decay=config['training']['optimizer']['weight_decay'])

    # Training loop
    pretrain_epochs = config['training'].get('pretrain_epochs', 0)
    max_epochs = config['training']['max_epochs']
    total_epochs = pretrain_epochs + max_epochs
    model_type = config['model']['type']

    for epoch in range(total_epochs):
        is_pretrain_phase = epoch < pretrain_epochs
        
        if model_type == 'baseline':
            loss = train_epoch_baseline(model, train_loader, losses_dict['spk'], optimizer, device, config)
        elif model_type == 'disentangler_ae':
            if is_pretrain_phase:
                # Phase 1: Speaker loss only
                loss = train_epoch_baseline(model, train_loader, losses_dict['spk'], optimizer, device, config)
            else:
                # Phase 2: Full disentanglement training
                grl_lambda = config['training'].get('grl_lambda', 1.0)
                loss = train_epoch_disentangler(model, train_loader, losses_dict, optimizer, device, config, grl_lambda)
        elif model_type == 'disentangler_vae':
            if is_pretrain_phase:
                # Phase 1: Speaker loss only
                loss = train_epoch_baseline(model, train_loader, losses_dict['spk'], optimizer, device, config)
            else:
                # Phase 2: Full VAE disentanglement training
                grl_lambda = config['training'].get('grl_lambda', 1.0)
                loss = train_epoch_vae(model, train_loader, losses_dict, optimizer, device, config, grl_lambda)

        phase_str = "Pretrain" if is_pretrain_phase else "Train"
        logger.info(f"Epoch {epoch+1}/{total_epochs} ({phase_str}) - Loss: {loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            ckpt_path = ckpt_dir / f"epoch{epoch+1:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save final checkpoint
    final_path = ckpt_dir / "best.pt"
    torch.save({
        'epoch': max_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, final_path)
    logger.info(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
