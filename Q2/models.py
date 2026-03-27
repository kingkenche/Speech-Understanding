"""Speaker embedding models and disentangler framework."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math

class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * (-ctx.lambda_), None


class TDNNEncoder(nn.Module):
    """Lightweight TDNN-based speaker encoder with attentive pooling."""
    def __init__(self, embedding_dim: int = 256, feat_dim: int = 80):
        super().__init__()
        self.feat_dim = feat_dim
        self.embedding_dim = embedding_dim

        # Time delay neural network layers
        self.tdnn1 = nn.Conv1d(feat_dim, 512, kernel_size=5, dilation=1, padding=2)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=5, dilation=2, padding=4)
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=5, dilation=3, padding=6)
        self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1)
        self.tdnn5 = nn.Conv1d(512, 1500, kernel_size=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1500)

        # Attentive pooling
        self.attention = nn.Sequential(
            nn.Linear(1500, 128),
            nn.ReLU(),
            nn.Linear(128, 1500),
            nn.Softmax(dim=1)
        )

        # Output projection
        self.fc = nn.Linear(1500, embedding_dim)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, channels, frames) -> (batch, embedding_dim)"""
        x = F.relu(self.bn1(self.tdnn1(x)))
        x = F.relu(self.bn2(self.tdnn2(x)))
        x = F.relu(self.bn3(self.tdnn3(x)))
        x = F.relu(self.bn4(self.tdnn4(x)))
        x = F.relu(self.bn5(self.tdnn5(x)))  # (batch, 1500, frames)

        # Attentive pooling
        x_t = x.transpose(1, 2)  # (batch, frames, 1500)
        w = self.attention(x_t)  # (batch, frames, 1500)
        pooled = torch.sum(w * x_t, dim=1)  # (batch, 1500)

        # Output projection
        embedding = self.bn_out(self.fc(pooled))
        return embedding


class SimpleResNet34(nn.Module):
    """Simplified ResNet34-like encoder for synthetic experiments."""
    def __init__(self, embedding_dim: int = 256, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.conv1 = nn.Conv1d(feat_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(256)
        self.fc_pool = nn.Linear(256, 1)
        self.mapping = nn.Linear(256 * 2, embedding_dim)
        self.bn_mapping = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.instance_norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        w = torch.softmax(self.fc_pool(x.transpose(1, 2)), dim=1)
        context = torch.sum(w * x.transpose(1, 2), dim=1)
        mean = torch.mean(x, dim=2)
        pooled = torch.cat([context, mean], dim=1)
        embedding = self.bn_mapping(self.mapping(pooled))
        return embedding

class AutoencoderDisentangler(nn.Module):
    """Auto-encoder based disentangler (paper's method)."""
    def __init__(self, embedding_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.component_dim = latent_dim // 2

        # Encoder
        self.encoder_bn = nn.BatchNorm1d(embedding_dim)
        self.encoder_fc = nn.Linear(embedding_dim, latent_dim)

        # Decoder
        self.decoder_bn = nn.BatchNorm1d(latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, embedding_dim)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode embedding into speaker and environment codes."""
        # Encode
        full_code = self.encoder_fc(self.encoder_bn(embedding))

        # Split and normalize
        speaker_code = F.normalize(full_code[:, :self.component_dim], p=2, dim=1)
        env_code = F.normalize(full_code[:, self.component_dim:], p=2, dim=1)
        full_code_norm = torch.cat([speaker_code, env_code], dim=1)

        # Decode
        reconstructed = self.decoder_fc(self.decoder_bn(full_code_norm))

        return reconstructed, speaker_code, env_code


class VAEDisentangler(nn.Module):
    """VAE-based disentangler (proposed improvement)."""
    def __init__(self, embedding_dim: int = 256, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.component_dim = latent_dim // 2

        # Encoder to mean and logvar (for each component)
        self.encoder_bn = nn.BatchNorm1d(embedding_dim)
        self.encoder_shared = nn.Linear(embedding_dim, hidden_dim)

        # Speaker component: mean and logvar
        self.spk_mean = nn.Linear(hidden_dim, self.component_dim)
        self.spk_logvar = nn.Linear(hidden_dim, self.component_dim)

        # Environment component: mean and logvar
        self.env_mean = nn.Linear(hidden_dim, self.component_dim)
        self.env_logvar = nn.Linear(hidden_dim, self.component_dim)

        # Decoder
        self.decoder_bn = nn.BatchNorm1d(latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, embedding_dim)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, (mean, logvar)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Encode embedding into disentangled speaker and environment codes."""
        # Encode
        h = F.relu(self.encoder_shared(self.encoder_bn(embedding)))

        # Sample speaker code
        spk_mean = self.spk_mean(h)
        spk_logvar = self.spk_logvar(h)
        speaker_code, spk_params = self.reparameterize(spk_mean, spk_logvar)
        speaker_code = F.normalize(speaker_code, p=2, dim=1)

        # Sample environment code
        env_mean = self.env_mean(h)
        env_logvar = self.env_logvar(h)
        env_code, env_params = self.reparameterize(env_mean, env_logvar)
        env_code = F.normalize(env_code, p=2, dim=1)

        # Combine
        full_code = torch.cat([speaker_code, env_code], dim=1)

        # Decode
        reconstructed = self.decoder_fc(self.decoder_bn(full_code))

        # Store for KL computation
        kl_info = {
            'spk_mean': spk_mean, 'spk_logvar': spk_logvar,
            'env_mean': env_mean, 'env_logvar': env_logvar
        }

        return reconstructed, speaker_code, env_code, kl_info

class SpeakerDiscriminator(nn.Module):
    """Speaker classifier on disentangled speaker code."""
    def __init__(self, component_dim: int = 128, num_speakers: int = 1000):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(component_dim),
            nn.Linear(component_dim, num_speakers)
        )

    def forward(self, speaker_code: torch.Tensor) -> torch.Tensor:
        return self.fc(speaker_code)


class EnvironmentDiscriminator(nn.Module):
    """Adversarial environment classifier."""
    def __init__(self, component_dim: int = 128, hidden_dim: int = 512, num_envs: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(component_dim),
            nn.Linear(component_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_envs)
        )

    def forward(self, component: torch.Tensor) -> torch.Tensor:
        return self.net(component)

class BaselineFramework(nn.Module):
    """Baseline: encoder + speaker loss only."""
    def __init__(self, embedding_extractor: nn.Module, embedding_dim: int = 256, num_speakers: int = 1000):
        super().__init__()
        self.embedding_extractor = embedding_extractor
        self.speaker_classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, num_speakers)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.embedding_extractor(x)
        speaker_logits = self.speaker_classifier(embedding)
        return {
            'embedding': embedding,
            'speaker_logits': speaker_logits
        }

    def extract_speaker_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_extractor(x)


class DisentanglementFramework(nn.Module):
    """Auto-encoder based disentanglement (paper's method)."""
    def __init__(self, embedding_extractor: nn.Module, embedding_dim: int = 256,
                 latent_dim: int = 256, num_speakers: int = 1000, num_envs: int = 100):
        super().__init__()
        self.embedding_extractor = embedding_extractor
        self.component_dim = latent_dim // 2
        self.disentangler = AutoencoderDisentangler(embedding_dim, latent_dim)
        self.speaker_discriminator = SpeakerDiscriminator(self.component_dim, num_speakers)
        self.env_discriminator_e = EnvironmentDiscriminator(self.component_dim, 512, num_envs)
        self.env_discriminator_s = EnvironmentDiscriminator(self.component_dim, 512, num_envs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.embedding_extractor(x)
        reconstructed, speaker_code, env_code = self.disentangler(embedding)
        return {
            'embedding': embedding,
            'reconstructed': reconstructed,
            'speaker_code': speaker_code,
            'environment_code': env_code,
            'speaker_logits': self.speaker_discriminator(speaker_code),
            'env_logits_e': self.env_discriminator_e(env_code),
            'env_logits_s': self.env_discriminator_s(speaker_code)
        }

    def extract_speaker_features(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_extractor(x)
        _, speaker_code, _ = self.disentangler(embedding)
        return speaker_code


class VAEDisentanglementFramework(nn.Module):
    """VAE-based disentanglement (proposed improvement)."""
    def __init__(self, embedding_extractor: nn.Module, embedding_dim: int = 256,
                 latent_dim: int = 256, num_speakers: int = 1000, num_envs: int = 100, hidden_dim: int = 512):
        super().__init__()
        self.embedding_extractor = embedding_extractor
        self.component_dim = latent_dim // 2
        self.disentangler = VAEDisentangler(embedding_dim, latent_dim, hidden_dim)
        self.speaker_discriminator = SpeakerDiscriminator(self.component_dim, num_speakers)
        self.env_discriminator_e = EnvironmentDiscriminator(self.component_dim, 512, num_envs)
        self.env_discriminator_s = EnvironmentDiscriminator(self.component_dim, 512, num_envs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.embedding_extractor(x)
        reconstructed, speaker_code, env_code, kl_info = self.disentangler(embedding)
        return {
            'embedding': embedding,
            'reconstructed': reconstructed,
            'speaker_code': speaker_code,
            'environment_code': env_code,
            'speaker_logits': self.speaker_discriminator(speaker_code),
            'env_logits_e': self.env_discriminator_e(env_code),
            'env_logits_s': self.env_discriminator_s(speaker_code),
            'kl_info': kl_info
        }

    def extract_speaker_features(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_extractor(x)
        reconstructed, speaker_code, env_code, _ = self.disentangler(embedding)
        return speaker_code
