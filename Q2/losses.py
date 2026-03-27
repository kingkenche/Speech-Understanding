"""Loss functions for disentangled speaker recognition."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AngularProtoLoss(nn.Module):
    """Angular Prototypical loss for speaker verification."""
    def __init__(self, num_speakers: int, embedding_dim: int = 256, margin: float = 0.2, scale: float = 32.0):
        super().__init__()
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.centers = nn.Parameter(torch.randn(num_speakers, embedding_dim))
        nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,) speaker IDs
        Returns:
            loss: scalar
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centers = F.normalize(self.centers, p=2, dim=1)

        # Cosine similarity with all centers
        cos_sim = torch.matmul(embeddings, centers.t())  # (batch, num_speakers)

        # Add margin
        target_logits = cos_sim.gather(1, labels.unsqueeze(1))  # (batch, 1)
        cos_sim[torch.arange(len(labels)), labels] -= self.margin

        # Scale and compute softmax loss
        logits = cos_sim * self.scale
        loss = F.cross_entropy(logits, labels)
        return loss


class TripletsLoss(nn.Module):
    """Simple triplet loss for speaker verification."""
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: (batch, embedding_dim)
            positive: (batch, embedding_dim)
            negative: (batch, embedding_dim)
        Returns:
            loss: scalar
        """
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss


class ReconstructionLoss(nn.Module):
    """L2 reconstruction loss."""
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.criterion(reconstructed, original)


class CorrelationLoss(nn.Module):
    """Minimize correlation between speaker and environment codes."""
    def __init__(self):
        super().__init__()

    def forward(self, speaker_code: torch.Tensor, env_code: torch.Tensor) -> torch.Tensor:
        """
        Compute mean absolute correlation between components.
        """
        # Normalize
        spk = F.normalize(speaker_code, p=2, dim=0)
        env = F.normalize(env_code, p=2, dim=0)

        # Compute correlation (cosine similarity)
        corr = torch.abs(torch.sum(spk * env, dim=0) / (spk.shape[0] + 1e-8))
        loss = torch.mean(corr)
        return loss


class KLDivergenceLoss(nn.Module):
    """KL divergence loss for VAE."""
    def __init__(self):
        super().__init__()

    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL(N(mu, sigma^2) || N(0, 1))
        """
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        return kl.mean()


class GradientReversalLoss(nn.Module):
    """Gradient reversal for adversarial training."""
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def apply_grl(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply GRL then compute cross-entropy loss."""
        logits_reversed = self.apply_grl(logits)
        return F.cross_entropy(logits_reversed, labels)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * (-ctx.lambda_), None


import math
