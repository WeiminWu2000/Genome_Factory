"""
Learnable normalization layer inserted between tokenizer embeddings and
the foundation-model encoder.

The layer uses a residual MLP so it starts as a near-identity function
and gradually learns a task-informed normalisation during fine-tuning.
"""

import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):
    """Lightweight residual MLP for learnable embedding normalization."""

    def __init__(self, hidden_size: int, mlp_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # Initialize last linear to near-zero so residual starts as identity
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply residual normalization: x + MLP(x)."""
        return embeddings + self.mlp(embeddings)
