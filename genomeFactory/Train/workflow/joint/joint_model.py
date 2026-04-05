"""
JointModel: wraps a HuggingFace classification / regression model with a
learnable NormalizationLayer and composite loss (task + batch MMD + bio
preservation).

Instead of using ``inputs_embeds`` (which some custom models do not fully
support), we hook the NormalizationLayer into the model's embedding layer
so the model is called normally with ``input_ids``.
"""

import torch
import torch.nn as nn

from .normalization_layer import NormalizationLayer
from .losses import compute_mmd_loss, compute_bio_preservation_loss


class _WrappedEmbedding(nn.Module):
    """Wraps an embedding layer and applies NormalizationLayer after it."""

    def __init__(self, original_embedding, norm_layer):
        super().__init__()
        self.original_embedding = original_embedding
        self.norm_layer = norm_layer
        # Storage for loss computation (set during forward, read by JointModel)
        self.last_original = None
        self.last_normalized = None

    def forward(self, input_ids):
        raw = self.original_embedding(input_ids)
        normalized = self.norm_layer(raw)
        # Store for auxiliary loss computation
        self.last_original = raw
        self.last_normalized = normalized
        return normalized


class JointModel(nn.Module):
    """
    Wrapper that inserts a learnable NormalizationLayer into the model's
    embedding pipeline, and computes a composite loss during training.

    Parameters
    ----------
    base_model   : HuggingFace AutoModelForSequenceClassification (or Regression).
    norm_layer   : NormalizationLayer instance.
    lambda_batch : float - weight for MMD batch-invariance loss.
    lambda_bio   : float - weight for biological-preservation loss.
    """

    def __init__(
        self,
        base_model: nn.Module,
        norm_layer: NormalizationLayer,
        lambda_batch: float = 0.1,
        lambda_bio: float = 0.05,
    ):
        super().__init__()
        self.base_model = base_model
        self.norm_layer = norm_layer
        self.lambda_batch = lambda_batch
        self.lambda_bio = lambda_bio

        # Wrap the embedding layer
        original_embed = base_model.get_input_embeddings()
        self.wrapped_embed = _WrappedEmbedding(original_embed, norm_layer)
        base_model.set_input_embeddings(self.wrapped_embed)

    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # Forward through the model normally (embedding hook applies norm)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if labels is not None and self.training:
            L_task = outputs.loss

            # Retrieve stored embeddings from the hook
            norm_embeds = self.wrapped_embed.last_normalized   # (B, L, D)
            raw_embeds = self.wrapped_embed.last_original      # (B, L, D)

            # Mean-pool for batch-level loss
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (norm_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = norm_embeds.mean(dim=1)

            L_batch = compute_mmd_loss(pooled.detach())

            vocab_size = self.base_model.config.vocab_size
            L_bio = compute_bio_preservation_loss(
                input_ids, raw_embeds.detach(), norm_embeds,
                vocab_size=vocab_size, k=3,
            )

            total_loss = (
                L_task
                + self.lambda_batch * L_batch
                + self.lambda_bio * L_bio
            )
            outputs.loss = total_loss

            # Store individual components for logging
            self._last_loss_task = L_task.detach().item()
            self._last_loss_batch = L_batch.detach().item()
            self._last_loss_bio = L_bio.detach().item()

        return outputs
