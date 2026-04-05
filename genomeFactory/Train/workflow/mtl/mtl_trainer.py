"""
Multi-task learning trainer.

Extends HuggingFace Trainer to:
  - route each batch to the correct task head via ``task_name``
  - weight losses by task-specific weights
  - evaluate per-task metrics separately
"""

import torch
import transformers
from torch.utils.data import DataLoader

from .mtl_dataset import MTLSampler


class MTLTrainer(transformers.Trainer):
    """Trainer subclass that handles multi-task batches."""

    def __init__(self, task_weights: dict = None, sampling_strategy: str = "round_robin", **kwargs):
        super().__init__(**kwargs)
        self.task_weights = task_weights or {}
        self.sampling_strategy = sampling_strategy

    def compute_loss(self, model, inputs, return_outputs=False):
        task_name = inputs.pop("task_name", None)
        outputs = model(task_name=task_name, **inputs)
        raw_loss = outputs.loss

        # Apply task weight
        if task_name and task_name in self.task_weights:
            loss = raw_loss * self.task_weights[task_name]
        else:
            loss = raw_loss

        # Store per-task loss for logging
        if task_name:
            self._last_task_name = task_name
            self._last_task_loss = raw_loss.detach().item()

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs):
        """Append per-task loss to the training logs."""
        if hasattr(self, '_last_task_name') and self._last_task_name:
            logs["task"] = self._last_task_name
            logs["loss_" + self._last_task_name] = self._last_task_loss
        super().log(logs)

    def get_train_dataloader(self) -> DataLoader:
        """Use MTLSampler for task-aware batch construction."""
        from .mtl_dataset import MTLDataset

        if not isinstance(self.train_dataset, MTLDataset):
            return super().get_train_dataloader()

        sampler = MTLSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            strategy=self.sampling_strategy,
            seed=self.args.seed,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
