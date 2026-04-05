"""
Joint-optimization training workflow.

Adds a learnable NormalizationLayer between tokenizer output and the
foundation-model encoder, and trains with a composite loss:

    L = L_task + lambda_batch * L_batch(MMD) + lambda_bio * L_bio(preservation)

Both the normalisation parameters and the model parameters are optimised
end-to-end, so the data normalisation adapts to the final task.
"""

import os
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.nn as nn
import transformers
import numpy as np
from torch.utils.data import Dataset

from genomeFactory.Train.metric.metric_classification import (
    preprocess_logits_for_metrics,
    compute_metrics,
)
from genomeFactory.Train.metric.metric_regression import (
    compute_metrics as compute_metrics_regression,
)
from .normalization_layer import NormalizationLayer
from .joint_model import JointModel

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Dataclass arguments
# -----------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="zhihan1996/DNABERT-2-117M")
    use_flash_attention: bool = field(default=False)


@dataclass
class JointArguments:
    lambda_batch: float = field(default=0.1, metadata={"help": "MMD batch-invariance loss weight"})
    lambda_bio: float = field(default=0.05, metadata={"help": "Biological-preservation loss weight"})
    norm_hidden_size: int = field(default=128, metadata={"help": "Hidden dim of NormalizationLayer MLP"})


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    classification: bool = field(default=True)
    regression: bool = field(default=False)


@dataclass
class JointTrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="joint_run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=16)
    num_train_epochs: int = field(default=3)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=100)
    remove_unused_columns: bool = field(default=False)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    learning_rate: float = field(default=3e-5)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output_joint")
    overwrite_output_dir: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    seed: int = field(default=42)
    saved_model_dir: str = field(default="")


# -----------------------------------------------------------------------
# Dataset (reused from workflow_classification but simplified)
# -----------------------------------------------------------------------

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 2:
            texts = [d[0] for d in data]
            labels = [float(d[1]) for d in data]
        elif len(data[0]) == 3:
            texts = [[d[0], d[1]] for d in data]
            labels = [float(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.labels = labels
        self.num_labels = len(set(int(l) for l in labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.tensor(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# -----------------------------------------------------------------------
# Custom Trainer with composite-loss logging
# -----------------------------------------------------------------------

class JointTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs):
        """Append individual loss components to the training logs."""
        if hasattr(self.model, '_last_loss_task'):
            logs["loss_task"] = self.model._last_loss_task
            logs["loss_batch_mmd"] = self.model._last_loss_batch
            logs["loss_bio_kmer"] = self.model._last_loss_bio
        super().log(logs)


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def train_joint():
    parser = transformers.HfArgumentParser((
        ModelArguments, JointArguments, DataArguments, JointTrainingArguments,
    ))
    model_args, joint_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # Datasets
    train_ds = SupervisedDataset(
        os.path.join(data_args.data_path, "train.csv"),
        tokenizer, training_args.model_max_length,
    )
    val_ds = SupervisedDataset(
        os.path.join(data_args.data_path, "dev.csv"),
        tokenizer, training_args.model_max_length,
    )
    test_ds = SupervisedDataset(
        os.path.join(data_args.data_path, "test.csv"),
        tokenizer, training_args.model_max_length,
    )

    # Determine num_labels and task type
    is_classification = data_args.classification and not data_args.regression
    num_labels = max(2, train_ds.num_labels) if is_classification else 1

    # Base model
    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        trust_remote_code=True,
    )

    # NormalizationLayer
    hidden_size = base_model.config.hidden_size
    norm_layer = NormalizationLayer(hidden_size, joint_args.norm_hidden_size)

    # Wrap in JointModel
    model = JointModel(
        base_model=base_model,
        norm_layer=norm_layer,
        lambda_batch=joint_args.lambda_batch,
        lambda_bio=joint_args.lambda_bio,
    )

    # Cast labels for classification
    if is_classification:
        train_ds.labels = [int(l) for l in train_ds.labels]
        val_ds.labels = [int(l) for l in val_ds.labels]
        test_ds.labels = [int(l) for l in test_ds.labels]

    collator = DataCollator(tokenizer=tokenizer)

    if is_classification:
        trainer = JointTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = JointTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics_regression,
        )

    # Train
    output = trainer.train()
    print(output.metrics)

    # Save
    save_dir = training_args.saved_model_dir.strip() or "./Trained_model_joint"
    os.makedirs(save_dir, exist_ok=True)
    base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(norm_layer.state_dict(), os.path.join(save_dir, "norm_layer.pt"))
    print(f"Model + NormalizationLayer saved to {save_dir}")

    # Evaluate on test set
    results = trainer.evaluate(eval_dataset=test_ds)
    results_dir = os.path.join(training_args.output_dir, "results", training_args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test results: {results}")
