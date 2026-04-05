"""
Multi-task learning workflow.

Users define multiple tasks in a single YAML config.  The framework
attaches one prediction head per task on top of the shared foundation
model backbone.  Each mini-batch is sampled from one task via round-robin
or proportional scheduling.

    genomefactory-cli train mtl_config.yaml
"""

import os
import json
import yaml
import logging
import argparse
from typing import Dict

import torch
import transformers

from .mtl_model import MultiTaskModel
from .mtl_dataset import TaskDataset, MTLDataset, MTLDataCollator
from .mtl_trainer import MTLTrainer
from genomeFactory.Train.metric.metric_classification import (
    calculate_metric_with_sklearn,
)
import numpy as np

logger = logging.getLogger(__name__)


def train_mtl():
    """Main entry point for multi-task training."""

    parser = argparse.ArgumentParser(description="Multi-task learning training")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the MTL YAML config file")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---- Parse config sections ----
    model_cfg = config.get("model", {})
    mtl_cfg = config.get("mtl", {})
    train_cfg = config.get("train", {})
    output_cfg = config.get("output", {})

    model_name = model_cfg.get("model_name_or_path", "zhihan1996/DNABERT-2-117M")
    task_configs = mtl_cfg.get("tasks", [])
    sampling_strategy = mtl_cfg.get("sampling_strategy", "round_robin")

    if not task_configs:
        raise ValueError("MTL config must define at least one task in mtl.tasks")

    # ---- Tokenizer ----
    max_length = train_cfg.get("model_max_length", 512)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # ---- Load per-task datasets ----
    train_task_ds = {}
    val_task_ds = {}
    test_task_ds = {}

    for tc in task_configs:
        name = tc["name"]
        data_path = tc["data_path"]
        task_type = tc["type"]

        train_task_ds[name] = TaskDataset(
            os.path.join(data_path, "train.csv"), tokenizer, max_length, task_type
        )
        val_task_ds[name] = TaskDataset(
            os.path.join(data_path, "dev.csv"), tokenizer, max_length, task_type
        )
        test_task_ds[name] = TaskDataset(
            os.path.join(data_path, "test.csv"), tokenizer, max_length, task_type
        )

        # Infer num_labels for classification
        if task_type == "classification" and "num_labels" not in tc:
            tc["num_labels"] = max(2, train_task_ds[name].num_labels)

        print(f"[MTL] Task '{name}' ({task_type}): "
              f"train={len(train_task_ds[name])}, "
              f"val={len(val_task_ds[name])}, "
              f"test={len(test_task_ds[name])}")

    # ---- Build multi-task datasets ----
    train_dataset = MTLDataset(train_task_ds)
    val_dataset = MTLDataset(val_task_ds)

    # ---- Model ----
    use_flash = train_cfg.get("use_flash_attention", False)
    model = MultiTaskModel(
        base_model_name=model_name,
        task_configs=task_configs,
        trust_remote_code=True,
        use_flash_attention=use_flash,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MTL] Model params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    for name, head in model.heads.items():
        hp = sum(p.numel() for p in head.parameters())
        print(f"  Head '{name}': {hp} params")

    # ---- Task weights ----
    task_weights = {tc["name"]: tc.get("weight", 1.0) for tc in task_configs}

    # ---- Training arguments ----
    output_dir = output_cfg.get("output_dir", "output_mtl")
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=output_cfg.get("overwrite_output_dir", True),
        run_name=train_cfg.get("run_name", "mtl_run"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg.get("learning_rate", 3e-5),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 50),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        evaluation_strategy=train_cfg.get("evaluation_strategy", "steps"),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=train_cfg.get("seed", 42),
        report_to="none",
    )

    # ---- Collator ----
    collator = MTLDataCollator(tokenizer=tokenizer)

    # ---- Metrics ----
    def compute_metrics_cls(eval_pred):
        """Classification metrics: apply argmax to logits, then sklearn."""
        preds, labels = eval_pred
        if preds.ndim == 2:
            preds = preds.argmax(axis=-1)
        # Guard against regression labels leaking in from mixed eval
        try:
            return calculate_metric_with_sklearn(preds, labels)
        except ValueError:
            return {"accuracy": 0.0, "f1": 0.0, "matthews_correlation": 0.0,
                    "precision": 0.0, "recall": 0.0}

    def compute_metrics_reg(eval_pred):
        """Regression metrics: MSE and MAE."""
        preds, labels = eval_pred
        preds = preds.squeeze()
        return {
            "mse": float(np.mean((preds - labels) ** 2)),
            "mae": float(np.mean(np.abs(preds - labels))),
        }

    def compute_metrics_mixed(eval_pred):
        """Metrics for mixed-task eval during training (best-effort)."""
        preds, labels = eval_pred
        try:
            if preds.ndim == 2:
                preds = preds.argmax(axis=-1)
            return calculate_metric_with_sklearn(preds, labels)
        except (ValueError, TypeError):
            preds_f = preds.squeeze() if preds.ndim > 1 else preds
            return {
                "mse": float(np.mean((preds_f - labels) ** 2)),
                "mae": float(np.mean(np.abs(preds_f - labels))),
            }

    # ---- Trainer ----
    trainer = MTLTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        task_weights=task_weights,
        sampling_strategy=sampling_strategy,
        compute_metrics=compute_metrics_mixed,  # handles mixed eval
    )

    # ---- Train ----
    output = trainer.train()
    print(f"\n[MTL] Training complete: {output.metrics}")

    # ---- Save ----
    save_dir = train_cfg.get("saved_model_dir", os.path.join(output_dir, "final_model"))
    os.makedirs(save_dir, exist_ok=True)
    model.backbone.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    # Save task heads
    torch.save(
        {name: head.state_dict() for name, head in model.heads.items()},
        os.path.join(save_dir, "task_heads.pt"),
    )
    print(f"[MTL] Model saved to {save_dir}")

    # ---- Per-task evaluation on test sets ----
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for tc in task_configs:
        name = tc["name"]
        eval_ds = MTLDataset({name: test_task_ds[name]})

        # Swap metrics based on task type
        if tc["type"] == "regression":
            trainer.compute_metrics = compute_metrics_reg
        else:
            trainer.compute_metrics = compute_metrics_cls

        results = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"test_{name}")
        all_results[name] = results
        print(f"[MTL] Test results for '{name}': {results}")

    with open(os.path.join(results_dir, "mtl_eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[MTL] All results saved to {results_dir}/mtl_eval_results.json")
