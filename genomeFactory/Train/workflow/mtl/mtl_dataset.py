"""
Multi-task dataset, sampler, and data collator.
"""

import csv
import math
import random
from typing import Dict, Sequence, List

import torch
import transformers
from torch.utils.data import Dataset, Sampler
from dataclasses import dataclass


# -----------------------------------------------------------------------
# Per-task dataset
# -----------------------------------------------------------------------

class TaskDataset(Dataset):
    """Load a single task's CSV into (input_ids, label) pairs."""

    def __init__(self, csv_path: str, tokenizer, max_length: int, task_type: str):
        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))[1:]

        if len(rows[0]) == 2:
            texts = [r[0] for r in rows]
            labels = [float(r[1]) for r in rows]
        elif len(rows[0]) == 3:
            texts = [[r[0], r[1]] for r in rows]
            labels = [float(r[2]) for r in rows]
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
        self.task_type = task_type

        if task_type == "classification":
            self.labels = [int(l) for l in labels]
        else:
            self.labels = labels

        self.num_labels = len(set(int(l) for l in labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# -----------------------------------------------------------------------
# Multi-task wrapper dataset
# -----------------------------------------------------------------------

class MTLDataset(Dataset):
    """
    Wraps multiple TaskDataset instances.  Each sample is tagged with
    its task name so the trainer knows which head to use.
    """

    def __init__(self, task_datasets: Dict[str, TaskDataset]):
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())

        # Build flat index: (task_name, local_idx)
        self._index = []
        for name, ds in task_datasets.items():
            for i in range(len(ds)):
                self._index.append((name, i))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        task_name, local_idx = self._index[i]
        item = self.task_datasets[task_name][local_idx]
        item["task_name"] = task_name
        return item


# -----------------------------------------------------------------------
# Round-robin or proportional sampler
# -----------------------------------------------------------------------

class MTLSampler(Sampler):
    """
    Samples batches in a round-robin or proportional fashion across tasks.

    Parameters
    ----------
    mtl_dataset : MTLDataset
    batch_size  : int
    strategy    : 'round_robin' or 'proportional'
    seed        : int
    """

    def __init__(self, mtl_dataset: MTLDataset, batch_size: int,
                 strategy: str = "round_robin", seed: int = 42):
        self.dataset = mtl_dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.rng = random.Random(seed)

        # Group indices by task
        self.task_indices = {}
        for flat_idx, (task_name, _) in enumerate(mtl_dataset._index):
            self.task_indices.setdefault(task_name, []).append(flat_idx)

    def __iter__(self):
        # Shuffle within each task
        shuffled = {}
        for name, idxs in self.task_indices.items():
            shuffled[name] = idxs.copy()
            self.rng.shuffle(shuffled[name])

        if self.strategy == "round_robin":
            task_names = list(shuffled.keys())
            task_ptrs = {n: 0 for n in task_names}
            task_cycle = 0

            while True:
                # Pick next task in round-robin order
                name = task_names[task_cycle % len(task_names)]
                task_cycle += 1

                ptr = task_ptrs[name]
                if ptr >= len(shuffled[name]):
                    # Check if all tasks exhausted
                    if all(task_ptrs[n] >= len(shuffled[n]) for n in task_names):
                        break
                    continue

                end = min(ptr + self.batch_size, len(shuffled[name]))
                for idx in shuffled[name][ptr:end]:
                    yield idx
                task_ptrs[name] = end

        else:  # proportional
            all_indices = []
            for idxs in shuffled.values():
                all_indices.extend(idxs)
            self.rng.shuffle(all_indices)
            yield from all_indices

    def __len__(self):
        return len(self.dataset)


# -----------------------------------------------------------------------
# Data collator
# -----------------------------------------------------------------------

@dataclass
class MTLDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        task_names = [inst["task_name"] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Determine if labels are ints or floats
        if isinstance(labels[0], int):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = torch.tensor(labels, dtype=torch.float)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            task_name=task_names[0],  # batch should be single-task
        )
