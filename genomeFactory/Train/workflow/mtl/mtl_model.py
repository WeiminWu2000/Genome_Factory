"""
Multi-task model: a shared foundation-model backbone with one prediction
head per task (classification or regression).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class MultiTaskModel(nn.Module):
    """
    Shared backbone + per-task prediction heads.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model name (e.g. ``zhihan1996/DNABERT-2-117M``).
    task_configs : list[dict]
        Each dict has keys: ``name``, ``type`` (classification / regression),
        ``num_labels`` (for classification), ``weight`` (loss weight).
    trust_remote_code : bool
    use_flash_attention : bool
    """

    def __init__(
        self,
        base_model_name: str,
        task_configs: list,
        trust_remote_code: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        # Load backbone (base model without classification head)
        self.backbone = transformers.AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        self.config = self.backbone.config
        hidden_size = self.config.hidden_size

        # Build task heads
        self.task_configs = {tc["name"]: tc for tc in task_configs}
        self.heads = nn.ModuleDict()
        for tc in task_configs:
            name = tc["name"]
            if tc["type"] == "classification":
                self.heads[name] = nn.Linear(hidden_size, tc["num_labels"])
            elif tc["type"] == "regression":
                self.heads[name] = nn.Linear(hidden_size, 1)
            else:
                raise ValueError(f"Unknown task type: {tc['type']}")

        # Loss functions per task
        self._loss_fns = {}
        for tc in task_configs:
            if tc["type"] == "classification":
                self._loss_fns[tc["name"]] = nn.CrossEntropyLoss()
            else:
                self._loss_fns[tc["name"]] = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        task_name: str = None,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Handle both tuple outputs and named outputs
        if isinstance(outputs, tuple):
            hidden = outputs[0]                                 # (B, L, D)
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]

        # Pool: mean pooling over sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.heads[task_name](pooled)                  # (B, C) or (B, 1)

        loss = None
        if labels is not None:
            tc = self.task_configs[task_name]
            if tc["type"] == "regression":
                loss = self._loss_fns[task_name](logits.squeeze(-1), labels.float())
            else:
                loss = self._loss_fns[task_name](logits, labels.long())

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
