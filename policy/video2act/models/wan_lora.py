"""Small trainable LoRA wrappers for frozen WAN Linear layers.

The checkpoint format intentionally mirrors the lightweight Lightewm convention:
wrapped modules expose ``lora_A`` and ``lora_B`` weights, while the frozen base
linear weight is left out by the Video2Act policy state_dict filter.
"""

import logging
import re
from typing import Iterable, Sequence

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.base_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def apply_lora_to_linear_modules(
    model: nn.Module,
    target_patterns: Iterable[str],
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> list[str]:
    patterns = list(target_patterns)
    if not patterns:
        raise ValueError("target_patterns must not be empty when WAN LoRA is enabled")
    alpha = float(alpha if alpha is not None else rank)

    replacements = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name, patterns):
            replacements.append((name, module))

    for name, module in replacements:
        _set_module(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))

    logger.info("Applied WAN LoRA to %d Linear layers", len(replacements))
    for name, _ in replacements[:20]:
        logger.info("  LoRA: %s", name)
    if len(replacements) > 20:
        logger.info("  ... and %d more", len(replacements) - 20)
    return [name for name, _ in replacements]


def lora_parameters(model: nn.Module):
    return [p for name, p in model.named_parameters() if ("lora_A" in name or "lora_B" in name) and p.requires_grad]


def is_lora_state_key(key: str) -> bool:
    return ".lora_A." in key or ".lora_B." in key
