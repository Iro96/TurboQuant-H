"""Shared helper functions for compression modules."""

from __future__ import annotations

import math
from typing import Optional

import torch


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def numel_from_shape(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 0


def token_length(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.shape[2])


def strip_empty_tokens(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or tensor.shape[2] == 0:
        return None
    return tensor
