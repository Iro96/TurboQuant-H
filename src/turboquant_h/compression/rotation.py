"""Rotation helpers used before quantization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..config import TurboQuantHConfig
from .common import is_power_of_two, numel_from_shape
from .packing import pack_lowbit_codes, unpack_lowbit_codes


def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    dimension = x.shape[-1]
    if not is_power_of_two(dimension):
        raise ValueError(f"Hadamard requires power-of-two last dim, got {dimension}")

    work = x.reshape(-1, dimension).clone()
    half_width = 1
    while half_width < dimension:
        work = work.view(-1, dimension // (half_width * 2), half_width * 2)
        left = work[..., :half_width].clone()
        right = work[..., half_width : 2 * half_width].clone()
        work[..., :half_width] = left + right
        work[..., half_width : 2 * half_width] = left - right
        work = work.view(-1, dimension)
        half_width *= 2
    return work.view_as(x)


@dataclass
class RotationState:
    mode: str
    packed_signs: Optional[torch.Tensor] = None
    sign_shape: Optional[tuple[int, ...]] = None

    def storage_bits(self) -> int:
        if self.packed_signs is None:
            return 0
        return int(self.packed_signs.numel() * 8)


def decode_rotation_signs(
    rotation: RotationState,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if rotation.packed_signs is None or rotation.sign_shape is None:
        return None
    num_values = numel_from_shape(rotation.sign_shape)
    signs = unpack_lowbit_codes(rotation.packed_signs.to(device), 1, num_values)
    signs = signs.reshape(rotation.sign_shape).to(torch.float32)
    return (signs * 2.0 - 1.0).to(dtype)


def select_rotation_batch(rotation: RotationState, beam_idx: torch.LongTensor) -> None:
    if rotation.packed_signs is None or rotation.sign_shape is None:
        return
    signs = decode_rotation_signs(rotation, device=beam_idx.device, dtype=torch.float32)
    assert signs is not None
    signs = signs.index_select(0, beam_idx)
    rotation.packed_signs = pack_lowbit_codes(((signs + 1.0) * 0.5).to(torch.uint8), 1)
    rotation.sign_shape = tuple(int(value) for value in signs.shape)


def make_rotation_state(x: torch.Tensor, cfg: TurboQuantHConfig, seed: int) -> RotationState:
    dimension = int(x.shape[-1])
    mode = cfg.rotation_mode

    if mode == "none" or not is_power_of_two(dimension):
        return RotationState(mode="none")
    if mode == "hadamard":
        return RotationState(mode="hadamard")
    if mode != "random_hadamard":
        raise ValueError(f"Unsupported rotation mode: {mode}")

    shape = (int(x.shape[0]), int(x.shape[1]), 1, dimension)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(0, 2, shape, generator=generator, dtype=torch.uint8)
    packed_signs = pack_lowbit_codes(signs.to(x.device), 1)
    return RotationState(mode="random_hadamard", packed_signs=packed_signs, sign_shape=shape)


def apply_rotation(x: torch.Tensor, rotation: RotationState) -> torch.Tensor:
    if rotation.mode == "none":
        return x
    work = x
    signs = decode_rotation_signs(rotation, device=x.device, dtype=x.dtype)
    if signs is not None:
        work = work * signs
    return fwht_last_dim(work) / math.sqrt(x.shape[-1])


def inverse_rotation(x: torch.Tensor, rotation: RotationState) -> torch.Tensor:
    if rotation.mode == "none":
        return x
    work = fwht_last_dim(x) / math.sqrt(x.shape[-1])
    signs = decode_rotation_signs(rotation, device=x.device, dtype=x.dtype)
    if signs is not None:
        work = work * signs
    return work
