"""Residual correction logic for compressed blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..config import TurboQuantHConfig
from .common import numel_from_shape
from .packing import pack_lowbit_codes, unpack_lowbit_codes


@dataclass
class CorrectionState:
    kind: str = "none"
    low_rank_left: Optional[torch.Tensor] = None
    low_rank_right: Optional[torch.Tensor] = None
    qjl_packed_signs: Optional[torch.Tensor] = None
    qjl_scale: Optional[torch.Tensor] = None
    qjl_shape: Optional[tuple[int, ...]] = None
    qjl_dim: int = 0
    qjl_seed: int = 0

    def storage_bits(self) -> int:
        bits = 0
        if self.low_rank_left is not None:
            bits += int(self.low_rank_left.numel() * self.low_rank_left.element_size() * 8)
        if self.low_rank_right is not None:
            bits += int(self.low_rank_right.numel() * self.low_rank_right.element_size() * 8)
        if self.qjl_packed_signs is not None:
            bits += int(self.qjl_packed_signs.numel() * 8)
        if self.qjl_scale is not None:
            bits += int(self.qjl_scale.numel() * self.qjl_scale.element_size() * 8)
        return bits


def make_qjl_projection(
    dim: int,
    proj_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(0, 2, (dim, proj_dim), generator=generator, dtype=torch.int8)
    projection = (signs.to(torch.float32) * 2.0 - 1.0) / math.sqrt(max(proj_dim, 1))
    return projection.to(device=device, dtype=dtype)


def _randomized_low_rank_factors(
    residual: torch.Tensor,
    rank: int,
    oversample: int,
    power_iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, tokens, dim = residual.shape
    matrices = residual.reshape(batch * heads, tokens, dim)
    sketch_rank = min(rank + oversample, min(tokens, dim))
    if sketch_rank <= 0:
        empty_left = residual.new_zeros(batch, heads, tokens, 0)
        empty_right = residual.new_zeros(batch, heads, 0, dim)
        return empty_left, empty_right

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    omega = torch.randn((batch * heads, dim, sketch_rank), generator=generator, dtype=torch.float32)
    omega = omega.to(device=residual.device, dtype=residual.dtype)

    sample = torch.matmul(matrices, omega)
    for _ in range(max(power_iters, 0)):
        sample = torch.matmul(matrices, torch.matmul(matrices.transpose(-1, -2), sample))

    orthogonal, _ = torch.linalg.qr(sample, mode="reduced")
    orthogonal = orthogonal[..., :rank]
    basis = torch.matmul(orthogonal.transpose(-1, -2), matrices)
    left = orthogonal.reshape(batch, heads, tokens, rank)
    right = basis.reshape(batch, heads, rank, dim)
    return left, right


def build_correction(
    residual: torch.Tensor,
    cfg: TurboQuantHConfig,
    seed: int,
    tensor_kind: str,
) -> CorrectionState:
    if residual.numel() == 0 or float(residual.abs().amax()) < 1e-8:
        return CorrectionState(kind="none")

    correction_type = cfg.resolved_correction_type(tensor_kind)
    if correction_type == "none":
        return CorrectionState(kind="none")

    if correction_type == "low_rank":
        rank = min(cfg.correction_rank, residual.shape[-2], residual.shape[-1])
        if rank <= 0:
            return CorrectionState(kind="none")
        left, right = _randomized_low_rank_factors(
            residual=residual,
            rank=rank,
            oversample=cfg.low_rank_oversample,
            power_iters=cfg.low_rank_power_iters,
            seed=seed,
        )
        return CorrectionState(
            kind="low_rank",
            low_rank_left=left.to(torch.float16),
            low_rank_right=right.to(torch.float16),
        )

    if correction_type == "qjl":
        proj_dim = min(max(cfg.qjl_dim, 1), residual.shape[-1])
        projection = make_qjl_projection(
            dim=residual.shape[-1],
            proj_dim=proj_dim,
            device=residual.device,
            dtype=residual.dtype,
            seed=seed,
        )
        projected = torch.matmul(residual, projection)
        signs = (projected >= 0).to(torch.uint8)
        scale = projected.norm(dim=-1, keepdim=True).div(math.sqrt(max(proj_dim, 1))).to(torch.float16)
        return CorrectionState(
            kind="qjl",
            qjl_packed_signs=pack_lowbit_codes(signs, 1),
            qjl_scale=scale,
            qjl_shape=tuple(int(value) for value in projected.shape),
            qjl_dim=proj_dim,
            qjl_seed=int(seed),
        )

    raise ValueError(f"Unsupported correction type: {correction_type}")


def apply_correction(x: torch.Tensor, correction: CorrectionState) -> torch.Tensor:
    if correction.kind == "none":
        return x

    if correction.kind == "low_rank":
        left = correction.low_rank_left.to(device=x.device, dtype=torch.float32)
        right = correction.low_rank_right.to(device=x.device, dtype=torch.float32)
        return x + torch.matmul(left, right)

    if correction.kind == "qjl":
        return x

    raise ValueError(f"Unsupported correction kind: {correction.kind}")


def select_correction_batch(correction: CorrectionState, beam_idx: torch.LongTensor) -> None:
    if correction.low_rank_left is not None:
        correction.low_rank_left = correction.low_rank_left.index_select(0, beam_idx)
    if correction.low_rank_right is not None:
        correction.low_rank_right = correction.low_rank_right.index_select(0, beam_idx)
    if correction.qjl_scale is not None:
        correction.qjl_scale = correction.qjl_scale.index_select(0, beam_idx)
    if correction.qjl_packed_signs is None or correction.qjl_shape is None:
        return

    num_values = numel_from_shape(correction.qjl_shape)
    signs = unpack_lowbit_codes(correction.qjl_packed_signs.to(beam_idx.device), 1, num_values)
    signs = signs.reshape(correction.qjl_shape).index_select(0, beam_idx)
    correction.qjl_packed_signs = pack_lowbit_codes(signs.to(torch.uint8), 1)
    correction.qjl_shape = tuple(int(value) for value in signs.shape)


def token_saliency_scores(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean(dim=-1).sqrt()
