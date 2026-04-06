"""Tensor quantization and dequantization helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..config import TurboQuantHConfig
from .common import numel_from_shape
from .packing import pack_lowbit_codes, unpack_lowbit_codes


@dataclass
class PackedQuantizedTensor:
    packed_codes: torch.Tensor
    bits: int
    shape: tuple[int, ...]
    quantizer: str
    grouping: str
    scale: Optional[torch.Tensor] = None
    offset: Optional[torch.Tensor] = None
    codebook: Optional[torch.Tensor] = None

    def storage_bits(self) -> int:
        bits = int(self.packed_codes.numel() * 8)
        if self.scale is not None:
            bits += int(self.scale.numel() * self.scale.element_size() * 8)
        if self.offset is not None:
            bits += int(self.offset.numel() * self.offset.element_size() * 8)
        if self.codebook is not None:
            bits += int(self.codebook.numel() * self.codebook.element_size() * 8)
        return bits


def _group_view(x: torch.Tensor, grouping: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    batch, heads, tokens, dim = x.shape
    if grouping == "head":
        return x.reshape(batch * heads, tokens * dim), (batch, heads, 1, 1)
    if grouping == "channel":
        return x.permute(0, 1, 3, 2).reshape(batch * heads * dim, tokens), (batch, heads, 1, dim)
    raise ValueError(f"Unsupported grouping: {grouping}")


def _reshape_group_stat(values: torch.Tensor, stat_shape: tuple[int, ...]) -> torch.Tensor:
    if stat_shape[-1] == 1:
        return values.reshape(stat_shape)
    return values.reshape(stat_shape[0], stat_shape[1], stat_shape[3]).unsqueeze(2)


def _estimate_abs_scale(flat: torch.Tensor, cfg: TurboQuantHConfig) -> torch.Tensor:
    abs_flat = flat.abs()
    if cfg.scale_estimator == "absmax":
        scale = abs_flat.amax(dim=-1)
    elif cfg.scale_estimator == "abs_quantile":
        scale = torch.quantile(abs_flat, q=cfg.scale_quantile, dim=-1)
    elif cfg.scale_estimator == "rms":
        scale = flat.pow(2).mean(dim=-1).sqrt() * 3.0
    else:
        raise ValueError(f"Unsupported scale estimator: {cfg.scale_estimator}")
    return torch.clamp(scale, min=1e-8)


def _static_codebook_levels(bits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if bits == 1:
        return torch.tensor([-1.0, 1.0], device=device, dtype=dtype)

    levels = 1 << bits
    quantiles = torch.linspace(
        0.5 / levels,
        1.0 - 0.5 / levels,
        levels,
        device=device,
        dtype=dtype,
    )
    values = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
    return values / values.abs().amax().clamp(min=1e-8)


def quantize_uniform_symmetric(
    x: torch.Tensor,
    bits: int,
    grouping: str,
    cfg: TurboQuantHConfig,
) -> PackedQuantizedTensor:
    if bits < 1 or bits > 8:
        raise ValueError(f"Only 1-8 bits are supported, got {bits}")

    flat, stat_shape = _group_view(x, grouping)
    scale = _reshape_group_stat(_estimate_abs_scale(flat, cfg), stat_shape)
    if bits == 1:
        signed = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x)).to(torch.int32)
        unsigned = ((signed + 1) // 2).to(torch.uint8)
        return PackedQuantizedTensor(
            packed_codes=pack_lowbit_codes(unsigned, bits),
            bits=bits,
            shape=tuple(int(value) for value in x.shape),
            quantizer="uniform",
            grouping=grouping,
            scale=scale.to(torch.float16),
        )

    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    denom = max(abs(qmin), abs(qmax))
    step = torch.clamp(scale / denom, min=1e-8)
    signed = torch.clamp(torch.round(x / step), qmin, qmax).to(torch.int32)
    unsigned = (signed - qmin).to(torch.uint8)
    return PackedQuantizedTensor(
        packed_codes=pack_lowbit_codes(unsigned, bits),
        bits=bits,
        shape=tuple(int(value) for value in x.shape),
        quantizer="uniform",
        grouping=grouping,
        scale=step.to(torch.float16),
    )


def quantize_codebook(
    x: torch.Tensor,
    bits: int,
    grouping: str,
    cfg: TurboQuantHConfig,
) -> PackedQuantizedTensor:
    if bits < 1 or bits > 8:
        raise ValueError(f"Only 1-8 bits are supported, got {bits}")

    flat, stat_shape = _group_view(x, grouping)
    center_flat = flat.mean(dim=-1)
    centered_flat = flat - center_flat.unsqueeze(-1)
    scale_flat = _estimate_abs_scale(centered_flat, cfg)

    center = _reshape_group_stat(center_flat, stat_shape)
    scale = _reshape_group_stat(scale_flat, stat_shape)

    levels = _static_codebook_levels(bits, device=x.device, dtype=torch.float32)
    if levels.numel() == 2:
        codes = (x >= center).to(torch.uint8)
    else:
        normalized = ((x - center) / scale).to(torch.float32)
        boundaries = ((levels[:-1] + levels[1:]) * 0.5).to(torch.float32)
        codes = torch.bucketize(normalized.contiguous(), boundaries).to(torch.uint8)

    return PackedQuantizedTensor(
        packed_codes=pack_lowbit_codes(codes, bits),
        bits=bits,
        shape=tuple(int(value) for value in x.shape),
        quantizer="codebook",
        grouping=grouping,
        scale=scale.to(torch.float16),
        offset=center.to(torch.float16),
    )


def quantize_tensor(x: torch.Tensor, cfg: TurboQuantHConfig, tensor_kind: str) -> PackedQuantizedTensor:
    quantizer = cfg.resolved_quantizer(tensor_kind)
    grouping = cfg.resolved_quant_scale(tensor_kind)
    if quantizer == "uniform":
        return quantize_uniform_symmetric(x, cfg.quant_bits_old, grouping, cfg)
    if quantizer == "codebook":
        return quantize_codebook(x, cfg.quant_bits_old, grouping, cfg)
    raise ValueError(f"Unsupported quantizer: {quantizer}")


def dequantize_tensor(quant: PackedQuantizedTensor, device: torch.device) -> torch.Tensor:
    num_values = numel_from_shape(quant.shape)
    codes = unpack_lowbit_codes(quant.packed_codes.to(device), quant.bits, num_values).reshape(quant.shape).long()

    if quant.quantizer == "uniform":
        scale = quant.scale.to(device=device, dtype=torch.float32)
        if quant.bits == 1:
            signed = codes.to(torch.float32) * 2.0 - 1.0
        else:
            qmin = -(1 << (quant.bits - 1))
            signed = codes.to(torch.float32) + float(qmin)
        return signed * scale

    if quant.quantizer == "codebook":
        if quant.codebook is not None:
            batch, heads, tokens, dim = quant.shape
            if quant.grouping == "head":
                codebook = quant.codebook.to(device=device, dtype=torch.float32)
                expanded = codebook.unsqueeze(2).unsqueeze(2).expand(-1, -1, tokens, dim, -1)
                return torch.gather(expanded, -1, codes.unsqueeze(-1)).squeeze(-1)
            if quant.grouping == "channel":
                codebook = quant.codebook.to(device=device, dtype=torch.float32)
                codes_bhdt = codes.permute(0, 1, 3, 2)
                expanded = codebook.unsqueeze(3).expand(-1, -1, -1, tokens, -1)
                values = torch.gather(expanded, -1, codes_bhdt.unsqueeze(-1)).squeeze(-1)
                return values.permute(0, 1, 3, 2).contiguous()

        levels = _static_codebook_levels(quant.bits, device=device, dtype=torch.float32)
        scale = quant.scale.to(device=device, dtype=torch.float32)
        offset = quant.offset.to(device=device, dtype=torch.float32)
        return offset + levels[codes] * scale

    raise ValueError(f"Unsupported quantizer: {quant.quantizer}")


def select_quantized_batch(quant: PackedQuantizedTensor, beam_idx: torch.LongTensor) -> None:
    num_values = numel_from_shape(quant.shape)
    codes = unpack_lowbit_codes(quant.packed_codes.to(beam_idx.device), quant.bits, num_values).reshape(quant.shape)
    codes = codes.index_select(0, beam_idx)
    quant.packed_codes = pack_lowbit_codes(codes.to(torch.uint8), quant.bits)
    quant.shape = tuple(int(value) for value in codes.shape)
    if quant.scale is not None:
        quant.scale = quant.scale.index_select(0, beam_idx)
    if quant.offset is not None:
        quant.offset = quant.offset.index_select(0, beam_idx)
    if quant.codebook is not None:
        quant.codebook = quant.codebook.index_select(0, beam_idx)
