"""Compressed KV-cache data structures."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
from transformers.cache_utils import Cache

from ..config import LatencyStats, TurboQuantHConfig
from .common import strip_empty_tokens, token_length
from .correction import (
    CorrectionState,
    apply_correction,
    build_correction,
    select_correction_batch,
    token_saliency_scores,
)
from .quantization import (
    PackedQuantizedTensor,
    dequantize_tensor,
    quantize_tensor,
    select_quantized_batch,
)
from .rotation import RotationState, apply_rotation, inverse_rotation, make_rotation_state, select_rotation_batch


@dataclass
class CompressedBlock:
    quant: PackedQuantizedTensor
    rotation: RotationState
    correction: CorrectionState
    retain_idx: Optional[torch.Tensor]
    retain_values: Optional[torch.Tensor]
    block_shape: tuple[int, ...]

    def storage_bits(self) -> int:
        bits = self.quant.storage_bits() + self.rotation.storage_bits() + self.correction.storage_bits()
        if self.retain_idx is not None:
            bits += int(self.retain_idx.numel() * self.retain_idx.element_size() * 8)
        if self.retain_values is not None:
            bits += int(self.retain_values.numel() * self.retain_values.element_size() * 8)
        return bits

    def _apply_retain_overlay(
        self,
        work: torch.Tensor,
        device: torch.device,
        rotated: bool,
    ) -> torch.Tensor:
        if self.retain_idx is None or self.retain_values is None:
            return work
        indices = self.retain_idx.to(device=device, dtype=torch.long)
        values = self.retain_values.to(device=device, dtype=torch.float32)
        if rotated:
            values = apply_rotation(values, self.rotation)
        work = work.clone()
        work.scatter_(2, indices.unsqueeze(-1).expand(-1, -1, -1, work.shape[-1]), values)
        return work

    def materialize_rotated_for_attention(self, device: torch.device) -> torch.Tensor:
        work = dequantize_tensor(self.quant, device=device)
        if self.correction.kind == "low_rank":
            work = apply_correction(work, self.correction)
        return self._apply_retain_overlay(work, device=device, rotated=True)

    def materialize(self, device: torch.device) -> torch.Tensor:
        work = dequantize_tensor(self.quant, device=device)
        work = apply_correction(work, self.correction)
        block = inverse_rotation(work, self.rotation)
        return self._apply_retain_overlay(block, device=device, rotated=False)

    def select_batch(self, beam_idx: torch.LongTensor) -> None:
        select_quantized_batch(self.quant, beam_idx)
        select_rotation_batch(self.rotation, beam_idx)
        select_correction_batch(self.correction, beam_idx)
        if self.retain_idx is not None:
            self.retain_idx = self.retain_idx.index_select(0, beam_idx)
        if self.retain_values is not None:
            self.retain_values = self.retain_values.index_select(0, beam_idx)
        batch, *rest = self.block_shape
        self.block_shape = (beam_idx.shape[0], *rest)


def _keep_token_count(length: int, keep_ratio: float) -> int:
    if length <= 0 or keep_ratio <= 0:
        return 0
    keep_count = int(math.ceil(keep_ratio * length))
    return max(1, min(length, keep_count))


def _retain_index_dtype(block_tokens: int) -> torch.dtype:
    if block_tokens <= 256:
        return torch.uint8
    return torch.int16


def compress_block(
    x: torch.Tensor,
    cfg: TurboQuantHConfig,
    seed: int,
    tensor_kind: str,
    storage_dtype: torch.dtype,
) -> CompressedBlock:
    block = x.to(torch.float32)
    rotation = make_rotation_state(block, cfg, seed=seed)
    rotated = apply_rotation(block, rotation)
    quant = quantize_tensor(rotated, cfg, tensor_kind=tensor_kind)
    approx = dequantize_tensor(quant, device=block.device)
    correction = build_correction(rotated - approx, cfg, seed=seed + 1, tensor_kind=tensor_kind)

    retain_idx = None
    retain_values = None
    keep_count = _keep_token_count(block.shape[2], cfg.keep_ratio_old)
    if keep_count > 0:
        scores = token_saliency_scores(block)
        topk = torch.topk(scores, k=keep_count, dim=-1)
        retain_idx = topk.indices.sort(dim=-1).values.to(_retain_index_dtype(block.shape[2]))
        retain_values = torch.gather(
            block,
            dim=2,
            index=retain_idx.to(torch.long).unsqueeze(-1).expand(-1, -1, -1, block.shape[-1]),
        ).to(storage_dtype)

    return CompressedBlock(
        quant=quant,
        rotation=rotation,
        correction=correction,
        retain_idx=retain_idx,
        retain_values=retain_values,
        block_shape=tuple(int(value) for value in block.shape),
    )


@dataclass
class CompressedSegment:
    old_blocks: list[CompressedBlock] = field(default_factory=list)
    pending_old_values: Optional[torch.Tensor] = None
    recent_values: Optional[torch.Tensor] = None
    original_shape: tuple[int, ...] = (0, 0, 0, 0)
    seed_base: int = 0
    tensor_kind: str = "k"
    storage_dtype: torch.dtype = torch.float32

    def old_token_count(self) -> int:
        block_tokens = sum(block.block_shape[2] for block in self.old_blocks)
        return int(block_tokens + token_length(self.pending_old_values))

    def total_tokens(self) -> int:
        return int(self.old_token_count() + token_length(self.recent_values))

    def _sync_shape(self) -> None:
        if not self.original_shape:
            return
        batch, heads, _, dim = self.original_shape
        self.original_shape = (batch, heads, self.total_tokens(), dim)

    def append_tokens(self, x: torch.Tensor, cfg: TurboQuantHConfig) -> None:
        tokens = x.to(self.storage_dtype)
        if self.recent_values is None:
            self.recent_values = tokens
        else:
            self.recent_values = torch.cat([self.recent_values, tokens], dim=2)

        while token_length(self.recent_values) > cfg.recent_fp_tokens:
            overflow = self.recent_values[:, :, :1, :]
            self.recent_values = strip_empty_tokens(self.recent_values[:, :, 1:, :])
            if self.pending_old_values is None:
                self.pending_old_values = overflow
            else:
                self.pending_old_values = torch.cat([self.pending_old_values, overflow], dim=2)

            while token_length(self.pending_old_values) >= cfg.block_size:
                block = self.pending_old_values[:, :, : cfg.block_size, :].to(torch.float32)
                self.pending_old_values = strip_empty_tokens(self.pending_old_values[:, :, cfg.block_size :, :])
                block_seed = self.seed_base + len(self.old_blocks) * 9973
                self.old_blocks.append(
                    compress_block(
                        block,
                        cfg,
                        seed=block_seed,
                        tensor_kind=self.tensor_kind,
                        storage_dtype=self.storage_dtype,
                    )
                )

        self.pending_old_values = strip_empty_tokens(self.pending_old_values)
        self.recent_values = strip_empty_tokens(self.recent_values)
        self._sync_shape()

    def materialize(self, device: torch.device) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for block in self.old_blocks:
            pieces.append(block.materialize(device=device))
        if self.pending_old_values is not None:
            pieces.append(self.pending_old_values.to(device=device, dtype=torch.float32))
        if self.recent_values is not None:
            pieces.append(self.recent_values.to(device=device, dtype=torch.float32))

        if pieces:
            return torch.cat(pieces, dim=2)

        batch, heads, _, dim = self.original_shape
        return torch.empty((batch, heads, 0, dim), device=device, dtype=torch.float32)

    def storage_bits(self) -> int:
        bits = sum(block.storage_bits() for block in self.old_blocks)
        if self.pending_old_values is not None:
            bits += int(self.pending_old_values.numel() * self.pending_old_values.element_size() * 8)
        if self.recent_values is not None:
            bits += int(self.recent_values.numel() * self.recent_values.element_size() * 8)
        return bits

    def select_batch(self, beam_idx: torch.LongTensor) -> None:
        for block in self.old_blocks:
            block.select_batch(beam_idx)
        if self.pending_old_values is not None:
            self.pending_old_values = self.pending_old_values.index_select(0, beam_idx)
        if self.recent_values is not None:
            self.recent_values = self.recent_values.index_select(0, beam_idx)
        self._sync_shape()


def compress_segment(x: torch.Tensor, cfg: TurboQuantHConfig, seed_base: int, tensor_kind: str) -> CompressedSegment:
    if x.dim() != 4:
        raise ValueError(f"Expected [B,H,T,D], got {tuple(x.shape)}")

    batch, heads, tokens, dim = x.shape
    recent_len = min(cfg.recent_fp_tokens, tokens)
    old_len = max(tokens - recent_len, 0)
    old = x[:, :, :old_len, :].to(torch.float32)
    recent = x[:, :, old_len:, :].to(x.dtype) if recent_len > 0 else None

    blocks: list[CompressedBlock] = []
    num_full_blocks = old_len // cfg.block_size
    for block_idx in range(num_full_blocks):
        start = block_idx * cfg.block_size
        stop = start + cfg.block_size
        block_seed = seed_base + block_idx * 9973
        blocks.append(
            compress_block(
                old[:, :, start:stop, :],
                cfg,
                seed=block_seed,
                tensor_kind=tensor_kind,
                storage_dtype=x.dtype,
            )
        )

    pending = None
    if old_len > num_full_blocks * cfg.block_size:
        pending = old[:, :, num_full_blocks * cfg.block_size :, :].to(x.dtype)

    segment = CompressedSegment(
        old_blocks=blocks,
        pending_old_values=pending,
        recent_values=recent,
        original_shape=(batch, heads, tokens, dim),
        seed_base=seed_base,
        tensor_kind=tensor_kind,
        storage_dtype=x.dtype,
    )
    segment._sync_shape()
    return segment


def decompress_segment(seg: CompressedSegment, device: torch.device) -> torch.Tensor:
    return seg.materialize(device=device)


class CompressedCacheLayer:
    def __init__(
        self,
        layer_idx: int,
        key_segment: CompressedSegment,
        value_segment: CompressedSegment,
        cfg: TurboQuantHConfig,
        latency: Optional[LatencyStats] = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.key_segment = key_segment
        self.value_segment = value_segment
        self.cfg = cfg
        self.latency = latency

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        start_time = time.perf_counter()
        self.key_segment.append_tokens(key_states, self.cfg)
        self.value_segment.append_tokens(value_states, self.cfg)
        if self.latency is not None:
            self.latency.compression_s += time.perf_counter() - start_time
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.key_segment.total_tokens()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_max_cache_shape(self) -> int:
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._select_batch(beam_idx)

    def _select_batch(self, beam_idx: torch.LongTensor) -> None:
        self.key_segment.select_batch(beam_idx)
        self.value_segment.select_batch(beam_idx)

    def iter_attention_entries(self) -> list[tuple[str, Any, Any]]:
        pieces: list[tuple[str, Any, Any]] = []
        for key_block, value_block in zip(self.key_segment.old_blocks, self.value_segment.old_blocks):
            pieces.append(("compressed", key_block, value_block))
        if self.key_segment.pending_old_values is not None:
            pieces.append(("dense", self.key_segment.pending_old_values, self.value_segment.pending_old_values))
        if self.key_segment.recent_values is not None:
            pieces.append(("dense", self.key_segment.recent_values, self.value_segment.recent_values))
        return pieces


class CompressedCache(Cache):
    def __init__(
        self,
        layers: list[CompressedCacheLayer],
        cfg: TurboQuantHConfig,
        baseline_dtype_bits: int,
        latency: Optional[LatencyStats] = None,
    ) -> None:
        super().__init__(layers=layers)
        self.cfg = cfg
        self.baseline_dtype_bits = baseline_dtype_bits
        self.latency = latency
        for layer in self.layers:
            layer.latency = latency

    @classmethod
    def from_past_key_values(
        cls,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        cfg: TurboQuantHConfig,
        baseline_dtype_bits: int,
        latency: Optional[LatencyStats] = None,
    ) -> "CompressedCache":
        layers: list[CompressedCacheLayer] = []
        entries = list(past_key_values)
        for layer_idx, (key, value) in enumerate(entries):
            key_seed = cfg.random_seed + layer_idx * 200003 + 17
            value_seed = cfg.random_seed + layer_idx * 200003 + 101
            layers.append(
                CompressedCacheLayer(
                    layer_idx=layer_idx,
                    key_segment=compress_segment(key, cfg, seed_base=key_seed, tensor_kind="k"),
                    value_segment=compress_segment(value, cfg, seed_base=value_seed, tensor_kind="v"),
                    cfg=cfg,
                    latency=latency,
                )
            )
        return cls(layers=layers, cfg=cfg, baseline_dtype_bits=baseline_dtype_bits, latency=latency)

    def to_dense_legacy_cache(self, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        dense_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            key = layer.key_segment.materialize(device=device)
            value = layer.value_segment.materialize(device=device)
            dense_layers.append((key, value))
        return dense_layers

    def estimate_baseline_bits(self) -> int:
        bits = 0
        for layer in self.layers:
            for segment in (layer.key_segment, layer.value_segment):
                batch, heads, tokens, dim = segment.original_shape
                bits += int(batch * heads * tokens * dim * self.baseline_dtype_bits)
        return bits

    def estimate_compressed_bits(self) -> int:
        bits = 0
        for layer in self.layers:
            bits += layer.key_segment.storage_bits()
            bits += layer.value_segment.storage_bits()
        return bits


def compress_past_key_values(
    past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    cfg: TurboQuantHConfig,
    baseline_dtype_bits: int,
    latency: Optional[LatencyStats] = None,
) -> CompressedCache:
    start_time = time.perf_counter()
    compressed = CompressedCache.from_past_key_values(
        past_key_values=past_key_values,
        cfg=cfg,
        baseline_dtype_bits=baseline_dtype_bits,
        latency=latency,
    )
    if latency is not None:
        latency.compression_s += time.perf_counter() - start_time
    return compressed


def decompress_past_key_values(
    compressed: CompressedCache,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    start_time = time.perf_counter()
    dense = compressed.to_dense_legacy_cache(device=device)
    if compressed.latency is not None:
        compressed.latency.dense_decompression_s += time.perf_counter() - start_time
    return dense


def estimate_compressed_bits(compressed: CompressedCache) -> tuple[int, int]:
    return compressed.estimate_baseline_bits(), compressed.estimate_compressed_bits()
