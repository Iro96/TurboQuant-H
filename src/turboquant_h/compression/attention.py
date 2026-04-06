"""Compressed attention execution and model patching."""

from __future__ import annotations

import math
import time
import types
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .cache import CompressedCache, CompressedCacheLayer
from .common import numel_from_shape
from .correction import CorrectionState, make_qjl_projection
from .packing import unpack_lowbit_codes
from .rotation import RotationState, decode_rotation_signs, fwht_last_dim


def _rotate_query_for_block(
    query: torch.Tensor,
    rotation: RotationState,
    num_key_value_groups: int,
) -> torch.Tensor:
    query_rot = query.to(torch.float32)
    if rotation.mode == "none":
        return query_rot
    signs = decode_rotation_signs(rotation, device=query.device, dtype=torch.float32)
    if signs is not None:
        query_rot = query_rot * repeat_kv(signs, num_key_value_groups)
    return fwht_last_dim(query_rot) / math.sqrt(query.shape[-1])


def _qjl_score_correction(
    query_rot: torch.Tensor,
    correction: CorrectionState,
    num_key_value_groups: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if (
        correction.kind != "qjl"
        or correction.qjl_shape is None
        or correction.qjl_packed_signs is None
        or correction.qjl_scale is None
    ):
        return None

    num_values = numel_from_shape(correction.qjl_shape)
    signs = unpack_lowbit_codes(correction.qjl_packed_signs.to(device), 1, num_values)
    signs = signs.reshape(correction.qjl_shape).to(torch.float32) * 2.0 - 1.0
    signs = repeat_kv(signs, num_key_value_groups)
    scales = repeat_kv(correction.qjl_scale.to(device=device, dtype=torch.float32), num_key_value_groups)

    projection = make_qjl_projection(
        dim=query_rot.shape[-1],
        proj_dim=correction.qjl_dim,
        device=device,
        dtype=torch.float32,
        seed=correction.qjl_seed,
    )
    q_proj = torch.matmul(query_rot.to(torch.float32), projection)
    logits = torch.einsum("bhqp,bhtp->bhqt", q_proj, signs)
    return logits * scales.squeeze(-1).unsqueeze(-2)


def compressed_eager_attention_forward(
    module: Any,
    query: torch.Tensor,
    cache_layer: CompressedCacheLayer,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if dropout != 0.0:
        raise NotImplementedError("Compressed streaming attention is implemented for inference-only benchmarking.")

    pieces = cache_layer.iter_attention_entries()
    if not pieces:
        raise RuntimeError("Compressed cache layer has no attention entries")

    start_time = time.perf_counter()
    device = query.device
    batch, num_heads, query_tokens, head_dim = query.shape
    running_max = torch.full((batch, num_heads, query_tokens), -torch.inf, device=device, dtype=torch.float32)
    running_norm = torch.zeros((batch, num_heads, query_tokens), device=device, dtype=torch.float32)
    running_out = torch.zeros((batch, num_heads, query_tokens, head_dim), device=device, dtype=torch.float32)

    base_query = query.to(torch.float32)
    offset = 0

    for piece_type, key_piece, value_piece in pieces:
        if piece_type == "compressed":
            query_chunk = _rotate_query_for_block(base_query, key_piece.rotation, module.num_key_value_groups)
            key_states = key_piece.materialize_rotated_for_attention(device=device)
            logits = torch.matmul(
                query_chunk,
                repeat_kv(key_states.to(device=device, dtype=torch.float32), module.num_key_value_groups).transpose(2, 3),
            ) * scaling
            score_correction = _qjl_score_correction(
                query_chunk,
                key_piece.correction,
                module.num_key_value_groups,
                device=device,
            )
            if score_correction is not None:
                logits = logits + score_correction
            value_states = value_piece.materialize(device=device, dtype=torch.float32)
        else:
            key_states = key_piece.to(device=device, dtype=torch.float32)
            value_states = value_piece.to(device=device, dtype=torch.float32)
            logits = torch.matmul(
                base_query,
                repeat_kv(key_states, module.num_key_value_groups).transpose(2, 3),
            ) * scaling

        length = key_states.shape[-2]
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, offset : offset + length]

        chunk_max = logits.max(dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)
        prev_scale = torch.exp(running_max - new_max)
        exp_logits = torch.exp(logits - new_max.unsqueeze(-1))
        repeated_value = repeat_kv(value_states, module.num_key_value_groups)
        running_out = running_out * prev_scale.unsqueeze(-1) + torch.matmul(exp_logits, repeated_value)
        running_norm = running_norm * prev_scale + exp_logits.sum(dim=-1)
        running_max = new_max
        offset += length

    attn_output = running_out / running_norm.unsqueeze(-1).clamp(min=1e-8)
    if cache_layer.latency is not None:
        cache_layer.latency.compressed_attention_s += time.perf_counter() - start_time
    return attn_output.transpose(1, 2).contiguous().to(query.dtype), None


def enable_compressed_attention(model: Any) -> bool:
    if getattr(model, "_turboquant_h_compressed_attention_enabled", False):
        return True

    patched_count = 0

    def make_patched_forward(original_forward: Any):
        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if not isinstance(past_key_values, CompressedCache):
                return original_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            cache_layer = past_key_values.layers[self.layer_idx]
            cache_layer.update(key_states, value_states, {"cache_position": cache_position})

            attn_output, attn_weights = compressed_eager_attention_forward(
                self,
                query_states,
                cache_layer,
                attention_mask,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        return patched_forward

    for module in model.modules():
        required_attrs = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "layer_idx",
            "num_key_value_groups",
            "head_dim",
            "scaling",
        )
        if "attention" not in module.__class__.__name__.lower():
            continue
        if not all(hasattr(module, attr) for attr in required_attrs):
            continue
        if getattr(module, "_turboquant_h_patched", False):
            continue

        original_forward = module.forward
        module._turboquant_h_original_forward = original_forward
        module.forward = types.MethodType(make_patched_forward(original_forward), module)
        module._turboquant_h_patched = True
        patched_count += 1

    model._turboquant_h_compressed_attention_enabled = patched_count > 0
    return patched_count > 0

