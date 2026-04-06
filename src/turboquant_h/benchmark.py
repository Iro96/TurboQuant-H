"""End-to-end benchmark execution."""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compression.attention import enable_compressed_attention
from .compression.cache import compress_past_key_values, decompress_past_key_values, estimate_compressed_bits
from .config import BenchmarkResult, LatencyStats, RuntimeConfig, TurboQuantHConfig


def select_device(force_cpu: bool) -> torch.device:
    return torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")


def select_model_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def load_model_and_tokenizer(runtime_cfg: RuntimeConfig) -> tuple[Any, Any, torch.device]:
    device = select_device(runtime_cfg.force_cpu)
    dtype = select_model_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(runtime_cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(runtime_cfg.model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device


def _sample_next_token(next_logits: torch.Tensor, cfg: TurboQuantHConfig) -> torch.Tensor:
    if not cfg.do_sample:
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    logits = next_logits / max(cfg.temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= cfg.top_p
    mask[..., 0] = True
    filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sampled = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)


@torch.no_grad()
def generate_with_compressed_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    cfg: TurboQuantHConfig,
    max_new_tokens: int = 64,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, int, int, bool, LatencyStats]:
    if device is None:
        device = next(model.parameters()).device

    latency = LatencyStats()

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    use_direct_attention = cfg.use_direct_compressed_attention and enable_compressed_attention(model)
    if cfg.resolved_correction_type("k") == "qjl" and not use_direct_attention:
        raise ValueError("QJL key correction requires direct compressed attention.")

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    start_time = time.perf_counter()
    out = model(**inputs, use_cache=True, return_dict=True)
    latency.prefill_s += time.perf_counter() - start_time
    past = out.past_key_values
    baseline_dtype_bits = int(next(iter(past))[0].element_size() * 8)
    compressed_cache = compress_past_key_values(
        past,
        cfg,
        baseline_dtype_bits=baseline_dtype_bits,
        latency=latency,
    )

    generated = inputs["input_ids"]
    next_logits = out.logits[:, -1, :]

    for _ in range(max_new_tokens):
        sample_start = time.perf_counter()
        next_token = _sample_next_token(next_logits, cfg)
        latency.sampling_s += time.perf_counter() - sample_start

        generated = torch.cat([generated, next_token], dim=-1)

        if use_direct_attention:
            forward_start = time.perf_counter()
            out = model(
                input_ids=next_token,
                past_key_values=compressed_cache,
                use_cache=True,
                return_dict=True,
            )
            latency.decode_forward_s += time.perf_counter() - forward_start
            compressed_cache = out.past_key_values
        else:
            dense_past = decompress_past_key_values(compressed_cache, device=device)
            forward_start = time.perf_counter()
            out = model(
                input_ids=next_token,
                past_key_values=dense_past,
                use_cache=True,
                return_dict=True,
            )
            latency.decode_forward_s += time.perf_counter() - forward_start
            compressed_cache = compress_past_key_values(
                out.past_key_values,
                cfg,
                baseline_dtype_bits=baseline_dtype_bits,
                latency=latency,
            )

        next_logits = out.logits[:, -1, :]

    baseline_bits, compressed_bits = estimate_compressed_bits(compressed_cache)
    return generated, baseline_bits, compressed_bits, use_direct_attention, latency


def run_benchmark(runtime_cfg: RuntimeConfig, compression_cfg: TurboQuantHConfig) -> BenchmarkResult:
    runtime_cfg.validate()
    compression_cfg.validate()
    torch.manual_seed(compression_cfg.random_seed)

    model, tokenizer, device = load_model_and_tokenizer(runtime_cfg)

    start_time = time.time()
    generated, baseline_bits, compressed_bits, used_direct_attention, latency = generate_with_compressed_cache(
        model,
        tokenizer,
        runtime_cfg.prompt,
        compression_cfg,
        max_new_tokens=runtime_cfg.max_new_tokens,
        device=device,
    )
    wall_time = time.time() - start_time

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return BenchmarkResult(
        text=text,
        baseline_bits=baseline_bits,
        compressed_bits=compressed_bits,
        used_direct_attention=used_direct_attention,
        latency=latency,
        wall_time_s=wall_time,
        device=device.type,
        model_name=runtime_cfg.model_name,
        prompt=runtime_cfg.prompt,
        max_new_tokens=runtime_cfg.max_new_tokens,
    )
