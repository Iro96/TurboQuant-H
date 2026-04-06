"""Terminal-friendly benchmark reporting helpers."""

from __future__ import annotations

from textwrap import shorten

from .config import BenchmarkResult


def _format_prompt(prompt: str, width: int = 88) -> str:
    return shorten(" ".join(prompt.split()), width=width, placeholder="...")


def format_benchmark_report(result: BenchmarkResult) -> str:
    lines = [
        "",
        "=== Output ===",
        "",
        result.text,
        "",
        "=== Run Summary ===",
        f"Model:                   {result.model_name}",
        f"Device:                  {result.device}",
        f"Prompt:                  {_format_prompt(result.prompt)}",
        f"Max new tokens:          {result.max_new_tokens}",
        f"Direct compressed attn:  {'enabled' if result.used_direct_attention else 'disabled'}",
        "",
        "=== Compression Estimate ===",
        f"Baseline bits:           {result.baseline_bits:,}",
        f"Compressed bits:         {result.compressed_bits:,}",
        f"Estimated ratio:         {result.compression_ratio:.2f}x",
        f"Wall time:               {result.wall_time_s:.2f}s",
        "",
        "=== Latency Breakdown ===",
        f"Prefill forward:         {result.latency.prefill_s:.4f}s",
        f"Decode forward:          {result.latency.decode_forward_s:.4f}s",
        f"Compression:             {result.latency.compression_s:.4f}s",
        f"Dense decompression:     {result.latency.dense_decompression_s:.4f}s",
        f"Compressed attention:    {result.latency.compressed_attention_s:.4f}s",
        f"Sampling:                {result.latency.sampling_s:.4f}s",
        "",
        "Note: this remains a research harness. Storage is bit-packed, correction is configurable,",
        "and decode can stream over compressed KV blocks without rebuilding a dense past cache.",
    ]
    return "\n".join(lines)
