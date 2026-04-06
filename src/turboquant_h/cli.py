"""Command-line interface for the TurboQuant-H benchmark."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .benchmark import run_benchmark
from .config import (
    CORRECTION_TYPES,
    QUANTIZERS,
    QUANT_SCALES,
    ROTATION_MODES,
    SCALE_ESTIMATORS,
    RuntimeConfig,
    TurboQuantHConfig,
)
from .reporting import format_benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turboquant-h-benchmark",
        description="Benchmark TurboQuant-H KV-cache compression with a Hugging Face causal language model.",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--model", default=RuntimeConfig.model_name)
    runtime_group.add_argument("--prompt", default=RuntimeConfig.prompt)
    runtime_group.add_argument("--max_new_tokens", type=int, default=RuntimeConfig.max_new_tokens)
    runtime_group.add_argument("--cpu", action="store_true")

    compression_group = parser.add_argument_group("compression")
    compression_group.add_argument("--recent_fp_tokens", type=int, default=TurboQuantHConfig.recent_fp_tokens)
    compression_group.add_argument("--block_size", type=int, default=TurboQuantHConfig.block_size)
    compression_group.add_argument("--keep_ratio_old", type=float, default=TurboQuantHConfig.keep_ratio_old)
    compression_group.add_argument("--quant_bits_old", type=int, default=TurboQuantHConfig.quant_bits_old)
    compression_group.add_argument("--rotation_mode", choices=ROTATION_MODES, default=TurboQuantHConfig.rotation_mode)
    compression_group.add_argument("--quantizer", choices=QUANTIZERS, default=TurboQuantHConfig.quantizer)
    compression_group.add_argument("--quant_scale", choices=QUANT_SCALES, default=TurboQuantHConfig.quant_scale)
    compression_group.add_argument("--disable_direct_compressed_attention", action="store_true")
    compression_group.add_argument("--random_seed", type=int, default=TurboQuantHConfig.random_seed)

    correction_group = parser.add_argument_group("correction")
    correction_group.add_argument("--correction_type", choices=CORRECTION_TYPES, default=TurboQuantHConfig.correction_type)
    correction_group.add_argument("--correction_rank", type=int, default=TurboQuantHConfig.correction_rank)
    correction_group.add_argument("--qjl_dim", type=int, default=TurboQuantHConfig.qjl_dim)
    correction_group.add_argument("--scale_estimator", choices=SCALE_ESTIMATORS, default=TurboQuantHConfig.scale_estimator)
    correction_group.add_argument("--scale_quantile", type=float, default=TurboQuantHConfig.scale_quantile)
    correction_group.add_argument("--low_rank_oversample", type=int, default=TurboQuantHConfig.low_rank_oversample)
    correction_group.add_argument("--low_rank_power_iters", type=int, default=TurboQuantHConfig.low_rank_power_iters)

    sampling_group = parser.add_argument_group("sampling")
    sampling_group.add_argument("--temperature", type=float, default=TurboQuantHConfig.temperature)
    sampling_group.add_argument("--top_p", type=float, default=TurboQuantHConfig.top_p)
    sampling_group.add_argument("--no_sample", action="store_true")

    override_group = parser.add_argument_group("per-tensor overrides")
    override_group.add_argument("--key_quantizer", choices=QUANTIZERS)
    override_group.add_argument("--value_quantizer", choices=QUANTIZERS)
    override_group.add_argument("--key_quant_scale", choices=QUANT_SCALES)
    override_group.add_argument("--value_quant_scale", choices=QUANT_SCALES)
    override_group.add_argument("--key_correction_type", choices=CORRECTION_TYPES)
    override_group.add_argument("--value_correction_type", choices=CORRECTION_TYPES)

    return parser


def build_configs(args: argparse.Namespace) -> tuple[RuntimeConfig, TurboQuantHConfig]:
    runtime_cfg = RuntimeConfig(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        force_cpu=args.cpu,
    )
    compression_cfg = TurboQuantHConfig(
        recent_fp_tokens=args.recent_fp_tokens,
        block_size=args.block_size,
        quant_bits_old=args.quant_bits_old,
        keep_ratio_old=args.keep_ratio_old,
        rotation_mode=args.rotation_mode,
        quantizer=args.quantizer,
        quant_scale=args.quant_scale,
        key_quantizer=args.key_quantizer,
        value_quantizer=args.value_quantizer,
        key_quant_scale=args.key_quant_scale,
        value_quant_scale=args.value_quant_scale,
        correction_type=args.correction_type,
        key_correction_type=args.key_correction_type,
        value_correction_type=args.value_correction_type,
        correction_rank=args.correction_rank,
        qjl_dim=args.qjl_dim,
        scale_estimator=args.scale_estimator,
        scale_quantile=args.scale_quantile,
        low_rank_oversample=args.low_rank_oversample,
        low_rank_power_iters=args.low_rank_power_iters,
        use_direct_compressed_attention=not args.disable_direct_compressed_attention,
        random_seed=args.random_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )
    runtime_cfg.validate()
    compression_cfg.validate()
    return runtime_cfg, compression_cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime_cfg, compression_cfg = build_configs(args)
    result = run_benchmark(runtime_cfg, compression_cfg)
    print(format_benchmark_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
