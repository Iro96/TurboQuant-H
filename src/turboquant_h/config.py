"""Configuration and result objects for the TurboQuant-H benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


ROTATION_MODES = ("none", "hadamard", "random_hadamard")
QUANTIZERS = ("uniform", "codebook")
QUANT_SCALES = ("head", "channel")
CORRECTION_TYPES = ("none", "low_rank", "qjl")
SCALE_ESTIMATORS = ("absmax", "abs_quantile", "rms")


def _validate_choice(value: Optional[str], allowed: tuple[str, ...], name: str) -> None:
    if value is not None and value not in allowed:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {allowed_values}. Got {value!r}.")


@dataclass(slots=True)
class TurboQuantHConfig:
    recent_fp_tokens: int = 32
    block_size: int = 16
    quant_bits_old: int = 2
    keep_ratio_old: float = 0.08
    rotation_mode: str = "hadamard"
    quantizer: str = "uniform"
    quant_scale: str = "channel"
    key_quantizer: Optional[str] = None
    value_quantizer: Optional[str] = None
    key_quant_scale: Optional[str] = None
    value_quant_scale: Optional[str] = None
    correction_type: str = "low_rank"
    key_correction_type: Optional[str] = None
    value_correction_type: Optional[str] = None
    correction_rank: int = 4
    qjl_dim: int = 16
    scale_estimator: str = "abs_quantile"
    scale_quantile: float = 0.995
    low_rank_oversample: int = 2
    low_rank_power_iters: int = 1
    use_direct_compressed_attention: bool = True
    random_seed: int = 0
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True

    def resolved_quantizer(self, tensor_kind: str) -> str:
        if tensor_kind == "k" and self.key_quantizer is not None:
            return self.key_quantizer
        if tensor_kind == "v" and self.value_quantizer is not None:
            return self.value_quantizer
        return self.quantizer

    def resolved_quant_scale(self, tensor_kind: str) -> str:
        if tensor_kind == "k" and self.key_quant_scale is not None:
            return self.key_quant_scale
        if tensor_kind == "v" and self.value_quant_scale is not None:
            return self.value_quant_scale
        return self.quant_scale

    def resolved_correction_type(self, tensor_kind: str) -> str:
        if tensor_kind == "k" and self.key_correction_type is not None:
            return self.key_correction_type
        if tensor_kind == "v" and self.value_correction_type is not None:
            return self.value_correction_type
        if tensor_kind == "v" and self.correction_type == "qjl":
            return "low_rank"
        return self.correction_type

    def validate(self) -> None:
        if self.recent_fp_tokens < 0:
            raise ValueError("recent_fp_tokens must be >= 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if not 0.0 <= self.keep_ratio_old <= 1.0:
            raise ValueError("keep_ratio_old must be between 0 and 1")
        if not 1 <= self.quant_bits_old <= 8:
            raise ValueError("quant_bits_old must be between 1 and 8")
        if self.correction_rank < 0:
            raise ValueError("correction_rank must be >= 0")
        if self.qjl_dim <= 0:
            raise ValueError("qjl_dim must be > 0")
        if self.low_rank_oversample < 0:
            raise ValueError("low_rank_oversample must be >= 0")
        if self.low_rank_power_iters < 0:
            raise ValueError("low_rank_power_iters must be >= 0")
        if not 0.0 < self.scale_quantile <= 1.0:
            raise ValueError("scale_quantile must be within (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be within (0, 1]")

        _validate_choice(self.rotation_mode, ROTATION_MODES, "rotation_mode")
        _validate_choice(self.quantizer, QUANTIZERS, "quantizer")
        _validate_choice(self.quant_scale, QUANT_SCALES, "quant_scale")
        _validate_choice(self.key_quantizer, QUANTIZERS, "key_quantizer")
        _validate_choice(self.value_quantizer, QUANTIZERS, "value_quantizer")
        _validate_choice(self.key_quant_scale, QUANT_SCALES, "key_quant_scale")
        _validate_choice(self.value_quant_scale, QUANT_SCALES, "value_quant_scale")
        _validate_choice(self.correction_type, CORRECTION_TYPES, "correction_type")
        _validate_choice(self.key_correction_type, CORRECTION_TYPES, "key_correction_type")
        _validate_choice(self.value_correction_type, CORRECTION_TYPES, "value_correction_type")
        _validate_choice(self.scale_estimator, SCALE_ESTIMATORS, "scale_estimator")


@dataclass(slots=True)
class RuntimeConfig:
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    prompt: str = "Explain why KV cache compression matters in one paragraph."
    max_new_tokens: int = 48
    force_cpu: bool = False

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")


@dataclass(slots=True)
class LatencyStats:
    prefill_s: float = 0.0
    decode_forward_s: float = 0.0
    compression_s: float = 0.0
    dense_decompression_s: float = 0.0
    compressed_attention_s: float = 0.0
    sampling_s: float = 0.0


@dataclass(slots=True)
class BenchmarkResult:
    text: str
    baseline_bits: int
    compressed_bits: int
    used_direct_attention: bool
    latency: LatencyStats = field(default_factory=LatencyStats)
    wall_time_s: float = 0.0
    device: str = "cpu"
    model_name: str = ""
    prompt: str = ""
    max_new_tokens: int = 0

    @property
    def compression_ratio(self) -> float:
        return self.baseline_bits / max(self.compressed_bits, 1)
