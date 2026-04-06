# Runtime Flow

This document shows how one benchmark run moves through the codebase.

## High-Level Flow

```text
wrapper script
  -> CLI parser
  -> config objects
  -> benchmark runner
  -> model prefill
  -> cache compression
  -> decode loop
  -> report formatting
  -> terminal output
```

## Step-By-Step Walkthrough

### 1. Entrypoint

File: `turboquant_h_smollm_benchmark.py`

Purpose:

- preserve the old script name,
- add `src/` to `sys.path`,
- delegate to `turboquant_h.cli.main()`.

This file is intentionally tiny. It exists for convenience, not business logic.

### 2. CLI Parsing

File: `src/turboquant_h/cli.py`

Important functions:

- `build_parser()`
- `build_configs()`
- `main()`

What happens:

1. command-line arguments are parsed,
2. runtime settings become a `RuntimeConfig`,
3. compression settings become a `TurboQuantHConfig`,
4. both configs are validated,
5. `run_benchmark(...)` is called.

### 3. Benchmark Setup

File: `src/turboquant_h/benchmark.py`

Important functions:

- `select_device()`
- `select_model_dtype()`
- `load_model_and_tokenizer()`
- `run_benchmark()`

What happens:

1. choose CPU or CUDA,
2. pick the model dtype,
3. load tokenizer and model,
4. seed PyTorch,
5. start timing,
6. call `generate_with_compressed_cache(...)`.

### 4. Prefill Pass

Still in `src/turboquant_h/benchmark.py`

Important function:

- `generate_with_compressed_cache()`

What happens:

1. the prompt is converted into model input text,
2. a prefill forward pass is run with `use_cache=True`,
3. `past_key_values` from the dense model output are collected,
4. baseline dtype size is measured,
5. dense cache is converted into a compressed cache.

### 5. Cache Compression

Main file:

- `src/turboquant_h/compression/cache.py`

Supporting files:

- `rotation.py`
- `quantization.py`
- `correction.py`
- `packing.py`

What happens conceptually:

1. split cache tensors into recent and old regions,
2. keep recent tokens in higher precision,
3. chop older regions into blocks,
4. rotate blocks if enabled,
5. quantize blocks into packed low-bit form,
6. optionally store correction information,
7. optionally retain selected salient tokens in higher precision,
8. store everything in compressed cache structures.

### 6. Decode Loop

File: `src/turboquant_h/benchmark.py`

For every generated token:

1. sample the next token from logits,
2. append it to the generated sequence,
3. run the next forward pass using either:
   - direct compressed attention, or
   - dense reconstruction of the cache first.

### 7. Direct Compressed Attention Path

File: `src/turboquant_h/compression/attention.py`

Important functions:

- `enable_compressed_attention()`
- `compressed_eager_attention_forward()`

What happens:

1. compatible attention modules are patched,
2. the patched attention layer receives a `CompressedCache`,
3. cached blocks are streamed piece by piece,
4. attention logits are accumulated without fully rebuilding a dense cache,
5. optional QJL score-space correction is applied when enabled.

### 8. Dense Reconstruction Path

Files:

- `src/turboquant_h/benchmark.py`
- `src/turboquant_h/compression/cache.py`

If direct compressed attention is disabled:

1. compressed cache is materialized back into dense tensors,
2. the model runs a normal forward step with dense `past_key_values`,
3. the new dense cache is compressed again for the next step.

This path is easier to reason about, but less efficient.

### 9. Final Result Assembly

Files:

- `src/turboquant_h/benchmark.py`
- `src/turboquant_h/reporting.py`

What happens:

1. baseline and compressed bit counts are estimated,
2. generated token ids are decoded into text,
3. a `BenchmarkResult` object is built,
4. `format_benchmark_report(...)` turns it into readable output,
5. the CLI prints the final report.

## Mental Model For Debugging

When something breaks, it usually helps to ask which layer the issue belongs to:

- argument or configuration issue: `cli.py` or `config.py`
- model loading or decode-loop issue: `benchmark.py`
- wrong cache size or weird reconstruction: `cache.py`
- strange bit-level values: `packing.py`
- strange rotated values: `rotation.py`
- quantization quality issue: `quantization.py`
- correction issue: `correction.py`
- direct compressed attention issue: `attention.py`
