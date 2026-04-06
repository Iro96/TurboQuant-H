# Project Map

This document is the "big picture" view of the repository. If you are new to the project, start here before reading the implementation details.

## One-Sentence Summary

TurboQuant-H is a research benchmark harness for testing hierarchical KV-cache compression ideas on Hugging Face causal language models.

## Top-Level Repository Layout

```text
TurboQuant-H/
|-- docs/
|   |-- README.md
|   |-- project-map.md
|   |-- runtime-flow.md
|   `-- compression-guide.md
|-- src/
|   `-- turboquant_h/
|       |-- __init__.py
|       |-- benchmark.py
|       |-- cli.py
|       |-- config.py
|       |-- reporting.py
|       `-- compression/
|           |-- __init__.py
|           |-- attention.py
|           |-- cache.py
|           |-- common.py
|           |-- correction.py
|           |-- packing.py
|           |-- quantization.py
|           `-- rotation.py
|-- tests/
|   |-- __init__.py
|   |-- test_config.py
|   `-- test_packing.py
|-- README.md
|-- pyproject.toml
|-- TurboQuant-H_paper_draft.md
`-- turboquant_h_smollm_benchmark.py
```

## What Each Top-Level Part Means

### `docs/`

The human guidebook for this repository. This folder exists to make onboarding easier and reduce the need to reverse-engineer the code from scratch.

### `src/turboquant_h/`

The real Python package. This is where the actual application logic lives.

### `tests/`

Small unit tests for stable utilities. Right now the tests focus on config behavior and bit-packing correctness.

### `README.md`

The quick public-facing project overview. It is shorter than the docs in this folder.

### `pyproject.toml`

The package metadata and script entrypoint definition. If you want to install this project or expose the CLI as `turboquant-h-benchmark`, this file is part of that setup.

### `TurboQuant-H_paper_draft.md`

Research context and design framing. This is not runtime code, but it helps explain the ideas the benchmark is trying to explore.

### `turboquant_h_smollm_benchmark.py`

The compatibility wrapper. It keeps the old entrypoint name working, but the real code now lives in the package under `src/turboquant_h/`.

## Package-Level Map

### `src/turboquant_h/__init__.py`

Package exports. This file exposes the main public objects and functions so other code can import them more cleanly.

### `src/turboquant_h/cli.py`

The command-line interface layer.

What it does:

- defines CLI arguments,
- groups them into runtime, compression, correction, sampling, and per-tensor override options,
- builds validated config objects,
- calls the benchmark runner,
- prints the formatted report.

If you want to add a new CLI flag, this is usually the first file to touch.

### `src/turboquant_h/config.py`

The source of truth for structured settings and benchmark results.

What lives here:

- allowed option values,
- `TurboQuantHConfig`,
- `RuntimeConfig`,
- `LatencyStats`,
- `BenchmarkResult`,
- validation logic,
- per-key/per-value override resolution.

If you want to add a new configurable behavior, this is usually the second file to touch after `cli.py`.

### `src/turboquant_h/benchmark.py`

The orchestration layer.

What it does:

- selects device and dtype,
- loads the tokenizer and model,
- runs prefill,
- creates the compressed cache,
- runs the decode loop,
- samples next tokens,
- returns the final benchmark result object.

If `cli.py` is the front door, `benchmark.py` is the conductor.

### `src/turboquant_h/reporting.py`

The presentation layer for terminal output. It turns a `BenchmarkResult` into a readable benchmark report.

### `src/turboquant_h/compression/`

The core subsystem. Everything inside this folder exists to compress, store, reconstruct, or directly attend over KV-cache data.

Read [Compression Guide](./compression-guide.md) for the detailed map.

## Best Reading Order For New People

1. `README.md`
2. `docs/project-map.md`
3. `src/turboquant_h/cli.py`
4. `src/turboquant_h/config.py`
5. `src/turboquant_h/benchmark.py`
6. `docs/compression-guide.md`
7. `src/turboquant_h/compression/cache.py`
8. `src/turboquant_h/compression/attention.py`

## If You Need To Change Something, Start Here

- Add or change CLI options: `src/turboquant_h/cli.py`
- Add config fields or validation: `src/turboquant_h/config.py`
- Change benchmark flow: `src/turboquant_h/benchmark.py`
- Change how output is printed: `src/turboquant_h/reporting.py`
- Change low-bit packing: `src/turboquant_h/compression/packing.py`
- Change rotations: `src/turboquant_h/compression/rotation.py`
- Change quantization behavior: `src/turboquant_h/compression/quantization.py`
- Change residual correction behavior: `src/turboquant_h/compression/correction.py`
- Change cache storage or reconstruction: `src/turboquant_h/compression/cache.py`
- Change direct compressed attention: `src/turboquant_h/compression/attention.py`
