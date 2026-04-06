# TurboQuant-H

TurboQuant-H is a research-oriented benchmark harness for experimenting with hierarchical KV-cache compression. A stronger hybrid design that keeps the mathematical core of TurboQuant. This algorithm uses data-oblivious, two-stage vector quantization scheme for large language model key–value (KV) cache compression, combining random rotation plus scalar quantization with a 1-bit Quantized Johnson–Lindenstrauss (QJL) residual to preserve both mean-squared error (MSE) and inner-product fidelity.

## New Contributor Docs

Start with the guides in [`docs/`](./docs/README.md):

- [`docs/project-map.md`](./docs/project-map.md)
- [`docs/runtime-flow.md`](./docs/runtime-flow.md)
- [`docs/compression-guide.md`](./docs/compression-guide.md)

## Project Layout

```text
TurboQuant-H/
|-- README.md
|-- pyproject.toml
|-- turboquant_h_smollm_benchmark.py
|-- docs/
|   |-- compression-guide.md
|   |-- project-map.md
|   |-- runtime-flow.md
|   |-- TurboQuant-H_paper_draft.md
|   `-- README.md
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
`-- tests/
    |-- __init__.py
    |-- test_cache.py
    |-- test_benchmark.py
    |-- test_config.py
    `-- test_packing.py
```

## Quick Start

Run the original script name:

```bash
python turboquant_h_smollm_benchmark.py --prompt "Explain KV cache compression."
```

Or use the package CLI with `src` on `PYTHONPATH`:

```bash
$env:PYTHONPATH = "src"
python -m turboquant_h.cli --prompt "Explain KV cache compression."
```

If you install the project:

```bash
pip install -e .
turboquant-h-benchmark --prompt "Explain KV cache compression."
```

## Example Options

```bash
python turboquant_h_smollm_benchmark.py \
  --model HuggingFaceTB/SmolLM-135M-Instruct \
  --quant_bits_old 2 \
  --rotation_mode hadamard \
  --correction_type low_rank \
  --prompt "Explain why KV cache compression matters in one paragraph."
```

## Module Guide

- `config.py`: benchmark configuration, validation, and result dataclasses.
- `benchmark.py`: model loading, generation loop, and end-to-end benchmark execution.
- `reporting.py`: terminal-friendly result formatting.
- `compression/packing.py`: bit-packing and unpacking utilities.
- `compression/rotation.py`: Hadamard-based rotation helpers.
- `compression/quantization.py`: tensor quantization and dequantization.
- `compression/correction.py`: low-rank and QJL-style residual correction logic.
- `compression/cache.py`: compressed KV-cache block, segment, and cache classes.
- `compression/attention.py`: compressed attention execution and model patching.

## Validation

The included tests are lightweight and focus on deterministic utilities:

```bash
python -m unittest discover -s tests
```

## Notes

- This is still a research harness, not a production CUDA implementation.
- The benchmark currently targets transformer attention modules with Llama-style projections.
- `TurboQuant-H_paper_draft.md` remains as supporting design context.
