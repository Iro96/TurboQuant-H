# TurboQuant-H

TurboQuant-H is a research-oriented benchmark harness for experimenting with hierarchical KV-cache compression on Hugging Face causal language models. This refactor turns the original single-file prototype into a small Python project with clearer module boundaries, a reusable CLI, and lightweight tests.

## Project Goals

- Keep the original benchmark behavior intact.
- Make the compression pipeline easier to read and extend.
- Separate CLI, reporting, benchmark flow, and compression internals.
- Preserve the old `turboquant_h_smollm_benchmark.py` entrypoint for convenience.

## Project Layout

```text
TurboQuant-H/
|-- README.md
|-- pyproject.toml
|-- turboquant_h_smollm_benchmark.py
|-- TurboQuant-H_paper_draft.md
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
