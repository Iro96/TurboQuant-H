# Compression Guide

This document explains the `src/turboquant_h/compression/` folder. If you plan to edit the compression logic, this is the most important guide in `docs/`.

## Folder Purpose

The compression package owns the full KV-cache compression lifecycle:

1. represent helper math and small shared utilities,
2. rotate tensors before quantization,
3. quantize tensors into low-bit packed storage,
4. store optional residual correction information,
5. manage compressed cache blocks and segments,
6. optionally compute attention directly on compressed cache data.

## Module Map

### `common.py`

Small helper utilities shared by the rest of the subsystem.

Examples:

- `is_power_of_two()`
- `numel_from_shape()`
- `token_length()`
- `strip_empty_tokens()`

This file should stay boring and reusable. If a helper is tiny and used across compression files, it probably belongs here.

### `packing.py`

The bit-packing layer.

What it owns:

- packing low-bit codes into `uint8` storage,
- unpacking them back into codes,
- handling fast paths for 1-, 2-, 4-, and 8-bit cases,
- handling generic packing for other supported widths.

Why it matters:

This file controls the most storage-centric part of the project. If packed values are wrong here, everything above it becomes hard to trust.

### `rotation.py`

The preprocessing rotation layer.

What it owns:

- fast Walsh-Hadamard transform,
- random sign state for randomized Hadamard,
- applying and inverting rotations,
- packing and restoring rotation signs,
- beam/batch selection support for cached rotation state.

Why it exists:

Some quantization strategies work better when values are rotated into a friendlier distribution first.

### `quantization.py`

The numeric compression layer.

What it owns:

- `PackedQuantizedTensor`,
- grouping rules such as head-level vs channel-level scaling,
- scale estimation,
- uniform quantization,
- codebook quantization,
- dequantization,
- beam/batch selection for quantized state.

Important idea:

This module decides how floating-point tensors become compact codes and how those codes turn back into approximate tensors.

### `correction.py`

The approximation repair layer.

What it owns:

- `CorrectionState`,
- randomized low-rank correction,
- QJL-style projected sign correction,
- reconstruction-time application of low-rank corrections,
- bookkeeping for correction storage and batch selection,
- saliency scoring for token retention decisions.

Important idea:

Quantization introduces error. This module stores extra information that can reduce the damage.

### `cache.py`

The storage and lifecycle layer.

What it owns:

- `CompressedBlock`
- `CompressedSegment`
- `CompressedCacheLayer`
- `CompressedCache`
- block compression,
- segment compression,
- cache append/update behavior,
- dense reconstruction,
- storage-bit estimation.

Important idea:

This is the main "state container" module. It defines what the compressed cache actually looks like in memory.

If you only read one compression file first, read this one.

### `attention.py`

The direct-compressed-attention layer.

What it owns:

- rotating queries to match compressed keys,
- optional QJL score-space correction,
- chunked attention over compressed and dense cache pieces,
- monkey-patching compatible transformer attention modules to use `CompressedCache`.

Important idea:

This module is what allows the project to avoid reconstructing a full dense KV cache on every decode step.

### `__init__.py`

The package export surface for compression utilities and cache types.

## Core Data Structures

These are the main objects a newcomer should recognize:

### `RotationState`

Lives in `rotation.py`.

Stores:

- which rotation mode is active,
- optional packed sign bits for randomized Hadamard.

### `PackedQuantizedTensor`

Lives in `quantization.py`.

Stores:

- packed codes,
- bit width,
- original shape,
- quantizer type,
- grouping strategy,
- optional scale, offset, or codebook data.

### `CorrectionState`

Lives in `correction.py`.

Stores:

- correction kind,
- optional low-rank factors,
- optional QJL sign and scale data.

### `CompressedBlock`

Lives in `cache.py`.

Represents one compressed cache block and combines:

- quantized tensor state,
- rotation state,
- correction state,
- optional retained full-precision token values.

### `CompressedSegment`

Lives in `cache.py`.

Represents a longer sequence region split into:

- compressed old blocks,
- pending old values not yet large enough to form a full block,
- recent higher-precision values.

### `CompressedCache`

Lives in `cache.py`.

Represents the full multi-layer compressed KV cache for the model.

## How The Compression Pieces Fit Together

```text
original KV tensors
  -> optional rotation
  -> quantization
  -> optional correction state
  -> optional salient-token retention
  -> compressed blocks
  -> compressed segments
  -> compressed cache
  -> direct compressed attention or dense reconstruction
```

## Where To Make Changes

- Change bit-packing format: `packing.py`
- Change rotation modes: `rotation.py`
- Change quantizer math: `quantization.py`
- Change correction methods: `correction.py`
- Change cache layout or update policy: `cache.py`
- Change streamed compressed attention behavior: `attention.py`

## Safe Reading Order Inside The Compression Package

1. `common.py`
2. `packing.py`
3. `rotation.py`
4. `quantization.py`
5. `correction.py`
6. `cache.py`
7. `attention.py`

That order moves from the simplest utilities to the most system-level behavior.
