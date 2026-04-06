"""Low-bit tensor packing helpers."""

from __future__ import annotations

import torch


def _pack_lowbit_codes_generic(codes: torch.Tensor, bits: int) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.int32)
    shifts = torch.arange(bits, device=flat.device, dtype=torch.int32)
    bit_positions = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64).unsqueeze(-1) * bits
    bit_positions = bit_positions + torch.arange(bits, device=flat.device, dtype=torch.int64)
    byte_positions = (bit_positions // 8).reshape(-1).to(torch.long)
    intra_byte = (bit_positions % 8).reshape(-1).to(torch.int32)
    bit_values = ((flat.unsqueeze(-1) >> shifts) & 1).reshape(-1)
    packed = torch.zeros(int((flat.numel() * bits + 7) // 8), device=flat.device, dtype=torch.int32)
    packed.scatter_add_(0, byte_positions, bit_values << intra_byte)
    return packed.to(torch.uint8)


def _unpack_lowbit_codes_generic(packed: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    bit_positions = torch.arange(num_values, device=packed.device, dtype=torch.int64).unsqueeze(-1) * bits
    bit_positions = bit_positions + torch.arange(bits, device=packed.device, dtype=torch.int64)
    byte_positions = bit_positions // 8
    intra_byte = bit_positions % 8
    selected = packed[byte_positions]
    bit_values = ((selected.to(torch.int64) >> intra_byte) & 1).to(torch.int64)
    shifts = torch.arange(bits, device=packed.device, dtype=torch.int64)
    codes = (bit_values << shifts).sum(dim=-1)
    return codes.to(torch.uint8)


def pack_lowbit_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 1 or bits > 8:
        raise ValueError(f"Only 1-8 bits are supported, got {bits}")

    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() == 0:
        return torch.empty(0, device=codes.device, dtype=torch.uint8)
    if bits == 8:
        return flat.contiguous()

    if bits in (1, 2, 4):
        values_per_byte = 8 // bits
        pad = (-flat.numel()) % values_per_byte
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)], dim=0)
        grouped = flat.view(-1, values_per_byte).to(torch.int32)
        shifts = torch.arange(values_per_byte, device=flat.device, dtype=torch.int32) * bits
        packed = torch.sum(grouped << shifts.unsqueeze(0), dim=-1)
        return packed.to(torch.uint8)

    return _pack_lowbit_codes_generic(flat, bits)


def unpack_lowbit_codes(packed: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    if bits < 1 or bits > 8:
        raise ValueError(f"Only 1-8 bits are supported, got {bits}")
    if num_values == 0:
        return torch.empty(0, device=packed.device, dtype=torch.uint8)
    if bits == 8:
        return packed[:num_values].to(torch.uint8)

    if bits in (1, 2, 4):
        values_per_byte = 8 // bits
        shifts = torch.arange(values_per_byte, device=packed.device, dtype=torch.int32) * bits
        unpacked = ((packed.to(torch.int32).unsqueeze(-1) >> shifts.unsqueeze(0)) & ((1 << bits) - 1)).reshape(-1)
        return unpacked[:num_values].to(torch.uint8)

    return _unpack_lowbit_codes_generic(packed, bits, num_values)
