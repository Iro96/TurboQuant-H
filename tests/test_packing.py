from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_h.compression.packing import pack_lowbit_codes, unpack_lowbit_codes


class PackingRoundTripTests(unittest.TestCase):
    def test_round_trip_supported_bits(self) -> None:
        base = torch.arange(37, dtype=torch.int32)
        for bits in (1, 2, 3, 4, 5, 7, 8):
            max_value = (1 << bits) - 1
            codes = (base % (max_value + 1)).to(torch.uint8)
            packed = pack_lowbit_codes(codes, bits)
            unpacked = unpack_lowbit_codes(packed, bits, codes.numel())
            self.assertTrue(torch.equal(unpacked, codes), msg=f"failed round trip for {bits}-bit packing")

    def test_empty_round_trip(self) -> None:
        codes = torch.empty(0, dtype=torch.uint8)
        packed = pack_lowbit_codes(codes, 2)
        unpacked = unpack_lowbit_codes(packed, 2, 0)
        self.assertEqual(packed.numel(), 0)
        self.assertEqual(unpacked.numel(), 0)


if __name__ == "__main__":
    unittest.main()
