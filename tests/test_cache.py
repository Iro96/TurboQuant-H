from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_h.compression.cache import compress_past_key_values, decompress_past_key_values
from turboquant_h.config import TurboQuantHConfig


class DenseCacheReconstructionTests(unittest.TestCase):
    def test_reconstruction_preserves_original_cache_dtype(self) -> None:
        cfg = TurboQuantHConfig(
            recent_fp_tokens=2,
            block_size=2,
            quant_bits_old=2,
            keep_ratio_old=0.0,
            rotation_mode="none",
            correction_type="none",
        )
        past = [
            (
                torch.randn(1, 2, 6, 4, dtype=torch.float16),
                torch.randn(1, 2, 6, 4, dtype=torch.float16),
            )
        ]

        compressed = compress_past_key_values(past, cfg, baseline_dtype_bits=16)
        dense = decompress_past_key_values(compressed, device=torch.device("cpu"))

        self.assertEqual(dense[0][0].dtype, torch.float16)
        self.assertEqual(dense[0][1].dtype, torch.float16)
        self.assertEqual(dense[0][0].shape, past[0][0].shape)
        self.assertEqual(dense[0][1].shape, past[0][1].shape)


if __name__ == "__main__":
    unittest.main()
