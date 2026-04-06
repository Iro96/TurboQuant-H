from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_h.config import RuntimeConfig, TurboQuantHConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_validate_rejects_negative_max_new_tokens(self) -> None:
        runtime = RuntimeConfig(max_new_tokens=-1)
        with self.assertRaises(ValueError):
            runtime.validate()


class TurboQuantHConfigTests(unittest.TestCase):
    def test_tensor_specific_overrides_take_precedence(self) -> None:
        cfg = TurboQuantHConfig(
            quantizer="uniform",
            key_quantizer="codebook",
            value_quant_scale="head",
            correction_type="qjl",
        )

        self.assertEqual(cfg.resolved_quantizer("k"), "codebook")
        self.assertEqual(cfg.resolved_quantizer("v"), "uniform")
        self.assertEqual(cfg.resolved_quant_scale("v"), "head")
        self.assertEqual(cfg.resolved_correction_type("v"), "low_rank")

    def test_validate_rejects_invalid_top_p(self) -> None:
        cfg = TurboQuantHConfig(top_p=0.0)
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
