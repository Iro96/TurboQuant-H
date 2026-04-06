from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_h.benchmark import _prepare_model_inputs, _sample_next_token
from turboquant_h.config import TurboQuantHConfig


class DummyBatch(dict):
    def to(self, device: torch.device) -> "DummyBatch":
        return DummyBatch({name: value.to(device) for name, value in self.items()})


class DummyTokenizerWithoutTemplate:
    chat_template = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str, return_tensors: str = "pt") -> DummyBatch:
        self.calls.append(text)
        return DummyBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    def apply_chat_template(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("apply_chat_template should not be used when no chat template is configured")


class DummyTokenizerWithTemplate:
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def __call__(self, text: str, return_tensors: str = "pt") -> DummyBatch:
        raise AssertionError("plain tokenization should not be used when a chat template is configured")

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.messages = messages
        self.kwargs = kwargs
        return DummyBatch({"input_ids": torch.tensor([[4, 5, 6]])})


class PrepareModelInputsTests(unittest.TestCase):
    def test_falls_back_to_plain_tokenization_without_chat_template(self) -> None:
        tokenizer = DummyTokenizerWithoutTemplate()

        model_inputs = _prepare_model_inputs(tokenizer, "Explain KV cache compression.", torch.device("cpu"))

        self.assertEqual(tokenizer.calls, ["Explain KV cache compression."])
        self.assertTrue(torch.equal(model_inputs["input_ids"], torch.tensor([[1, 2, 3]])))

    def test_uses_chat_template_directly_when_available(self) -> None:
        tokenizer = DummyTokenizerWithTemplate()

        model_inputs = _prepare_model_inputs(tokenizer, "Explain KV cache compression.", torch.device("cpu"))

        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Explain KV cache compression."}])
        self.assertEqual(
            tokenizer.kwargs,
            {
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            },
        )
        self.assertTrue(torch.equal(model_inputs["input_ids"], torch.tensor([[4, 5, 6]])))


class SamplingTests(unittest.TestCase):
    def test_top_p_sampling_keeps_boundary_token(self) -> None:
        torch.manual_seed(0)
        logits = torch.log(torch.tensor([[0.72, 0.26, 0.02]], dtype=torch.float32))
        cfg = TurboQuantHConfig(top_p=0.8, temperature=1.0, do_sample=True)

        seen = set()
        for _ in range(256):
            token = int(_sample_next_token(logits, cfg).item())
            seen.add(token)

        self.assertIn(0, seen)
        self.assertIn(1, seen)
        self.assertNotIn(2, seen)


if __name__ == "__main__":
    unittest.main()
