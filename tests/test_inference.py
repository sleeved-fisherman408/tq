from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tq.core.turbo import TQConfig
from tq.server.inference import GenerationMetrics, InferenceEngine


def _mock_tensor(shape=(1, 5)):
    t = MagicMock()
    t.shape = shape
    t.__getitem__ = lambda self, idx: MagicMock()
    t.to = MagicMock(return_value=t)
    return t


class TestInferenceEngineInit:
    def test_defaults(self):
        eng = InferenceEngine("test-model")
        assert eng.model_name == "test-model"
        assert eng.model is None
        assert eng.tokenizer is None
        assert eng.tq_config is None
        assert eng.device == "auto"

    def test_with_tq_config(self):
        tq = TQConfig(
            key_bits=4,
            value_bits=4,
            key_method="mse",
            value_method="mse",
            outlier_channels=32,
            outlier_bits=8,
            compression_ratio=4.0,
            estimated_quality="lossless",
        )
        eng = InferenceEngine("m", tq_config=tq, device="cpu")
        assert eng.tq_config is tq
        assert eng.device == "cpu"


class TestLoad:
    @patch("tq.server.inference.AutoTokenizer")
    @patch("tq.server.inference.AutoModelForCausalLM")
    @patch("tq.server.inference.torch")
    def test_load_no_tq(self, mock_torch, mock_lm_cls, mock_tok_cls):
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_model = MagicMock()
        mock_lm_cls.from_pretrained.return_value = mock_model
        mock_tok = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tok

        eng = InferenceEngine("test-model", device="cpu")
        eng.load()

        mock_lm_cls.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()
        assert eng.model is mock_model
        assert eng.tokenizer is mock_tok

    @patch("tq.server.inference.AutoTokenizer")
    @patch("tq.server.inference.AutoModelForCausalLM")
    @patch("tq.server.inference.torch")
    def test_load_with_tq_importerror(self, mock_torch, mock_lm_cls, mock_tok_cls):
        mock_torch.float16 = "float16"
        mock_model = MagicMock()
        mock_lm_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        tq = TQConfig(
            key_bits=4,
            value_bits=4,
            key_method="mse",
            value_method="mse",
            outlier_channels=32,
            outlier_bits=8,
            compression_ratio=4.0,
            estimated_quality="lossless",
        )
        eng = InferenceEngine("m", tq_config=tq)
        with patch.dict("sys.modules", {"turboquant": None}):
            eng.load()
        assert eng._tq_cache is None


class TestGenerate:
    def test_not_loaded_raises(self):
        eng = InferenceEngine("m")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            eng.generate("hello")

    def test_generate_cpu_device(self):
        eng = InferenceEngine("m")
        eng.model = MagicMock()
        eng.tokenizer = MagicMock()
        eng.model.device = MagicMock()
        eng.model.device.type = "cpu"

        input_ids = MagicMock()
        input_ids.shape = (1, 10)
        eng.tokenizer.return_value = MagicMock(input_ids=input_ids)

        completion_ids = MagicMock()
        completion_ids.__len__ = lambda self: 5
        output_ids = MagicMock()
        output_ids.__getitem__ = MagicMock(return_value=completion_ids)
        eng.model.generate.return_value = output_ids

        with patch("tq.server.inference.torch"):
            text, metrics = eng.generate("test prompt", max_tokens=50)
        assert isinstance(metrics, GenerationMetrics)
        assert metrics.prompt_tokens == 10

    def test_generate_cuda_device(self):
        eng = InferenceEngine("m")
        eng.model = MagicMock()
        eng.tokenizer = MagicMock()
        eng.model.device = MagicMock()
        eng.model.device.type = "cuda"

        input_ids = MagicMock()
        input_ids.shape = (1, 10)
        moved_ids = MagicMock()
        input_ids.to.return_value = moved_ids
        eng.tokenizer.return_value = MagicMock(input_ids=input_ids)

        completion_ids = MagicMock()
        completion_ids.__len__ = lambda self: 5
        output_ids = MagicMock()
        output_ids.__getitem__ = MagicMock(return_value=completion_ids)
        eng.model.generate.return_value = output_ids

        with patch("tq.server.inference.torch"):
            text, metrics = eng.generate("test prompt")
        input_ids.to.assert_called_once_with(eng.model.device)

    def test_generate_stream_flag_redirects(self):
        eng = InferenceEngine("m")
        eng.model = MagicMock()
        eng.tokenizer = MagicMock()
        eng.model.device = MagicMock()
        eng.model.device.type = "cpu"

        input_ids = MagicMock()
        input_ids.shape = (1, 5)
        eng.tokenizer.return_value = MagicMock(input_ids=input_ids)

        with (
            patch("tq.server.inference.torch"),
            patch.object(eng, "_generate_stream", return_value=("text", GenerationMetrics())),
        ):
            text, metrics = eng.generate("test", stream=True)
        assert text == "text"


class TestGenerateStream:
    def test_not_loaded_raises(self):
        eng = InferenceEngine("m")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            list(eng.generate_stream("hello"))

    def test_internal_stream_raises(self):
        eng = InferenceEngine("m")
        with pytest.raises(NotImplementedError, match="Use generate_stream"):
            eng._generate_stream(MagicMock(), 10, 0.7, 0.9, 5, 0.0)

    def test_generate_stream_cpu_device(self):
        eng = InferenceEngine("m")
        eng.model = MagicMock()
        eng.tokenizer = MagicMock()
        eng.model.device = MagicMock()
        eng.model.device.type = "cpu"

        input_ids = MagicMock()
        input_ids.shape = (1, 3)
        eng.tokenizer.return_value = MagicMock(input_ids=input_ids)

        next_token_val = MagicMock()
        next_token_val.item.return_value = 2

        next_token = MagicMock()
        next_token.__getitem__ = MagicMock(return_value=next_token_val)
        next_token[0] = next_token_val

        logits = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = logits
        eng.model.return_value = mock_output

        mock_torch = MagicMock()
        mock_torch.argmax.return_value = next_token
        mock_torch.cat.return_value = MagicMock()
        mock_torch.no_grad = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)

        with patch("tq.server.inference.torch", mock_torch):
            eng.tokenizer.eos_token_id = 2
            tokens = list(eng.generate_stream("hello", max_tokens=5, temperature=0))
        assert isinstance(tokens, list)
        eng.tokenizer.decode.assert_called()


class TestEstimateKVSize:
    def test_no_model(self):
        eng = InferenceEngine("m")
        assert eng._estimate_kv_size(1000) == 0.0

    def test_with_model_no_tq(self):
        eng = InferenceEngine("m")
        eng.model = MagicMock()
        eng.model.config = MagicMock(
            num_hidden_layers=32,
            num_key_value_heads=8,
            hidden_size=4096,
            num_attention_heads=32,
        )
        size = eng._estimate_kv_size(1000)
        assert size > 0

    def test_with_tq_compression(self):
        eng = InferenceEngine("m")
        eng.model = MagicMock()
        eng.model.config = MagicMock(
            num_hidden_layers=32,
            num_key_value_heads=8,
            hidden_size=4096,
            num_attention_heads=32,
        )
        eng.tq_config = TQConfig(
            key_bits=4,
            value_bits=4,
            key_method="mse",
            value_method="mse",
            outlier_channels=32,
            outlier_bits=8,
            compression_ratio=4.0,
            estimated_quality="lossless",
        )
        size_no_tq = eng._estimate_kv_size(1000)
        eng.tq_config = None
        size_plain = eng._estimate_kv_size(1000)
        assert size_no_tq == pytest.approx(size_plain / 4.0)


class TestGetStatus:
    def test_status_no_tq(self):
        eng = InferenceEngine("test-model")
        status = eng.get_status()
        assert status["model"] == "test-model"
        assert status["turboquant"]["enabled"] is False
        assert status["turboquant"]["key_bits"] is None
        assert "performance" in status

    def test_status_with_tq(self):
        tq = TQConfig(
            key_bits=4,
            value_bits=4,
            key_method="mse",
            value_method="mse",
            outlier_channels=32,
            outlier_bits=8,
            compression_ratio=4.0,
            estimated_quality="lossless",
        )
        eng = InferenceEngine("m", tq_config=tq)
        status = eng.get_status()
        assert status["turboquant"]["enabled"] is False
        assert status["turboquant"]["key_bits"] == 4
        assert status["turboquant"]["compression_ratio"] == 4.0
