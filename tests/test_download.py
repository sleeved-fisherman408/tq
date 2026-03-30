from __future__ import annotations

from unittest.mock import MagicMock, patch

from tq.core.config import InstalledModel
from tq.core.download import _gguf_filename, list_models, pull_model, remove_model


class TestGGUFFilename:
    def test_basic(self):
        assert _gguf_filename("TheBloke/Llama-2-7B", "Q4_K_M") == "Llama-2-7B-Q4_K_M.gguf"

    def test_different_quant(self):
        assert _gguf_filename("Qwen/Qwen3-8B", "Q8_0") == "Qwen3-8B-Q8_0.gguf"


class TestPullModel:
    @patch("tq.core.download.find_installed_model")
    @patch("tq.core.download.resolve_model")
    @patch("tq.core.download.ensure_dirs")
    def test_already_installed(self, mock_dirs, mock_resolve, mock_find):
        mock_resolve.return_value = MagicMock(model_id="Qwen/Qwen3-8B")
        mock_find.return_value = InstalledModel(
            model_id="Qwen/Qwen3-8B",
            shortname="qwen3-8b",
            gguf_path="/tmp/existing.gguf",
            quant="Q4_K_M",
            size_bytes=1000,
            downloaded_at="2026-01-01",
        )
        with patch("tq.core.download.Path") as mock_path_cls:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path_cls.return_value = mock_path
            result = pull_model("qwen3-8b")
        assert result == mock_path

    @patch("tq.core.download.register_installed_model")
    @patch("tq.core.download.find_installed_model")
    @patch("tq.core.download.resolve_model")
    @patch("tq.core.download.ensure_dirs")
    @patch("tq.core.download.hf_hub_download")
    def test_download_new(self, mock_hf, mock_dirs, mock_resolve, mock_find, mock_register):
        mock_resolve.return_value = MagicMock(model_id="Qwen/Qwen3-8B")
        mock_find.return_value = None
        mock_path = MagicMock()
        mock_path.stat.return_value = MagicMock(st_size=5000000000)
        mock_hf.return_value = str(mock_path)

        with patch("tq.core.download.Path", return_value=mock_path):
            pull_model("qwen3-8b")

        mock_hf.assert_called_once()
        mock_register.assert_called_once()
        installed = mock_register.call_args[0][0]
        assert installed.model_id == "Qwen/Qwen3-8B"
        assert installed.quant == "Q4_K_M"

    @patch("tq.core.download.register_installed_model")
    @patch("tq.core.download.find_installed_model")
    @patch("tq.core.download.resolve_model")
    @patch("tq.core.download.ensure_dirs")
    @patch("tq.core.download.hf_hub_download")
    def test_download_custom_quant(
        self, mock_hf, mock_dirs, mock_resolve, mock_find, mock_register
    ):
        mock_resolve.return_value = MagicMock(model_id="Qwen/Qwen3-8B")
        mock_find.return_value = None
        mock_path = MagicMock()
        mock_path.stat.return_value = MagicMock(st_size=8000000000)
        mock_hf.return_value = str(mock_path)

        with patch("tq.core.download.Path", return_value=mock_path):
            pull_model("qwen3-8b", quant="Q8_0")

        hf_call = mock_hf.call_args
        assert "Q8_0" in hf_call.kwargs.get("filename", hf_call[1].get("filename", ""))


class TestListModels:
    @patch("tq.core.download.load_installed_models")
    def test_list(self, mock_load):
        mock_load.return_value = [
            InstalledModel(
                model_id="test/model",
                shortname="test",
                gguf_path="/tmp/test.gguf",
                quant="Q4_K_M",
                size_bytes=1000,
                downloaded_at="2026-01-01",
            )
        ]
        result = list_models()
        assert len(result) == 1
        assert result[0].model_id == "test/model"

    @patch("tq.core.download.load_installed_models")
    def test_list_empty(self, mock_load):
        mock_load.return_value = []
        assert list_models() == []


class TestRemoveModel:
    @patch("tq.core.download.find_installed_model")
    @patch("tq.core.download.unregister_installed_model")
    @patch("tq.core.download.resolve_model")
    def test_remove_success_no_file(self, mock_resolve, mock_unregister, mock_find):
        mock_resolve.return_value = MagicMock(model_id="test/model")
        mock_unregister.return_value = True
        mock_find.return_value = None
        assert remove_model("test") is True

    @patch("tq.core.download.find_installed_model")
    @patch("tq.core.download.unregister_installed_model")
    @patch("tq.core.download.resolve_model")
    def test_remove_success_with_file(self, mock_resolve, mock_unregister, mock_find):
        mock_resolve.return_value = MagicMock(model_id="test/model")
        mock_unregister.return_value = True
        mock_installed = MagicMock()
        mock_installed.gguf_path = "/tmp/test.gguf"
        mock_find.return_value = mock_installed

        with patch("tq.core.download.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.exists.return_value = True
            mock_path_cls.return_value = mock_p
            assert remove_model("test") is True
            mock_p.unlink.assert_called_once()

    @patch("tq.core.download.unregister_installed_model")
    @patch("tq.core.download.resolve_model")
    def test_remove_not_registered(self, mock_resolve, mock_unregister):
        mock_resolve.return_value = MagicMock(model_id="test/model")
        mock_unregister.return_value = False
        assert remove_model("test") is False
