"""Tests for --models all and the /models endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import ServerConfig
from pyzm.models.detection import BBox, Detection, DetectionResult


# ---------------------------------------------------------------------------
# ServerConfig validation
# ---------------------------------------------------------------------------

class TestServerConfigModelsAll:
    def test_all_alone_is_valid(self):
        config = ServerConfig(models=["all"])
        assert config.models == ["all"]

    def test_all_mixed_with_other_raises(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            ServerConfig(models=["all", "yolov4"])

    def test_normal_models_still_work(self):
        config = ServerConfig(models=["yolov4", "yolov7"])
        assert config.models == ["yolov4", "yolov7"]


# ---------------------------------------------------------------------------
# /models endpoint
# ---------------------------------------------------------------------------

def _mock_detector(lazy: bool = False, processor: str = "cpu", requested: str = "cpu"):
    """Return a mock Detector with a mock pipeline.

    *processor* is what the backend is running on now and *requested* what it
    was configured with -- they differ once a GPU model has fallen back to CPU.
    """
    det = MagicMock()
    det._pipeline = MagicMock()
    det._config = MagicMock()
    det._config.models = [MagicMock()]
    det.detect.return_value = DetectionResult(
        detections=[
            Detection(
                label="person", confidence=0.95,
                bbox=BBox(10, 20, 50, 80), model_name="yolov4",
            )
        ],
        frame_id="single",
    )

    # Mock the pipeline's _backends list
    mc_mock = MagicMock()
    mc_mock.name = "yolov4"
    mc_mock.type.value = "object"
    mc_mock.framework.value = "opencv"
    mc_mock.processor.value = requested

    backend_mock = MagicMock()
    backend_mock.is_loaded = not lazy
    backend_mock.processor = processor

    det._pipeline._backends = [(mc_mock, backend_mock)]
    return det


@pytest.fixture
def client_eager():
    """TestClient with normal eager loading."""
    config = ServerConfig(models=["yolov4"])
    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector(lazy=False)
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

        from pyzm.serve.app import create_app
        application = create_app(config)

        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


@pytest.fixture
def client_all():
    """TestClient simulating --models all (lazy mode)."""
    config = ServerConfig(models=["all"])
    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector(lazy=True)
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

        from pyzm.serve.app import create_app
        application = create_app(config)

        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


class TestModelsEndpoint:
    def test_models_returns_list(self, client_eager):
        resp = client_eager.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "yolov4"
        assert data["models"][0]["loaded"] is True

    def test_models_lazy_shows_not_loaded(self, client_all):
        resp = client_all.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["loaded"] is False

    def test_models_reports_the_processor_in_use(self, client_eager):
        """The live processor is the only signal a gateway is degraded. Refs #66"""
        entry = client_eager.get("/models").json()["models"][0]
        assert entry["processor"] == "cpu"
        assert entry["requested_processor"] == "cpu"

    def test_models_reports_a_gpu_model_that_fell_back_to_cpu(self):
        """A GPU model running on CPU must say so, not report what it asked for."""
        config = ServerConfig(models=["yolov4"])
        with patch("pyzm.serve.app.Detector") as MockDetector:
            mock_det = _mock_detector(processor="cpu", requested="gpu")
            MockDetector.return_value = mock_det
            mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

            from pyzm.serve.app import create_app
            from fastapi.testclient import TestClient

            with TestClient(create_app(config)) as tc:
                entry = tc.get("/models").json()["models"][0]

        assert entry["processor"] == "cpu"
        assert entry["requested_processor"] == "gpu"


class TestGpuPolicy:
    """The server's GPU-fallback policy reaches the model configs. Refs #66"""

    def _models(self):
        from pyzm.models.config import ModelConfig

        return [ModelConfig(name="a"), ModelConfig(name="b")]

    def test_strict_mode_and_retry_interval_are_stamped_on_every_model(self):
        from pyzm.serve.app import _apply_gpu_policy

        config = ServerConfig(allow_cpu_fallback=False, gpu_retry_seconds=30)
        models = _apply_gpu_policy(self._models(), config)

        assert [m.allow_cpu_fallback for m in models] == [False, False]
        assert [m.gpu_retry_seconds for m in models] == [30, 30]

    def test_defaults_leave_the_model_configs_alone(self):
        """Without a server-level override, per-model values must survive."""
        from pyzm.serve.app import _apply_gpu_policy

        models = self._models()
        assert _apply_gpu_policy(models, ServerConfig()) is models

    def _config_from_cli(self, argv):
        """Run the serve CLI with *argv* and return the ServerConfig it built."""
        import sys

        from pyzm.serve import __main__ as serve_main

        with patch.object(sys, "argv", ["pyzm.serve", *argv]), \
                patch("uvicorn.run"), \
                patch("pyzm.serve.app.create_app") as create_app:
            serve_main.main()
        return create_app.call_args[0][0]

    def test_cli_defaults_keep_the_fallback_on(self):
        config = self._config_from_cli([])

        assert config.allow_cpu_fallback is True
        assert config.gpu_retry_seconds is None

    def test_cli_flags_set_the_policy(self):
        config = self._config_from_cli(
            ["--processor", "gpu", "--no-cpu-fallback", "--gpu-retry-seconds", "30"]
        )

        assert config.allow_cpu_fallback is False
        assert config.gpu_retry_seconds == 30

    def test_the_lifespan_applies_the_policy(self):
        from pyzm.models.config import ModelConfig

        config = ServerConfig(models=["yolov4"], allow_cpu_fallback=False)
        with patch("pyzm.serve.app.Detector") as MockDetector:
            mock_det = _mock_detector()
            mock_det._config.models = [ModelConfig(name="yolov4")]
            MockDetector.return_value = mock_det
            mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

            from pyzm.serve.app import create_app
            from fastapi.testclient import TestClient

            with TestClient(create_app(config)):
                assert mock_det._config.models[0].allow_cpu_fallback is False


class TestModelsAllLifespan:
    def test_all_creates_detector_with_none_models(self):
        """When config.models == ['all'], Detector is created with models=None."""
        config = ServerConfig(models=["all"])
        with patch("pyzm.serve.app.Detector") as MockDetector:
            mock_det = _mock_detector(lazy=True)
            MockDetector.return_value = mock_det
            mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

            from pyzm.serve.app import create_app
            from fastapi.testclient import TestClient

            application = create_app(config)
            with TestClient(application):
                MockDetector.assert_called_once_with(
                    models=None,
                    base_path=config.base_path,
                    processor=config.processor,
                )
                mock_det._ensure_pipeline.assert_called_once_with(lazy=True)

    def test_normal_creates_detector_with_models(self):
        """When config.models is normal, Detector is created with model names."""
        config = ServerConfig(models=["yolov4"])
        with patch("pyzm.serve.app.Detector") as MockDetector:
            mock_det = _mock_detector(lazy=False)
            MockDetector.return_value = mock_det
            mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

            from pyzm.serve.app import create_app
            from fastapi.testclient import TestClient

            application = create_app(config)
            with TestClient(application):
                MockDetector.assert_called_once_with(
                    models=["yolov4"],
                    base_path=config.base_path,
                    processor=config.processor,
                )
                mock_det._ensure_pipeline.assert_called_once_with(lazy=False)
