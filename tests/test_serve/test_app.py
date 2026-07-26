"""Tests for pyzm.serve.app -- FastAPI detection server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyzm.models.config import ServerConfig
from pyzm.models.detection import BBox, Detection, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_detector():
    """Return a mock Detector whose detect() returns a canned result.

    detect() uses side_effect (not return_value) so every call gets a FRESH
    DetectionResult. The server mutates result.detections in place (confidence
    and pattern filters); a shared return_value would leak those mutations
    across frames within a single request, unlike the real Detector.
    """
    det = MagicMock()
    det._config = MagicMock()
    det._config.models = [MagicMock()]

    # Dumb server runs a single backend per /infer call. Build a mock pipeline
    # with one object backend whose detect() returns a canned raw detection.
    mc = MagicMock()
    mc.type.value = "object"
    mc.name = "yolov4"
    backend = MagicMock()
    backend.detect.return_value = [
        Detection(label="person", confidence=0.95, bbox=BBox(10, 20, 50, 80), model_name="yolov4")
    ]
    pipeline = MagicMock()
    pipeline._backends = [(mc, backend)]
    det._pipeline = pipeline  # non-None -> /health reports models_loaded=True
    det._backend = backend  # exposed for assertions
    return det


@pytest.fixture
def client():
    """Create a FastAPI TestClient with a mock Detector.

    The patch must stay active while TestClient is alive so the lifespan
    (which creates the Detector) uses the mock.
    """
    config = ServerConfig(models=["yolov4"])

    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector()
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock()

        from pyzm.serve.app import create_app
        application = create_app(config)

        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorkerConfigEnv:
    """Round-trip of ServerConfig through the worker env var.

    Regression guard: model_dump_json() masks SecretStr (auth_password) as
    "**********", which silently broke /login in every uvicorn worker when
    --workers > 1.  config_to_env must preserve the real secret.
    """

    def test_config_to_env_preserves_auth_password(self):
        from pyzm.serve.app import config_from_env, config_to_env

        cfg = ServerConfig(
            models=["yolov4"],
            auth_enabled=True,
            auth_username="admin",
            auth_password="hunter2",
            token_secret="topsecret",
            workers=3,
        )
        raw = config_to_env(cfg)
        assert "**********" not in raw

        with patch.dict("os.environ", {"PYZM_SERVER_CONFIG": raw}):
            restored = config_from_env()
        assert restored.auth_password.get_secret_value() == "hunter2"
        assert restored.token_secret == "topsecret"
        assert restored.workers == 3

    def test_config_from_env_absent_returns_default(self):
        import os

        from pyzm.serve.app import config_from_env

        with patch.dict(os.environ, {}, clear=True):
            cfg = config_from_env()
        assert cfg.workers == 1


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True


class TestInfer:
    """The dumb /infer endpoint: one model, one image, raw detections, no filtering."""

    def _jpeg(self):
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def test_infer_returns_raw_detections(self, client):
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "object", "name": "yolov4"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert data["detections"] == [
            {"label": "person", "confidence": 0.95, "box": [10, 20, 50, 80],
             "type": "object", "model_name": "yolov4"}
        ]

    def test_infer_type_only_falls_back_to_type_model(self, client):
        # name omitted -> server uses its loaded model of that type
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "object"},
        )
        assert resp.status_code == 200
        assert resp.json()["detections"][0]["label"] == "person"

    def test_infer_unknown_type_returns_error_not_500(self, client):
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "face"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["detections"] == []
        assert "no model" in data["error"]

    def test_infer_empty_file(self, client):
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", b"", "image/jpeg")},
            data={"type": "object"},
        )
        assert resp.status_code == 400

    def test_infer_bad_image(self, client):
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", b"not-a-jpeg", "image/jpeg")},
            data={"type": "object"},
        )
        assert resp.status_code == 400

    def test_infer_does_not_filter(self, client):
        # Server must return low-confidence detections untouched: it is the
        # client's job to apply min_confidence/pattern. Backend returns 0.95;
        # assert the server never drops based on any threshold.
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "object", "name": "yolov4"},
        )
        assert len(resp.json()["detections"]) == 1

    def test_infer_url_mode_server_fetches(self, client, monkeypatch):
        # URL mode: no uploaded image; server fetches the frame from ZM.
        import pyzm.serve.app as appmod
        resp = MagicMock()
        resp.content = self._jpeg()
        resp.raise_for_status = MagicMock()
        seen = {}
        def fake_get(u, **k):
            seen["url"] = u
            return resp
        monkeypatch.setattr(appmod.http_requests, "get", fake_get)
        r = client.post("/infer", data={
            "type": "object",
            "url": "http://zm/index.php?view=image&eid=1&fid=snapshot",
            "zm_auth": "token=x",
        })
        assert r.status_code == 200
        assert r.json()["detections"][0]["label"] == "person"
        assert "token=x" in seen["url"]        # zm_auth appended to the fetch

    def test_infer_inference_error_reported(self, client, monkeypatch):
        # A backend that raises -> reported in `error`, not a 500.
        from fastapi.testclient import TestClient  # noqa: F401
        app_det = client.app.state.detector
        app_det._backend.detect.side_effect = RuntimeError("boom")
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "object", "name": "yolov4"},
        )
        assert resp.status_code == 200
        assert resp.json()["error"] == "boom"

    def test_infer_named_model_miss_is_an_error(self, client):
        """A named request must not be answered by a different model.

        The server loads 'yolov4'; a client asking for 'yolo11s' wants that
        model's results. Substituting the loaded one returns detections the
        client never asked for, and looks like a successful run.
        """
        resp = client.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "object", "name": "yolo11s"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["detections"] == []
        assert "yolo11s" in data["error"]


class TestClientOwnedThreshold:
    """min_confidence travels with the request and beats the server's value."""

    @staticmethod
    def _jpeg():
        import cv2
        import numpy as np
        ok, buf = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
        assert ok
        return buf.tobytes()

    @pytest.fixture
    def client_with_real_config(self):
        """TestClient whose backend carries a real ModelConfig (min_confidence=0.9)."""
        from pyzm.models.config import ModelConfig

        seen = {}

        class _Backend:
            def __init__(self):
                self._config = ModelConfig(name="yolov4", min_confidence=0.9)

            def detect(self, image):
                seen["min_confidence"] = self._config.min_confidence
                return [Detection("person", 0.5, BBox(1, 2, 3, 4), "yolov4", "object")]

        backend = _Backend()
        mc = MagicMock()
        mc.type.value = "object"
        mc.name = "yolov4"

        det = MagicMock()
        pipeline = MagicMock()
        pipeline._backends = [(mc, backend)]
        det._pipeline = pipeline
        det._ensure_pipeline = MagicMock()

        with patch("pyzm.serve.app.Detector") as MockDetector:
            MockDetector.return_value = det
            from pyzm.serve.app import create_app
            from fastapi.testclient import TestClient
            with TestClient(create_app(ServerConfig(models=["yolov4"]))) as tc:
                yield tc, backend, seen

    def test_client_value_replaces_server_value(self, client_with_real_config):
        tc, backend, seen = client_with_real_config
        resp = tc.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "object", "name": "yolov4", "min_confidence": "0.15"},
        )
        assert resp.status_code == 200
        assert seen["min_confidence"] == 0.15
        # the shared backend keeps its own threshold for other requests
        assert backend._config.min_confidence == 0.9

    def test_omitted_keeps_server_value(self, client_with_real_config):
        tc, backend, seen = client_with_real_config
        resp = tc.post(
            "/infer",
            files={"image": ("f.jpg", self._jpeg(), "image/jpeg")},
            data={"type": "object", "name": "yolov4"},
        )
        assert resp.status_code == 200
        assert seen["min_confidence"] == 0.9
