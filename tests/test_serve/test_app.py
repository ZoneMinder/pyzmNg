"""Tests for pyzm.serve.app -- FastAPI detection server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import ServerConfig
from pyzm.models.detection import BBox, Detection, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_detector():
    """Return a mock Detector whose detect() returns a canned result."""
    det = MagicMock()
    det._pipeline = True  # so /health reports models_loaded=True
    det._config = MagicMock()
    det._config.models = [MagicMock()]
    det.detect.return_value = DetectionResult(
        detections=[
            Detection(
                label="person",
                confidence=0.95,
                bbox=BBox(10, 20, 50, 80),
                model_name="yolov4",
            )
        ],
        frame_id="single",
    )
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

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True


class TestDetect:
    def _make_jpeg(self):
        """Create a minimal valid JPEG bytes payload."""
        import cv2
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    @pytest.mark.integration
    def test_detect_returns_result(self, client):
        jpeg = self._make_jpeg()
        resp = client.post("/detect", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert "labels" in data
        assert "boxes" in data
        assert data["labels"] == ["person"]

    def test_detect_empty_file(self, client):
        resp = client.post("/detect", files={"file": ("test.jpg", b"", "image/jpeg")})
        assert resp.status_code == 400

    @pytest.mark.integration
    def test_detect_bad_image(self, client):
        resp = client.post("/detect", files={"file": ("test.jpg", b"not-a-jpeg", "image/jpeg")})
        assert resp.status_code == 400

    @pytest.mark.integration
    def test_detect_no_image_field(self, client):
        """image key is stripped from response."""
        jpeg = self._make_jpeg()
        resp = client.post("/detect", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        assert "image" not in resp.json()


class TestDetectUrls:
    """Tests for the /detect_urls endpoint."""

    def _make_jpeg_bytes(self):
        import cv2
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def test_detect_urls_empty_list(self, client):
        resp = client.post("/detect_urls", json={"urls": []})
        assert resp.status_code == 400

    def test_detect_urls_no_urls_key(self, client):
        resp = client.post("/detect_urls", json={})
        assert resp.status_code == 400

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_fetches_and_detects(self, mock_http, client):
        """Server fetches image from URL and returns per-frame raw results."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [
                {"frame_id": "snapshot", "url": "http://zm.example.com/image?eid=1&fid=snapshot"},
            ],
            "zm_auth": "token=abc123",
            "verify_ssl": False,
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["labels"] == ["person"]
        assert data["results"][0]["frame_id"] == "snapshot"

        # Verify auth was appended to URL
        fetched_url = mock_http.get.call_args[0][0]
        assert "token=abc123" in fetched_url

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_appends_auth_with_ampersand(self, mock_http, client):
        """zm_auth is appended with & when URL already has query params."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image?eid=1&fid=1"}],
            "zm_auth": "token=xyz",
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200

        fetched_url = mock_http.get.call_args[0][0]
        assert "&token=xyz" in fetched_url

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_appends_auth_with_question_mark(self, mock_http, client):
        """zm_auth is appended with ? when URL has no query params."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image"}],
            "zm_auth": "token=xyz",
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200

        fetched_url = mock_http.get.call_args[0][0]
        assert "?token=xyz" in fetched_url

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_no_auth(self, mock_http, client):
        """When zm_auth is empty, URL is not modified."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image?eid=1"}],
            "zm_auth": "",
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200

        fetched_url = mock_http.get.call_args[0][0]
        assert fetched_url == "http://zm/image?eid=1"

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_all_fetch_failures(self, mock_http, client):
        """When all URL fetches fail, returns empty results list."""
        mock_http.get.side_effect = ConnectionError("unreachable")

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image"}],
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_image_stripped(self, mock_http, client):
        """image key is stripped from each per-frame response."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {"urls": [{"frame_id": "1", "url": "http://zm/image"}]}
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert "image" not in data["results"][0]

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_returns_all_frames(self, mock_http, client):
        """Multiple frames each get their own result entry."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [
                {"frame_id": "snapshot", "url": "http://zm/image?fid=snapshot"},
                {"frame_id": "alarm", "url": "http://zm/image?fid=alarm"},
            ],
            "stop_on_match": False,
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["frame_id"] == "snapshot"
        assert data["results"][1]["frame_id"] == "alarm"

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_404_stops_after_consecutive_misses(self, mock_http, client):
        """Stops after contig_frames_before_error consecutive 404s."""
        not_found = MagicMock()
        not_found.status_code = 404
        mock_http.get.return_value = not_found

        payload = {
            "urls": [
                {"frame_id": str(i), "url": f"http://zm/image?fid={i}"}
                for i in range(10)
            ],
            "contig_frames_before_error": 3,
            "max_attempts": 1,
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        # Should have stopped after 3 consecutive 404s, not processed all 10
        assert mock_http.get.call_count == 3

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_404_resets_on_success(self, mock_http, client):
        """Consecutive 404 counter resets when a frame is fetched successfully."""
        jpeg_bytes = self._make_jpeg_bytes()
        ok_resp = MagicMock()
        ok_resp.content = jpeg_bytes
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()

        not_found = MagicMock()
        not_found.status_code = 404

        # 2 404s, then 1 success, then 2 more 404s → should not stop (counter resets)
        # With contig_frames_before_error=3, after reset only 2 consecutive → no stop
        mock_http.get.side_effect = [not_found, not_found, ok_resp, not_found, not_found]

        payload = {
            "urls": [
                {"frame_id": str(i), "url": f"http://zm/image?fid={i}"}
                for i in range(5)
            ],
            "contig_frames_before_error": 3,
            "max_attempts": 1,
            "stop_on_match": False,
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        # All 5 frames should have been attempted
        assert mock_http.get.call_count == 5

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_stop_on_match_stops_at_first_match(self, mock_http, client):
        """stop_on_match=True stops after first frame with detection."""
        jpeg_bytes = self._make_jpeg_bytes()
        ok_resp = MagicMock()
        ok_resp.content = jpeg_bytes
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = ok_resp
        payload = {
            "urls": [
                {"frame_id": "1", "url": "http://zm/image?fid=1"},
                {"frame_id": "2", "url": "http://zm/image?fid=2"},
                {"frame_id": "3", "url": "http://zm/image?fid=3"},
            ],
            "stop_on_match": True,
            "min_confidence": 0.5,
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Mock detector returns person:0.95; should stop after first match
        assert len(data["results"]) == 1
        assert mock_http.get.call_count == 1
    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_confidence_filter(self, mock_http, client):
        """Detections below min_confidence are filtered out."""
        jpeg_bytes = self._make_jpeg_bytes()
        ok_resp = MagicMock()
        ok_resp.content = jpeg_bytes
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = ok_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image?fid=1"}],
            "min_confidence": 0.99,  # Very high threshold — model won't reach it
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        results = resp.json()["results"]
        # All detections should be filtered out
        for r in results:
            assert not r.get("labels")
