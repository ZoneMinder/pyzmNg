"""E2E: Remote detection via pyzm.serve -- health, detect, models, auth, gateway."""

from __future__ import annotations

import pytest

from tests.test_ml_e2e.conftest import (
    BIRD_IMAGE, BASE_PATH,
    find_one_model, start_serve, stop_serve, wait_for_serve,
)


@pytest.mark.serve
class TestRemoteDetection:

    PORT = 15200

    def test_health_endpoint(self):
        import requests
        port = self.PORT
        proc = start_serve([find_one_model()], port)
        try:
            assert wait_for_serve(port), "Server failed to start"
            r = requests.get(f"http://127.0.0.1:{port}/health")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ok"
            assert data["models_loaded"] is True
        finally:
            stop_serve(proc)

    def test_detect_endpoint(self):
        import requests
        port = self.PORT + 1
        proc = start_serve([find_one_model()], port)
        try:
            assert wait_for_serve(port), "Server failed to start"
            with open(BIRD_IMAGE, "rb") as f:
                r = requests.post(
                    f"http://127.0.0.1:{port}/infer",
                    files={"image": ("bird.jpg", f, "image/jpeg")},
                    data={"type": "object"},
                )
            assert r.status_code == 200
            data = r.json()
            assert "detections" in data
            assert isinstance(data["detections"], list)
        finally:
            stop_serve(proc)

    def test_models_endpoint_eager(self):
        import requests
        port = self.PORT + 2
        proc = start_serve([find_one_model()], port)
        try:
            assert wait_for_serve(port), "Server failed to start"
            r = requests.get(f"http://127.0.0.1:{port}/models")
            assert r.status_code == 200
            data = r.json()
            models = data["models"]
            assert len(models) > 0
            assert models[0]["loaded"] is True
        finally:
            stop_serve(proc)

    def test_models_all_lazy(self):
        import requests
        port = self.PORT + 3
        proc = start_serve(["all"], port)
        try:
            assert wait_for_serve(port), "Server failed to start"
            r = requests.get(f"http://127.0.0.1:{port}/models")
            data = r.json()
            models = data["models"]
            assert len(models) > 0
            loaded_before = [m["name"] for m in models if m["loaded"]]
            assert len(loaded_before) == 0, "Lazy models should not be loaded before detect"

            with open(BIRD_IMAGE, "rb") as f:
                r = requests.post(
                    f"http://127.0.0.1:{port}/infer",
                    files={"image": ("bird.jpg", f, "image/jpeg")},
                    data={"type": "object"},
                )
            assert r.status_code == 200

            r = requests.get(f"http://127.0.0.1:{port}/models")
            data = r.json()
            loaded_after = [m["name"] for m in data["models"] if m["loaded"]]
            assert len(loaded_after) > 0
        finally:
            stop_serve(proc)

    def test_gateway_image_mode(self):
        import requests
        from pyzm.ml.detector import Detector
        port = self.PORT + 4
        model = find_one_model()
        proc = start_serve([model], port)
        try:
            assert wait_for_serve(port), "Server failed to start"
            det = Detector(
                models=[model],
                base_path=BASE_PATH,
                gateway=f"http://127.0.0.1:{port}",
            )
            result = det.detect(BIRD_IMAGE)
            assert isinstance(result.detections, list)
            assert result.frame_id == "single"
        finally:
            stop_serve(proc)

    def test_detect_with_zones_via_gateway(self):
        """Zones are applied client-side when using gateway mode."""
        from pyzm.ml.detector import Detector
        from pyzm.models.zm import Zone
        import cv2
        port = self.PORT + 5
        model = find_one_model()
        proc = start_serve([model], port)
        try:
            assert wait_for_serve(port), "Server failed to start"
            det = Detector(
                models=[model],
                base_path=BASE_PATH,
                gateway=f"http://127.0.0.1:{port}",
            )
            img = cv2.imread(BIRD_IMAGE)
            h, w = img.shape[:2]
            # Zone covering full image but only allowing "bird" pattern
            zones = [Zone(name="full", points=[(0, 0), (w, 0), (w, h), (0, h)], pattern="bird")]
            result = det.detect(img, zones=zones)
            for d in result.detections:
                assert d.label == "bird"
        finally:
            stop_serve(proc)

    def test_auth_flow(self):
        import requests
        port = self.PORT + 6
        proc = start_serve(
            [find_one_model()], port,
            extra_args=["--auth", "--auth-user", "testuser", "--auth-password", "testpass",
                        "--token-secret", "e2e-test-secret"],
        )
        try:
            assert wait_for_serve(port), "Server failed to start"

            # Login
            r = requests.post(
                f"http://127.0.0.1:{port}/login",
                json={"username": "testuser", "password": "testpass"},
            )
            assert r.status_code == 200
            token = r.json()["access_token"]

            # Detect with token
            with open(BIRD_IMAGE, "rb") as f:
                r = requests.post(
                    f"http://127.0.0.1:{port}/infer",
                    files={"image": ("bird.jpg", f, "image/jpeg")},
                    data={"type": "object"},
                    headers={"Authorization": f"Bearer {token}"},
                )
            assert r.status_code == 200

            # Detect without token should fail
            with open(BIRD_IMAGE, "rb") as f:
                r = requests.post(
                    f"http://127.0.0.1:{port}/infer",
                    files={"image": ("bird.jpg", f, "image/jpeg")},
                    data={"type": "object"},
                )
            assert r.status_code in (401, 403)

            # Bad credentials should fail
            r = requests.post(
                f"http://127.0.0.1:{port}/login",
                json={"username": "wrong", "password": "wrong"},
            )
            assert r.status_code in (401, 403)
        finally:
            stop_serve(proc)

    def test_local_remote_parity(self):
        """Same image + model + zones must yield identical results local vs remote."""
        from pyzm.ml.detector import Detector
        import cv2
        port = self.PORT + 8
        model = find_one_model()
        proc = start_serve([model], port)
        try:
            assert wait_for_serve(port), "Server failed to start"
            img = cv2.imread(BIRD_IMAGE)
            local = Detector(models=[model], base_path=BASE_PATH)
            remote = Detector(models=[model], base_path=BASE_PATH,
                              gateway=f"http://127.0.0.1:{port}")
            rl = local.detect(img)
            rr = remote.detect(img)
            assert rl.labels == rr.labels
            assert [d.bbox.as_list() for d in rl.detections] == \
                   [d.bbox.as_list() for d in rr.detections]
            assert [round(d.confidence, 4) for d in rl.detections] == \
                   [round(d.confidence, 4) for d in rr.detections]
        finally:
            stop_serve(proc)

    def test_gateway_with_auth(self):
        from pyzm.ml.detector import Detector
        port = self.PORT + 7
        proc = start_serve(
            [find_one_model()], port,
            extra_args=["--auth", "--auth-user", "admin", "--auth-password", "secret",
                        "--token-secret", "e2e-secret"],
        )
        try:
            assert wait_for_serve(port), "Server failed to start"
            det = Detector(
                models=[find_one_model()],
                base_path=BASE_PATH,
                gateway=f"http://127.0.0.1:{port}",
                gateway_username="admin",
                gateway_password="secret",
            )
            result = det.detect(BIRD_IMAGE)
            assert isinstance(result.detections, list)
        finally:
            stop_serve(proc)
