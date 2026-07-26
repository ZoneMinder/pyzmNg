"""Remote inference: client-side glue for the dumb pyzm.serve gateway.

The gateway is a pure inference engine: given one image + one model reference it
runs that model and returns raw detections. All orchestration (model sequence,
pre_existing_labels gating, pattern/zone/size/past filtering, frame strategy)
stays on the client, inside :class:`~pyzm.ml.pipeline.ModelPipeline`.

:class:`RemoteInferenceBackend` implements the :class:`MLBackend` interface, so
the pipeline treats a remotely-served model exactly like a local one -- which is
what makes local and remote produce identical results (structural parity).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests

from pyzm.ml.backends.base import MLBackend
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

    from pyzm.models.config import ModelConfig

logger = logging.getLogger("pyzm.ml")


class GatewayUnreachable(Exception):
    """The gateway could not be reached (connection/timeout/HTTP error).

    Raised for *transport* failures so the pipeline lets it propagate to the
    event-level fallback (ml_fallback_local).
    """


class GatewayModelError(RuntimeError):
    """The gateway reached us but could not run this model (e.g. it has no model
    of that type loaded, or fetching the frame failed). The pipeline logs a
    clear one-line warning and skips this model -- it is not a crash.
    """


class GatewayClient:
    """Talks to a remote pyzm.serve gateway: auth + single-model inference."""

    def __init__(
        self,
        url: str,
        username: str | None = None,
        password: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.url = url.rstrip("/")
        self._username = username
        self._password = password
        self._timeout = timeout
        self._token: str | None = None
        # URL mode: when set (per frame, by the Detector), infer() tells the
        # gateway to FETCH the frame from ZM instead of uploading pixels.
        # Shape: {"url": <zm image url>, "zm_auth": <token>, "verify_ssl": bool}.
        self.current_frame: dict | None = None

    def _auth_headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        if not self._username:
            return {}
        resp = requests.post(
            f"{self.url}/login",
            json={"username": self._username, "password": self._password or ""},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        self._token = resp.json().get("access_token")
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def infer(
        self,
        image: "np.ndarray",
        mtype: str,
        name: str,
        min_confidence: float | None = None,
    ) -> list[Detection]:
        """Run one model on one frame on the gateway, return raw detections.

        URL mode (``current_frame`` set): send a ZM frame reference and let the
        gateway fetch it. Image mode: upload the decoded frame as lossless PNG
        (identical pixels -> exact local<->remote parity).

        ``min_confidence`` is a client-owned decision: the gateway applies the
        value we send instead of whatever its own copy of the model was loaded
        with, so the same config threshold applies locally and remotely.
        """
        frame = self.current_frame
        data = {"type": mtype, "name": name or ""}
        if min_confidence is not None:
            data["min_confidence"] = str(min_confidence)
        files = None
        if frame:
            data["url"] = frame["url"]
            data["zm_auth"] = frame.get("zm_auth", "")
            data["verify_ssl"] = "1" if frame.get("verify_ssl", True) else "0"
        else:
            import cv2  # lazy
            ok, buf = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("Failed to encode frame for remote inference")
            files = {"image": ("frame.png", buf.tobytes(), "image/png")}
        try:
            resp = requests.post(
                f"{self.url}/infer",
                data=data,
                files=files,
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # connect/timeout/HTTP error
            raise GatewayUnreachable(str(exc)) from exc
        payload = resp.json()
        if payload.get("error"):
            raise GatewayModelError(payload["error"])
        return [_detection_from_dict(d, name) for d in payload.get("detections", [])]


def _detection_from_dict(d: dict, model_name: str) -> Detection:
    box = d["box"]
    return Detection(
        label=d["label"],
        confidence=float(d["confidence"]),
        bbox=BBox(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
        model_name=d.get("model_name") or model_name,
        detection_type=d.get("type", "object"),
    )


class RemoteInferenceBackend(MLBackend):
    """MLBackend proxy that runs a model on the remote gateway instead of locally."""

    def __init__(self, model_config: "ModelConfig", client: GatewayClient) -> None:
        self._config = model_config
        self._client = client

    @property
    def name(self) -> str:
        return self._config.name or self._config.framework.value

    def load(self) -> None:  # nothing loads locally
        return None

    @property
    def is_loaded(self) -> bool:
        return True

    def detect(self, image: "np.ndarray") -> list[Detection]:
        return self._client.infer(
            image,
            self._config.type.value,
            self._config.name or "",
            min_confidence=self._config.min_confidence,
        )
