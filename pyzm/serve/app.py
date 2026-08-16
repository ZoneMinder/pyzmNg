"""FastAPI application factory for the pyzm ML detection server.

The server is a dumb inference engine: given one image and one model reference
(type + optional name) it runs that single model and returns raw detections.
It performs no filtering, no model-sequence orchestration, and no frame
selection -- all of that stays on the client (see pyzm.ml.pipeline /
pyzm.ml.remote). This is what keeps local and remote detection identical.
"""

from __future__ import annotations
import copy as _copy
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import requests as http_requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from pyzm.ml.detector import Detector
from pyzm.models.config import ModelConfig, ServerConfig
from pyzm.serve.auth import create_login_route, create_token_dependency

logger = logging.getLogger("pyzm.serve")

# Env var used to hand the parsed ServerConfig to uvicorn worker processes.
_CONFIG_ENV_VAR = "PYZM_SERVER_CONFIG"


def config_to_env(config: ServerConfig) -> str:
    """Serialise a ServerConfig to JSON for the worker env var.

    ``model_dump_json()`` masks SecretStr fields (auth_password) as
    ``"**********"``, which would silently break /login in every worker.
    We inject the real secret value so workers authenticate correctly.
    Paired with :func:`config_from_env`.
    """
    import json

    data = config.model_dump(mode="json")
    data["auth_password"] = config.auth_password.get_secret_value()
    return json.dumps(data)


def config_from_env() -> ServerConfig:
    """Reconstruct a ServerConfig from the worker env var.

    Falls back to a default ServerConfig when the variable is absent
    (single-worker mode).  Paired with :func:`config_to_env`.
    """
    import json
    import os

    raw = os.environ.get(_CONFIG_ENV_VAR)
    return ServerConfig.model_validate(json.loads(raw)) if raw else ServerConfig()


def _apply_gpu_policy(
    models: list["ModelConfig"], config: ServerConfig
) -> list["ModelConfig"]:
    """Stamp the server-wide GPU-fallback policy onto every model config.

    ``--no-cpu-fallback`` / ``--gpu-retry-seconds`` are operator decisions about
    the whole gateway, but the behaviour lives on each backend's ModelConfig.
    Only options the operator actually set are applied, so a per-model value
    from a ``detector_config`` is never silently overwritten. Refs #66
    """
    update: dict[str, Any] = {}
    if not config.allow_cpu_fallback:
        update["allow_cpu_fallback"] = False
    if config.gpu_retry_seconds is not None:
        update["gpu_retry_seconds"] = config.gpu_retry_seconds
    if not update:
        return models
    return [mc.model_copy(update=update) for mc in models]


def get_app() -> FastAPI:
    """Factory entry point for uvicorn multi-worker mode.

    Reads the server configuration from the PYZM_SERVER_CONFIG environment
    variable (JSON-serialised ServerConfig) so that each worker
    process spawned by uvicorn receives the same configuration as the parent
    without having to re-parse CLI arguments.  Falls back to a default
    ServerConfig when the variable is absent (single-worker mode).

    Logging is configured here so that each worker inherits the correct level
    independently of the parent process.
    """
    config = config_from_env()
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("pyzm").setLevel(level)
    return create_app(config)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    The :class:`Detector` is created during the lifespan startup phase so
    models are loaded once and persist across requests.
    """
    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.detector_config is not None:
            detector = Detector(config=config.detector_config)
            lazy = not config.detector_config.models
        else:
            lazy = config.models == ["all"]
            detector = Detector(
                models=None if lazy else config.models,
                base_path=config.base_path,
                processor=config.processor,
            )
        detector._config.models = _apply_gpu_policy(detector._config.models, config)
        detector._ensure_pipeline(lazy=lazy)
        app.state.detector = detector
        mode = "lazy" if lazy else "eager"
        logger.info(
            "Detector ready (%s): %d model(s)", mode, len(detector._config.models)
        )
        yield

    app = FastAPI(title="pyzm ML Detection Server", lifespan=lifespan)

    # -- Optional auth -------------------------------------------------------
    auth_deps: list[Any] = []
    if config.auth_enabled:
        verify_token = create_token_dependency(config)
        auth_deps = [Depends(verify_token)]
    # Always register /login so clients with credentials configured don't
    # get a 404.  When auth is disabled the route accepts any credentials.
    app.post("/login")(create_login_route(config))

    # -- Routes --------------------------------------------------------------

    @app.get("/health")
    def health():
        models_loaded = (
            hasattr(app.state, "detector") and app.state.detector._pipeline is not None
        )
        return {"status": "ok", "models_loaded": models_loaded}

    @app.get("/models")
    def list_models():
        """Return the list of available models and their load status."""
        detector: Detector = app.state.detector
        pipeline = detector._pipeline
        if pipeline is None:
            return {"models": []}
        result = []
        for mc, backend in pipeline._backends:
            result.append({
                "name": mc.name or mc.framework.value,
                "type": mc.type.value,
                "framework": mc.framework.value,
                "loaded": backend.is_loaded,
                # What inference is running on *now* vs what was asked for. A
                # GPU model that hit a CUDA error reports processor="cpu" here,
                # which is the only thing that makes a degraded gateway visible
                # to a healthcheck. Refs #66
                "processor": getattr(backend, "processor", mc.processor.value),
                "requested_processor": mc.processor.value,
            })
        return {"models": result}

    def _find_backend(pipeline, mtype: str, name: str):
        """Return the loaded backend matching (type, optional name), or None.

        An unnamed request takes the first loaded model of that type. A *named*
        request must match exactly: substituting a different model would answer
        with detections the client never asked for, and it would look like a
        successful run.
        """
        if pipeline is None:
            return None
        for mc, backend in pipeline._backends:
            if mc.type.value != mtype:
                continue
            if not name or (mc.name or "") == name:
                return backend
        return None

    # Tiny decoded-frame cache keyed by URL so multiple models on the same
    # frame (URL mode) fetch+decode it once. Bounded; transparent optimization.
    _frame_cache: "OrderedDict[str, Any]" = OrderedDict()

    def _fetch_frame(url: str, zm_auth: str, verify_ssl: bool):
        import cv2
        import numpy as np
        if url in _frame_cache:
            _frame_cache.move_to_end(url)
            return _frame_cache[url]
        full = url if not zm_auth else f"{url}{'&' if '?' in url else '?'}{zm_auth}"
        resp = http_requests.get(full, timeout=10, verify=verify_ssl)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode fetched frame")
        _frame_cache[url] = frame
        if len(_frame_cache) > 8:
            _frame_cache.popitem(last=False)
        return frame

    @app.post("/infer", dependencies=auth_deps)
    async def infer(
        type: str = Form(...),
        name: str = Form(""),
        image: UploadFile = File(None),
        url: str = Form(""),
        zm_auth: str = Form(""),
        verify_ssl: str = Form("1"),
        min_confidence: float = Form(None),
    ):
        """Run ONE model on ONE frame, return raw (unfiltered) detections.

        URL mode: `url` set -> the server fetches the frame from ZM.
        Image mode: `image` uploaded -> inference on the uploaded frame.

        `min_confidence` is owned by the client: when sent it replaces the
        threshold this server loaded the model with, so one config drives both
        local and remote runs. Omitted (an older client) keeps the server's.
        """
        import cv2
        import numpy as np

        if url:
            try:
                frame = _fetch_frame(url, zm_auth, verify_ssl not in ("0", "false", "False", ""))
            except Exception as exc:
                logger.exception("URL-mode fetch failed: %s", url)
                return {"detections": [], "error": f"fetch failed: {exc}"}
        else:
            contents = await image.read() if image is not None else b""
            if not contents:
                raise HTTPException(status_code=400, detail="Empty file")
            arr = np.frombuffer(contents, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(status_code=400, detail="Could not decode image")

        detector: Detector = app.state.detector
        backend = _find_backend(detector._pipeline, type, name)
        if backend is None:
            return {"detections": [], "error": f"no model loaded for type={type} name={name!r}"}

        if min_confidence is not None and min_confidence != backend._config.min_confidence:
            # Shallow copy so the request gets its own threshold while sharing
            # the loaded weights; never mutate the backend other requests use.
            backend = _copy.copy(backend)
            backend._config = backend._config.model_copy(
                update={"min_confidence": min_confidence}
            )

        try:
            detections = backend.detect(frame)
        except Exception as exc:  # inference failure -> report, don't 500
            logger.exception("Inference failed for type=%s name=%s", type, name)
            return {"detections": [], "error": str(exc)}

        return {
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "box": d.bbox.as_list(),
                    "type": d.detection_type,
                    "model_name": d.model_name,
                }
                for d in detections
            ],
            "error": None,
        }

    return app
