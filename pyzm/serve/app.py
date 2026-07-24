"""FastAPI application factory for the pyzm ML detection server.

The server is a dumb inference engine: given one image and one model reference
(type + optional name) it runs that single model and returns raw detections.
It performs no filtering, no model-sequence orchestration, and no frame
selection -- all of that stays on the client (see pyzm.ml.pipeline /
pyzm.ml.remote). This is what keeps local and remote detection identical.
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from pyzm.ml.detector import Detector
from pyzm.models.config import ServerConfig
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
            })
        return {"models": result}

    def _find_backend(pipeline, mtype: str, name: str):
        """Return the loaded backend matching (type, optional name), or None."""
        if pipeline is None:
            return None
        # name is a preference: exact (type,name) wins; otherwise fall back to
        # the first loaded model of that type (server model names need not equal
        # the client's config names).
        fallback = None
        for mc, backend in pipeline._backends:
            if mc.type.value != mtype:
                continue
            if name and (mc.name or "") == name:
                return backend
            if fallback is None:
                fallback = backend
        return fallback

    @app.post("/infer", dependencies=auth_deps)
    async def infer(
        image: UploadFile = File(...),
        type: str = Form(...),
        name: str = Form(""),
    ):
        """Run ONE model on ONE image, return raw (unfiltered) detections."""
        import cv2
        import numpy as np

        contents = await image.read()
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
