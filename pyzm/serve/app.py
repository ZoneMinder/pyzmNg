"""FastAPI application factory for the pyzm ML detection server."""

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import requests as http_requests
from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile

from pyzm.ml.detector import Detector
from pyzm.ml.filters import filter_by_pattern, filter_by_zone
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

    @app.post("/detect", dependencies=auth_deps)
    async def detect(file: UploadFile = File(...)):
        import cv2
        import numpy as np

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        detector: Detector = app.state.detector
        result = detector.detect(image)

        data = result.to_dict()
        data.pop("image", None)
        return data

    @app.post("/detect_urls", dependencies=auth_deps)
    async def detect_urls(payload: dict = Body(...)):
        """Fetch images from URLs, run inference, return per-frame raw results."""
        import cv2
        import numpy as np

        urls = payload.get("urls", [])
        zm_auth = payload.get("zm_auth", "")
        verify_ssl = payload.get("verify_ssl", True)

        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        detector: Detector = app.state.detector
        results = []
        # Track consecutive 404s to detect end-of-event early
        consecutive_404 = 0
        max_consecutive_404 = payload.get("contig_frames_before_error", 5)

        for entry in urls:
            fid = str(entry.get("frame_id", ""))
            url = entry.get("url", "")
            if not url:
                continue

            if zm_auth:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}{zm_auth}"

            try:
                resp = http_requests.get(url, timeout=10, verify=verify_ssl)
                if resp.status_code == 404:
                    # Frame may not exist yet (live event still being written).
                    # Retry up to max_attempts times before counting as missing.
                    max_attempts = payload.get("max_attempts", 1)
                    sleep_secs = payload.get("sleep_between_attempts", 3)
                    fetched = False
                    for attempt in range(1, max_attempts + 1):
                        if attempt > 1:
                            logger.info("Frame %s: does not exist yet, retrying in %ds (attempt %d/%d)", fid, sleep_secs, attempt, max_attempts)
                            await asyncio.sleep(sleep_secs)
                            resp2 = http_requests.get(url, timeout=10, verify=verify_ssl)
                            if resp2.status_code != 404:
                                resp = resp2
                                fetched = True
                                break
                    if not fetched:
                        consecutive_404 += 1
                        logger.info("Frame %s: does not exist yet, skipping (%d consecutive)", fid, consecutive_404)
                        if consecutive_404 >= max_consecutive_404:
                            logger.info("Stopping after %d consecutive missing frames", consecutive_404)
                            break
                        continue
                consecutive_404 = 0
                logger.info("Frame %s: fetched OK, running inference", fid)
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is None:
                    logger.warning("Could not decode image from URL for frame %s", fid)
                    continue
            except Exception:
                logger.exception("Failed to fetch frame %s", fid)
                continue

            result = detector.detect(image)

            # --- Server-side filters ---
            # 1. Confidence filter: drop detections below the threshold
            #    forwarded by the client (derived from model min_confidence).
            min_confidence = payload.get("min_confidence", 0.3)
            filtered_low = [d for d in result.detections if d.confidence < min_confidence]
            if filtered_low:
                logger.info(
                    "Frame %s: %d detection(s) below min_confidence %.0f%% filtered out: %s",
                    fid, len(filtered_low), min_confidence * 100,
                    [(d.label, round(d.confidence * 100)) for d in filtered_low],
                )
            result.detections = [d for d in result.detections if d.confidence >= min_confidence]

            # 2. Pattern filter: keep only labels matching the client regex
            #    (e.g. "(person)" to ignore dog/cat detections).
            pattern = payload.get("pattern", ".*")
            if pattern and pattern != ".*":
                result.detections = filter_by_pattern(result.detections, pattern)

            data = result.to_dict()
            data.pop("image", None)
            data["frame_id"] = fid

            # 3. Zone short-circuit: check whether any surviving detection
            #    falls inside the requested zone polygons.
            #    NOTE: we do NOT reassign result.detections here — all detections
            #    (in-zone and out-of-zone) are returned to the client so it can
            #    apply its own zone filtering and log what was discarded.
            #    The zone check here is only used to drive stop_on_match.
            zones_data = payload.get("zones", [])
            has_zone_match = True
            if zones_data and result.detections:
                h, w = image.shape[:2]
                kept, _ = filter_by_zone(result.detections, zones_data, (h, w))
                has_zone_match = len(kept) > 0

            results.append(data)
            if payload.get("stop_on_match", True) and data.get("labels") and has_zone_match:
                break

        return {"results": results}

    return app
