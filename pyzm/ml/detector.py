"""Top-level Detector -- public API for pyzm v2 ML detection.

Usage::

    from pyzm import Detector

    # Auto-discover all models in the default path
    det = Detector()

    # Pick specific models by name (resolved from base_path)
    det = Detector(models=["yolo11s", "yolo26s"])

    # Custom model directory
    det = Detector(models=["yolo11s"], base_path="/my/models")

    # Detect
    result = det.detect("/path/to/image.jpg")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pyzm.ml.pipeline import ModelPipeline
from pyzm.models.config import (
    DetectorConfig,
    FrameStrategy,
    ModelConfig,
    ModelFramework,
    ModelType,
    Processor,
)

from pyzm.ml.filters import filter_by_pattern, filter_by_size, filter_by_zone, filter_past_per_type
from pyzm.models.detection import DetectionResult

if TYPE_CHECKING:
    import numpy as np
    from pyzm.models.config import StreamConfig
    from pyzm.models.zm import Zone

logger = logging.getLogger("pyzm.ml")

DEFAULT_BASE_MODEL_PATH = "/var/lib/zmeventnotification/models"

# ---------------------------------------------------------------------------
# Model file discovery
# ---------------------------------------------------------------------------

# Extensions we recognise as model weight files
_WEIGHT_EXTS = {".weights", ".onnx", ".tflite"}
# Extensions we recognise as label files
_LABEL_EXTS = {".names", ".txt", ".labels"}


def _find_file(directory: Path, ext: str) -> Path | None:
    """Return the first file in *directory* matching the extension."""
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix == ext:
            return f
    return None


def _find_labels(directory: Path) -> str | None:
    """Find a labels file in *directory*, preferring .names > .txt > .labels."""
    for ext in (".names", ".txt", ".labels"):
        p = _find_file(directory, ext)
        if p:
            return str(p)
    return None


def _model_config_from_file(
    weights_path: Path,
    directory: Path,
    processor: Processor = Processor.CPU,
) -> ModelConfig:
    """Build a ModelConfig from a discovered weights file."""
    suffix = weights_path.suffix.lower()
    name = weights_path.stem

    if suffix == ".onnx":
        return ModelConfig(
            name=name,
            type=ModelType.OBJECT,
            framework=ModelFramework.OPENCV,
            processor=processor,
            weights=str(weights_path),
        )
    elif suffix == ".tflite":
        return ModelConfig(
            name=name,
            type=ModelType.OBJECT,
            framework=ModelFramework.CORAL,
            processor=Processor.TPU,
            weights=str(weights_path),
            labels=_find_labels(directory),
        )
    else:
        # .weights (Darknet)
        cfg = _find_file(directory, ".cfg")
        return ModelConfig(
            name=name,
            type=ModelType.OBJECT,
            framework=ModelFramework.OPENCV,
            processor=processor,
            weights=str(weights_path),
            config=str(cfg) if cfg else None,
            labels=_find_labels(directory),
        )


def _discover_models(
    base_path: Path,
    processor: Processor = Processor.CPU,
) -> list[ModelConfig]:
    """Scan *base_path* for model files and return ModelConfigs.

    Walks one level of subdirectories looking for weight files
    (.weights, .onnx, .tflite).
    """
    if not base_path.is_dir():
        logger.warning("Model base path %s does not exist", base_path)
        return []

    models: list[ModelConfig] = []
    for entry in sorted(base_path.iterdir()):
        if not entry.is_dir():
            continue
        for f in sorted(entry.iterdir()):
            if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
                mc = _model_config_from_file(f, entry, processor)
                logger.debug("Discovered model: %s (%s)", mc.name, f)
                models.append(mc)

    if not models:
        logger.warning("No models found in %s", base_path)
    return models


def _resolve_model_name(
    name: str,
    base_path: Path,
    processor: Processor = Processor.CPU,
) -> ModelConfig:
    """Resolve a model spec, optionally prefixed with the name to publish it as.

    ``"<published name>=<spec>"`` loads *spec* but calls the model *published
    name*. A remote client asks for a model by name, so this is how a gateway
    serves the names that appear in the client's config without renaming files:
    ``--models "YOLOv11 ONNX=yolo11s"``.

    A plain spec keeps whatever name the resolver derives from the weights.
    """
    alias = None
    if "=" in name:
        alias, name = name.split("=", 1)
        alias, name = alias.strip(), name.strip()

    resolved = _resolve_model_spec(name, base_path, processor)
    if alias:
        return resolved.model_copy(update={"name": alias})
    return resolved


def _resolve_model_spec(
    name: str,
    base_path: Path,
    processor: Processor = Processor.CPU,
) -> ModelConfig:
    """Resolve a model name string against a base directory.

    Search order:
    0. Absolute path to a weight file (e.g. ``/path/to/best.onnx``)
    1. Directory named *name* containing model files
    2. Any weight file whose stem matches *name* in any subdirectory
    3. Fall back to a bare ModelConfig with just the name and processor
    """
    # 0. Absolute path to a weight file
    abs_path = Path(name)
    if abs_path.is_absolute() and abs_path.is_file() and abs_path.suffix.lower() in _WEIGHT_EXTS:
        return _model_config_from_file(abs_path, abs_path.parent, processor)

    # 1. Direct directory match: e.g. "yolo11" -> base_path/yolo11/
    candidate_dir = base_path / name
    if candidate_dir.is_dir():
        for f in sorted(candidate_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
                return _model_config_from_file(f, candidate_dir, processor)

    # 2. File stem match across all subdirs: e.g. "yolo26s" -> ultralytics/yolo26s.onnx
    if base_path.is_dir():
        for subdir in sorted(base_path.iterdir()):
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.iterdir()):
                if f.is_file() and f.stem == name and f.suffix.lower() in _WEIGHT_EXTS:
                    return _model_config_from_file(f, subdir, processor)

    # 3. Fallback: bare config with no paths — _create_backend rejects this for
    #    any framework that needs weights, with a message naming the fix.
    logger.warning(
        "Model '%s' matched no weights file under %s. A plain model name is "
        "resolved against disk, so this falls back to the defaults "
        "(type=object, framework=opencv) and will fail to load. Models with no "
        "weights file of their own (dlib face recognition, cloud ALPR) must be "
        "declared under 'detector_config' with an explicit type and framework.",
        name, base_path,
    )
    return ModelConfig(name=name, processor=processor)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """Top-level detection API.

    Parameters
    ----------
    config:
        A fully specified :class:`DetectorConfig`.  Takes precedence over
        *models* and *base_path*.
    models:
        A convenience shorthand -- a list of model name strings
        (e.g. ``["yolo11s", "yolo26s"]``) or :class:`ModelConfig` objects.
        String names are resolved against *base_path* to find weight,
        config, and label files automatically.
    base_path:
        Directory containing model subdirectories.  Defaults to
        ``/var/lib/zmeventnotification/models``.  When *models* is
        ``None`` and *config* is ``None``, all models in this directory
        are auto-discovered.  When *models* contains name strings, they
        are resolved against this path.
    processor:
        Hardware target for auto-discovered/resolved models.  Accepts
        ``"cpu"``, ``"gpu"``, ``"tpu"`` or a :class:`Processor` enum.
        Ignored when *config* is provided or when *models* contains
        :class:`ModelConfig` objects (which carry their own processor).
    gateway:
        URL of a remote ``pyzm.serve`` server (e.g. ``http://gpu:5000``).
        When set, ``detect()`` sends images to the remote server instead
        of running inference locally.
    gateway_mode:
        ``"url"`` (default) sends frame URLs so the server fetches images
        directly from ZoneMinder.  ``"image"`` sends JPEG-encoded frames
        instead.  Only applies to ``detect_event()``; single-image
        ``detect()`` calls always use image mode.
    gateway_timeout:
        HTTP timeout in seconds for remote detection requests.
    gateway_username:
        Username for remote server authentication (optional).
    gateway_password:
        Password for remote server authentication (optional).
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        models: list[str | ModelConfig] | None = None,
        base_path: str | Path = DEFAULT_BASE_MODEL_PATH,
        processor: str | Processor = Processor.CPU,
        *,
        gateway: str | None = None,
        gateway_mode: str = "url",
        gateway_timeout: int = 60,
        gateway_username: str | None = None,
        gateway_password: str | None = None,
    ) -> None:
        bp = Path(base_path)
        proc = Processor(processor) if isinstance(processor, str) else processor

        if config is not None:
            self._config = config
        elif models is not None:
            model_configs: list[ModelConfig] = []
            for m in models:
                if isinstance(m, str):
                    model_configs.append(_resolve_model_name(m, bp, proc))
                else:
                    model_configs.append(m)
            self._config = DetectorConfig(models=model_configs)
        else:
            # Auto-discover all models from base_path
            discovered = _discover_models(bp, proc)
            self._config = DetectorConfig(models=discovered)

        self._pipeline: ModelPipeline | None = None

        # Remote gateway (dumb inference server). When set, remote-capable
        # models run on the gateway via RemoteInferenceBackend; all
        # orchestration (sequence, gating, filtering, frame strategy) stays
        # local, so local and remote produce identical results.
        from pyzm.ml.remote import GatewayClient
        self._gateway = gateway.rstrip("/") if gateway else None
        self._gateway_mode = gateway_mode  # reserved for URL-mode server-fetch
        self._gateway_timeout = gateway_timeout
        self._gateway_username = gateway_username
        self._gateway_password = gateway_password
        self._gw_client = (
            GatewayClient(self._gateway, gateway_username, gateway_password, gateway_timeout)
            if self._gateway else None
        )

    # -- private helpers ------------------------------------------------------

    def _ensure_pipeline(self, lazy: bool = False) -> ModelPipeline:
        if self._pipeline is None:
            self._pipeline = ModelPipeline(self._config, gateway_client=self._gw_client)
            if lazy:
                self._pipeline.prepare()
            else:
                self._pipeline.load()
        return self._pipeline

    @staticmethod
    def _load_image(path: str) -> "np.ndarray":
        import cv2  # lazy
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img

    def _apply_filters(
        self,
        detections: list,
        zones: list["Zone"] | None,
        image_shape: tuple[int, int],
        original_shape: tuple[int, int] | None = None,
    ) -> tuple[list, list]:
        """Apply client-side filters: pattern, size, zones, past detections.

        Parameters
        ----------
        original_shape:
            ``(height, width)`` of the image before resizing.  When provided
            and different from *image_shape*, zone polygons are rescaled from
            original coordinates to match the (resized) detection space.

        Returns (kept_detections, error_boxes).
        """
        h, w = image_shape
        zone_dicts = [z.as_dict() for z in zones] if zones else []

        # Rescale zone polygons when the image was resized
        if original_shape and zone_dicts and (original_shape[0] != h or original_shape[1] != w):
            orig_h, orig_w = original_shape
            xfactor = w / orig_w
            yfactor = h / orig_h
            for zd in zone_dicts:
                pts = zd.get("value") or zd.get("points", [])
                zd["value"] = [(int(x * xfactor), int(y * yfactor)) for x, y in pts]

        detections = filter_by_pattern(detections, self._config.pattern)
        detections = filter_by_size(detections, self._config.max_detection_size, (h, w))
        detections, error_boxes = filter_by_zone(
            detections, zone_dicts, (h, w),
            strategy=self._config.zone_match_strategy,
        )
        detections = filter_past_per_type(detections, self._config)

        return detections, error_boxes

    # -- public API -----------------------------------------------------------

    def detect(
        self,
        input: "str | np.ndarray | list[tuple[int | str, np.ndarray]]",
        zones: list["Zone"] | None = None,
    ) -> DetectionResult:
        """Run detection on one or more images.

        Parameters
        ----------
        input:
            - ``str``: path to an image file.
            - ``np.ndarray``: a single BGR image array.
            - ``list[tuple[frame_id, np.ndarray]]``: multiple frames.
              The best frame is chosen by ``frame_strategy``.
        zones:
            Optional detection zone polygons.

        Returns
        -------
        DetectionResult
        """
        import numpy as np  # lazy

        pipeline = self._ensure_pipeline()

        # Single image path
        if isinstance(input, str):
            image = self._load_image(input)
            result = pipeline.run(image, zones=zones)
            result.frame_id = "single"
            return result

        # Single numpy array
        if isinstance(input, np.ndarray):
            result = pipeline.run(input, zones=zones)
            result.frame_id = "single"
            return result

        # Multiple frames: list of (frame_id, image) tuples
        if isinstance(input, list):
            return self._detect_multi_frame(input, zones, pipeline)

        raise TypeError(f"Unsupported input type: {type(input)}")

    def detect_audio(
        self,
        audio_path: str,
        event_week: int = -1,
        lat: float = -1.0,
        lon: float = -1.0,
    ) -> DetectionResult:
        """Run audio detection on a standalone audio file.

        This is a convenience method for running audio backends (e.g.
        BirdNET) on an audio file without a ZoneMinder event.  Any format
        that ffmpeg can read (WAV, MP3, MP4, etc.) is supported.

        Parameters
        ----------
        audio_path:
            Path to an audio file.
        event_week:
            ISO week number (1–48) for seasonal filtering.  -1 disables.
        lat, lon:
            Latitude/longitude for location-based species filtering.
            -1 disables.

        Returns
        -------
        DetectionResult
        """
        import numpy as np  # lazy

        pipeline = self._ensure_pipeline()
        pipeline.set_audio_context(audio_path, event_week, lat, lon)

        # Audio backends ignore the image; pass a 1x1 dummy.
        dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        result = pipeline.run(dummy)
        result.frame_id = "audio"
        return result

    def detect_event(
        self,
        zm_client: "ZMClient",
        event_id: int,
        zones: list["Zone"] | None = None,
        stream_config: "StreamConfig | None" = None,
    ) -> DetectionResult:
        """Extract frames from a ZM event and run detection.

        Parameters
        ----------
        zm_client:
            A :class:`pyzm.client.ZMClient`.
        event_id:
            ZoneMinder event ID.
        zones:
            Optional detection zones.
        stream_config:
            Controls frame extraction (which frames, resize, etc.).

        Returns
        -------
        DetectionResult
        """
        from pyzm.models.config import StreamConfig as SC  # lazy

        sc = stream_config or SC()

        # URL mode: the gateway fetches frames from ZM, so we never download
        # them here. Only possible when every enabled model is remote-capable
        # (a client-side model, e.g. cloud ALPR or audio, needs local pixels).
        if self._gateway and self._gateway_mode == "url":
            from pyzm.ml.pipeline import REMOTE_CAPABLE_FRAMEWORKS
            enabled = [mc for mc in self._config.models if mc.enabled]
            if sc.resize:
                # The gateway fetches frames straight from ZM at full size, so a
                # configured resize could not be applied -- remote would infer on
                # different pixels than local. Download and upload instead.
                logger.info(
                    "URL mode: stream resize=%s is set; downloading frames locally "
                    "so local and remote see the same pixels.", sc.resize,
                )
            elif enabled and all(mc.framework in REMOTE_CAPABLE_FRAMEWORKS for mc in enabled):
                return self._detect_event_url(zm_client, event_id, zones, sc)
            else:
                logger.info(
                    "URL mode: a client-side model is enabled; downloading frames locally."
                )

        # Get Event object and extract frames via OOP API
        ev = zm_client.event(event_id)
        result = ev.extract_frames(stream_config=sc)

        # Unpack (frames, image_dims) tuple
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            frames, image_dims = result
            original_shape = image_dims.get("original")
        else:
            frames = result
            original_shape = None

        if not frames:
            logger.warning("No frames extracted for event %d", event_id)
            return DetectionResult()

        pipeline = self._ensure_pipeline()

        # Extract audio if any enabled model needs it
        wav_path = None
        has_audio_model = any(
            mc.type == ModelType.AUDIO and mc.enabled
            for mc in self._config.models
        )
        if has_audio_model:
            wav_path, week, mon_lat, mon_lon = self._extract_event_audio(
                zm_client, event_id,
            )
            pipeline.set_audio_context(wav_path, week, mon_lat, mon_lon)

        try:
            return self._detect_multi_frame(frames, zones, pipeline, original_shape=original_shape)
        finally:
            if wav_path:
                import os
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    @staticmethod
    def _url_frame_ids(sc) -> list[str]:
        """Frame ids to analyse in URL mode (no local download)."""
        if sc.frame_set:
            return [str(f) for f in sc.frame_set]
        if sc.max_frames:
            start = sc.start_frame or 1
            skip = sc.frame_skip or 1
            return [str(start + i * skip) for i in range(sc.max_frames)]
        return ["snapshot", "alarm", "1"]

    def _detect_event_url(self, zm_client, event_id, zones, sc) -> DetectionResult:
        """URL mode: the gateway fetches each frame; we never download pixels.

        Frame dimensions (needed for client-side size/zone filters) come from the
        monitor, not the image. The pipeline runs on a correctly-sized blank
        frame; each RemoteInferenceBackend fetches the real frame from ZM via the
        gateway using the per-frame context set below.
        """
        import numpy as np
        from pyzm.ml.remote import GatewayUnreachable

        api = zm_client.api
        portal = api.portal_url
        auth = api.auth.get_auth_string()
        verify = api.config.verify_ssl
        ev = zm_client.event(event_id)

        h = w = 0
        try:
            mon = zm_client.monitor(ev.monitor_id)
            h, w = int(mon.height), int(mon.width)
        except Exception:
            logger.debug("URL mode: could not read monitor dimensions", exc_info=True)
        if not (h and w):
            logger.info("URL mode: monitor dimensions unavailable; downloading frames.")
            result = ev.extract_frames(stream_config=sc)
            frames = result[0] if isinstance(result, tuple) else result
            if not frames:
                return DetectionResult()
            return self._detect_multi_frame(frames, zones, self._ensure_pipeline())

        pipeline = self._ensure_pipeline()
        strategy = self._config.frame_strategy
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        results: list[DetectionResult] = []
        try:
            for fid in self._url_frame_ids(sc):
                self._gw_client.current_frame = {
                    "url": f"{portal}/index.php?view=image&eid={event_id}&fid={fid}",
                    "zm_auth": auth, "verify_ssl": verify,
                }
                try:
                    r = pipeline.run(blank, zones=zones)
                except GatewayUnreachable:
                    raise  # let event-level fallback (ml_fallback_local) handle it
                except Exception:
                    logger.exception("URL-mode detection failed for frame %s", fid)
                    continue
                r.frame_id = fid
                results.append(r)
                if strategy in (FrameStrategy.FIRST, FrameStrategy.FIRST_NEW) and r.matched:
                    break
        finally:
            self._gw_client.current_frame = None

        if not results:
            return DetectionResult()
        best = results[0]
        for r in results[1:]:
            if _is_better(r, best, strategy):
                best = r
        return best

    # -- class methods --------------------------------------------------------

    @classmethod
    def from_config(cls, path: str) -> "Detector":
        """Load a Detector from a YAML configuration file.

        The YAML is expected to have a top-level structure that can be parsed
        into a :class:`DetectorConfig`.
        """
        import yaml  # lazy

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            raw = {}

        detector_config = DetectorConfig.model_validate(raw)
        return cls(config=detector_config)

    @classmethod
    def from_dict(cls, ml_options: dict) -> "Detector":
        """Build a Detector from an ``ml_sequence`` dict.

        This delegates to :meth:`DetectorConfig.from_dict` so existing
        YAML configurations work directly.  If ``ml_options["general"]``
        contains ``ml_gateway``, the detector is created in remote mode.
        """
        detector_config = DetectorConfig.from_dict(ml_options)
        general = ml_options.get("general", {})
        return cls(
            config=detector_config,
            gateway=general.get("ml_gateway"),
            gateway_mode=general.get("ml_gateway_mode", "url"),
            gateway_username=general.get("ml_user"),
            gateway_password=general.get("ml_password"),
            gateway_timeout=int(general.get("ml_timeout", 60)),
        )

    # -- audio extraction -----------------------------------------------------

    @staticmethod
    def _extract_event_audio(
        zm_client: "ZMClient",
        event_id: int,
    ) -> tuple[str | None, int, float, float]:
        """Extract audio from an event's video file for BirdNET analysis.

        Returns ``(wav_path, week, monitor_lat, monitor_lon)`` or
        ``(None, -1, -1.0, -1.0)`` on failure.
        """
        import os
        import subprocess
        import tempfile
        from datetime import datetime

        # Query DB for event video file and monitor location
        try:
            conn = zm_client._get_db()
        except Exception:
            logger.debug("Could not get DB connection, skipping audio extraction")
            return None, -1, -1.0, -1.0
        if conn is None:
            logger.debug("Could not connect to ZM database, skipping audio extraction")
            return None, -1, -1.0, -1.0

        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT E.DefaultVideo, E.StartDateTime, "
                "M.Latitude, M.Longitude "
                "FROM Events E JOIN Monitors M ON E.MonitorId = M.Id "
                "WHERE E.Id = %s",
                (event_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
        except Exception:
            logger.debug("Failed to query event %d for audio extraction", event_id, exc_info=True)
            return None, -1, -1.0, -1.0

        if not row or not row.get("DefaultVideo"):
            logger.debug("Event %d has no DefaultVideo", event_id)
            return None, -1, -1.0, -1.0

        # Build the video file path via Event.path()
        try:
            ev = zm_client.event(event_id)
            video_dir = ev.path()
        except Exception:
            logger.debug("Failed to get event path for %d", event_id, exc_info=True)
            return None, -1, -1.0, -1.0

        video_path = os.path.join(video_dir, row["DefaultVideo"])
        if not os.path.isfile(video_path):
            logger.debug("Video file not found: %s", video_path)
            return None, -1, -1.0, -1.0

        # Probe for audio stream
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            if "audio" not in probe.stdout:
                logger.debug("No audio stream in %s", video_path)
                return None, -1, -1.0, -1.0
        except Exception:
            logger.debug("ffprobe failed for %s", video_path, exc_info=True)
            return None, -1, -1.0, -1.0

        # Extract audio to temp WAV (48 kHz mono, PCM s16le)
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="zm_birdnet_")
        os.close(wav_fd)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "48000", "-ac", "1",
                    wav_path,
                ],
                capture_output=True, timeout=60, check=True,
            )
        except Exception:
            logger.debug("ffmpeg audio extraction failed for %s", video_path, exc_info=True)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            return None, -1, -1.0, -1.0

        # Compute week number (1-48, clamped for BirdNET)
        week = -1
        start_dt = row.get("StartDateTime")
        if start_dt:
            if isinstance(start_dt, str):
                try:
                    start_dt = datetime.fromisoformat(start_dt)
                except ValueError:
                    start_dt = None
            if start_dt is not None:
                week = min((start_dt.timetuple().tm_yday // 7) + 1, 48)

        monitor_lat = float(row.get("Latitude") or -1.0)
        monitor_lon = float(row.get("Longitude") or -1.0)

        logger.debug(
            "Extracted audio for event %d: %s (week=%d, lat=%.2f, lon=%.2f)",
            event_id, wav_path, week, monitor_lat, monitor_lon,
        )
        return wav_path, week, monitor_lat, monitor_lon

    # -- multi-frame logic ----------------------------------------------------

    def _detect_multi_frame(
        self,
        frames: list[tuple[int | str, "np.ndarray"]],
        zones: list["Zone"] | None,
        pipeline: ModelPipeline,
        original_shape: tuple[int, int] | None = None,
    ) -> DetectionResult:
        """Run detection on multiple frames and pick the best result using
        ``frame_strategy``."""
        strategy = self._config.frame_strategy
        all_results: list[DetectionResult] = []

        # Acquire session-level locks for exclusive-hardware backends
        # so the lock is held across ALL frames, not per-frame.
        locked_backends = []
        for _mc, backend in pipeline._backends:
            if backend.needs_exclusive_lock:
                backend.acquire_lock()
                locked_backends.append(backend)

        try:
            for frame_id, image in frames:
                try:
                    result = pipeline.run(image, zones=zones, original_shape=original_shape)
                    result.frame_id = frame_id
                    all_results.append(result)
                except Exception:
                    logger.exception("Error detecting frame %s", frame_id)
                    continue

                # Short-circuit for 'first' / 'first_new' strategies
                if strategy in (FrameStrategy.FIRST, FrameStrategy.FIRST_NEW) and result.matched:
                    logger.debug("Frame strategy %r: returning frame %s", strategy.value, frame_id)
                    return result

            if not all_results:
                return DetectionResult()

            # Pick best according to strategy
            best = all_results[0]
            for result in all_results[1:]:
                if _is_better(result, best, strategy):
                    best = result

            return best
        finally:
            for backend in locked_backends:
                backend.release_lock()


def _is_better(
    candidate: DetectionResult,
    current: DetectionResult,
    strategy: FrameStrategy,
) -> bool:
    """Return True if *candidate* is a better result than *current* under the
    given frame strategy."""
    if strategy in (FrameStrategy.FIRST, FrameStrategy.FIRST_NEW):
        # Already handled by short-circuit above; fallback to first match
        return candidate.matched and not current.matched

    if strategy == FrameStrategy.MOST:
        if len(candidate.detections) != len(current.detections):
            return len(candidate.detections) > len(current.detections)
        return sum(candidate.confidences) > sum(current.confidences)

    if strategy == FrameStrategy.MOST_UNIQUE:
        cand_unique = len(set(candidate.labels))
        curr_unique = len(set(current.labels))
        if cand_unique != curr_unique:
            return cand_unique > curr_unique
        return sum(candidate.confidences) > sum(current.confidences)

    if strategy == FrameStrategy.MOST_MODELS:
        candidate_models = {d.model_name for d in candidate.detections}
        current_models = {d.model_name for d in current.detections}
        if len(candidate_models) != len(current_models):
            return len(candidate_models) > len(current_models)
        if len(candidate.detections) != len(current.detections):
            return len(candidate.detections) > len(current.detections)
        return sum(candidate.confidences) > sum(current.confidences)

    return False
