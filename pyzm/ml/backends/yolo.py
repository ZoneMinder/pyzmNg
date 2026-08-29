"""Merged YOLO backend — absorbs pyzm.ml.yolo base logic.

Provides :class:`YoloBase` (shared blob creation, NMS, GPU setup, locking)
and a :func:`create_yolo_backend` factory that dispatches to
:class:`YoloOnnx` or :class:`YoloDarknet` based on weights extension.

Refs #23
"""

from __future__ import annotations

import logging
import re
import time as _time
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend, PortalockerMixin
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")

# Ceiling for the escalating GPU-retry backoff, in seconds. A genuinely broken
# GPU is re-probed at most this often; a transient fault still heals in one
# gpu_retry_seconds. Refs #66
_GPU_RETRY_BACKOFF_CAP = 900


class _GpuState:
    """GPU fallback state, shared by every shallow copy of a backend.

    ``pyzm.serve`` shallow-copies a loaded backend to give one request its own
    ``min_confidence``. The copies share the same ``net``, so they must share
    this too: a fallback on a copy switches the shared net to CPU, and without
    shared state the original -- the object ``/models`` reports on, and the one
    that would retry the GPU -- would go on claiming it runs on GPU.
    """

    __slots__ = ("processor", "retry_at", "delay")

    def __init__(self, processor: str, delay: int) -> None:
        self.processor = processor
        # monotonic deadline after which the GPU is worth retrying;
        # None = not degraded, or degraded with no retry scheduled.
        self.retry_at: float | None = None
        self.delay = delay


def _cv2_version() -> tuple[int, int, int]:
    """Return ``(major, minor, patch)`` from ``cv2.__version__``."""
    import cv2

    parts = [re.sub(r"[^0-9]", "", p) or "0" for p in cv2.__version__.split(".")]
    return (
        int(parts[0]) if len(parts) > 0 else 0,
        int(parts[1]) if len(parts) > 1 else 0,
        int(parts[2]) if len(parts) > 2 else 0,
    )


class YoloBase(MLBackend, PortalockerMixin):
    """Shared base for Darknet and ONNX YOLO backends.

    Subclasses must implement:
      - ``_load_model()``
      - ``_forward_and_parse(blob, width, height, conf_threshold)``
            → ``(class_ids, confidences, boxes)``
    """

    _DEFAULT_DIM = 416

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.net = None
        self.classes: list[str] | None = None
        self._gpu = _GpuState(config.processor.value, config.gpu_retry_seconds)
        self.model_height = config.model_height or self._DEFAULT_DIM
        self.model_width = config.model_width or self._DEFAULT_DIM
        self._init_lock()

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "yolo"

    @property
    def is_loaded(self) -> bool:
        return self.net is not None

    @property
    def processor(self) -> str:
        """The processor inference is *actually* running on right now."""
        return self._gpu.processor

    @processor.setter
    def processor(self, value: str) -> None:
        self._gpu.processor = value

    @property
    def requested_processor(self) -> str:
        """The processor this model was configured with, fallback or not."""
        return self._config.processor.value

    def load(self) -> None:
        logger.info(
            "%s: loading YOLO model (processor=%s, weights=%s)",
            self.name,
            self.processor,
            self._config.weights,
        )
        self._load_model()

        # Detect GPU→CPU fallback
        if self.processor != self.requested_processor:
            logger.warning(
                "%s: requested processor=%s but fell back to %s",
                self.name,
                self.requested_processor,
                self.processor,
            )
        else:
            logger.debug("%s: running on %s", self.name, self.processor)

    def detect(self, image: "np.ndarray") -> list[Detection]:
        import cv2
        import numpy as np

        if self.net is None:
            self.load()

        Height, Width = image.shape[:2]
        logger.debug(
            "%s: detect extracted image dimensions as: %dw x %dh",
            self.name,
            Width,
            Height,
        )

        if self._auto_lock:
            self.acquire_lock()

        try:
            blob = self._create_blob(image)

            nms_threshold = 0.4
            conf_threshold = 0.2
            if self._config.min_confidence < conf_threshold:
                conf_threshold = self._config.min_confidence

            self._maybe_restore_gpu()

            _t0 = _time.perf_counter()
            try:
                class_ids, confidences, boxes = self._forward_and_parse(
                    blob, Width, Height, conf_threshold
                )
                if self.processor == "gpu":
                    # A healthy run clears the escalating retry backoff, so the
                    # next transient fault is retried after the base delay.
                    self._gpu.delay = self._config.gpu_retry_seconds
            except cv2.error as e:
                if self.processor != "gpu":
                    raise
                if not self._config.allow_cpu_fallback:
                    logger.error(
                        "%s: GPU inference failed and CPU fallback is disabled: %s",
                        self.name,
                        e,
                    )
                    raise
                self._fall_back_to_cpu(e)
                class_ids, confidences, boxes = self._forward_and_parse(
                    blob, Width, Height, conf_threshold
                )

            diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
            logger.debug(
                "perf: processor:%s %s detection took: %s",
                self.processor,
                self.name,
                diff_time,
            )

            if self._auto_lock:
                self.release_lock()
        except:
            if self._auto_lock:
                self.release_lock()
            raise

        # NMS
        _t0 = _time.perf_counter()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug(
            "perf: processor:%s %s NMS filtering took: %s",
            self.processor,
            self.name,
            diff_time,
        )
        indices = np.array(indices).flatten()

        detections: list[Detection] = []
        for i in indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            conf = confidences[i]
            label = str(self.classes[class_ids[i]])

            if conf < self._config.min_confidence:
                logger.debug(
                    "%s: dropping %s (%.2f < %.2f)",
                    self.name,
                    label,
                    conf,
                    self._config.min_confidence,
                )
                continue

            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=BBox(
                        x1=int(round(x)),
                        y1=int(round(y)),
                        x2=int(round(x + w)),
                        y2=int(round(y + h)),
                    ),
                    model_name=self.name,
                    detection_type="object",
                )
            )
        return detections

    # -- internal helpers (shared) --------------------------------------------

    def populate_class_labels(self) -> None:
        labels_path = self._config.labels
        with open(labels_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def _fall_back_to_cpu(self, reason: object) -> None:
        """Move the net to CPU after a GPU failure and schedule a GPU retry.

        The fallback used to be permanent: a single transient CUDA error pinned
        a long-lived server to CPU for the rest of the process, several times
        slower, with nothing but one log line to say so. Scheduling a retry lets
        a momentary fault heal itself; the wait doubles on each further failure
        so a genuinely broken GPU is not re-probed on every request.

        A retry is always scheduled unless ``gpu_retry_seconds`` is 0. The one
        condition that cannot heal -- an OpenCV build with no CUDA support at
        all -- never reaches here; ``_setup_gpu()`` handles it directly. Refs #66
        """
        import cv2

        self.processor = "cpu"
        if self.net is not None:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        wait = self._gpu.delay
        if wait <= 0:
            self._gpu.retry_at = None
            logger.error(
                "%s: GPU failed: %s. Falling back to CPU for the life of this "
                "process (GPU retry disabled).",
                self.name,
                reason,
            )
            return

        self._gpu.retry_at = _time.monotonic() + wait
        self._gpu.delay = min(wait * 2, _GPU_RETRY_BACKOFF_CAP)
        logger.error(
            "%s: GPU failed: %s. Falling back to CPU; retrying GPU in %d seconds.",
            self.name,
            reason,
            wait,
        )

    def _maybe_restore_gpu(self) -> None:
        """Put the net back on CUDA once the fallback backoff has elapsed."""
        if self._gpu.retry_at is None or _time.monotonic() < self._gpu.retry_at:
            return

        import cv2

        self._gpu.retry_at = None
        logger.info("%s: retrying GPU inference after an earlier CPU fallback", self.name)
        self.processor = "gpu"
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception as e:
            self._fall_back_to_cpu(e)

    def _setup_gpu(self, cv2_ver: tuple[int, int, int]) -> None:
        """Configure CUDA backend if processor is 'gpu' and OpenCV supports it."""
        import cv2

        if self.processor == "gpu":
            if cv2_ver < (4, 2, 0):
                logger.error(
                    "%s: OpenCV %s does not support CUDA for DNNs (need 4.2+)",
                    self.name,
                    cv2.__version__,
                )
                self.processor = "cpu"
        else:
            logger.debug("%s: using CPU for detection", self.name)

        if self.processor == "gpu":
            logger.debug("%s: setting CUDA backend for OpenCV", self.name)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception as e:
                self._fall_back_to_cpu(f"CUDA backend setup failed: {e}")

    def _create_blob(self, image: "np.ndarray"):
        import cv2

        scale = 0.00392  # 1/255
        return cv2.dnn.blobFromImage(
            image, scale, (self.model_width, self.model_height), (0, 0, 0), True, crop=False
        )

    # -- abstract (subclass) --------------------------------------------------

    def _load_model(self) -> None:
        raise NotImplementedError

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        raise NotImplementedError


def create_yolo_backend(config: ModelConfig) -> YoloBase:
    """Factory: return :class:`YoloOnnx` or :class:`YoloDarknet` based on weights extension."""
    weights = config.weights or ""
    if weights.lower().endswith(".onnx"):
        from pyzm.ml.backends.yolo_onnx import YoloOnnx

        return YoloOnnx(config)
    else:
        from pyzm.ml.backends.yolo_darknet import YoloDarknet

        return YoloDarknet(config)
