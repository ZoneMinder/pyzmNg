"""GPU -> CPU fallback behaviour of the YOLO backends.

A transient CUDA error used to pin a backend to CPU for the life of the
object, which on a long-running ``pyzm.serve`` process turned a momentary
glitch into an indefinite slowdown that nothing reported. These tests cover
the recoverable fallback (retry after a backoff), the strict mode that refuses
to degrade at all, and the state sharing the server's per-request backend copy
depends on.

Refs #66
"""

from __future__ import annotations

import copy
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pyzm.ml.backends.yolo import YoloBase  # noqa: E402
from pyzm.models.config import ModelConfig, Processor  # noqa: E402


IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)

# One person detection, in the (class_ids, confidences, boxes) shape
# _forward_and_parse returns.
DETECTION = ([0], [0.9], [[10.0, 10.0, 20.0, 20.0]])


class _ScriptedYolo(YoloBase):
    """A YoloBase whose forward pass is scripted by the test.

    Each entry of *results* is consumed by one ``_forward_and_parse`` call and
    is either an exception to raise or a ``(class_ids, confidences, boxes)``
    tuple to return.
    """

    def __init__(self, config: ModelConfig, results: list) -> None:
        super().__init__(config)
        self._results = list(results)
        self.net = MagicMock()
        self.classes = ["person"]
        # processor in effect at each forward call, in order
        self.forward_processors: list[str] = []

    def _load_model(self) -> None:  # pragma: no cover - net is pre-set
        raise AssertionError("load() must not run: net is already set")

    def _forward_and_parse(self, blob, width, height, conf_threshold):
        self.forward_processors.append(self.processor)
        outcome = self._results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _make(results, *, processor=Processor.GPU, **cfg) -> _ScriptedYolo:
    config = ModelConfig(
        name="yolo11m",
        processor=processor,
        disable_locks=True,
        **cfg,
    )
    return _ScriptedYolo(config, results)


class TestTransientGpuFailure:
    """A GPU error degrades to CPU, answers the request, and schedules a retry."""

    def test_falls_back_to_cpu_and_still_answers(self):
        b = _make([cv2.error("CUDA-capable device(s) is/are busy"), DETECTION])

        detections = b.detect(IMAGE)

        assert [d.label for d in detections] == ["person"]
        assert b.processor == "cpu"
        assert b.forward_processors == ["gpu", "cpu"]
        assert b.net.setPreferableTarget.call_args[0][0] == cv2.dnn.DNN_TARGET_CPU

    def test_requested_processor_survives_the_fallback(self):
        b = _make([cv2.error("boom"), DETECTION])

        b.detect(IMAGE)

        assert b.processor == "cpu"
        assert b.requested_processor == "gpu"

    def test_fallback_schedules_a_gpu_retry(self):
        b = _make([cv2.error("boom"), DETECTION])

        b.detect(IMAGE)

        assert b._gpu.retry_at is not None
        assert b._gpu.retry_at - time.monotonic() == pytest.approx(60, abs=5)

    def test_gpu_is_not_retried_before_the_backoff_elapses(self):
        b = _make([cv2.error("boom"), DETECTION, DETECTION])

        b.detect(IMAGE)
        b.detect(IMAGE)

        assert b.processor == "cpu"
        assert b.forward_processors == ["gpu", "cpu", "cpu"]

    def test_gpu_is_retried_once_the_backoff_elapses(self):
        b = _make([cv2.error("boom"), DETECTION, DETECTION])
        b.detect(IMAGE)
        b._gpu.retry_at = time.monotonic() - 1  # backoff has elapsed

        b.detect(IMAGE)

        assert b.processor == "gpu"
        assert b.forward_processors == ["gpu", "cpu", "gpu"]
        assert b.net.setPreferableBackend.call_args[0][0] == cv2.dnn.DNN_BACKEND_CUDA
        assert b.net.setPreferableTarget.call_args[0][0] == cv2.dnn.DNN_TARGET_CUDA

    def test_repeated_failures_escalate_the_backoff(self):
        b = _make([cv2.error("boom"), DETECTION, cv2.error("boom"), DETECTION])
        b.detect(IMAGE)
        first_wait = b._gpu.retry_at - time.monotonic()
        b._gpu.retry_at = time.monotonic() - 1

        b.detect(IMAGE)  # retries GPU, fails again

        second_wait = b._gpu.retry_at - time.monotonic()
        assert second_wait == pytest.approx(first_wait * 2, abs=5)

    def test_a_healthy_gpu_run_resets_the_backoff(self):
        b = _make([cv2.error("boom"), DETECTION, DETECTION, cv2.error("boom"), DETECTION])
        b.detect(IMAGE)
        b._gpu.retry_at = time.monotonic() - 1
        b.detect(IMAGE)  # GPU healthy again -> backoff back to the base delay

        b.detect(IMAGE)  # fails once more

        assert b._gpu.retry_at - time.monotonic() == pytest.approx(60, abs=5)

    def test_retry_seconds_zero_makes_the_fallback_permanent(self):
        b = _make([cv2.error("boom"), DETECTION, DETECTION], gpu_retry_seconds=0)

        b.detect(IMAGE)
        b.detect(IMAGE)

        assert b._gpu.retry_at is None
        assert b.processor == "cpu"
        assert b.forward_processors == ["gpu", "cpu", "cpu"]


class TestStrictMode:
    """``allow_cpu_fallback=False``: fail the request rather than answer slowly."""

    def test_gpu_error_is_raised_instead_of_degrading(self):
        b = _make([cv2.error("boom")], allow_cpu_fallback=False)

        with pytest.raises(cv2.error):
            b.detect(IMAGE)

        assert b.processor == "gpu"
        assert b.forward_processors == ["gpu"]

    def test_a_cpu_error_is_never_swallowed(self):
        b = _make([cv2.error("bad blob")], processor=Processor.CPU)

        with pytest.raises(cv2.error):
            b.detect(IMAGE)

        assert b.forward_processors == ["cpu"]


class TestSharedFallbackState:
    """pyzm.serve shallow-copies a backend to give one request its own threshold.

    The copy shares the loaded ``net``, so it must share the fallback state too:
    otherwise a fallback on the copy switches the shared net to CPU while the
    original -- the one ``/models`` reports on and the one that would retry the
    GPU -- still claims to be running on GPU.
    """

    def test_a_shallow_copy_shares_the_fallback_state(self):
        b = _make([cv2.error("boom"), DETECTION])
        request_copy = copy.copy(b)

        request_copy.detect(IMAGE)

        assert b.processor == "cpu"
        assert b._gpu.retry_at is not None


class TestSetupGpuFallback:
    """Load-time GPU setup degrades the same way, but only when it may recover."""

    def test_a_failed_cuda_setup_schedules_a_retry(self):
        b = _make([])
        b.net.setPreferableBackend.side_effect = [cv2.error("no CUDA context"), None]

        b._setup_gpu((4, 12, 0))

        assert b.processor == "cpu"
        assert b._gpu.retry_at is not None

    def test_opencv_without_cuda_support_never_retries(self):
        b = _make([])

        b._setup_gpu((4, 1, 0))  # < 4.2: this build cannot ever use CUDA

        assert b.processor == "cpu"
        assert b._gpu.retry_at is None
