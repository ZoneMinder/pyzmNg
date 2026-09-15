"""Tests for YoloDarknet model loading.

OpenCV 5 removed the Darknet importer, so ``cv2.dnn.readNet(weights, cfg)``
can no longer load ``.weights``/``.cfg`` models. The backend must say so in
terms the user can act on instead of letting a raw ``cv2.error`` escape.

Refs #70
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_darknet():
    """Build a minimal YoloDarknet with the attrs _load_model touches."""
    from pyzm.ml.backends.yolo_darknet import YoloDarknet
    from pyzm.models.config import ModelConfig

    obj = YoloDarknet(ModelConfig(
        name="YoloV4 GPU/CPU",
        weights="/var/lib/zmeventnotification/models/yolov4/yolov4.weights",
        config="/var/lib/zmeventnotification/models/yolov4/yolov4.cfg",
        disable_locks=True,
    ))
    # _setup_gpu touches the real cv2 DNN backend constants; class labels want
    # a names file on disk. Neither is what these tests are about.
    obj._setup_gpu = MagicMock()
    obj.populate_class_labels = MagicMock()
    return obj


def _load_with_cv2_version(obj, version):
    """Run _load_model against a faked cv2 of the given version.

    Returns the fake cv2 module so callers can assert on readNet.
    """
    fake_cv2 = MagicMock()
    fake_cv2.__version__ = version
    parts = [int(p) for p in version.split("-")[0].split(".")]
    with patch.dict(sys.modules, {"cv2": fake_cv2}), \
         patch("pyzm.ml.backends.yolo_darknet._cv2_version", return_value=tuple(parts)):
        obj._load_model()
    return fake_cv2


class TestDarknetOnOpenCV5:
    def test_opencv5_raises_actionable_error(self):
        obj = _make_darknet()
        with pytest.raises(RuntimeError) as exc:
            _load_with_cv2_version(obj, "5.1.0-dev")

        msg = str(exc.value)
        assert "Darknet importer" in msg
        assert "5.1.0-dev" in msg          # the version actually installed
        assert '"opencv-contrib-python<5"' in msg  # a version that works
        assert "ONNX" in msg               # the alternative
        assert "YoloV4 GPU/CPU" in msg     # which model failed
        assert "yolov4.weights" in msg     # and which file

    def test_opencv5_does_not_call_readnet(self):
        # readNet is what emits the unreadable cv2.error; the guard must run
        # before it, not translate its output afterwards.
        obj = _make_darknet()
        fake_cv2 = MagicMock()
        fake_cv2.__version__ = "5.0.0"
        with patch.dict(sys.modules, {"cv2": fake_cv2}), \
             patch("pyzm.ml.backends.yolo_darknet._cv2_version", return_value=(5, 0, 0)):
            with pytest.raises(RuntimeError):
                obj._load_model()
        fake_cv2.dnn.readNet.assert_not_called()

    def test_opencv413_still_loads(self):
        obj = _make_darknet()
        fake_cv2 = _load_with_cv2_version(obj, "4.13.0")

        fake_cv2.dnn.readNet.assert_called_once_with(
            obj._config.weights, obj._config.config
        )
        assert obj.net is fake_cv2.dnn.readNet.return_value
        obj._setup_gpu.assert_called_once_with((4, 13, 0))
        obj.populate_class_labels.assert_called_once()

    def test_opencv454_api_flag_still_set(self):
        # Regression lock: the getUnconnectedOutLayers() API switch must keep
        # working after the version lookup moved above readNet.
        obj = _make_darknet()
        _load_with_cv2_version(obj, "4.13.0")
        assert obj._is_get_unconnected_api_list is True

    def test_old_opencv_keeps_legacy_api_flag(self):
        obj = _make_darknet()
        _load_with_cv2_version(obj, "4.4.0")
        assert obj._is_get_unconnected_api_list is False
