"""Local <-> remote parity at the pipeline level (no models needed).

The remote path reuses ModelPipeline verbatim; only raw inference is swapped for
a RemoteInferenceBackend. Given identical raw detections, all downstream gating
and filtering (pattern, size, zone) must produce identical results. That is what
makes "your config works the same locally and remotely" true by construction.

A real-model end-to-end parity check lives in
tests/test_ml_e2e/test_remote_serve.py::TestRemoteDetection::test_local_remote_parity.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyzm.ml.backends.base import MLBackend
from pyzm.ml.pipeline import ModelPipeline
from pyzm.ml.remote import GatewayClient
from pyzm.models.config import DetectorConfig
from pyzm.models.detection import BBox, Detection

# Two raw detections: a big person and a small dog, at known positions.
RAW = [
    Detection("person", 0.9, BBox(10, 10, 50, 90), "m", "object"),
    Detection("dog", 0.8, BBox(60, 60, 80, 80), "m", "object"),
]


class _FakeBackend(MLBackend):
    """Local backend returning fixed detections (stands in for a real model)."""

    def __init__(self, dets):
        self._dets = dets

    @property
    def name(self):
        return "fake"

    def load(self):
        return None

    @property
    def is_loaded(self):
        return True

    def detect(self, image):
        return list(self._dets)


def _cfg(pattern=".*", max_size=None):
    obj_general = {"pattern": pattern}
    if max_size:
        obj_general["max_detection_size"] = max_size
    return DetectorConfig.from_dict({
        "general": {"model_sequence": "object", "same_model_sequence_strategy": "first",
                    "pattern": pattern},
        "object": {"general": obj_general,
                   "sequence": [{"name": "m", "object_framework": "opencv"}]},
    })


def _run_local(cfg, img):
    with patch.object(ModelPipeline, "_make_backend", lambda self, mc: _FakeBackend(RAW)):
        pipe = ModelPipeline(cfg)
        pipe.load()
        return pipe.run(img)


def _run_remote(cfg, img, zones=None):
    # Real RemoteInferenceBackend is used (object=opencv is remote-capable);
    # only the network call is stubbed to return the same RAW detections.
    with patch.object(GatewayClient, "infer", lambda self, image, t, n: list(RAW)):
        pipe = ModelPipeline(cfg, gateway_client=GatewayClient("http://gpu:5000"))
        pipe.load()
        return pipe.run(img, zones=zones)


def _key(result):
    return (
        result.labels,
        [d.bbox.as_list() for d in result.detections],
        [round(d.confidence, 5) for d in result.detections],
        [d.detection_type for d in result.detections],
    )


@pytest.mark.parametrize("pattern,expected", [
    (".*", ["person", "dog"]),
    ("(person)", ["person"]),
    ("(person|dog)", ["person", "dog"]),
])
def test_parity_pattern_filter(pattern, expected):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = _cfg(pattern=pattern)
    local = _run_local(cfg, img)
    remote = _run_remote(cfg, img)
    assert local.labels == expected          # filter actually did something
    assert _key(local) == _key(remote)       # and remote matches local exactly


def test_parity_size_filter():
    # max_detection_size drops the big person box; both paths must agree.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = _cfg(max_size="30%")
    local = _run_local(cfg, img)
    remote = _run_remote(cfg, img)
    assert _key(local) == _key(remote)


def test_parity_zone_filter():
    from pyzm.models.zm import Zone
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = _cfg()
    # Zone covering only the top-left quadrant -> keeps person, drops dog.
    zones = [Zone(name="tl", points=[(0, 0), (55, 0), (55, 95), (0, 95)])]
    local = None
    with patch.object(ModelPipeline, "_make_backend", lambda self, mc: _FakeBackend(RAW)):
        pipe = ModelPipeline(cfg)
        pipe.load()
        local = pipe.run(img, zones=zones)
    remote = _run_remote(cfg, img, zones=zones)
    assert local.labels == ["person"]
    assert _key(local) == _key(remote)
