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
    with patch.object(
        GatewayClient, "infer", lambda self, image, t, n, min_confidence=None: list(RAW)
    ):
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


# ---------------------------------------------------------------------------
# Multi-type and multi-frame parity (no models / no license / no images).
#
# Inference is stubbed identically on both paths; the REAL RemoteInferenceBackend
# + ModelPipeline + Detector do everything else. This proves that object, face
# and alpr all route and sequence identically local vs remote, and that frame
# strategy picks the same frame -- without a face model or an ALPR key.
# The real PNG->server->model transport is proven separately by the object
# e2e parity test (test_ml_e2e/test_remote_serve.py::test_local_remote_parity).
# ---------------------------------------------------------------------------

from pyzm.models.detection import BBox as _BBox, Detection as _Det

RAW_BY_TYPE = {
    "object": [_Det("person", 0.9, _BBox(10, 10, 40, 90), "obj", "object"),
               _Det("car", 0.8, _BBox(50, 50, 70, 70), "obj", "object")],
    "face": [_Det("john", 0.7, _BBox(12, 12, 30, 40), "face", "face")],
    "alpr": [_Det("ABC123", 0.6, _BBox(52, 52, 68, 60), "alpr", "alpr")],
}


class _TypedFake(MLBackend):
    def __init__(self, mtype):
        self._t = mtype

    @property
    def name(self):
        return self._t

    def load(self):
        return None

    @property
    def is_loaded(self):
        return True

    def detect(self, image):
        return list(RAW_BY_TYPE.get(self._t, []))


def _multitype_cfg():
    return DetectorConfig.from_dict({
        "general": {"model_sequence": "object,face,alpr",
                    "same_model_sequence_strategy": "union", "pattern": ".*"},
        "object": {"general": {"pattern": ".*"},
                   "sequence": [{"name": "o", "object_framework": "opencv"}]},
        "face": {"general": {"pattern": ".*"},
                 "sequence": [{"name": "f", "face_detection_framework": "dlib"}]},
        "alpr": {"general": {"pattern": ".*"},
                 "sequence": [{"name": "a", "alpr_service": "plate_recognizer"}]},
    })


def test_parity_multitype_object_face_alpr():
    """object+face -> remote; alpr -> local; combined result identical."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = _multitype_cfg()

    # Local: every backend is a local fake.
    with patch("pyzm.ml.pipeline._create_backend", lambda mc: _TypedFake(mc.type.value)):
        lp = ModelPipeline(cfg)
        lp.load()
        local = lp.run(img)

    # Remote: object+face become real RemoteInferenceBackend (HTTP stubbed);
    # alpr stays local via the patched _create_backend.
    with patch("pyzm.ml.pipeline._create_backend", lambda mc: _TypedFake(mc.type.value)), \
         patch.object(GatewayClient, "infer",
                      lambda self, image, t, n, min_confidence=None: list(RAW_BY_TYPE.get(t, []))):
        rp = ModelPipeline(cfg, gateway_client=GatewayClient("http://gpu:5000"))
        rp.load()
        remote = rp.run(img)

    assert set(local.labels) == {"person", "car", "john", "ABC123"}
    assert _key(local) == _key(remote)


def test_parity_alpr_never_sent_remote():
    """The gateway must never be asked to run alpr (it's client-only)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = _multitype_cfg()
    seen_types = []

    def spy_infer(self, image, t, n, min_confidence=None):
        seen_types.append(t)
        return list(RAW_BY_TYPE.get(t, []))

    with patch("pyzm.ml.pipeline._create_backend", lambda mc: _TypedFake(mc.type.value)), \
         patch.object(GatewayClient, "infer", spy_infer):
        rp = ModelPipeline(cfg, gateway_client=GatewayClient("http://gpu:5000"))
        rp.load()
        rp.run(img)

    assert "alpr" not in seen_types      # alpr ran locally, not on the gateway
    assert set(seen_types) == {"object", "face"}


def test_parity_multiframe_frame_strategy():
    """Same frame is selected local vs remote under a 'most' frame strategy."""
    from pyzm.ml.detector import Detector

    cfg = DetectorConfig.from_dict({
        "general": {"model_sequence": "object", "same_model_sequence_strategy": "first",
                    "pattern": ".*", "frame_strategy": "most"},
        "object": {"general": {"pattern": ".*"},
                   "sequence": [{"name": "o", "object_framework": "opencv"}]},
    })

    def dets_for(image):
        n = int(image[0, 0, 0])  # frame marker -> detection count
        return [_Det(f"o{k}", 0.9, _BBox(k, k, k + 2, k + 2), "o", "object") for k in range(n)]

    frames = [("f1", np.full((12, 12, 3), 1, np.uint8)),
              ("f2", np.full((12, 12, 3), 3, np.uint8)),   # richest -> should win
              ("f3", np.full((12, 12, 3), 2, np.uint8))]

    class _CountFake(MLBackend):
        @property
        def name(self):
            return "o"
        def load(self):
            return None
        @property
        def is_loaded(self):
            return True
        def detect(self, image):
            return dets_for(image)

    with patch("pyzm.ml.pipeline._create_backend", lambda mc: _CountFake()):
        local = Detector(config=cfg).detect(list(frames))

    with patch.object(
        GatewayClient, "infer",
        lambda self, image, t, n, min_confidence=None: dets_for(image),
    ):
        remote = Detector(config=cfg, gateway="http://gpu:5000").detect(list(frames))

    assert local.frame_id == remote.frame_id == "f2"
    assert _key(local) == _key(remote)


def test_client_min_confidence_is_sent_to_gateway():
    """min_confidence is a client decision, so it must travel with the request.

    Without this the gateway silently applies whatever threshold it was started
    with, and a config that detects locally returns nothing remotely.
    """
    cfg = DetectorConfig.from_dict({
        "general": {"model_sequence": "object", "same_model_sequence_strategy": "first"},
        "object": {"general": {"object_min_confidence": 0.15},
                   "sequence": [{"name": "m", "object_framework": "opencv",
                                 "object_min_confidence": 0.15}]},
    })
    sent = {}

    def _capture(self, image, t, n, min_confidence=None):
        sent["min_confidence"] = min_confidence
        return list(RAW)

    with patch.object(GatewayClient, "infer", _capture):
        pipe = ModelPipeline(cfg, gateway_client=GatewayClient("http://gpu:5000"))
        pipe.load()
        pipe.run(np.zeros((100, 100, 3), dtype=np.uint8))

    assert sent["min_confidence"] == 0.15
