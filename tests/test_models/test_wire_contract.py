"""Wire-format contract for DetectionResult.to_dict()/from_dict().

This is the exact dict shape that crosses the pyzm <-> zmeventnotification (ES)
boundary: the ML gateway serialises a DetectionResult with ``to_dict()`` and the
client rebuilds it with ``from_dict()``; ES's ``format_detection_output`` and
``zm_detect`` read these keys directly by name.

A silent rename/removal of any key here is a production KeyError in ES with the
rest of the suite green. These tests lock the contract with real objects (no
mocks) so such a change fails loudly and on purpose.
"""

from __future__ import annotations

from pyzm.models.detection import BBox, Detection, DetectionResult


# The frozen wire contract. If you change this set you are changing what ES
# consumes -- update ES (hook/zmes_hook_helpers/utils.py, zm_detect.py) in the
# same change and update this test deliberately.
WIRE_KEYS = {
    "boxes",
    "labels",
    "confidences",
    "frame_id",
    "image_dimensions",
    "image",
    "error_boxes",
    "model_names",
    "detection_types",
    "polygons",
}


def _sample_result() -> DetectionResult:
    return DetectionResult(
        detections=[
            Detection("person", 0.91, BBox(10, 20, 110, 220), model_name="yolov4", detection_type="object"),
            Detection("dog", 0.63, BBox(300, 40, 360, 120), model_name="yolov4", detection_type="object"),
        ],
        frame_id="snapshot",
        image_dimensions={"original": (600, 800)},
        error_boxes=[BBox(5, 5, 15, 15)],
    )


def test_to_dict_key_set_is_exactly_the_wire_contract():
    """to_dict() must emit exactly the keys ES reads -- no more, no less."""
    keys = set(_sample_result().to_dict().keys())
    assert keys == WIRE_KEYS, (
        f"wire contract drift: missing={WIRE_KEYS - keys}, extra={keys - WIRE_KEYS}"
    )


def test_empty_result_still_emits_full_wire_contract():
    """Even with no detections, every key ES reads must be present."""
    assert set(DetectionResult().to_dict().keys()) == WIRE_KEYS


def test_round_trip_preserves_detections():
    """from_dict(to_dict(r)) reconstructs the detection payload faithfully."""
    original = _sample_result()
    rebuilt = DetectionResult.from_dict(original.to_dict())

    assert rebuilt.labels == ["person", "dog"]
    assert rebuilt.confidences == [0.91, 0.63]
    assert rebuilt.boxes == [[10, 20, 110, 220], [300, 40, 360, 120]]
    assert [d.model_name for d in rebuilt.detections] == ["yolov4", "yolov4"]
    assert [d.detection_type for d in rebuilt.detections] == ["object", "object"]
    assert rebuilt.frame_id == "snapshot"
    assert rebuilt.image_dimensions == {"original": (600, 800)}
    assert [eb.as_list() for eb in rebuilt.error_boxes] == [[5, 5, 15, 15]]


def test_to_dict_values_align_by_index():
    """labels/boxes/confidences/model_names stay index-aligned (ES zips them)."""
    d = _sample_result().to_dict()
    n = len(d["labels"])
    assert len(d["boxes"]) == n
    assert len(d["confidences"]) == n
    assert len(d["model_names"]) == n
    assert len(d["detection_types"]) == n
