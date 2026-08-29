"""Tests for pyzm.ml.filters -- detection filtering functions."""

from __future__ import annotations

import os
import pickle
from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.detection import BBox, Detection


# ===================================================================
# Helpers
# ===================================================================

def _det(label: str, x1: int, y1: int, x2: int, y2: int, conf: float = 0.9) -> Detection:
    """Shortcut to create a Detection."""
    return Detection(
        label=label,
        confidence=conf,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        model_name="test",
    )


# ===================================================================
# TestFilterByZone
# ===================================================================

@pytest.mark.integration
class TestFilterByZone:
    """Tests for filter_by_zone. Requires shapely."""

    def test_detection_inside_zone_passes(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]
        zones = [{"name": "zone1", "points": [(0, 0), (100, 0), (100, 100), (0, 100)]}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"
        assert len(error_boxes) == 0

    def test_detection_outside_zone_filtered(self):
        from pyzm.ml.filters import filter_by_zone

        # Detection is at (200, 200) - (250, 250), zone is at (0,0)-(100,100)
        dets = [_det("person", 200, 200, 250, 250)]
        zones = [{"name": "zone1", "points": [(0, 0), (100, 0), (100, 100), (0, 100)]}]

        kept, error_boxes = filter_by_zone(dets, zones, (300, 300))
        assert len(kept) == 0
        assert len(error_boxes) == 1

    def test_zone_pattern_filters_labels(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [
            _det("person", 10, 10, 40, 40),
            _det("car", 10, 10, 40, 40),
        ]
        zones = [{"name": "driveway", "points": [(0, 0), (100, 0), (100, 100), (0, 100)], "pattern": "person"}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"
        assert len(error_boxes) == 1

    def test_no_zones_synthesises_full_image(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]

        kept, error_boxes = filter_by_zone(dets, [], (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"

    def test_multiple_zones_first_match_wins(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("dog", 10, 10, 40, 40)]
        zones = [
            {"name": "zone1", "points": [(0, 0), (50, 0), (50, 50), (0, 50)], "pattern": "person"},
            {"name": "zone2", "points": [(0, 0), (50, 0), (50, 50), (0, 50)], "pattern": "dog"},
        ]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "dog"

    def test_zone_with_value_key(self):
        """Test backward compat: zone dict uses 'value' instead of 'points'."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]
        zones = [{"name": "zone1", "value": [(0, 0), (100, 0), (100, 100), (0, 100)]}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1

    def test_zone_pattern_none_matches_all(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("anything", 10, 10, 40, 40)]
        zones = [{"name": "zone1", "points": [(0, 0), (100, 0), (100, 100), (0, 100)], "pattern": None}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1

    # -- ignore_pattern tests (Ref: ZoneMinder/pyzm#37) --

    def test_ignore_pattern_suppresses_matching_label(self):
        """Detection matching ignore_pattern in an intersecting zone is suppressed."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", 10, 10, 40, 40)]
        zones = [{
            "name": "driveway",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ignore_pattern": "(car|truck)",
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 0
        assert len(error_boxes) == 1

    def test_ignore_pattern_does_not_suppress_non_matching(self):
        """Detection NOT matching ignore_pattern passes through normally."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]
        zones = [{
            "name": "driveway",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ignore_pattern": "(car|truck)",
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"

    def test_ignore_pattern_with_positive_pattern(self):
        """ignore_pattern takes precedence over positive pattern for matching labels."""
        from pyzm.ml.filters import filter_by_zone

        dets = [
            _det("car", 10, 10, 40, 40),
            _det("person", 10, 10, 40, 40),
        ]
        zones = [{
            "name": "driveway",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "pattern": "(person|car)",
            "ignore_pattern": "car",
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"

    def test_ignore_pattern_none_does_not_suppress(self):
        """When ignore_pattern is None, no suppression occurs."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", 10, 10, 40, 40)]
        zones = [{
            "name": "zone1",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ignore_pattern": None,
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1



# ===================================================================
# TestFilterByZoneStrategies  (Ref: ZoneMinder/pyzmNg#68)
# ===================================================================

# The real-world sliver case from the issue: a car in the street on a
# 1920x1080 front-porch camera.  "street" holds 70.7% of the box and
# rejects it by pattern; "drivewayfar" clips 10.4% of the box and is
# patternless, so under "any_matching" it rescues a detection the
# operator meant to exclude.
_CAR_IN_STREET = (1554, 59, 1718, 167)
_FRAME = (1080, 1920)


def _street(pattern: str | None = "(NeverMatchThis)", ignore: str | None = None) -> dict:
    return {
        "name": "street",
        "points": [(1489, 84), (1919, 86), (1919, 296), (1483, 107)],
        "pattern": pattern,
        "ignore_pattern": ignore,
    }


def _driveway_far(pattern: str | None = None, ignore: str | None = None) -> dict:
    return {
        "name": "drivewayfar",
        "points": [(1446, 86), (1483, 92), (1912, 294), (1912, 345), (1424, 246), (1394, 244)],
        "pattern": pattern,
        "ignore_pattern": ignore,
    }


def _porch() -> dict:
    """Large zone that does NOT intersect the car box."""
    return {
        "name": "porch",
        "points": [(0, 0), (546, 2), (595, 256), (1426, 244), (1919, 347), (1919, 1079), (0, 1079)],
        "pattern": None,
    }


@pytest.mark.integration
class TestFilterByZoneStrategies:
    """Zone-resolution strategies. Requires shapely."""

    # -- any_matching (default, pre-2.6 behaviour) --

    def test_default_strategy_lets_a_sliver_zone_rescue(self):
        """Default is any_matching: drivewayfar rescues what street rejected."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]

        kept, error_boxes = filter_by_zone(dets, [_street(), _driveway_far()], _FRAME)
        assert [d.label for d in kept] == ["car"]
        assert error_boxes == []

    def test_any_matching_explicit_matches_default(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]

        kept, error_boxes = filter_by_zone(
            dets, [_street(), _driveway_far()], _FRAME, strategy="any_matching",
        )
        assert [d.label for d in kept] == ["car"]
        assert error_boxes == []

    # -- first_intersecting (pyzm 0.3.x / ES 6 parity) --

    def test_first_intersecting_rejection_is_terminal(self):
        """street decides and rejects; drivewayfar never gets a say."""
        from pyzm.ml.filters import filter_by_zone

        det = _det("car", *_CAR_IN_STREET)

        kept, error_boxes = filter_by_zone(
            [det], [_street(), _driveway_far()], _FRAME, strategy="first_intersecting",
        )
        assert kept == []
        assert error_boxes == [det.bbox]

    def test_first_intersecting_skips_non_intersecting_zones(self):
        """A zone the box misses does not get to decide."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]

        kept, _ = filter_by_zone(
            dets, [_porch(), _street(), _driveway_far()], _FRAME,
            strategy="first_intersecting",
        )
        assert kept == []

    def test_first_intersecting_is_order_dependent(self):
        """Reversing zone order flips the outcome -- the documented drawback."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]

        kept, error_boxes = filter_by_zone(
            dets, [_driveway_far(), _street()], _FRAME, strategy="first_intersecting",
        )
        assert [d.label for d in kept] == ["car"]
        assert error_boxes == []

    def test_first_intersecting_ignore_pattern_is_terminal(self):
        """ignore_pattern in the deciding zone suppresses outright."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]
        zones = [_street(pattern=None, ignore="(car|truck)"), _driveway_far()]

        kept, error_boxes = filter_by_zone(dets, zones, _FRAME, strategy="first_intersecting")
        assert kept == []
        assert len(error_boxes) == 1

    def test_first_intersecting_keeps_when_deciding_zone_matches(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]
        zones = [_street(pattern="(car|person)"), _driveway_far(pattern="(NeverMatchThis)")]

        kept, error_boxes = filter_by_zone(dets, zones, _FRAME, strategy="first_intersecting")
        assert [d.label for d in kept] == ["car"]
        assert error_boxes == []

    # -- largest_overlap --

    def test_largest_overlap_rejects_when_dominant_zone_rejects(self):
        """street covers 70.7% of the box, drivewayfar 10.4% -- street decides."""
        from pyzm.ml.filters import filter_by_zone

        det = _det("car", *_CAR_IN_STREET)

        kept, error_boxes = filter_by_zone(
            [det], [_street(), _driveway_far()], _FRAME, strategy="largest_overlap",
        )
        assert kept == []
        assert error_boxes == [det.bbox]

    def test_largest_overlap_is_order_independent(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]

        kept, _ = filter_by_zone(
            dets, [_driveway_far(), _street()], _FRAME, strategy="largest_overlap",
        )
        assert kept == []

    def test_largest_overlap_keeps_when_dominant_zone_matches(self):
        """Swap the patterns: the dominant zone now accepts the car."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]
        zones = [_street(pattern=None), _driveway_far(pattern="(NeverMatchThis)")]

        kept, error_boxes = filter_by_zone(dets, zones, _FRAME, strategy="largest_overlap")
        assert [d.label for d in kept] == ["car"]
        assert error_boxes == []

    def test_largest_overlap_ignores_non_intersecting_zones(self):
        """A big zone the box misses cannot win -- drivewayfar decides."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]
        zones = [_porch(), _driveway_far(pattern="(car|person)")]

        kept, _ = filter_by_zone(dets, zones, _FRAME, strategy="largest_overlap")
        assert [d.label for d in kept] == ["car"]

    def test_largest_overlap_ignore_pattern_is_terminal(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]
        zones = [_street(pattern=None, ignore="(car|truck)"), _driveway_far()]

        kept, error_boxes = filter_by_zone(dets, zones, _FRAME, strategy="largest_overlap")
        assert kept == []
        assert len(error_boxes) == 1

    def test_largest_overlap_tie_goes_to_first_zone(self):
        """Two identical polygons: the earlier one decides."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("dog", 10, 10, 40, 40)]
        square = [(0, 0), (50, 0), (50, 50), (0, 50)]
        zones = [
            {"name": "first", "points": square, "pattern": "person"},
            {"name": "second", "points": square, "pattern": "dog"},
        ]

        kept, _ = filter_by_zone(dets, zones, (100, 100), strategy="largest_overlap")
        assert kept == []

    # -- shared behaviour --

    @pytest.mark.parametrize(
        "strategy", ["any_matching", "first_intersecting", "largest_overlap"],
    )
    def test_detection_outside_every_zone_is_dropped(self, strategy):
        from pyzm.ml.filters import filter_by_zone

        det = _det("car", 10, 10, 40, 40)

        kept, error_boxes = filter_by_zone([det], [_street()], _FRAME, strategy=strategy)
        assert kept == []
        assert error_boxes == [det.bbox]

    @pytest.mark.parametrize(
        "strategy", ["any_matching", "first_intersecting", "largest_overlap"],
    )
    def test_empty_zone_list_passes_everything(self, strategy):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", *_CAR_IN_STREET)]

        kept, error_boxes = filter_by_zone(dets, [], _FRAME, strategy=strategy)
        assert kept == dets
        assert error_boxes == []

    def test_strategy_accepts_enum(self):
        from pyzm.ml.filters import filter_by_zone
        from pyzm.models.config import ZoneMatchStrategy

        dets = [_det("car", *_CAR_IN_STREET)]

        kept, _ = filter_by_zone(
            dets, [_street(), _driveway_far()], _FRAME,
            strategy=ZoneMatchStrategy.LARGEST_OVERLAP,
        )
        assert kept == []

    def test_unknown_strategy_raises(self):
        from pyzm.ml.filters import filter_by_zone

        with pytest.raises(ValueError):
            filter_by_zone(
                [_det("car", *_CAR_IN_STREET)], [_street()], _FRAME,
                strategy="closest_zone",
            )

# ===================================================================
# TestFilterBySize
# ===================================================================

class TestFilterBySize:
    """Tests for filter_by_size."""

    def test_filter_by_size_percentage(self):
        from pyzm.ml.filters import filter_by_size

        # Image is 100x100 = 10000px area. Detection is 50x50 = 2500px.
        # 50% of 10000 = 5000. Detection area (2500) < 5000 -> kept.
        dets = [_det("person", 0, 0, 50, 50)]
        result = filter_by_size(dets, "50%", (100, 100))
        assert len(result) == 1

    def test_filter_by_size_percentage_too_large(self):
        from pyzm.ml.filters import filter_by_size

        # 90x90 = 8100px area. 50% of 10000 = 5000. 8100 > 5000 -> filtered.
        dets = [_det("person", 0, 0, 90, 90)]
        result = filter_by_size(dets, "50%", (100, 100))
        assert len(result) == 0

    def test_filter_by_size_pixels(self):
        from pyzm.ml.filters import filter_by_size

        # Detection is 50x50 = 2500px. 300px threshold -> 2500 > 300 -> filtered.
        dets = [_det("person", 0, 0, 50, 50)]
        result = filter_by_size(dets, "300px", (100, 100))
        assert len(result) == 0

    def test_filter_by_size_pixels_passes(self):
        from pyzm.ml.filters import filter_by_size

        # Detection is 5x5 = 25px. 300px threshold -> 25 < 300 -> kept.
        dets = [_det("person", 0, 0, 5, 5)]
        result = filter_by_size(dets, "300px", (100, 100))
        assert len(result) == 1

    def test_filter_by_size_none_passes_all(self):
        from pyzm.ml.filters import filter_by_size

        dets = [_det("person", 0, 0, 99, 99)]
        result = filter_by_size(dets, None, (100, 100))
        assert len(result) == 1

    def test_filter_by_size_empty_string_passes_all(self):
        from pyzm.ml.filters import filter_by_size

        dets = [_det("person", 0, 0, 99, 99)]
        result = filter_by_size(dets, "", (100, 100))
        assert len(result) == 1

    def test_filter_by_size_multiple_detections(self):
        from pyzm.ml.filters import filter_by_size

        dets = [
            _det("small", 0, 0, 10, 10),   # area = 100
            _det("large", 0, 0, 80, 80),    # area = 6400
        ]
        result = filter_by_size(dets, "1000px", (100, 100))
        assert len(result) == 1
        assert result[0].label == "small"


# ===================================================================
# TestFilterByPattern
# ===================================================================

class TestFilterByPattern:
    """Tests for filter_by_pattern."""

    def test_regex_matching(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [
            _det("person", 0, 0, 10, 10),
            _det("car", 10, 10, 20, 20),
            _det("dog", 20, 20, 30, 30),
        ]
        result = filter_by_pattern(dets, "(person|car)")
        assert len(result) == 2
        assert result[0].label == "person"
        assert result[1].label == "car"

    def test_wildcard_matches_all(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10), _det("car", 10, 10, 20, 20)]
        result = filter_by_pattern(dets, ".*")
        assert len(result) == 2

    def test_empty_pattern_matches_all(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10)]
        result = filter_by_pattern(dets, "")
        assert len(result) == 1

    def test_no_match(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10)]
        result = filter_by_pattern(dets, "truck")
        assert len(result) == 0

    def test_partial_match_with_prefix(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10)]
        result = filter_by_pattern(dets, "per.*")
        assert len(result) == 1


# ===================================================================
# TestLoadSavePastDetections
# ===================================================================

class TestLoadSavePastDetections:
    """Tests for the composable load/save helpers."""

    def test_round_trip(self, tmp_path):
        from pyzm.ml.filters import load_past_detections, save_past_detections

        past_file = str(tmp_path / "past.pkl")
        dets = [_det("person", 10, 10, 50, 50), _det("car", 60, 60, 100, 100)]
        save_past_detections(past_file, dets)

        boxes, labels = load_past_detections(past_file)
        assert len(boxes) == 2
        assert boxes[0] == [10, 10, 50, 50]
        assert boxes[1] == [60, 60, 100, 100]
        assert labels == ["person", "car"]

    def test_load_missing_file(self, tmp_path):
        from pyzm.ml.filters import load_past_detections

        boxes, labels = load_past_detections(str(tmp_path / "nonexistent.pkl"))
        assert boxes == []
        assert labels == []

    def test_load_empty_file(self, tmp_path):
        from pyzm.ml.filters import load_past_detections

        past_file = str(tmp_path / "empty.pkl")
        with open(past_file, "wb"):
            pass
        boxes, labels = load_past_detections(past_file)
        assert boxes == []
        assert labels == []

    def test_save_empty_detections_is_noop(self, tmp_path):
        from pyzm.ml.filters import save_past_detections

        past_file = str(tmp_path / "past.pkl")
        save_past_detections(past_file, [])
        assert not os.path.exists(past_file)


# ===================================================================
# TestMatchPastDetectionsPure
# ===================================================================

@pytest.mark.integration
class TestMatchPastDetectionsPure:
    """Tests for the pure match_past_detections logic (no I/O)."""

    def test_no_saved_data_returns_all(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("person", 10, 10, 50, 50)]
        result = match_past_detections(dets, [], [], "5%")
        assert len(result) == 1

    def test_duplicate_removed(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("person", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["person"], "5%",
        )
        assert len(result) == 0

    def test_different_label_kept(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("car", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["person"], "5%",
        )
        assert len(result) == 1

    def test_aliases_match_across_labels(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("car", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["bus"], "5%",
            aliases=[["car", "bus"]],
        )
        assert len(result) == 0

    def test_ignore_labels_always_kept(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("dog", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["dog"], "5%",
            ignore_labels=["dog"],
        )
        assert len(result) == 1
