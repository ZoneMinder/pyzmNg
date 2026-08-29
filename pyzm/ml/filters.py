"""Pure functions for filtering detections.

Every function takes and returns :class:`Detection` / :class:`BBox` objects
from :mod:`pyzm.models.detection`.  Heavy dependencies (Shapely, pickle) are
imported at function level so they remain optional.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from pyzm.models.config import ZoneMatchStrategy
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    pass

logger = logging.getLogger("pyzm.ml")


# ---------------------------------------------------------------------------
# Zone filtering
# ---------------------------------------------------------------------------

def _zone_points(zone: dict) -> list:
    """Polygon points of *zone*, accepting either the ``points`` or ``value`` key."""
    return zone.get("points") or zone.get("value", [])


def _zone_verdict(det: Detection, zone: dict) -> bool:
    """Apply *zone*'s patterns to *det*.  ``True`` keeps the detection.

    Used by the strategies where a single zone decides the outcome:
    ``ignore_pattern`` suppresses, then ``pattern`` (default ``.*``) keeps.
    """
    zone_name = zone.get("name", "unnamed")
    zone_ignore = zone.get("ignore_pattern")
    if zone_ignore and re.match(zone_ignore, det.label):
        logger.debug(
            "filter_by_zone: %s matches ignore_pattern %s of deciding zone %s, suppressing",
            det.label, zone_ignore, zone_name,
        )
        return False

    pattern = zone.get("pattern") or ".*"
    if re.match(pattern, det.label):
        logger.debug(
            "filter_by_zone: %s matches pattern %s of deciding zone %s",
            det.label, pattern, zone_name,
        )
        return True

    logger.debug(
        "filter_by_zone: %s does NOT match pattern %s of deciding zone %s, rejecting",
        det.label, pattern, zone_name,
    )
    return False


def _keep_any_matching(det: Detection, bbox_poly, zones: list[dict], Polygon) -> bool:
    """Keep as soon as any intersecting zone's pattern matches."""
    for zone in zones:
        zone_name = zone.get("name", "unnamed")
        zone_poly = Polygon(_zone_points(zone))
        if not bbox_poly.intersects(zone_poly):
            logger.debug(
                "filter_by_zone: %s does NOT intersect zone %s",
                det.label, zone_name,
            )
            continue

        # Check ignore_pattern first -- suppress matching labels in this zone
        zone_ignore = zone.get("ignore_pattern")
        if zone_ignore and re.match(zone_ignore, det.label):
            logger.debug(
                "filter_by_zone: %s intersects zone %s and matches ignore_pattern %s, suppressing",
                det.label, zone_name, zone_ignore,
            )
            continue  # try other zones

        # Zone intersects -- now check the pattern
        pattern = zone.get("pattern") or ".*"
        if re.match(pattern, det.label):
            logger.debug(
                "filter_by_zone: %s intersects zone %s and matches pattern %s",
                det.label, zone_name, pattern,
            )
            return True  # matched on first zone is enough

        logger.debug(
            "filter_by_zone: %s intersects zone %s but does NOT match pattern %s",
            det.label, zone_name, pattern,
        )
    return False


def _keep_first_intersecting(det: Detection, bbox_poly, zones: list[dict], Polygon) -> bool:
    """The first zone the box intersects decides, whether or not it matches."""
    for zone in zones:
        zone_poly = Polygon(_zone_points(zone))
        if not bbox_poly.intersects(zone_poly):
            logger.debug(
                "filter_by_zone: %s does NOT intersect zone %s",
                det.label, zone.get("name", "unnamed"),
            )
            continue
        return _zone_verdict(det, zone)
    return False


def _keep_largest_overlap(det: Detection, bbox_poly, zones: list[dict], Polygon) -> bool:
    """The zone covering most of the bounding box decides.  Ties go to the
    earlier zone."""
    best_zone: dict | None = None
    best_area = -1.0

    for zone in zones:
        zone_poly = Polygon(_zone_points(zone))
        if not bbox_poly.intersects(zone_poly):
            logger.debug(
                "filter_by_zone: %s does NOT intersect zone %s",
                det.label, zone.get("name", "unnamed"),
            )
            continue
        area = bbox_poly.intersection(zone_poly).area
        if area > best_area:
            best_zone, best_area = zone, area

    if best_zone is None:
        return False

    box_area = bbox_poly.area
    logger.debug(
        "filter_by_zone: zone %s covers most of %s (%.1f%% of the box), it decides",
        best_zone.get("name", "unnamed"), det.label,
        (best_area / box_area * 100) if box_area else 0.0,
    )
    return _zone_verdict(det, best_zone)


_ZONE_STRATEGIES = {
    ZoneMatchStrategy.ANY_MATCHING: _keep_any_matching,
    ZoneMatchStrategy.FIRST_INTERSECTING: _keep_first_intersecting,
    ZoneMatchStrategy.LARGEST_OVERLAP: _keep_largest_overlap,
}


def filter_by_zone(
    detections: list[Detection],
    zones: list[dict],
    image_shape: tuple[int, int],
    strategy: ZoneMatchStrategy | str = ZoneMatchStrategy.ANY_MATCHING,
) -> tuple[list[Detection], list[BBox]]:
    """Keep only detections whose bounding box intersects at least one zone.

    Parameters
    ----------
    detections:
        Raw detections from a backend.
    zones:
        Each zone is a dict-like with ``name`` (str), ``points``
        (list of (x, y) tuples), and optionally ``pattern`` (str | None) and
        ``ignore_pattern`` (str | None).  These can be
        :class:`pyzm.models.zm.Zone` objects turned into dicts via
        :meth:`as_dict`, or simple dicts.
    image_shape:
        ``(height, width)`` of the analysed image.
    strategy:
        How to resolve a box that intersects several zones
        (:class:`~pyzm.models.config.ZoneMatchStrategy`):

        ``any_matching`` (default)
            Keep as soon as any intersecting zone's ``pattern`` matches.  A
            zone that rejects the detection does not stop a later zone from
            keeping it.
        ``first_intersecting``
            The first intersecting zone decides, keep or reject.  This is the
            pyzm 0.3.x / ES 6 behaviour.  Order-dependent.
        ``largest_overlap``
            The zone covering the largest share of the bounding box decides.
            Order-independent; ties go to the earlier zone.

        Under the latter two, a rejection -- by ``ignore_pattern`` or by a
        ``pattern`` mismatch -- is final, so an exclusion zone cannot be
        overridden by a zone the box merely clips.

    Returns
    -------
    kept:
        Detections kept by the deciding zone(s).
    error_boxes:
        Bounding boxes of detections that were filtered out.
    """
    # No zones = no filtering, pass everything through.
    if not zones:
        return detections, []

    from shapely.geometry import Polygon  # optional dependency

    keep_fn = _ZONE_STRATEGIES[ZoneMatchStrategy(strategy)]

    kept: list[Detection] = []
    error_boxes: list[BBox] = []

    for det in detections:
        bbox_poly = Polygon(det.bbox.as_polygon_coords())
        if keep_fn(det, bbox_poly, zones, Polygon):
            kept.append(det)
        else:
            error_boxes.append(det.bbox)

    return kept, error_boxes


# ---------------------------------------------------------------------------
# Size filtering
# ---------------------------------------------------------------------------

def _parse_size_spec(spec: str, total_area: int) -> float:
    """Parse a size spec like ``"50%"`` or ``"300px"`` into absolute pixels."""
    m = re.match(r"(\d*\.?\d+)(px|%)?$", spec, re.IGNORECASE)
    if not m:
        logger.error("Invalid size spec: %s", spec)
        return 0.0
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "%":
        return value / 100.0 * total_area
    # Default (no unit or "px") -> absolute pixels
    return value


def filter_by_size(
    detections: list[Detection],
    max_size: str | None,
    image_shape: tuple[int, int],
) -> list[Detection]:
    """Filter detections whose area exceeds *max_size*.

    *max_size* may be ``"50%"`` (of image area) or ``"300px"`` (absolute
    pixel area).  If *max_size* is ``None`` or empty, all detections pass.
    """
    if not max_size:
        return detections

    h, w = image_shape
    max_area = _parse_size_spec(max_size, h * w)
    if max_area <= 0:
        return detections

    kept: list[Detection] = []
    for det in detections:
        if det.bbox.area > max_area:
            logger.debug(
                "filter_by_size: dropping %s (area %d > max %d)",
                det.label, det.bbox.area, int(max_area),
            )
        else:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Pattern filtering
# ---------------------------------------------------------------------------

def filter_by_pattern(
    detections: list[Detection],
    pattern: str,
) -> list[Detection]:
    """Keep only detections whose label matches *pattern* (regex)."""
    if not pattern or pattern == ".*":
        return detections

    compiled = re.compile(pattern)
    kept: list[Detection] = []
    for det in detections:
        if compiled.match(det.label):
            kept.append(det)
        else:
            logger.debug(
                "filter_by_pattern: dropping %s (does not match %s)",
                det.label, pattern,
            )
    return kept


# ---------------------------------------------------------------------------
# Past-detection filtering
# ---------------------------------------------------------------------------

def load_past_detections(past_file: str) -> tuple[list[list[int]], list[str]]:
    """Load ``(saved_boxes, saved_labels)`` from a pickle file.

    Returns ``([], [])`` on missing file, empty file, or read error.
    """
    import pickle  # lazy import

    try:
        with open(past_file, "rb") as fh:
            saved_boxes: list[list[int]] = pickle.load(fh)
            saved_labels: list[str] = pickle.load(fh)
        return saved_boxes, saved_labels
    except FileNotFoundError:
        logger.debug("No past-detection file found at %s", past_file)
    except EOFError:
        logger.debug("Empty past-detection file at %s, removing", past_file)
        try:
            os.remove(past_file)
        except OSError:
            pass
    except Exception:
        logger.exception("Error reading past detections from %s", past_file)
    return [], []


def save_past_detections(past_file: str, detections: list[Detection]) -> None:
    """Save current detections to a pickle file for future comparisons."""
    import pickle  # lazy import

    if not detections:
        return
    try:
        with open(past_file, "wb") as fh:
            pickle.dump([d.bbox.as_list() for d in detections], fh)
            pickle.dump([d.label for d in detections], fh)
        logger.debug("Saved %d detections to %s", len(detections), past_file)
    except Exception:
        logger.exception("Error saving past detections to %s", past_file)


def match_past_detections(
    detections: list[Detection],
    saved_boxes: list[list[int]],
    saved_labels: list[str],
    max_diff_area: str = "5%",
    label_area_overrides: dict[str, str] | None = None,
    ignore_labels: list[str] | None = None,
    aliases: list[list[str]] | None = None,
) -> list[Detection]:
    """Filter detections against past data.  Pure logic, no I/O.

    Parameters
    ----------
    saved_boxes, saved_labels:
        Previously saved detection data (from :func:`load_past_detections`).
    max_diff_area:
        Default area tolerance, e.g. ``"5%"`` or ``"300px"``.
    label_area_overrides:
        Per-label area tolerance, e.g. ``{"car": "10%"}``.
    ignore_labels:
        Labels to skip entirely (always kept, never matched).
    aliases:
        Groups of equivalent labels, e.g. ``[["car","bus","truck"]]``.
    """
    from shapely.geometry import Polygon  # optional dependency

    if not saved_boxes:
        return list(detections)

    label_area_overrides = label_area_overrides or {}
    ignore_labels = ignore_labels or []
    alias_map: dict[str, str] = {}
    for group in (aliases or []):
        canonical = group[0]
        for label in group:
            alias_map[label] = canonical

    kept: list[Detection] = []
    for det in detections:
        if det.label in ignore_labels:
            kept.append(det)
            continue

        det_poly = Polygon(det.bbox.as_polygon_coords())
        det_canonical = alias_map.get(det.label, det.label)
        found_match = False

        for saved_idx, saved_box in enumerate(saved_boxes):
            saved_canonical = alias_map.get(saved_labels[saved_idx], saved_labels[saved_idx])
            if saved_canonical != det_canonical:
                continue

            saved_bbox = BBox(x1=saved_box[0], y1=saved_box[1], x2=saved_box[2], y2=saved_box[3])
            saved_poly = Polygon(saved_bbox.as_polygon_coords())

            if not saved_poly.intersects(det_poly):
                continue

            if det_poly.contains(saved_poly):
                diff_area = det_poly.difference(saved_poly).area
                ref_area = det_poly.area
            else:
                diff_area = saved_poly.difference(det_poly).area
                ref_area = saved_poly.area

            effective_max = label_area_overrides.get(det.label, max_diff_area)
            max_pixels = _parse_size_spec(effective_max, int(ref_area)) if ref_area > 0 else 0

            if diff_area <= max_pixels:
                logger.debug(
                    "match_past_detections: %s at %s matches saved %s at %s (diff=%.0f <= max=%.0f), removing",
                    det.label, det.bbox, saved_labels[saved_idx], saved_box,
                    diff_area, max_pixels,
                )
                found_match = True
                break

        if not found_match:
            kept.append(det)

    return kept


# ---------------------------------------------------------------------------
# Composite past-detection filter (per model-type with global fallback)
# ---------------------------------------------------------------------------

def filter_past_per_type(
    detections: list[Detection],
    config: "DetectorConfig",
) -> list[Detection]:
    """Apply past-detection dedup per model-type using config settings.

    This is the standalone version of the logic formerly in
    ``ModelPipeline._filter_past_per_type()``.
    """
    from collections import defaultdict

    from pyzm.models.config import ModelType, TypeOverrides

    if not detections:
        return detections

    # Quick check: is past-detection matching enabled for any type?
    any_enabled = config.match_past_detections
    if not any_enabled:
        for tov in config.type_overrides.values():
            if tov.match_past_detections is True:
                any_enabled = True
                break
    if not any_enabled:
        return detections

    if config.monitor_id:
        past_file = os.path.join(config.image_path, f"past_detections_mid{config.monitor_id}.pkl")
    else:
        past_file = os.path.join(config.image_path, "past_detections.pkl")
    saved_boxes, saved_labels = load_past_detections(past_file)

    by_type: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        by_type[det.detection_type].append(det)

    kept: list[Detection] = []
    for dtype, dets in by_type.items():
        try:
            mtype = ModelType(dtype)
        except ValueError:
            mtype = None

        tov = config.type_overrides.get(mtype, TypeOverrides()) if mtype else TypeOverrides()

        enabled = tov.match_past_detections if tov.match_past_detections is not None else config.match_past_detections
        if not enabled:
            kept.extend(dets)
            continue

        max_diff = tov.past_det_max_diff_area if tov.past_det_max_diff_area is not None else config.past_det_max_diff_area
        label_overrides = tov.past_det_max_diff_area_labels or config.past_det_max_diff_area_labels
        ignore = tov.ignore_past_detection_labels if tov.ignore_past_detection_labels is not None else config.ignore_past_detection_labels
        aliases = tov.aliases if tov.aliases is not None else config.aliases

        kept.extend(match_past_detections(
            dets, saved_boxes, saved_labels,
            max_diff_area=max_diff,
            label_area_overrides=label_overrides,
            ignore_labels=ignore,
            aliases=aliases,
        ))

    save_past_detections(past_file, detections)
    return kept
