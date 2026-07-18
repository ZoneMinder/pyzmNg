## Overview

This PR improves the remote gateway URL mode (`ml_gateway_mode: url`) with several enhancements that make it production-ready for real-world deployments. All changes are **strictly scoped to the URL gateway path** (`detect_event` → `_remote_detect_urls` → `pyzm.serve`). The local mode and the image gateway mode (`ml_gateway_mode: image`) are completely unaffected.

## Changes

### 1. Frame selection with `start_frame` / `frame_skip` / `max_frames`

In URL gateway mode, the client builds a list of frame URLs and sends them to the gateway. Previously, when `frame_set` was empty the client fell back to a hardcoded `["snapshot"]` — a single frame.

This PR introduces a three-way selection strategy:

1. **`frame_set` has values** — use them as-is (existing behaviour, unchanged)
2. **`frame_set` is empty AND `max_frames > 0`** — use `start_frame` + `frame_skip` + `max_frames` to build a sparse frame list distributed across the event.
   Example: `start_frame=50, frame_skip=25, max_frames=30` generates `[50, 75, 100, ..., 775]`.
   This is more effective than snapshots for detecting objects that appear briefly.
3. **`frame_set` is empty AND `max_frames` is 0 (not set)** — fall back to the default `["snapshot", "alarm", "1"]` with a warning, avoiding silent single-frame analysis.

Existing installations that do not explicitly set `frame_set: []` are completely unaffected.

### 2. Retry for missing frames (live events)

In live events frames are still being written to disk when detection starts. The gateway now retries missing frames (HTTP 404) up to `max_attempts` times, sleeping `sleep_between_attempts` seconds between attempts. After exhausting retries, consecutive 404s are counted and processing stops after `contig_frames_before_error` consecutive misses.

New `StreamConfig` fields forwarded to the gateway:
- `max_attempts` (default: 1)
- `sleep_between_attempts` (default: 3)
- `contig_frames_before_error` (default: 5)

### 3. Server-side filtering (confidence, pattern, zones) and short-circuit

Previously the gateway returned all raw detections and the client applied all filters. This meant the gateway always processed every frame, even when an early frame already had a clear match in zone.

The gateway now applies filters in order before the short-circuit check:

1. **Confidence filter** — drops detections below `min_confidence` (derived from the lowest `min_confidence` across enabled models)
2. **Pattern filter** — drops labels not matching the client pattern (e.g. `"(person)"`)
3. **Zone filter** — checks whether surviving detections intersect the zone polygons forwarded by the client
4. **Short-circuit** — when `stop_on_match=True` (default), stops processing as soon as a frame passes all filters, avoiding unnecessary inference on remaining frames

New `StreamConfig` field: `stop_on_match: bool = True`

### 4. Multi-worker support for `pyzm.serve` (CPU)

Added `--workers N` CLI argument to `pyzm.serve`. This is primarily useful when running on **CPU**, where each YOLOv4 inference call is blocking and takes ~500ms per frame. With a single worker, simultaneous events queue up and wait. With multiple workers, each event is handled by an independent process with its own model copy in memory, enabling truly parallel inference.

**Note:** On GPU the inference is fast enough that a single worker is generally sufficient. Multi-worker is not recommended on GPU as it multiplies VRAM usage without meaningful throughput gain.

Each worker loads the model independently (~443MB per worker for YOLOv4 at 576×576). The server configuration is serialised to the `PYZM_SERVER_CONFIG` environment variable so each worker can reconstruct it via `get_app()` without re-parsing CLI arguments.

New `ServerConfig` fields:
- `workers: int = 1`
- `log_level: str = "info"`

## Files changed

| File | Scope |
|------|-------|
| `pyzm/ml/detector.py` | URL gateway path only |
| `pyzm/serve/app.py` | Gateway server only |
| `pyzm/serve/__main__.py` | Gateway server only |
| `pyzm/models/config.py` | New fields, backwards compatible |

## Testing

Tested in production on nvr-raimon (Debian 13, Python 3.13, ~34 cameras) with:
- YOLOv4 custom trained model (2 classes: person/dog)
- 3 workers running in parallel
- Live event detection with retry
- Short-circuit confirmed stopping at first in-zone match
- Pattern filter confirmed suppressing non-person detections
