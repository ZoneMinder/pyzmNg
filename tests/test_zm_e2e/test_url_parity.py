"""E2E: URL-mode local<->remote parity against a live ZoneMinder.

Local mode downloads the frame; URL mode has the gateway fetch the SAME ZM URL.
Same bytes -> same decode -> same inference, so the final results must match.

Needs both a live ZM (zm_client / object_event fixtures) and a pyzm.serve
subprocess with real models -> marked both zm_e2e and serve, so it runs under
`make release-gate`. Uses no zones / no size filter on purpose: URL mode sizes
its blank frame from the monitor W×H, so a percentage zone/size filter could
diverge if the event was recorded at a different resolution. Detection boxes are
in the fetched frame's pixel space (identical image), so label/box/confidence
parity is dimension-independent and isolates the transport.
"""

from __future__ import annotations

import pytest

from pyzm.ml.detector import Detector
from pyzm.models.config import StreamConfig
from tests.test_ml_e2e.conftest import (
    BASE_PATH, find_one_model, start_serve, stop_serve, wait_for_serve,
)

pytestmark = [pytest.mark.zm_e2e, pytest.mark.serve]

PORT = 15300


def test_url_mode_matches_local(zm_client, object_event):
    model = find_one_model()
    proc = start_serve([model], PORT)
    try:
        assert wait_for_serve(PORT), "Server failed to start"
        sc = StreamConfig(frame_set=["snapshot"])  # deterministic single frame

        local = Detector(models=[model], base_path=BASE_PATH)
        remote = Detector(
            models=[model], base_path=BASE_PATH,
            gateway=f"http://127.0.0.1:{PORT}", gateway_mode="url",
        )

        rl = local.detect_event(zm_client, object_event.id, stream_config=sc)
        rr = remote.detect_event(zm_client, object_event.id, stream_config=sc)

        assert rl.labels == rr.labels
        assert [d.bbox.as_list() for d in rl.detections] == \
               [d.bbox.as_list() for d in rr.detections]
        assert [round(d.confidence, 4) for d in rl.detections] == \
               [round(d.confidence, 4) for d in rr.detections]
    finally:
        stop_serve(proc)
