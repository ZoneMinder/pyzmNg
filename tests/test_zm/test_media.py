"""Tests for pyzm.zm.media -- FrameExtractor."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from pyzm.models.config import StreamConfig
from pyzm.zm.media import FrameExtractor


# ===================================================================
# Helpers
# ===================================================================

def _make_mock_api() -> MagicMock:
    api = MagicMock()
    api.portal_url = "https://zm.example.com/zm"
    return api


def _image_response() -> MagicMock:
    """A requests.Response-alike carrying JPEG bytes."""
    resp = MagicMock()
    resp.content = b"\xff\xd8\xff\xe0jpeg-ish"
    return resp


def _mock_cv2() -> MagicMock:
    """cv2 stub whose imdecode returns an image with a real shape."""
    cv2 = MagicMock()
    img = MagicMock()
    img.shape = (480, 640, 3)
    cv2.imdecode.return_value = img
    return cv2


# ===================================================================
# _fetch_frame_image
# ===================================================================

class TestFetchFrameImage:
    """A frame fetch must never raise a transport error at the caller."""

    @pytest.mark.parametrize("exc", [
        requests.ReadTimeout("read timed out"),
        requests.ConnectionError("connection aborted"),
        requests.HTTPError("500 server error"),
    ])
    def test_transport_error_returns_none(self, exc):
        api = _make_mock_api()
        api.request.side_effect = exc
        extractor = FrameExtractor(
            api=api,
            stream_config=StreamConfig(max_attempts=1, sleep_between_attempts=0),
        )

        assert extractor._fetch_frame_image("url", _mock_cv2(), MagicMock()) is None

    def test_transport_error_retried_up_to_max_attempts(self):
        api = _make_mock_api()
        api.request.side_effect = requests.ReadTimeout("read timed out")
        extractor = FrameExtractor(
            api=api,
            stream_config=StreamConfig(max_attempts=3, sleep_between_attempts=0),
        )

        assert extractor._fetch_frame_image("url", _mock_cv2(), MagicMock()) is None
        assert api.request.call_count == 3

    def test_recovers_on_later_attempt(self):
        api = _make_mock_api()
        api.request.side_effect = [
            requests.ReadTimeout("read timed out"),
            _image_response(),
        ]
        cv2 = _mock_cv2()
        extractor = FrameExtractor(
            api=api,
            stream_config=StreamConfig(max_attempts=2, sleep_between_attempts=0),
        )

        img = extractor._fetch_frame_image("url", cv2, MagicMock())
        assert img is cv2.imdecode.return_value
        assert api.request.call_count == 2

    def test_relogin_still_propagates(self):
        """Only BAD_IMAGE is swallowed; RELOGIN must reach the caller."""
        api = _make_mock_api()
        api.request.side_effect = ValueError("RELOGIN")
        extractor = FrameExtractor(
            api=api,
            stream_config=StreamConfig(max_attempts=1, sleep_between_attempts=0),
        )

        with pytest.raises(ValueError, match="RELOGIN"):
            extractor._fetch_frame_image("url", _mock_cv2(), MagicMock())

    def test_json_decode_error_returns_none(self):
        """JSONDecodeError is both a RequestException and a ValueError.

        The RequestException clause has to be matched first, or the
        ValueError clause sees a message that is not BAD_IMAGE and re-raises.
        """
        api = _make_mock_api()
        api.request.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "", 0,
        )
        extractor = FrameExtractor(
            api=api,
            stream_config=StreamConfig(max_attempts=1, sleep_between_attempts=0),
        )

        assert extractor._fetch_frame_image("url", _mock_cv2(), MagicMock()) is None


# ===================================================================
# _read_zm_event
# ===================================================================

class TestReadZMEventTransportFailure:

    def _extract(self, api, cfg):
        extractor = FrameExtractor(api=api, stream_config=cfg)
        with patch.dict(sys.modules, {"cv2": _mock_cv2(), "numpy": MagicMock()}):
            return list(extractor.extract_frames("12345"))

    def test_partial_frames_survive_a_persistent_timeout(self):
        """Frames fetched before the ZM host stalls are still returned."""
        api = _make_mock_api()
        api.request.side_effect = [
            _image_response(),
            requests.ReadTimeout("read timed out"),
            requests.ReadTimeout("read timed out"),
        ]
        cfg = StreamConfig(
            frame_set=["1", "2", "3"],
            contig_frames_before_error=2,
            max_attempts=1,
            sleep_between_attempts=0,
        )

        frames = self._extract(api, cfg)

        assert [f.frame_id for f, _ in frames] == ["1"]
        assert api.request.call_count == 3

    def test_all_frames_timing_out_yields_nothing_without_raising(self):
        api = _make_mock_api()
        api.request.side_effect = requests.ReadTimeout("read timed out")
        cfg = StreamConfig(
            frame_set=["1", "2", "3", "4"],
            contig_frames_before_error=2,
            max_attempts=1,
            sleep_between_attempts=0,
        )

        assert self._extract(api, cfg) == []
        # Stops at the contiguous-error budget rather than walking the set.
        assert api.request.call_count == 2
