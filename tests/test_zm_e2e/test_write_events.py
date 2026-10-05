"""E2E write tests for event deletion (requires ZM_E2E_WRITE=1).

NOTE: Do NOT use ZM_E2E_WRITE env var in test logic -- the user will
run these manually with ``ZM_E2E_WRITE=1 pytest tests/test_zm_e2e/ -v``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zm_e2e, pytest.mark.zm_e2e_write]


class TestWriteEvents:
    def test_delete_event(self, zm_client, any_event):
        """delete() should delete an event. Picks the oldest event other than
        the session-scoped ``any_event``, which later tests still use."""
        events = [e for e in zm_client.events(limit=2) if e.id != any_event.id]
        if not events:
            pytest.skip("Need a second event to delete (any_event is in use)")

        ev = events[0]
        ev.delete()

        # Verify it's gone
        with pytest.raises(ValueError, match="not found"):
            zm_client.event(ev.id)
