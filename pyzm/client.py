"""Top-level ZoneMinder client -- the one-stop public API.

Usage::

    from pyzm import ZMClient

    zm = ZMClient(api_url="https://zm.example.com/zm/api", user="admin", password="secret")

    for m in zm.monitors():
        print(m.name, m.function)
        for z in m.get_zones():
            print(f"  zone: {z.name}")

    for ev in zm.events(monitor_id=1, since="1 hour ago"):
        print(ev.id, ev.cause)
        ev.update_notes("detected")
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import quote, urlparse, urlencode

import requests

from pyzm.models.config import StreamConfig, ZMClientConfig
from pyzm.models.zm import Event, Frame, Monitor, Notification, PTZCapabilities, Zone
from pyzm.zm.api import ZMAPI
from pyzm.zm.media import FrameExtractor
from pyzm.zm.timeparse import parse_human_time as _parse_human_time

logger = logging.getLogger("pyzm")


class ZMClient:
    """High-level ZoneMinder client.

    Parameters
    ----------
    api_url:
        Full ZM API URL (e.g. ``https://server/zm/api``).
    user:
        ZM username.  ``None`` when auth is disabled.
    password:
        ZM password.
    portal_url:
        Full portal URL (e.g. ``https://server/zm``).  Auto-derived from
        *api_url* when not provided.
    verify_ssl:
        Whether to verify SSL certificates.  Set to ``False`` for
        self-signed certs.
    config:
        A pre-built :class:`ZMClientConfig`.  When provided, all other
        keyword args are ignored.
    """

    def __init__(
        self,
        api_url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        portal_url: str | None = None,
        verify_ssl: bool = True,
        timeout: int = 30,
        db_user: str | None = None,
        db_password: str | None = None,
        db_host: str | None = None,
        db_name: str | None = None,
        conf_path: str | None = None,
        config: ZMClientConfig | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            if api_url is None:
                raise ValueError("Either 'api_url' or 'config' must be provided")
            self._config = ZMClientConfig(
                api_url=api_url.rstrip("/"),
                portal_url=portal_url,
                user=user,
                password=password,
                verify_ssl=verify_ssl,
                timeout=timeout,
                db_user=db_user,
                db_password=db_password,
                db_host=db_host,
                db_name=db_name,
                conf_path=conf_path,
            )

        self._api = ZMAPI(self._config)

        # Caches
        self._monitors: list[Monitor] | None = None
        self._zms_path: str | None = None
        self._servers_by_id: dict[int, dict] | None = None
        self._min_streaming_port: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def api(self) -> ZMAPI:
        """The underlying low-level API (for advanced use)."""
        return self._api

    @property
    def zm_version(self) -> str | None:
        return self._api.zm_version

    @property
    def api_version(self) -> str | None:
        return self._api.api_version

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_db(self):
        """Return a DB connection using config-level credential overrides."""
        from pyzm.zm.db import get_zm_db

        cfg = self._config
        pw = cfg.db_password.get_secret_value() if cfg.db_password else None
        return get_zm_db(
            db_user=cfg.db_user,
            db_password=pw,
            db_host=cfg.db_host,
            db_name=cfg.db_name,
            conf_path=cfg.conf_path,
        )

    # ------------------------------------------------------------------
    # Monitors
    # ------------------------------------------------------------------

    def monitors(self, *, force_reload: bool = False) -> list[Monitor]:
        """Return all monitors.  Cached after first call."""
        if self._monitors is not None and not force_reload:
            return self._monitors

        data = self._api.get("monitors.json")
        raw_list = data.get("monitors", []) if data else []
        self._monitors = [Monitor.from_api_dict(m, client=self) for m in raw_list]
        return self._monitors

    def monitor(self, monitor_id: int | str) -> Monitor:
        """Return a single monitor by ID (int) or name (str, case-insensitive)."""
        if isinstance(monitor_id, str):
            needle = monitor_id.lower()
            for m in self.monitors():
                if m.name.lower() == needle:
                    return m
            raise ValueError(f"Monitor {monitor_id!r} not found")
        for m in self.monitors():
            if m.id == monitor_id:
                return m
        # Fallback: direct API call
        data = self._api.get(f"monitors/{monitor_id}.json")
        if data and data.get("monitor"):
            return Monitor.from_api_dict(data["monitor"], client=self)
        raise ValueError(f"Monitor {monitor_id} not found")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def events(
        self,
        *,
        event_id: int | None = None,
        monitor_id: int | None = None,
        since: str | None = None,
        until: str | None = None,
        min_alarm_frames: int | None = None,
        object_only: bool = False,
        limit: int = 100,
    ) -> list[Event]:
        """Query events with optional filters."""
        # Use CakePHP URL-path filter syntax (e.g. events/index/Field:val)
        # instead of filter[Query][terms] query strings which are broken
        # on ZM >= 1.38.  REGEXP replaces LIKE to avoid % in URL paths.
        path_parts: list[str] = []

        if event_id is not None:
            path_parts.append(f"Id:{event_id}")

        if monitor_id is not None:
            path_parts.append(f"MonitorId:{monitor_id}")

        if since:
            parsed = _parse_human_time(since)
            if parsed:
                path_parts.append(f"StartTime >=:{parsed}")

        if until:
            parsed = _parse_human_time(until)
            if parsed:
                path_parts.append(f"StartTime <=:{parsed}")

        if min_alarm_frames is not None:
            path_parts.append(f"AlarmFrames >=:{min_alarm_frames}")

        if object_only:
            path_parts.append("Notes REGEXP:detected")

        path_filter = "/".join(path_parts)
        endpoint = f"events/index/{path_filter}.json" if path_filter else "events/index.json"
        params = {"page": "1", "limit": str(limit)}

        data = self._api.get(endpoint, params=params)
        events_list = data.get("events", []) if data else []
        return [Event.from_api_dict(e, client=self) for e in events_list]

    def event_counts(self, interval: str, *, include_archived: bool = False) -> dict[int, int]:
        """Count events per monitor that started within *interval*.

        *interval* is a MySQL interval such as ``"1 hour"``, ``"1 day"`` or
        ``"1 week"``.  Returns ``{monitor_id: count}`` from one API call;
        monitors with no events are left out.  Archived events are excluded
        unless *include_archived* is set.
        """
        if not _INTERVAL_RE.match(interval.strip()):
            raise ValueError(f"Not a MySQL interval like '1 hour': {interval!r}")
        endpoint = f"events/consoleEvents/{quote(interval.strip())}"
        if not include_archived:
            endpoint += "/Archived:0"
        data = self._api.get(f"{endpoint}.json")
        results = data.get("results") if data else None
        # ZM sends an empty PHP array, [], when there are no events
        if not isinstance(results, dict):
            return {}
        return {int(mid): int(count) for mid, count in results.items()}

    def event(self, event_id: int) -> Event:
        """Fetch a single event by ID."""
        data = self._api.get(f"events/{event_id}.json")
        if data and data.get("event"):
            return Event.from_api_dict(data["event"], client=self)
        raise ValueError(f"Event {event_id} not found")

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def notifications(self) -> list[Notification]:
        """Fetch all push notification token registrations."""
        data = self._api.get("notifications.json")
        items = data.get("notifications", []) if data else []
        return [Notification.from_api_dict(n, client=self) for n in items]

    def notification(self, notification_id: int) -> Notification:
        """Fetch a single notification registration by ID."""
        data = self._api.get(f"notifications/{notification_id}.json")
        if data and data.get("notification"):
            return Notification.from_api_dict(data["notification"], client=self)
        raise ValueError(f"Notification {notification_id} not found")

    def _create_notification(self, **kwargs) -> dict:
        """Create or upsert a notification registration."""
        data = {f"Notification[{k}]": v for k, v in kwargs.items()}
        return self._api.post("notifications.json", data=data)

    def _update_notification(self, notification_id: int, **kwargs) -> dict:
        """Update fields on a notification registration."""
        data = {f"Notification[{k}]": v for k, v in kwargs.items()}
        return self._api.put(f"notifications/{notification_id}.json", data=data)

    def _delete_notification(self, notification_id: int) -> None:
        """Delete a notification registration."""
        self._api.delete(f"notifications/{notification_id}.json")

    def _update_notification_last_sent(self, notification_id: int, badge: int | None = None) -> None:
        """Update LastNotifiedAt to now and optionally set BadgeCount."""
        from datetime import datetime
        data = {"LastNotifiedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if badge is not None:
            data["BadgeCount"] = str(badge)
        self._update_notification(notification_id, **data)

    def delete_events(
        self,
        *,
        monitor_id: int | None = None,
        before: str | None = None,
        min_alarm_frames: int | None = None,
        limit: int = 100,
    ) -> int:
        """Query events matching filters and delete each one. Returns count deleted."""
        matched = self.events(
            monitor_id=monitor_id,
            until=before,
            min_alarm_frames=min_alarm_frames,
            limit=limit,
        )
        for ev in matched:
            self._api.delete(f"events/{ev.id}.json")
        return len(matched)

    # ------------------------------------------------------------------
    # Internal resource operations (called by Monitor/Event methods)
    # ------------------------------------------------------------------

    def _delete_event(self, event_id: int) -> None:
        """Delete a single event."""
        self._api.delete(f"events/{event_id}.json")

    def _monitor_zones(self, monitor_id: int) -> list[Zone]:
        """Return detection zones for a monitor."""
        data = self._api.get(f"zones/forMonitor/{monitor_id}.json")
        raw_zones = data.get("zones", []) if data else []
        # Only fetch monitor dimensions when at least one zone uses percentages
        has_pct = any(
            z.get("Zone", z).get("Units") == "Percent" for z in raw_zones
        )
        mon_w = mon_h = 0
        if has_pct:
            mon_data = self._api.get(f"monitors/{monitor_id}.json")
            mon = mon_data.get("monitor", {}).get("Monitor", {}) if mon_data else {}
            mon_w = int(mon.get("Width", 0))
            mon_h = int(mon.get("Height", 0))
        zones: list[Zone] = []
        for z in raw_zones:
            zd = z.get("Zone", z)
            coords_str = zd.get("Coords", "")
            points = _parse_zone_coords(coords_str)
            if zd.get("Units") == "Percent" and mon_w and mon_h:
                points = [(int(round(x * mon_w / 100.0)), int(round(y * mon_h / 100.0))) for x, y in points]
            zones.append(Zone(
                name=zd.get("Name", ""),
                points=points,
                _raw=z,
            ))
        return zones

    def _arm(self, monitor_id: int, cause: str | None = None) -> dict:
        """Trigger alarm ON for a monitor, with an optional event cause."""
        endpoint = f"monitors/alarm/id:{monitor_id}/command:on.json"
        if cause:
            return self._api.get(endpoint, params={"cause": cause})
        return self._api.get(endpoint)

    def _disarm(self, monitor_id: int) -> dict:
        """Cancel alarm for a monitor."""
        return self._api.get(f"monitors/alarm/id:{monitor_id}/command:off.json")

    def _alarm_status(self, monitor_id: int) -> dict:
        """Get alarm status for a monitor."""
        return self._api.get(f"monitors/alarm/id:{monitor_id}/command:status.json")

    def _update_monitor(self, monitor_id: int, **kwargs) -> None:
        """Update monitor fields (function, enabled, name, etc.)."""
        data = {f"Monitor[{k}]": v for k, v in kwargs.items()}
        self._api.put(f"monitors/{monitor_id}.json", data=data)
        self._monitors = None  # invalidate cache

    def _zms_url(
        self, monitor_id: int, mode: str, *, server_id: int | None = None,
        authenticated: bool = False, **kwargs,
    ) -> str:
        """Build a ZMS CGI URL for a monitor.

        When ZM has more than one server, a monitor's URL uses its own
        server's ``Servers`` row (see :meth:`_server_base_url`).  Otherwise
        the portal URL and ``ZM_PATH_ZMS`` are used, with any path prefix
        they share deduplicated.  *authenticated* appends the current auth
        string.
        """
        params = {"mode": mode, "monitor": str(monitor_id), **kwargs}
        query = urlencode(params)
        if authenticated and (auth := self._api.auth.get_auth_string()):
            query += f"&{auth}"

        server = self._server(server_id) if server_id else None
        if server is not None:
            base = self._server_base_url(server, monitor_id)
            return f"{base}{server.get('PathToZMS') or self._zms_config_path()}?{query}"

        zms_path = self._zms_config_path()
        portal_url = self._api.portal_url
        portal_path = urlparse(portal_url).path.rstrip("/")
        # Deduplicate overlapping prefix (e.g. portal /zm + zms /zm/cgi-bin/...)
        if portal_path and zms_path.startswith(portal_path):
            zms_path = zms_path[len(portal_path):]
        return f"{portal_url}{zms_path}?{query}"

    def _zms_config_path(self) -> str:
        """``ZM_PATH_ZMS``, fetched once per client.

        An API user without permission to view ZM's options can't read it.
        Then the ZMS path of the ``Servers`` entries is used, which ZM's web
        UI prefers over ``ZM_PATH_ZMS`` anyway, when they agree on one.
        """
        if self._zms_path is None:
            try:
                self._zms_path = self.config("ZM_PATH_ZMS")["Value"]
            except (requests.HTTPError, ValueError) as exc:
                paths = {s.get("PathToZMS") for s in self.servers() if s.get("PathToZMS")}
                if len(paths) != 1:
                    raise ValueError(
                        f"Can't build a stream URL: ZM_PATH_ZMS not found ({exc}) and no "
                        "Servers entry gives a ZMS path. The API user needs permission "
                        "to view ZM's options."
                    ) from None
                self._zms_path = paths.pop()
        return self._zms_path

    def _server(self, server_id: int) -> dict | None:
        """The ``Servers`` row for *server_id* when ZM has more than one
        server, else ``None``.  Read once per client."""
        if self._servers_by_id is None:
            try:
                rows = self.servers()
            except (requests.ConnectionError, requests.Timeout) as exc:
                # Transient: use the portal this time and read them again next time
                logger.warning("Can't read ZM servers (%s); using the portal URL", exc)
                return None
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Can't read ZM servers (%s); stream URLs use the portal URL", exc)
                rows = []
            # With one server the portal URL already reaches it, perhaps by a
            # name (proxy, public DNS) the Servers row doesn't know.
            self._servers_by_id = (
                {int(s["Id"]): s for s in rows if s.get("Id")} if len(rows) > 1 else {}
            )
        server = self._servers_by_id.get(server_id)
        if server is None and self._servers_by_id:
            logger.warning("Monitor references unknown ServerId %d; using the portal URL", server_id)
        return server

    def _server_base_url(self, server: dict, monitor_id: int) -> str:
        """``Protocol://Hostname[:Port]`` from a ``Servers`` row.

        As ZM's web UI does: Name stands in for an empty Hostname, and
        ``ZM_MIN_STREAMING_PORT``, when set, gives each monitor the port
        ``ZM_MIN_STREAMING_PORT + monitor id``.  An empty Protocol takes the
        portal's; an empty Port is left out (ZM's UI would use the
        browser's).  The scheme's default port isn't written out.
        """
        protocol = server.get("Protocol") or urlparse(self._api.portal_url).scheme or "http"
        host = server.get("Hostname") or server.get("Name")
        min_port = self._min_streaming_port_value()
        port = min_port + monitor_id if min_port else server.get("Port")
        if port and str(port) != {"https": "443", "http": "80"}.get(protocol):
            return f"{protocol}://{host}:{port}"
        return f"{protocol}://{host}"

    def _min_streaming_port_value(self) -> int:
        """``ZM_MIN_STREAMING_PORT`` as an int (0 when unset), read once."""
        if self._min_streaming_port is None:
            try:
                self._min_streaming_port = int(self.config("ZM_MIN_STREAMING_PORT").get("Value") or 0)
            except (requests.ConnectionError, requests.Timeout):
                return 0  # transient: read it again next time
            except (requests.RequestException, ValueError):
                self._min_streaming_port = 0
        return self._min_streaming_port

    def _streaming_url_mjpeg(self, monitor_id: int, **kwargs) -> str:
        """Build an MJPEG streaming URL for a monitor."""
        return self._zms_url(monitor_id, "jpeg", **kwargs)

    def _snapshot_url(self, monitor_id: int, **kwargs) -> str:
        """Build a single-frame snapshot URL for a monitor."""
        return self._zms_url(monitor_id, "single", **kwargs)

    def _daemon_status(self, monitor_id: int) -> dict:
        """Get daemon status for a monitor's capture daemon (zmc)."""
        return self._api.get(
            f"monitors/daemonStatus/id:{monitor_id}/daemon:zmc.json"
        ) or {}

    def _ptz_capabilities(self, control_id: int) -> PTZCapabilities:
        """Fetch PTZ capabilities for a control profile."""
        data = self._api.get(f"controls/{control_id}.json")
        if not data or "control" not in data:
            raise ValueError(f"Control profile {control_id} not found")
        ctrl = data["control"].get("Control", data["control"])
        return PTZCapabilities.from_api_dict(ctrl)

    def _ptz_command(
        self, monitor_id: int, command: str, *, mode: str = "con",
        preset: int = 1, stop_after: float | None = None,
    ) -> None:
        """Send a PTZ command to a monitor via the ZM portal.

        Translates user-friendly command names to ZM's internal names:

        - ``"up"`` → ``moveConUp`` / ``moveRelUp`` / ``moveAbsUp``
        - ``"zoom_in"`` → ``zoomConTele`` / ``zoomRelTele`` / ``zoomAbsTele``
        - ``"stop"`` → ``moveStop``
        - ``"home"`` → ``presetHome``
        - ``"preset"`` (with preset=N) → ``presetGotoN``

        If *stop_after* is set, sleeps that many seconds then sends
        ``moveStop``.
        """
        zm_command = _ptz_command_name(command, mode=mode, preset=preset)
        portal_url = self._api.portal_url
        url = f"{portal_url}/index.php"
        params = {
            "view": "request",
            "request": "control",
            "id": str(monitor_id),
            "control": zm_command,
            "xge": "0",
            "yge": "0",
        }
        self._api.request(url, params=params)

        if stop_after is not None and stop_after > 0 and command != "stop":
            time.sleep(stop_after)
            self._api.request(url, params={**params, "control": "moveStop"})

    def _update_event_notes(self, event_id: int, notes: str) -> None:
        """Update the Notes field of an event."""
        url = f"events/{event_id}.json"
        self._api.put(url, data={"Event[Notes]": notes})

    def _tag_event(self, event_id: int, labels: list[str]) -> None:
        """Tag an event with detected object labels.

        For each unique label, creates the tag if it doesn't exist and
        associates it with the event.  Requires ZM >= 1.37.44.
        """
        unique = list(dict.fromkeys(labels))  # dedupe, preserve order
        if not unique:
            return
        logger.debug("Tagging event %s with %s", event_id, unique)
        for label in unique:
            self._tag_one(label, event_id)

    def _tag_one(self, label: str, event_id: int) -> None:
        """Create or find a tag by name and link it to an event via direct DB."""
        import datetime

        from pyzm.zm.db import dict_cursor

        conn = self._get_db()
        if conn is None:
            logger.warning("Could not connect to ZM database, skipping tagging")
            return

        cur = dict_cursor(conn)
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Find or create tag
            cur.execute("SELECT Id FROM Tags WHERE Name=%s", (label,))
            row = cur.fetchone()
            if row:
                tag_id = row["Id"]
                cur.execute("UPDATE Tags SET LastAssignedDate=%s WHERE Id=%s", (now, tag_id))
                logger.debug("Tag '%s' exists (id=%s), linked to event %s", label, tag_id, event_id)
            else:
                cur.execute(
                    "INSERT INTO Tags (Name, CreateDate, LastAssignedDate) VALUES (%s, %s, %s)",
                    (label, now, now),
                )
                tag_id = cur.lastrowid
                logger.debug("Created tag '%s' (id=%s), linked to event %s", label, tag_id, event_id)

            # Link tag to event (ignore duplicate)
            try:
                cur.execute(
                    "INSERT INTO Events_Tags (TagId, EventId) VALUES (%s, %s)",
                    (tag_id, event_id),
                )
            except Exception:
                logger.debug("Tag '%s' already linked to event %s", label, event_id)

            conn.commit()
        finally:
            cur.close()
            conn.close()

    def _save_objdetect(
        self,
        event: Event,
        image: Any,
        metadata: dict,
        path_override: str | None = None,
    ) -> str | None:
        """Write ``objdetect.jpg`` and ``objects.json`` for an event.

        Parameters
        ----------
        event:
            The :class:`Event` whose directory will be used.
        image:
            A NumPy image array (BGR, as from OpenCV).
        metadata:
            Detection data dict.  The standard keys ``labels``, ``boxes``,
            ``frame_id``, ``confidences``, and ``image_dimensions`` are
            extracted and serialised.
        path_override:
            If given, write to this directory instead of ``event.path()``.

        Returns the directory path written to, or ``None`` if no path was
        available.
        """
        import json as _json

        import cv2 as _cv2

        eventpath = path_override or self._event_path(event.id)
        if not eventpath:
            logger.debug("No event path available, skipping save_objdetect")
            return None

        os.makedirs(eventpath, exist_ok=True)

        objdetect_path = os.path.join(eventpath, "objdetect.jpg")
        _cv2.imwrite(objdetect_path, image)
        logger.debug("Wrote objdetect image to %s", objdetect_path)

        keys = ("labels", "boxes", "frame_id", "confidences", "image_dimensions")
        obj_json = {k: metadata[k] for k in keys if k in metadata}
        json_path = os.path.join(eventpath, "objects.json")
        with open(json_path, "w") as f:
            _json.dump(obj_json, f)

        return eventpath

    def _event_path(self, event_id: int) -> str | None:
        """Construct the filesystem path for an event."""
        from datetime import datetime as _dt

        from pyzm.zm.db import dict_cursor

        conn = self._get_db()
        if conn is None:
            logger.warning("Cannot resolve event path: DB unavailable")
            return None

        cur = dict_cursor(conn)
        try:
            cur.execute(
                "SELECT E.MonitorId, E.StartDateTime, E.StorageId, "
                "S.Path AS StoragePath, S.Scheme "
                "FROM Events E LEFT JOIN Storage S ON E.StorageId = S.Id "
                "WHERE E.Id=%s",
                (event_id,),
            )
            row = cur.fetchone()

            if not row or not row["StartDateTime"]:
                logger.warning("Cannot resolve event path: missing DB fields for event %s", event_id)
                return None

            storage_path = row.get("StoragePath")
            scheme = row.get("Scheme")

            if not storage_path:
                logger.error("Event %s has StorageId=%s which does not map to a valid "
                             "Storage row — the monitor's storage configuration may be "
                             "invalid. Falling back to Storage Id=1.",
                             event_id, row.get("StorageId"))
                cur.execute(
                    "SELECT Path, Scheme FROM Storage WHERE Id=1"
                )
                fallback = cur.fetchone()
                if fallback and fallback["Path"]:
                    storage_path = fallback["Path"]
                    scheme = fallback.get("Scheme") or scheme
        finally:
            cur.close()
            conn.close()

        if not storage_path:
            logger.warning("Cannot resolve event path: no storage path for event %s", event_id)
            return None

        monitor_id = row["MonitorId"]
        scheme = (scheme or "Medium").capitalize()
        start_dt = row["StartDateTime"]
        if isinstance(start_dt, str):
            try:
                start_dt = _dt.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # PyMySQL returns MySQL's zero date as the string 0000-00-00 ...
                logger.warning("Cannot resolve event path: event %s has StartDateTime %r",
                               event_id, start_dt)
                return None

        if scheme == "Deep":
            relative = "{}/{}".format(monitor_id, start_dt.strftime("%y/%m/%d/%H/%M/%S"))
        elif scheme == "Medium":
            relative = "{}/{}/{}".format(monitor_id, start_dt.strftime("%Y-%m-%d"), event_id)
        else:
            relative = "{}/{}".format(monitor_id, event_id)

        path = os.path.join(storage_path, relative)
        logger.debug("Event %s path (scheme=%s): %s", event_id, scheme, path)
        return path

    def _event_frames(self, event_id: int) -> list[Frame]:
        """Fetch per-frame metadata (Score, Type, Delta) for an event."""
        data = self._api.get(f"frames/index/EventId:{event_id}.json")
        raw_list = data.get("frames", []) if data else []
        frames: list[Frame] = []
        for entry in raw_list:
            fd = entry.get("Frame", entry)
            frames.append(Frame(
                frame_id=int(fd.get("FrameId", 0)),
                event_id=int(fd.get("EventId", event_id)),
                type=fd.get("Type", ""),
                score=int(fd.get("Score", 0)),
                delta=float(fd.get("Delta", 0)),
                _raw=entry,
            ))
        return frames

    def _get_event_frames(
        self,
        event_id: int,
        stream_config: StreamConfig | None = None,
    ) -> tuple[list[tuple[int | str, Any]], dict[str, tuple[int, int] | None]]:
        """Extract frames from a ZM event as numpy arrays."""
        sc = stream_config or StreamConfig()
        extractor = FrameExtractor(api=self._api, stream_config=sc)
        frames: list[tuple[int | str, Any]] = []
        for frame, img in extractor.extract_frames(str(event_id)):
            frames.append((frame.frame_id, img))

        orig = extractor.original_shape
        if frames and orig:
            resized_h, resized_w = frames[0][1].shape[:2]
            resized = (resized_h, resized_w) if (resized_h, resized_w) != orig else None
        else:
            resized = None

        image_dims: dict[str, tuple[int, int] | None] = {
            "original": orig,
            "resized": resized,
        }
        return frames, image_dims

    # ------------------------------------------------------------------
    # System health
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Check if the ZM daemon is running."""
        data = self._api.get("host/daemonCheck.json")
        return bool(data and data.get("result") == 1)

    def system_load(self) -> dict[str, float]:
        """Get system load averages."""
        data = self._api.get("host/getLoad.json")
        load = data.get("load", []) if data else []
        keys = ("1min", "5min", "15min")
        return {k: float(load[i]) for i, k in enumerate(keys) if i < len(load)}

    def disk_usage(self) -> dict:
        """Get disk usage info."""
        return self._api.get("host/getDiskPercent.json") or {}

    def timezone(self) -> str:
        """Get server timezone."""
        data = self._api.get("host/getTimeZone.json")
        if not data:
            return ""
        # ZM 1.38+ uses "tz", older versions may use "timezone"
        return data.get("tz") or data.get("timezone") or ""

    # ------------------------------------------------------------------
    # Configuration management
    # ------------------------------------------------------------------

    def configs(self) -> list[dict]:
        """List all ZM configuration parameters."""
        data = self._api.get("configs.json")
        raw = data.get("configs", []) if data else []
        return [c.get("Config", c) for c in raw]

    def config(self, name: str) -> dict:
        """Get a single config parameter by name."""
        data = self._api.get(f"configs/viewByName/{name}.json")
        if data and data.get("config"):
            cfg = data["config"].get("Config", data["config"])
            cfg.setdefault("Name", name)
            return cfg
        raise ValueError(f"Config '{name}' not found")

    def set_config(self, name: str, value: str) -> None:
        """Set a config parameter value."""
        cfg = self.config(name)
        config_id = cfg.get("Id")
        if not config_id:
            for c in self.configs():
                if c.get("Name") == name:
                    config_id = c.get("Id")
                    break
        if not config_id:
            raise ValueError(f"Config '{name}' not found or has no Id")
        self._api.put(f"configs/{config_id}.json", data={"Config[Value]": value})

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def states(self) -> list[dict]:
        """List all available ZM states."""
        data = self._api.get("states.json")
        raw = data.get("states", []) if data else []
        return [s.get("State", s) for s in raw]

    def set_state(self, state: str) -> Any:
        """Set ZM to a named state (e.g. 'start', 'stop', 'restart')."""
        return self._api.get(f"states/change/{state}.json")

    def start(self) -> Any:
        return self.set_state("start")

    def stop(self) -> Any:
        return self.set_state("stop")

    def restart(self) -> Any:
        return self.set_state("restart")

    # ------------------------------------------------------------------
    # Servers & Storage
    # ------------------------------------------------------------------

    def servers(self) -> list[dict]:
        """List all ZM servers (multi-server setups)."""
        data = self._api.get("servers.json")
        raw = data.get("servers", []) if data else []
        return [s.get("Server", s) for s in raw]

    def storage(self) -> list[dict]:
        """List all storage areas with disk usage."""
        data = self._api.get("storage.json")
        raw = data.get("storage", []) if data else []
        return [s.get("Storage", s) for s in raw]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_zone_coords(coords_str: str) -> list[tuple[float, float]]:
    """Parse ZM zone coordinate string like ``"0,0 639,0 639,479 0,479"``.

    Coordinates may be integers (pixels) or floats (percentages, ZM 1.37+).
    Ref: ZoneMinder/zmeventnotificationNg#18
    """
    if not coords_str:
        return []
    points: list[tuple[float, float]] = []
    for pair in coords_str.strip().split():
        parts = pair.split(",")
        if len(parts) == 2:
            points.append((float(parts[0]), float(parts[1])))
    return points


# ZM's consoleEvents accepts a MySQL interval: a number and a unit
_INTERVAL_RE = re.compile(r"^[ \-.:0-9]+\s+[_a-z]+$", re.IGNORECASE)

_PTZ_MOVE_MAP = {
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "up_left": "UpLeft", "up_right": "UpRight",
    "down_left": "DownLeft", "down_right": "DownRight",
}

_PTZ_ZOOM_MAP = {"zoom_in": "Tele", "zoom_out": "Wide"}

_PTZ_MODE_PREFIX = {"con": "Con", "rel": "Rel", "abs": "Abs"}


def _ptz_command_name(command: str, *, mode: str = "con", preset: int = 1) -> str:
    """Translate a user-friendly PTZ command to a ZM internal command name."""
    cmd = command.lower().strip()

    if cmd == "stop":
        return "moveStop"
    if cmd == "home":
        return "presetHome"
    if cmd == "preset":
        return f"presetGoto{preset}"

    mode_suffix = _PTZ_MODE_PREFIX.get(mode)
    if mode_suffix is None:
        raise ValueError(f"Unknown PTZ mode {mode!r}; use 'con', 'rel', or 'abs'")

    if cmd in _PTZ_MOVE_MAP:
        return f"move{mode_suffix}{_PTZ_MOVE_MAP[cmd]}"
    if cmd in _PTZ_ZOOM_MAP:
        return f"zoom{mode_suffix}{_PTZ_ZOOM_MAP[cmd]}"

    raise ValueError(
        f"Unknown PTZ command {command!r}; use one of: "
        f"{', '.join(sorted({*_PTZ_MOVE_MAP, *_PTZ_ZOOM_MAP, 'stop', 'home', 'preset'}))}"
    )
