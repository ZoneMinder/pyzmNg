Client & Authentication
=======================

Connecting to ZoneMinder
-------------------------

.. code-block:: python

   from pyzm import ZMClient

   zm = ZMClient(
       api_url="https://zm.example.com/zm/api",
       user="admin",
       password="secret",
       # verify_ssl=False,  # for self-signed certs
   )

   print(f"ZM {zm.zm_version}, API {zm.api_version}")

``api_url`` must be the full ZM API URL (ending in ``/api``).

Constructor parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``api_url``
     - (required)
     - Full ZM API URL (e.g. ``https://server/zm/api``)
   * - ``user``
     - ``None``
     - ZM username. ``None`` when auth is disabled.
   * - ``password``
     - ``None``
     - ZM password
   * - ``portal_url``
     - auto
     - Full portal URL (e.g. ``https://server/zm``). Auto-derived from
       ``api_url`` when not provided.
   * - ``verify_ssl``
     - ``True``
     - Set to ``False`` for self-signed certificates
   * - ``timeout``
     - ``30``
     - HTTP request timeout in seconds
   * - ``db_user``
     - ``None``
     - Override the ZM database username (normally read from ``zm.conf``)
   * - ``db_password``
     - ``None``
     - Override the ZM database password
   * - ``db_host``
     - ``None``
     - Override the ZM database host (e.g. ``"dbhost"`` or ``"dbhost:3307"``)
   * - ``db_name``
     - ``None``
     - Override the ZM database name
   * - ``conf_path``
     - ``None``
     - Path to the ZM config directory. When unset, pyzm honors the
       ``PYZM_CONFPATH`` env var, then auto-discovers ``zm.conf`` across
       ``/etc/zm``, ``/config``, and ``/etc/zoneminder``. Set this to force a
       non-standard location.
   * - ``config``
     - ``None``
     - A pre-built ``ZMClientConfig``. When provided, all other keyword
       args are ignored.

Authentication
---------------

pyzm handles authentication internally.  When ``user`` and ``password``
are provided, it obtains an API token and refreshes it automatically.
Legacy (non-token) authentication is also supported for older ZM
installations.

Set ``verify_ssl=False`` if your ZM server uses a self-signed TLS
certificate.

Database access
----------------

Some operations — ``ev.tag()``, ``ev.path()``, ``ev.save_objdetect()``
(when no ``path_override`` is given), and audio extraction for BirdNET —
require a direct MySQL connection to the ZM database.  By
default, pyzm reads credentials from ``zm.conf`` (the same file
ZoneMinder uses).  It looks for that file first at ``PYZM_CONFPATH`` (if
the env var is set), then auto-discovers it across ``/etc/zm``,
``/config`` (common in containerized ZoneMinder), and ``/etc/zoneminder``.
Pass ``conf_path`` to force a specific directory.

If the user running pyzm cannot read ``zm.conf`` (e.g. permission
denied), you can pass database credentials explicitly:

.. code-block:: python

   zm = ZMClient(
       api_url="https://zm.example.com/zm/api",
       user="admin",
       password="secret",
       db_user="zmuser",
       db_password="zmpass",
       db_host="localhost",
   )

   ev = zm.event(12345)
   ev.tag(["person"])   # uses the explicit DB credentials
   path = ev.path()     # same

The merge strategy is:

1. Resolve the config directory: ``conf_path`` if set, else
   ``PYZM_CONFPATH``, else the first of ``/etc/zm`` / ``/config`` /
   ``/etc/zoneminder`` that contains a ``zm.conf``.
2. Try to read ``zm.conf`` from that directory.
3. Overlay any explicit ``db_*`` parameters — explicit values always win.
4. Fall back to ``"localhost"`` for host and ``"zm"`` for database name.

Accessing the full API response
--------------------------------

pyzm models only extract a subset of the fields returned by the ZM API.
Every API-sourced object carries a ``raw()`` method that returns the full,
unmodified API response dict — useful for accessing fields like ``Path``,
``Protocol``, ``StorageId``, etc.:

.. code-block:: python

   m = zm.monitor(1)
   m.raw()["Monitor"]["Path"]        # e.g. "rtsp://cam/stream"
   m.raw()["Monitor"]["Protocol"]    # e.g. "rtsp"
   m.status.raw()                    # full Monitor_Status sub-dict

   ev = zm.event(12345)
   ev.raw()["Event"]["DiskSpace"]    # field not on the Event dataclass

   frames = ev.get_frames()
   frames[0].raw()                   # full Frame API dict

   zones = m.get_zones()
   zones[0].raw()                    # full Zone API dict including AlarmRGB, etc.

``raw()`` is available on ``Monitor``, ``MonitorStatus``, ``Event``,
``Frame``, ``Zone``, ``PTZCapabilities``, and ``Notification``.

Notifications (Push Tokens)
----------------------------

*Requires ZoneMinder 1.39.2+.*

ZoneMinder's ``Notifications`` API stores FCM push tokens registered by client
apps (e.g. zmNinjaNg). pyzm provides methods to query and manage these tokens —
primarily used by ``zm_detect`` to send push notifications after detection.

.. code-block:: python

   # List all registered tokens for the authenticated user
   tokens = zm.notifications()
   for t in tokens:
       print(f"Id={t.id}, platform={t.platform}, token={t.token[:20]}...")

   # Get a specific notification by ID
   notif = zm.notification(1)

   # Check if a token should receive a notification for a given monitor
   if notif.should_notify(monitor_id=3):
       print("Token is registered for monitor 3")

   # Check throttle (returns True if Interval has not elapsed since LastNotifiedAt)
   if notif.is_throttled():
       print("Skipping — too soon since last notification")

   # Update LastNotifiedAt and increment BadgeCount after sending
   notif.update_last_sent(badge=notif.badge_count + 1)

   # Delete a token (e.g. when FCM reports it as invalid)
   notif.delete()

Notification fields
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``id``
     - ``int``
     - Primary key
   * - ``user_id``
     - ``int | None``
     - Owning ZM user (``None`` when auth is disabled)
   * - ``token``
     - ``str``
     - FCM registration token
   * - ``platform``
     - ``str``
     - ``"android"``, ``"ios"``, or ``"web"``
   * - ``monitor_list``
     - ``str | None``
     - Comma-separated monitor IDs, or ``None`` for all monitors
   * - ``interval``
     - ``int``
     - Minimum seconds between notifications (0 = no throttle)
   * - ``push_state``
     - ``str``
     - ``"enabled"`` or ``"disabled"``
   * - ``app_version``
     - ``str | None``
     - Client app version
   * - ``badge_count``
     - ``int``
     - Current badge count
   * - ``last_notified_at``
     - ``datetime | None``
     - When the last push was sent to this token

Helper methods
~~~~~~~~~~~~~~~

- ``monitors()`` — returns the monitor list as a ``list[int]``, or empty list if all monitors
- ``should_notify(monitor_id)`` — ``True`` if this token should receive notifications for the given monitor (checks ``push_state``, ``monitor_list``)
- ``is_throttled()`` — ``True`` if ``interval`` seconds have not elapsed since ``last_notified_at``
- ``update_last_sent(badge)`` — updates ``LastNotifiedAt`` to now and sets ``BadgeCount``
- ``delete()`` — deletes this notification record from ZM
