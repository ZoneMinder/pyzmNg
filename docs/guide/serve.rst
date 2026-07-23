Remote ML Detection Server
===========================

``pyzm.serve`` is a built-in FastAPI server that loads ML models once
and serves detection requests over HTTP. This lets you offload GPU-heavy
inference to a dedicated machine.

.. code-block:: text

   URL mode (default)                     GPU box
   +-----------------+   frame URLs      +------------------+
   | zm_detect.py    | ----------------> | pyzm.serve       |
   | Detector(       |                   |  fetch from ZM   |
   |   gateway=...)  | <---------------- |  detect & return |
   +-----------------+  DetectionResult  +------------------+
                                                |
                                         +------v-----------+
                                         | ZoneMinder API   |
                                         +------------------+

   Image mode (gateway_mode="image")      GPU box
   +-----------------+     HTTP/JPEG     +------------------+
   | zm_detect.py    | ----------------> | pyzm.serve       |
   | Detector(       |                   |   YOLO11 (GPU)  |
   |   gateway=...,  | <---------------- |   Coral TPU      |
   |   gateway_mode= |  DetectionResult  +------------------+
   |     "image")    |
   +-----------------+

Two detection modes are available:

- **URL mode** (default) -- the client sends frame URLs to the
  ``/detect_urls`` endpoint and the *server* fetches images directly from
  ZoneMinder.  This avoids transferring every frame through the client.
- **Image mode** -- the client fetches frames from ZM, JPEG-encodes them,
  and uploads each one to the ``/detect`` endpoint.  Use this when the
  server cannot reach ZoneMinder directly.

.. list-table:: URL mode vs Image mode trade-offs
   :header-rows: 1
   :widths: 30 35 35

   * -
     - URL mode (default)
     - Image mode
   * - Network requirement
     - Server must reach ZoneMinder
     - Only client needs ZM access
   * - Bandwidth
     - Low — client sends only URLs
     - Higher — client uploads JPEG per frame
   * - Latency
     - Server fetches from ZM (one extra hop)
     - Single client → server transfer
   * - Security
     - ZM credentials forwarded via ``zm_auth``
     - Images leave ZM network
   * - Configuration
     - ``gateway_mode`` omitted or ``"url"``
     - ``gateway_mode="image"`` (Python) or
       ``ml_gateway_mode: "image"`` (YAML)
   * - Best for
     - Same network / VPN between server and ZM
     - Server on a different network or cloud

**When to choose Image mode:**
Use Image mode when the GPU server cannot reach the ZoneMinder API
directly (e.g., server is in the cloud, or firewall rules prevent it).
The client handles frame fetching and uploads JPEG images to ``/detect``.

**When to stay with URL mode (default):**
Use URL mode when the server and ZoneMinder are on the same network.
This minimises bandwidth on the client side and lets the server fetch
only the frames it needs.


Deployment scenarios
---------------------

Scenario 1: ZM + EventServerNg + hooks + pyzm (same box)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything runs on the same machine. The ZoneMinder EventServerNg (zmesNg)
triggers hook scripts which call ``zm_detect.py``, and detection runs
locally via the ``Detector`` class.

.. code-block:: text

   ZoneMinder --> zmeventnotification.pl (zmesNg)
                     |
                     v
                  zm_event_start.sh
                     |
                     v
                  zm_detect.py --> Detector (local GPU/CPU)

**objectconfig.yml** (no ``remote`` section needed):

.. code-block:: yaml

   ml:
     ml_sequence:
       general:
         model_sequence: "object"
       object:
         general:
           pattern: "(person|car)"
         sequence:
           - name: YOLO11s
             object_weights: "/var/lib/zmeventnotification/models/ultralytics/yolo11s.onnx"
             object_labels: "/var/lib/zmeventnotification/models/yolov4/coco.names"
             object_framework: opencv
             object_processor: gpu

**Test locally:**

.. code-block:: bash

   sudo -u www-data /opt/zoneminder/venv/bin/python /path/to/zm_detect.py \
       --config /etc/zm/objectconfig.yml \
       --eventid 12345


Scenario 2: ZM + hooks + pyzm (same box, no zmesNg)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as Scenario 1 but without EventServerNg. ZoneMinder calls
``zm_detect.py`` directly via its ``EventStartCommand`` / ``EventEndCommand``
recording settings.

.. code-block:: text

   ZoneMinder EventStartCommand --> zm_detect.py --> Detector (local)

**ZoneMinder Console -> Click on Monitor Source -> Recording:**

.. code-block:: text

   EventStartCommand = /opt/zoneminder/venv/bin/python /path/to/zm_detect.py -c /etc/zm/objectconfig.yml -e %EID% -m %MID% -r "%EC%" -n

**objectconfig.yml** is the same as Scenario 1.


Scenario 3: ZM box + remote GPU box (split architecture)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detection runs on a separate GPU machine. The ZM box runs ``zm_detect.py``
which sends requests to the remote ``pyzm.serve`` server over HTTP.

.. code-block:: text

   ZM box                              GPU box
   +-------------------+               +------------------------+
   | zm_detect.py      |   HTTP        | pyzm.serve             |
   | Detector(         | ------------> |   --models all         |
   |   gateway=...)    |               |   --processor gpu      |
   |                   | <------------ |   --port 5000          |
   +-------------------+  JSON result  +------------------------+

**GPU box setup:**

.. code-block:: bash

   pip install pyzm[serve]
   python -m pyzm.serve --models all --processor gpu --port 5000

Or with specific models and auth:

.. code-block:: bash

   python -m pyzm.serve \
       --models yolo11s yolo26s \
       --processor gpu \
       --port 5000 \
       --auth --auth-user admin --auth-password secret \
       --token-secret my-jwt-secret

**ZM box objectconfig.yml:**

.. code-block:: yaml

   ml:
     ml_sequence:
       general:
         model_sequence: "object"
         ml_gateway: "http://192.168.1.100:5000"
         # ml_gateway_mode: "image"   # uncomment if server can't reach ZM directly
         ml_fallback_local: "yes"
       object:
         general:
           pattern: "(person|car)"
         sequence:
           - object_framework: opencv
             object_weights: "/var/lib/zmeventnotification/models/ultralytics/yolo11s.onnx"

By default, URL mode is used -- the server fetches frames directly from ZM.
Set ``ml_gateway_mode: "image"`` if the server cannot reach ZoneMinder
(the client will JPEG-encode and upload frames instead).


Available models
-----------------

Model names and discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

Model names passed via ``--models`` (or ``Detector(models=[...])``) are
resolved against ``--base-path`` on disk. There are no hardcoded presets --
any name you pass is looked up as follows:

1. **Directory match** -- ``base_path/<name>/`` containing a weight file
2. **File stem match** -- any ``<name>.onnx``, ``<name>.weights``, or
   ``<name>.tflite`` in any subdirectory of ``base_path``

The framework is inferred from the file extension:

- ``.onnx`` -- OpenCV DNN (ONNX runtime)
- ``.weights`` -- OpenCV DNN (Darknet format, also needs a ``.cfg`` file)
- ``.tflite`` -- Coral Edge TPU runtime (processor forced to ``tpu``)

Label files are auto-detected from the same directory (``.names``,
``.txt``, ``.labels``). For Darknet models, ``.cfg`` files are also
discovered automatically.

The ``--processor`` flag (``cpu``, ``gpu``, ``tpu``) applies to all
discovered models (except ``.tflite`` which always uses ``tpu``).


``--models all`` (lazy loading)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you pass ``--models all`` to the server, **every** model in
``--base-path`` is discovered and registered, but weights are **not**
loaded into memory at startup. Instead, each backend loads its weights
on the first request that uses it.

This is useful when you have many models but don't want to consume
GPU memory for all of them upfront.

.. code-block:: bash

   python -m pyzm.serve --models all --base-path /data/models --processor gpu

Use the ``GET /models`` endpoint to check which models are available
and whether their weights have been loaded.


Model directory layout
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   /var/lib/zmeventnotification/models/
   +-- yolov4/
   |   +-- yolov4.weights
   |   +-- yolov4.cfg
   |   +-- coco.names
   +-- ultralytics/
   |   +-- yolo11s.onnx
   |   +-- yolo11n.onnx
   |   +-- yolo26s.onnx
   +-- coral_edgetpu/
   |   +-- ssd_mobilenet_v2.tflite
   |   +-- coco_labels.txt


Server setup
-------------

Installation
~~~~~~~~~~~~~

The ``[serve]`` extra automatically includes all ML dependencies.

.. code-block:: bash

   pip install "pyzm[serve]"


CLI options
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Description
   * - ``--host``
     - ``0.0.0.0``
     - Bind address
   * - ``--port``
     - ``5000``
     - Bind port
   * - ``--models``
     - ``yolo11s``
     - Model names (space-separated). Use ``all`` to auto-discover every
       model in ``--base-path`` (loaded lazily on first request).
   * - ``--base-path``
     - ``/var/lib/zmeventnotification/models``
     - Directory containing model subdirectories
   * - ``--processor``
     - ``cpu``
     - ``cpu``, ``gpu``, or ``tpu``
   * - ``--auth``
     - off
     - Enable JWT authentication
   * - ``--auth-user``
     - ``admin``
     - Username (when auth enabled)
   * - ``--auth-password``
     - (empty)
     - Password (when auth enabled)
   * - ``--token-secret``
     - ``change-me``
     - Secret key used to sign JWT tokens. **Change this in production.**
   * - ``--debug``
     - off
     - Enable debug logging for both pyzm and uvicorn
   * - ``--workers``
     - ``1``
     - Number of uvicorn worker processes. Each worker loads its own copy of
       the model(s) into memory, enabling parallel inference across
       simultaneous events. Useful on **CPU**; see *Multi-worker* below.
   * - ``--config``
     - (none)
     - Path to a YAML config file (``ServerConfig``). Overrides CLI flags.


YAML config file
~~~~~~~~~~~~~~~~~

Instead of CLI flags, you can provide a YAML config file via ``--config``:

.. code-block:: bash

   python -m pyzm.serve --config /etc/pyzm/serve.yml

Example ``serve.yml``:

.. code-block:: yaml

   host: "0.0.0.0"
   port: 5000
   models:
     - yolo11s
     - yolo26s
   base_path: "/var/lib/zmeventnotification/models"
   processor: gpu
   auth_enabled: true
   auth_username: admin
   auth_password: "my-secret-password"
   token_secret: "a-strong-random-secret"
   token_expiry_seconds: 3600
   workers: 3          # parallel worker processes (CPU)
   log_level: info     # debug, info, warning, error, critical

All fields correspond to ``ServerConfig`` attributes. When using
``--config``, CLI flags are ignored.


Multi-worker (CPU parallelism)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the server runs a single process. On **CPU**, each inference
call is blocking (~500 ms/frame for YOLOv4), so simultaneous events queue
up behind one another. Passing ``--workers N`` spawns ``N`` uvicorn worker
processes, each with its own copy of the model(s), so events are handled
truly in parallel:

.. code-block:: bash

   python -m pyzm.serve --models yolov4 --processor cpu --workers 3

Notes:

- Each worker loads the model independently (e.g. ~443 MB per worker for
  YOLOv4 at 576×576), so memory scales with the worker count.
- On **GPU**, a single worker is normally sufficient; extra workers
  multiply VRAM usage without a meaningful throughput gain.
- The parsed configuration (including auth credentials) is handed to the
  workers automatically -- authentication works the same as single-worker
  mode.
- ``--workers`` is ignored on Windows.


Client usage
-------------

Using the ``Detector`` API:

.. code-block:: python

   from pyzm import Detector

   # URL mode (default) -- server fetches frames from ZM
   detector = Detector(models=["yolo11s"], gateway="http://gpu-box:5000")

   # Image mode -- client uploads JPEG-encoded frames
   # detector = Detector(models=["yolo11s"], gateway="http://gpu-box:5000",
   #                     gateway_mode="image")

   # With authentication:
   # detector = Detector(models=["yolo11s"], gateway="http://gpu-box:5000",
   #                     gateway_username="admin", gateway_password="secret")

   # detect() always uploads the image (single-image mode)
   result = detector.detect("/path/to/image.jpg")
   print(result.summary)

   # detect_event() uses URL mode by default -- sends frame URLs,
   # server fetches them directly from ZM
   result = detector.detect_event(zm_client, event_id=12345,
                                   stream_config=stream_cfg)

URL mode only applies to ``detect_event()`` calls.  Single-image
``detect()`` calls always upload the image regardless of this setting.

URL mode: frame selection, retry, and short-circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In URL mode the following ``StreamConfig`` (``stream_sequence``) fields
control which frames are analysed and how the gateway processes them.
They are forwarded to the server as part of the request. All are optional
and default to backwards-compatible behaviour.

**Frame selection** -- how the list of frames to analyse is built:

1. ``frame_set`` has values (default ``["snapshot", "alarm", "1"]``) --
   the list is used as-is. Supports named frames (``snapshot``, ``alarm``)
   and numeric frame IDs.
2. ``frame_set`` is empty **and** ``max_frames > 0`` -- a sparse frame list
   is generated from ``start_frame`` + ``frame_skip`` + ``max_frames``.
   For example ``start_frame=50, frame_skip=25, max_frames=30`` analyses
   frames ``[50, 75, 100, …, 775]``. This distributes analysis across a
   long event without downloading every frame.
3. ``frame_set`` is empty **and** ``max_frames`` is 0 -- falls back to
   ``["snapshot", "alarm", "1"]`` (with a warning), so an empty
   ``frame_set`` never silently collapses to a single frame.

Existing installations that do not explicitly set ``frame_set: []`` are
unaffected.

**Retry for live events** -- when detection starts while frames are still
being written to disk, the gateway retries missing frames (HTTP 404):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Field
     - Default
     - Description
   * - ``max_attempts``
     - ``1``
     - How many times the gateway fetches a 404 frame before treating it as
       missing (``1`` = no retry).
   * - ``sleep_between_attempts``
     - ``3``
     - Seconds to wait between retry attempts for a missing frame.
   * - ``contig_frames_before_error``
     - ``5``
     - Stop processing after this many **consecutive** missing frames
       (end-of-event detection). The counter resets on any frame fetched
       successfully.

**Short-circuit** -- ``stop_on_match`` (default ``True``) lets the gateway
stop as soon as one frame produces a detection that passes all server-side
filters (confidence, pattern, and zone), avoiding inference on the
remaining frames.

.. note::

   Short-circuiting only takes effect for the ``first`` and ``first_new``
   frame strategies. The ``most``, ``most_unique`` and ``most_models``
   strategies must examine every frame to pick the best one, so the client
   automatically suppresses ``stop_on_match`` for them regardless of the
   configured value.

Using ``from_dict()``
~~~~~~~~~~~~~~~~~~~~~~

The ``ml_gateway`` key in the ``general`` section of ``ml_options``
automatically enables remote mode:

.. code-block:: python

   ml_options = {
       "general": {
           "model_sequence": "object",
           "ml_gateway": "http://gpu-box:5000",
           # "ml_gateway_mode": "image",  # uncomment if server can't reach ZM
           # "ml_user": "admin",
           # "ml_password": "secret",
       },
       "object": {
           "general": {"pattern": ".*"},
           "sequence": [...],
       },
   }

   detector = Detector.from_dict(ml_options)
   result = detector.detect(image)


Authentication
---------------

When the server is started with ``--auth``, clients must first obtain a
JWT token via ``/login``, then pass it as a Bearer token on subsequent
requests. The ``Detector`` gateway mode handles this automatically.

The ``--token-secret`` flag controls the secret key used to sign JWT
tokens. **Always set this to a strong random value in production.**
The default (``change-me``) is insecure.

Manual flow:

.. code-block:: bash

   # Login
   TOKEN=$(curl -s -X POST http://gpu-box:5000/login \
       -H 'Content-Type: application/json' \
       -d '{"username":"admin","password":"secret"}' \
       | jq -r .access_token)

   # Detect
   curl -X POST http://gpu-box:5000/detect \
       -H "Authorization: Bearer $TOKEN" \
       -F file=@/path/to/image.jpg

Tokens expire after ``token_expiry_seconds`` (default 3600), configurable
via the YAML config file.


API reference
--------------

``GET /health``
~~~~~~~~~~~~~~~~

Health check. Returns:

.. code-block:: json

   {"status": "ok", "models_loaded": true}

``GET /models``
~~~~~~~~~~~~~~~~

Returns the list of available models and their load status. Useful with
``--models all`` to check which backends have been lazily loaded.

.. code-block:: json

   {
     "models": [
       {"name": "yolo11s", "type": "object", "framework": "opencv", "loaded": true},
       {"name": "yolo26s", "type": "object", "framework": "opencv", "loaded": false}
     ]
   }

``POST /detect``
~~~~~~~~~~~~~~~~~

Run detection on an uploaded image (image mode).

- **Content-Type:** ``multipart/form-data``
- **Parameters:**
  - ``file`` (required) -- JPEG/PNG image
  - ``zones`` (optional) -- JSON string of zone list
- **Auth:** Bearer token (when auth enabled)
- **Returns:** ``DetectionResult`` as JSON (image field excluded)

``POST /detect_urls``
~~~~~~~~~~~~~~~~~~~~~~

Run detection on images fetched from URLs (URL mode).

- **Content-Type:** ``application/json``
- **Body:**

  .. code-block:: json

     {
       "urls": [
         {"frame_id": "snapshot", "url": "https://zm.example.com/zm/index.php?view=image&eid=123&fid=snapshot"},
         {"frame_id": "1", "url": "https://zm.example.com/zm/index.php?view=image&eid=123&fid=1"}
       ],
       "zm_auth": "token=abc123...",
       "zones": [{"name": "driveway", "value": [[0,0],[100,0],[100,100],[0,100]]}],
       "verify_ssl": false
     }

- **Auth:** Bearer token (when auth enabled)
- **Returns:** ``DetectionResult`` as JSON (best frame selected by ``frame_strategy``)

The server appends ``zm_auth`` to each URL and fetches the image via HTTP
GET (10-second timeout per URL; failures are logged and skipped).
Frame strategy (``first``, ``first_new``, ``most``, ``most_unique``,
``most_models``) is applied server-side to pick the best result.

.. note::

   The ``zones`` field accepts both ``"value"`` and ``"points"`` as the key
   for polygon coordinates (they are interchangeable in both ``/detect``
   and ``/detect_urls``).

``POST /login``
~~~~~~~~~~~~~~~~

Obtain a JWT token. This endpoint is always registered, even when
``--auth`` is not enabled (so clients with pre-configured credentials
don't get a 404). When auth is disabled, any credentials are accepted.

- **Content-Type:** ``application/json``
- **Body:** ``{"username": "...", "password": "..."}``
- **Returns:** ``{"access_token": "...", "expires": 3600}``


objectconfig.yml remote section
---------------------------------

In a ZoneMinder event notification setup, configure the remote gateway
in ``objectconfig.yml``:

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object"
       ml_gateway: "http://gpu-box:5000"
       # ml_gateway_mode: "image"        # uncomment if server can't reach ZM
       ml_user: "admin"
       ml_password: "secret"
       ml_fallback_local: yes

     object:
       general:
         pattern: "(person|car)"
       sequence:
         - object_framework: opencv
           object_weights: /path/to/yolo11s.onnx
           object_labels: /path/to/coco.names

When ``ml_gateway`` is set, detection requests are sent to the remote
server.  URL mode is the default -- the server fetches frames directly
from ZM.  Set ``ml_gateway_mode: "image"`` if the server cannot reach
ZoneMinder.  If ``ml_fallback_local`` is ``yes`` and the remote server
is unreachable, detection falls back to local inference.
