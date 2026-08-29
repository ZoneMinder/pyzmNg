Remote ML Detection Server
===========================

``pyzm.serve`` is a built-in FastAPI server that loads ML models once
and serves detection requests over HTTP. This lets you offload GPU-heavy
inference to a dedicated machine.

.. note::

   **The server is a dumb inference engine.** It exposes a single
   ``POST /infer`` endpoint that runs *one* model on *one* frame and returns
   raw detections. All orchestration -- the model sequence,
   ``pre_existing_labels`` gating, and every filter (pattern, zone, size,
   past-detection dedup) plus frame selection -- runs on the **client** using
   your ``objectconfig.yml``, which also supplies the detection threshold with
   each request. The server holds only model files and a processor setting.
   This is what makes local and remote **object** detection identical (the test
   suite asserts local/remote parity); face recognition runs against the
   server's own encodings and is the exception. Two transports feed ``/infer``:
   **URL mode** (default) sends a frame reference and the server fetches it
   from ZM; **image mode** uploads the decoded frame as lossless PNG. URL mode
   needs every enabled model to be gateway-run and no ``resize`` in
   ``stream_sequence`` (a client-side model such as cloud ALPR, or a resize,
   forces image mode for that event).

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
   +-----------------+     HTTP/PNG      +------------------+
   | zm_detect.py    | ----------------> | pyzm.serve       |
   | Detector(       |                   |   YOLO11 (GPU)  |
   |   gateway=...,  | <---------------- |   Coral TPU      |
   |   gateway_mode= |  DetectionResult  +------------------+
   |     "image")    |
   +-----------------+

Both modes post to the same ``/infer`` endpoint; they differ only in how the
frame gets there:

- **URL mode** (default) -- the client sends a frame URL plus a ZM auth token
  and the *server* fetches the image directly from ZoneMinder. This avoids
  transferring every frame through the client.
- **Image mode** -- the client fetches frames from ZM and uploads each one as
  lossless PNG. Use this when the server cannot reach ZoneMinder directly, or
  when a ``resize`` means the server must not fetch full-size frames itself.

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
     - Higher — client uploads a PNG per frame
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
The client handles frame fetching and uploads lossless PNGs.

**When to stay with URL mode (default):**
Use URL mode when the server and ZoneMinder are on the same network.
This minimises bandwidth on the client side and lets the server fetch
only the frames it needs.

**Automatic fallback to image mode.** Two situations switch a single event to
image mode and log the reason:

- A client-side model is enabled (cloud ALPR, AWS Rekognition, audio). Those
  need local pixels, so the frames are downloaded anyway.
- ``stream_sequence.resize`` is set. The server fetches frames from ZM at full
  resolution and never sees the resize, so staying in URL mode would run
  inference on different pixels than a local run.


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

   pip install "pyzm[serve]"        # or "pyzm[full]" if this box also trains models
   python -m pyzm.serve --models all --processor gpu --port 5000

Or with specific models and auth:

.. code-block:: bash

   python -m pyzm.serve \
       --models "YOLO11s=yolo11s" yolo26s \
       --processor gpu \
       --port 5000 \
       --auth --auth-user admin --auth-password secret \
       --token-secret my-jwt-secret

**ZM box** -- install the client without the ``serve`` extra:

.. code-block:: bash

   pip install pyzm

**ZM box objectconfig.yml:**

.. code-block:: yaml

   remote:
     ml_gateway: "http://192.168.1.100:5000"
     ml_gateway_mode: "url"          # "image" if the server can't reach ZM
     ml_fallback_local: "yes"
     ml_timeout: 60

   ml:
     ml_sequence:
       general:
         model_sequence: "object"
       object:
         general:
           pattern: "(person|car)"
         sequence:
           - name: "YOLO11s"
             object_framework: opencv
             object_weights: "/var/lib/zmeventnotification/models/ultralytics/yolo11s.onnx"
             object_min_confidence: 0.5

By default, URL mode is used -- the server fetches frames directly from ZM.
Set ``ml_gateway_mode: "image"`` if the server cannot reach ZoneMinder
(the client then uploads lossless PNGs instead).

Note how the two sides line up: the sequence entry is named ``YOLO11s`` and the
server publishes ``"YOLO11s=yolo11s"``, so the name the client asks for exists
on the gateway. ``object_min_confidence`` stays on the ZM box -- it is sent with
each request. ``object_weights`` is used only if this event falls back to local
detection; for a remote run the gateway's own copy of the model is what loads,
so point the published name at equivalent weights.


Available models
-----------------

Model names and discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

Model names passed via ``--models`` (or ``Detector(models=[...])``) are
resolved against ``--base-path`` on disk. There are no hardcoded presets --
any name you pass is looked up as follows:

0. **Published name** -- an entry written ``<published name>=<spec>`` loads
   *spec* by the rules below but registers it under *published name*. See
   :ref:`serve-model-names`.
1. **Directory match** -- ``base_path/<name>/`` containing a weight file
2. **File stem match** -- any ``<name>.onnx``, ``<name>.weights``, or
   ``<name>.tflite`` in any subdirectory of ``base_path``

This lookup only works for models that *have* a weights file, and it always
yields ``type: object`` -- see
:ref:`serve-correlation` for what that means for a client, and
:ref:`Declaring models <serve-declaring>` for the models that need an explicit
declaration instead.

The framework is inferred from the file extension:

- ``.onnx`` -- OpenCV DNN (ONNX runtime)
- ``.weights`` -- OpenCV DNN (Darknet format, also needs a ``.cfg`` file)
- ``.tflite`` -- Coral Edge TPU runtime (processor forced to ``tpu``)

Label files are auto-detected from the same directory (``.names``,
``.txt``, ``.labels``). For Darknet models, ``.cfg`` files are also
discovered automatically.

The ``--processor`` flag (``cpu``, ``gpu``, ``tpu``) applies to all
discovered models (except ``.tflite`` which always uses ``tpu``).


.. _serve-model-names:

Matching model names with the client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A client asks for a model by name.** The ``name`` of each entry in the
client's sequence must match a name this server publishes. A name the server
has not loaded is reported as an error and that model is skipped -- the server
never substitutes a different model, because answering with one the client did
not ask for looks like success while returning the wrong detections.

Check what a running server publishes:

.. code-block:: bash

   curl -s http://gpu-box:5000/models | python3 -m json.tool

To serve a model under the name the client uses, write the entry as
``<published name>=<spec>``. The server loads *spec* and answers to *published
name*, so no weight files are renamed:

.. code-block:: bash

   python -m pyzm.serve --models "YOLOv11 ONNX=yolo11l"

The same syntax works in a YAML config file:

.. code-block:: yaml

   models:
     - "YOLOv11 ONNX=yolo11l"
     - "MobileDet=ssdlite_mobiledet_coco_qat_postprocess_edgetpu"

.. warning::

   Matching the *name* is enforced; matching the *weights* is not. Publishing
   ``yolo11s`` under a name the client's config points at ``yolo11l`` is
   accepted and runs, but the smaller model scores differently -- detections
   that pass the client's threshold locally can fall below it remotely, with no
   error anywhere. For exact local/remote parity, point the published name at
   the same weights the client would load.


.. _serve-correlation:

How a client entry and a server model correlate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correlation key is the pair **(type, name)**, and nothing else. For every
enabled model in its sequence, the client sends that pair; the server looks for
a loaded model with the same pair and runs it, or returns an error.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Client ``objectconfig.yml`` sequence entry
     - Must exist on the server as
   * - ``name: "YOLOv11 ONNX"`` in the ``object`` sequence
     - a model published as name ``YOLOv11 ONNX``, type ``object``
   * - ``name: "DLIB face recognition"`` in the ``face`` sequence
     - a model published as name ``DLIB face recognition``, type ``face``

The client's *type* comes from which sequence the entry sits in (``object``,
``face``, ``alpr``, ``audio``), not from anything you write on the entry. So a
face model registered on the server as type ``object`` can never be reached,
even when the names match exactly.

Confirm both halves of the pair with ``GET /models``, which reports the ``name``
and ``type`` of everything loaded.

.. _serve-declaring:

Declaring models: two forms, and when each works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**``models:`` (or ``--models``) is a filename shorthand.** Each entry is looked
up on disk under ``base_path``, and the *type* and *framework* are then inferred
from the file that was found:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - File found
     - Registered as
   * - ``<name>.onnx``
     - ``type: object``, ``framework: opencv``
   * - ``<name>.weights`` (+ ``.cfg``)
     - ``type: object``, ``framework: opencv`` (Darknet)
   * - ``<name>.tflite``
     - ``type: object``, ``framework: coral_edgetpu``, ``processor: tpu``
   * - *nothing*
     - ``type: object``, ``framework: opencv``, ``weights: None`` -- broken

Two consequences follow, and they explain why a gateway can serve YOLO happily
while failing on everything else:

- **A YOLO model needs nothing more than the shorthand.** ``yolo11l`` finds
  ``yolo11l.onnx``, and the extension alone establishes both facts the server
  needs. Nothing about that model has to be stated.
- **The shorthand can only ever produce** ``type: object``. There is no branch
  that yields ``face``, ``alpr`` or ``audio`` -- not even for a face-detection
  ``.tflite``, which registers as ``object`` like any other Coral model. So a
  face model named in ``models:`` is unreachable from a client's ``face``
  sequence even when the names match perfectly.

**``detector_config:`` declares models explicitly**, and is **required** --
not merely preferred -- whenever the shorthand cannot express what you need:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Model
     - Why the shorthand cannot express it
   * - dlib face recognition
     - Has no weights file at all -- it is driven by ``known_faces_dir`` and the
       ``faces.dat`` encodings trained from it. Nothing on disk to match, and
       nothing in a filename that says "face model, dlib framework".
   * - TPU face detection
     - The ``.tflite`` file exists, but it registers as ``type: object``. Only
       an explicit declaration gives it ``type: face``.
   * - Cloud ALPR / Rekognition
     - Configured by API key, not by a local file. (These run client-side
       anyway and are never requested from a gateway.)

A name in ``models:`` that matches no file on disk lands in the last row of the
first table: the ``ModelConfig`` defaults, with no weights. Loading it fails
with an explicit error naming ``detector_config`` as the fix; the gateway's
other models are unaffected and keep serving.

.. note::

   Older versions instead failed deep inside the OpenCV DNN loader with
   ``(-2:Unspecified error) Cannot determine an origin framework of files`` and
   a traceback through ``yolo_darknet.py`` -- confusing, because the model in
   question had nothing to do with YOLO. If you see that, this is the cause.

.. important::

   ``detector_config`` **replaces** the ``models`` list entirely. When it is
   present, top-level ``models``, ``base_path`` and ``processor`` are ignored,
   so every model needs its own absolute ``weights`` path and its own
   ``processor``.

Worked example -- object detection plus dlib face recognition on one gateway:

.. code-block:: yaml

   host: "0.0.0.0"
   port: 5000
   log_level: debug
   workers: 1

   detector_config:
     models:
       - name: "YOLOv11 ONNX"          # matches the client's object sequence entry
         type: object
         framework: opencv
         processor: gpu
         weights: "/var/lib/zmeventnotification/models/ultralytics/yolo11l.onnx"

       - name: "DLIB face recognition" # matches the client's face sequence entry
         type: face
         framework: face_dlib
         known_faces_dir: "/var/lib/zmeventnotification/known_faces"
         unknown_faces_dir: "/var/lib/zmeventnotification/unknown_faces"
         face_model: cnn

The matching client ``objectconfig.yml``:

.. code-block:: yaml

   remote:
     ml_gateway: "http://gpu-box:5000"
     ml_fallback_local: "yes"

   ml:
     ml_sequence:
       general:
         model_sequence: "object,face"
       object:
         sequence:
           - name: "YOLOv11 ONNX"      # same name, object sequence
             object_framework: opencv
             object_min_confidence: 0.5
       face:
         sequence:
           - name: "DLIB face recognition"   # same name, face sequence
             face_detection_framework: dlib

dlib face recognition also needs ``dlib`` and ``face_recognition`` importable on
the gateway -- they are imported lazily when the model loads and are **not** part
of the ``[serve]`` extra. The encodings are trained on the gateway from
``known_faces_dir`` the first time the model loads, so the face images must be
present *there*, not on the ZM box.


Which side owns which setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every setting has exactly one owner; there is no merging and no overriding.

**The client owns the outcome** -- sent with each request or applied to the
returned detections, so it is identical local and remote:

- ``min_confidence`` (sent on the ``/infer`` call and applied in place of the
  value this server loaded the model with)
- ``pattern``, zones (including ``zone_match_strategy``), ``max_detection_size``
- ``model_sequence``, ``same_model_sequence_strategy``, ``frame_strategy``
- past-detection filtering and ``pre_existing_labels`` gating

**The server owns the machine** -- never sent by the client, never derived from
the client's config:

- ``weights``, ``config``, ``labels`` and ``--base-path``. Paths never cross the
  wire in either direction; a client path is meaningless here.
- ``--processor`` (cpu/gpu/tpu) and the model input dimensions.
- Face recognition data: ``known_faces_dir``, ``unknown_faces_dir``, the trained
  encodings, and the face tuning parameters. Face recognition runs entirely on
  this box, matches against **this box's** encodings, and writes unknown-face
  crops to **this box's** disk. Train faces on the server -- which needs the
  ``[full]`` extra, not just ``[serve]``. Face results are therefore the one
  thing that need not match a local run.


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

Install into whichever environment will run the server. On a box that also runs
ZoneMinder that is usually its virtualenv, e.g.
``/opt/zoneminder/venv/bin/pip``. A client-only box (the ZM machine in a split
setup) needs plain ``pyzm``, without the ``[serve]`` extra.

If this box will also **train** models -- YOLO fine-tuning, or the face
recognition encodings, which must be built on the box that runs the face model
-- install ``[full]`` instead. It is ``[ml]`` + ``[serve]`` + ``[train]``, so it
covers the server as well:

.. code-block:: bash

   pip install "pyzm[full]"

.. important::

   A running server holds its code in memory, so **restart it** after an
   upgrade. Otherwise the new version is on disk while the old one keeps
   answering requests -- typically seen as an endpoint that 404s even though the
   installed copy has it. Confirm which version is loaded with:

   .. code-block:: bash

      python -c "import pyzm; print(pyzm.__version__, pyzm.__file__)"

   Client and server should run the same pyzm version. A newer client sending
   settings an older server does not understand (for example
   ``min_confidence``) falls back to the server's own values, which quietly
   changes results.


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
   * - ``--no-cpu-fallback``
     - off
     - Fail a request instead of degrading to CPU when GPU inference errors.
       See :ref:`serve-gpu-fallback`.
   * - ``--gpu-retry-seconds``
     - ``60``
     - Seconds on CPU after a GPU failure before the GPU is retried, doubling
       after each further failure. ``0`` makes a fallback permanent.
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


.. _serve-gpu-fallback:

When the GPU fails
~~~~~~~~~~~~~~~~~~~

CUDA errors happen: a driver hiccup, another process holding the device, a
failed unified-memory allocation. When an inference call on a ``gpu`` model
raises, the server does not fail the request -- it moves that model to CPU,
re-runs the frame, and answers. Detection keeps working, several times slower.

That fallback is **temporary**. After ``--gpu-retry-seconds`` (60 by default)
the next request puts the model back on CUDA. If it fails again the wait
doubles, up to 15 minutes, so a genuinely broken GPU is not re-probed on every
request while a momentary fault heals itself within a minute. Both events are
logged at ``ERROR`` / ``INFO`` by the ``pyzm.ml`` logger:

.. code-block:: text

   pyzm.ml ERROR yolo11m: GPU failed: OpenCV(4.12.0) ... CUDA-capable device(s)
     is/are busy or unavailable. Falling back to CPU; retrying GPU in 60 seconds.
   pyzm.ml INFO  yolo11m: retrying GPU inference after an earlier CPU fallback

**Detecting a degraded server.** Logs are not a monitor. ``GET /models``
reports, per model, the ``processor`` in use right now and the
``requested_processor`` it was configured with; a gateway that has degraded is
one where they differ. The endpoint needs no authentication, so a container
healthcheck can use it directly:

.. code-block:: bash

   curl -sf http://localhost:5000/models \
     | python3 -c 'import json,sys; m=json.load(sys.stdin)["models"]; \
       sys.exit(any(x["processor"] != x["requested_processor"] for x in m))'

**Refusing to degrade.** Some deployments would rather fail a request than
answer it slowly and silently -- a caller cannot tell a slow answer from a fast
one, but it can act on an error. ``--no-cpu-fallback`` makes a GPU failure
propagate instead: ``/infer`` returns ``{"detections": [], "error": "..."}``
and the model stays on GPU for the next request.

.. code-block:: bash

   # never answer from CPU; retry the GPU after 30s instead of 60s
   python -m pyzm.serve --models yolo11m --processor gpu \
       --no-cpu-fallback --gpu-retry-seconds 30

Both settings also exist per-model as ``allow_cpu_fallback`` and
``gpu_retry_seconds`` on a ``ModelConfig`` (in a ``detector_config`` block or
an ``ml_sequence`` entry). The CLI flags apply to every model the server loads
and are only stamped onto the model configs when you pass them.


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

URL mode: frame selection and short-circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In URL mode the client builds the frame list from ``StreamConfig``
(``stream_sequence``) and sends the server one frame reference per ``/infer``
call. The server has no notion of an event, a frame list, or a retry policy --
it fetches the one URL it was given.

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

.. note::

   ``max_attempts``, ``sleep_between_attempts`` and
   ``contig_frames_before_error`` apply to the **client's** frame download, so
   they take effect in image mode only. In URL mode nothing is downloaded on
   the client, and the server does not retry a frame it cannot fetch -- that
   ``/infer`` call returns an ``error`` and the client moves on.

**Short-circuit** -- ``stop_on_match`` (default ``True``) stops the client
requesting further frames once one frame produces a match. Filtering is always
client-side; the server only runs inference.

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

   # Infer (one model, one image)
   curl -X POST http://gpu-box:5000/infer \
       -H "Authorization: Bearer $TOKEN" \
       -F image=@/path/to/image.png \
       -F type=object

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

Returns the list of available models, their load status, and the processor each
one is running on. Useful with ``--models all`` to check which backends have
been lazily loaded, to confirm the ``name`` and ``type`` a client must ask for
(:ref:`serve-correlation`), and to detect a GPU model that has degraded to CPU
(:ref:`serve-gpu-fallback`).

.. code-block:: json

   {
     "models": [
       {"name": "yolo11s", "type": "object", "framework": "opencv",
        "loaded": true, "processor": "gpu", "requested_processor": "gpu"},
       {"name": "yolo26s", "type": "object", "framework": "opencv",
        "loaded": false, "processor": "cpu", "requested_processor": "cpu"}
     ]
   }

``processor`` is what inference is running on **now**; ``requested_processor``
is what the model was configured with. They differ only when a GPU model has
fallen back to CPU. This endpoint is never authenticated, so a container
healthcheck can compare the two without a token.

``POST /infer``
~~~~~~~~~~~~~~~~

Run **one** model on **one** frame and return detections that have had no
filtering applied beyond the requested ``min_confidence``. The server does no
pattern, zone, size or past-detection filtering, no model-sequence
orchestration and no frame selection -- the client's :class:`ModelPipeline`
does all of that, so local and remote object detection produce identical
results.

- **Content-Type:** ``multipart/form-data``
- **Parameters:**

  - ``type`` (required) -- model type: ``object``, ``face``, ``alpr``, ``audio``
  - ``name`` (optional) -- model name. Omitted, the server uses its first loaded
    model of that ``type``. Given, it must match a published name exactly; an
    unknown name is an error, never a substitution.
  - ``min_confidence`` (optional) -- replaces the threshold this server loaded
    the model with, so the client's configured value applies. Omitted (an older
    client), the server's own value is used.
  - **URL mode:** ``url`` (ZM image URL) + ``zm_auth`` (token) + ``verify_ssl``
    (``"1"``/``"0"``) -- the server fetches the frame from ZM
  - **Image mode:** ``image`` -- an uploaded frame (PNG lossless, or JPEG)

- **Auth:** Bearer token (when auth enabled)
- **Returns:**

  .. code-block:: json

     {
       "detections": [
         {"label": "person", "confidence": 0.93, "box": [10, 20, 50, 80],
          "type": "object", "model_name": "yolo11s"}
       ],
       "error": null
     }

  When the server has no model for the requested ``(type, name)``,
  ``detections`` is empty and ``error`` explains why (the client logs it and
  continues, like a local model that fails to load). Both halves of that pair
  must match something loaded here -- see :ref:`serve-correlation`.

.. note::

   The client sends model **references** (``type``/``name``) plus the settings
   it owns, never file paths and never its whole config. The server resolves the
   reference against its own loaded models. For local/remote parity the server
   must publish the names the client's config references, backed by equivalent
   weights -- see :ref:`serve-model-names`.

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

The gateway keys live in their own top-level ``remote`` section, not inside
``ml_sequence``:

.. code-block:: yaml

   remote:
     ml_gateway: "http://gpu-box:5000"
     ml_gateway_mode: "url"            # "image" if the server can't reach ZM
     ml_user: "admin"
     ml_password: "secret"
     ml_timeout: 60
     ml_fallback_local: "yes"

   ml:
     ml_sequence:
       general:
         model_sequence: "object"
       object:
         general:
           pattern: "(person|car)"
         sequence:
           - name: "YOLO11s"           # must match a name the server publishes
             object_framework: opencv
             object_weights: /path/to/yolo11s.onnx
             object_labels: /path/to/coco.names
             object_min_confidence: 0.5

When ``ml_gateway`` is set, detection requests are sent to the remote
server.  URL mode is the default -- the server fetches frames directly
from ZM.  Set ``ml_gateway_mode: "image"`` if the server cannot reach
ZoneMinder.  If ``ml_fallback_local`` is ``yes`` and the remote server
is unreachable, detection falls back to local inference.

While first setting a gateway up, use ``ml_fallback_local: "no"`` so a gateway
problem fails loudly instead of quietly running locally and looking like it
worked.

The full ZoneMinder-side guide is in the zmeventnotificationNg docs:

- `Using the remote ML detection server <https://zmeventnotificationng.readthedocs.io/en/latest/guides/hooks.html#remote-ml-config>`__
  -- end-to-end setup for both boxes
- `Matching model names <https://zmeventnotificationng.readthedocs.io/en/latest/guides/hooks.html#remote-model-names>`__
  -- the same correlation rule as :ref:`serve-correlation`, from the client side
- `Which side owns which setting <https://zmeventnotificationng.readthedocs.io/en/latest/guides/hooks.html#remote-config-ownership>`__
- `Verifying a remote setup <https://zmeventnotificationng.readthedocs.io/en/latest/guides/hooks.html#verifying-a-remote-setup>`__
  -- commands that prove a run went remote, plus a symptom/cause table
