"""Direct MySQL connection to the ZoneMinder database.

Reads credentials from ``zm.conf`` (and ``conf.d/*.conf``) — the same files
that ZM itself uses.  The directory is resolved by :func:`_resolve_conf_path`:
an explicit ``conf_path`` or the ``PYZM_CONFPATH`` env var wins, otherwise a
few well-known locations are probed (handles containerized ZM, which keeps
config under ``/config``).  Explicit credential overrides take precedence.
"""

from __future__ import annotations

import configparser
import glob
import logging
import os

logger = logging.getLogger("pyzm.zm")

_CONF_PATH = os.environ.get("PYZM_CONFPATH", "/etc/zm")

# Well-known directories to probe for zm.conf, in order, when neither an
# explicit conf_path nor PYZM_CONFPATH is provided. /etc/zm is the default
# install location; /config is the common container layout; /etc/zoneminder
# is used by some distro packages.
_CONF_SEARCH_PATHS = ("/etc/zm", "/config", "/etc/zoneminder")


def _resolve_conf_path(conf_path: str | None = None) -> str:
    """Return the directory to read ZM config from.

    Priority: explicit ``conf_path`` argument, then the ``PYZM_CONFPATH`` env
    var, then auto-discovery across :data:`_CONF_SEARCH_PATHS` (the first that
    actually contains a ``zm.conf``). Falls back to ``/etc/zm`` when nothing is
    found, preserving prior behaviour and its warnings.
    """
    if conf_path:
        return conf_path
    env = os.environ.get("PYZM_CONFPATH")
    if env:
        return env
    for candidate in _CONF_SEARCH_PATHS:
        if os.path.exists(os.path.join(candidate, "zm.conf")):
            logger.debug("Auto-discovered ZM config at %s", candidate)
            return candidate
    return _CONF_SEARCH_PATHS[0]


def _read_zm_conf(conf_path: str = _CONF_PATH) -> dict[str, str]:
    """Parse ZM config files and return DB credentials."""
    files = sorted(glob.glob(os.path.join(conf_path, "conf.d", "*.conf")))
    files.insert(0, os.path.join(conf_path, "zm.conf"))

    parser = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=("#",)
    )
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            parser.read_string("[zm_root]\n" + fh.read())

    section = parser["zm_root"] if parser.has_section("zm_root") else {}
    return {
        "user": section.get("ZM_DB_USER", "zmuser"),
        "password": section.get("ZM_DB_PASS", "zmpass"),
        "host": section.get("ZM_DB_HOST", "localhost"),
        "database": section.get("ZM_DB_NAME", "zm"),
    }


def get_zm_db(
    db_user: str | None = None,
    db_password: str | None = None,
    db_host: str | None = None,
    db_name: str | None = None,
    conf_path: str | None = None,
):
    """Return a ``mysql.connector`` connection to the ZM database, or ``None``.

    Merge strategy: always try to read zm.conf first. If it's unreadable
    (permission denied, missing), log a warning and continue. Then overlay
    any explicit values — explicit values win over zm.conf values. Final
    fallback for host is ``"localhost"``, for db name is ``"zm"``.
    """
    try:
        import mysql.connector
    except ImportError:
        logger.warning("mysql-connector-python not installed, DB access unavailable")
        return None

    # Start with zm.conf values (best-effort)
    has_explicit = any(v is not None for v in [db_user, db_password, db_host, db_name])
    try:
        creds = _read_zm_conf(_resolve_conf_path(conf_path))
    except (PermissionError, OSError) as exc:
        if has_explicit:
            logger.debug("Could not read zm.conf: %s (using explicit credentials)", exc)
        else:
            logger.warning("Could not read zm.conf: %s", exc)
        creds = {"user": "zmuser", "password": "zmpass", "host": "localhost", "database": "zm"}

    # Overlay explicit overrides
    if db_user is not None:
        creds["user"] = db_user
    if db_password is not None:
        creds["password"] = db_password
    if db_host is not None:
        creds["host"] = db_host
    if db_name is not None:
        creds["database"] = db_name

    host = creds["host"]
    port = 3306

    # ZM_DB_HOST can be "hostname:port" or "hostname:/path/to/socket"
    if ":" in host:
        host, suffix = host.split(":", 1)
        if suffix.startswith("/"):
            # Unix socket path — pass as unix_socket
            try:
                return mysql.connector.connect(
                    user=creds["user"],
                    password=creds["password"],
                    database=creds["database"],
                    unix_socket=suffix,
                )
            except mysql.connector.Error as exc:
                logger.warning("DB connect via socket %s failed: %s", suffix, exc)
                return None
        else:
            try:
                port = int(suffix)
            except ValueError:
                logger.warning("Invalid port in db_host %r, using default 3306", creds["host"])
                port = 3306

    try:
        return mysql.connector.connect(
            host=host,
            port=port,
            user=creds["user"],
            password=creds["password"],
            database=creds["database"],
        )
    except mysql.connector.Error as exc:
        logger.warning("DB connect to %s:%s failed: %s", host, port, exc)
        return None
