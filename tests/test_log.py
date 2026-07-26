"""Tests for pyzm.log -- ZM-native logging via setup_zm_logging."""

from __future__ import annotations

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

from pyzm.log import (
    _python_to_zm,
    _read_zm_conf_full, _zm_config_to_handler_level, _ZM_OFF,
    _ZMDBHandler, _ZMFileFormatter, _ZMSyslogFormatter,
    ZMLogAdapter, setup_zm_logging,
    get_logpath, get_log_file,
)


# ===================================================================
# TestPythonToZmMapping
# ===================================================================

class TestPythonToZm:
    """Tests for _python_to_zm level mapping."""

    def test_debug_maps_to_dbg(self):
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "DBG"

    def test_info_maps_to_inf(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "INF"

    def test_warning_maps_to_war(self):
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "WAR"

    def test_error_maps_to_err(self):
        record = logging.LogRecord("test", logging.ERROR, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "ERR"

    def test_critical_maps_to_fat(self):
        record = logging.LogRecord("test", logging.CRITICAL, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "FAT"

    def test_below_debug_maps_to_dbg(self):
        record = logging.LogRecord("test", 5, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "DBG"

    def test_between_info_and_warning(self):
        """Level 25 is between INFO(20) and WARNING(30) -> WAR."""
        record = logging.LogRecord("test", 25, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "WAR"


# ===================================================================
# TestReadZmConfFull
# ===================================================================

class TestReadZmConfFull:
    """Tests for _read_zm_conf_full."""

    def test_reads_zm_conf(self, tmp_path):
        (tmp_path / "zm.conf").write_text(
            "ZM_DB_USER=testuser\n"
            "ZM_DB_PASS=testpass\n"
            "ZM_DB_HOST=dbhost\n"
            "ZM_DB_NAME=testdb\n"
            "ZM_WEB_USER=apache\n"
            "ZM_WEB_GROUP=apache\n"
            "ZM_PATH_LOGS=/tmp/zmlogs\n"
        )
        (tmp_path / "conf.d").mkdir()
        result = _read_zm_conf_full(str(tmp_path))
        assert result["dbuser"] == "testuser"
        assert result["dbpassword"] == "testpass"
        assert result["dbhost"] == "dbhost"
        assert result["dbname"] == "testdb"
        assert result["webuser"] == "apache"
        assert result["webgroup"] == "apache"
        assert result["logpath"] == "/tmp/zmlogs"

    def test_conf_d_overrides(self, tmp_path):
        (tmp_path / "zm.conf").write_text("ZM_DB_USER=original\n")
        (tmp_path / "conf.d").mkdir()
        (tmp_path / "conf.d" / "01-override.conf").write_text(
            "ZM_DB_USER=overridden\n"
        )
        result = _read_zm_conf_full(str(tmp_path))
        assert result["dbuser"] == "overridden"

    def test_defaults_when_missing(self, tmp_path):
        (tmp_path / "zm.conf").write_text("")
        (tmp_path / "conf.d").mkdir()
        result = _read_zm_conf_full(str(tmp_path))
        assert result["dbuser"] == "zmuser"
        assert result["dbpassword"] == "zmpass"
        assert result["dbhost"] == "localhost"
        assert result["dbname"] == "zm"
        assert result["webuser"] == "www-data"
        assert result["logpath"] == "/var/log/zm"

    def test_missing_conf_path(self, tmp_path):
        """Non-existent path returns defaults."""
        result = _read_zm_conf_full(str(tmp_path / "nonexistent"))
        assert result["dbuser"] == "zmuser"


# ===================================================================
# TestZmConfigToHandlerLevel
# ===================================================================

class TestZmConfigToHandlerLevel:
    """Tests for _zm_config_to_handler_level."""

    def test_debug(self):
        assert _zm_config_to_handler_level(1) == logging.DEBUG
        assert _zm_config_to_handler_level(5) == logging.DEBUG

    def test_info(self):
        assert _zm_config_to_handler_level(0) == logging.INFO

    def test_warning(self):
        assert _zm_config_to_handler_level(-1) == logging.WARNING

    def test_error(self):
        assert _zm_config_to_handler_level(-2) == logging.ERROR

    def test_critical(self):
        assert _zm_config_to_handler_level(-3) == logging.CRITICAL
        assert _zm_config_to_handler_level(-4) == logging.CRITICAL


# ===================================================================
# TestZMDBHandler
# ===================================================================

class TestZMDBHandler:
    """Tests for _ZMDBHandler."""

    @patch("mysql.connector.connect")
    def test_emit_inserts_row(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 42, "hello", (), None,
        )
        handler.emit(record)

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO Logs" in sql
        vals = mock_cursor.execute.call_args[0][1]
        assert vals[1] == "test"  # Component
        assert vals[4] == 0       # Level (INF=0)
        assert vals[5] == "INF"   # Code
        assert vals[6] == "hello" # Message
        mock_conn.commit.assert_called_once()
        handler.close()

    @patch("mysql.connector.connect")
    def test_emit_debug_code_matches_perl(self, mock_connect):
        """Perl uses DB1..DB9 codes and stores actual sub-level in Level."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.DEBUG, "test.py", 1, "dbg", (), None,
        )
        record.zm_debug_level = 3  # type: ignore[attr-defined]
        handler.emit(record)

        vals = mock_cursor.execute.call_args[0][1]
        assert vals[4] == 3      # Level = actual debug sub-level
        assert vals[5] == "DB3"  # Code = DB prefix (not DBG)
        handler.close()

    @patch("mysql.connector.connect", side_effect=Exception("no db"))
    def test_connect_failure_skips_emit(self, mock_connect):
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        assert handler._conn is None
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        # Should not raise
        handler.emit(record)
        handler.close()

    @patch("mysql.connector.connect")
    def test_reconnect_on_write_failure(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("write failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        # Connection should be cleared for retry on next emit
        assert handler._conn is None
        handler.close()

    @patch("mysql.connector.connect")
    def test_connect_kwargs_host_port(self, mock_connect):
        mock_connect.return_value = MagicMock()
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="dbhost:3307", user="u", password="p", database="zm",
        )
        kw = handler._connect_kwargs()
        assert kw["host"] == "dbhost"
        assert kw["port"] == 3307
        handler.close()

    @patch("mysql.connector.connect")
    def test_connect_kwargs_unix_socket(self, mock_connect):
        mock_connect.return_value = MagicMock()
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost:/var/run/mysqld.sock",
            user="u", password="p", database="zm",
        )
        kw = handler._connect_kwargs()
        assert kw["unix_socket"] == "/var/run/mysqld.sock"
        assert "host" not in kw
        handler.close()

    @patch("mysql.connector.connect")
    def test_close_closes_connection(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        handler.close()
        mock_conn.close.assert_called_once()
        assert handler._conn is None


# ===================================================================
# TestZMFileFormatter
# ===================================================================

class TestZMFileFormatter:
    """Tests for _ZMFileFormatter matching Perl's Logger.pm format."""

    def test_format_info(self):
        fmt = _ZMFileFormatter("zmesdetect_m1")
        record = logging.LogRecord(
            "test", logging.INFO, "/path/to/test.py", 42,
            "hello world", (), None,
        )
        result = fmt.format(record)
        # Perl format: timestamp.usec id[pid].CODE [caller:line] [msg]
        assert "zmesdetect_m1[" in result
        assert "].INF" in result      # dot before code
        assert "[test.py:42]" in result  # brackets around caller
        assert "[hello world]" in result

    def test_format_includes_microseconds(self):
        fmt = _ZMFileFormatter("test")
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        result = fmt.format(record)
        # Should contain .NNNNNN after the time
        import re
        assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{6}", result)

    def test_format_debug_uses_db_prefix(self):
        """Perl uses DB3, not DBG3."""
        fmt = _ZMFileFormatter("test")
        record = logging.LogRecord(
            "test", logging.DEBUG, "test.py", 1, "dbg", (), None,
        )
        record.zm_debug_level = 3  # type: ignore[attr-defined]
        result = fmt.format(record)
        assert "].DB3" in result
        assert "DBG" not in result


# ===================================================================
# TestZMSyslogFormatter
# ===================================================================

class TestZMSyslogFormatter:
    """Tests for _ZMSyslogFormatter matching Perl's Logger.pm format."""

    def test_format_info(self):
        """Perl syslog: CODE [message] -- ident/pid added by syslog."""
        fmt = _ZMSyslogFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "syslog msg", (), None,
        )
        result = fmt.format(record)
        assert result == "INF [syslog msg]"

    def test_format_debug_uses_db_prefix(self):
        """Perl uses DB2, not DBG2."""
        fmt = _ZMSyslogFormatter()
        record = logging.LogRecord(
            "test", logging.DEBUG, "test.py", 1, "dbg", (), None,
        )
        record.zm_debug_level = 2  # type: ignore[attr-defined]
        result = fmt.format(record)
        assert result == "DB2 [dbg]"


# ===================================================================
# TestZMLogAdapter
# ===================================================================

class TestZMLogAdapter:
    """Tests for ZMLogAdapter."""

    def _make_adapter(self, **config_overrides):
        logger = logging.getLogger("zm.test_adapter")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        handler = logging.handlers.MemoryHandler(capacity=100)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        config = {
            "log_debug": 1,
            "log_level_debug": 5,
            "log_debug_target": "",
            "dump_console": False,
        }
        config.update(config_overrides)
        return ZMLogAdapter(logger, config, "test_process"), handler

    def test_debug_emits_when_enabled(self):
        adapter, handler = self._make_adapter()
        adapter.Debug(1, "test debug")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].getMessage() == "test debug"
        adapter._logger.handlers.clear()

    def test_debug_suppressed_when_disabled(self):
        adapter, handler = self._make_adapter(log_debug=0)
        adapter.Debug(1, "should not appear")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_suppressed_when_level_too_high(self):
        adapter, handler = self._make_adapter(log_level_debug=2)
        adapter.Debug(3, "level 3 debug")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_target_filtering(self):
        adapter, handler = self._make_adapter(log_debug_target="zmc_m1|zmc_m2")
        # process_name is "test_process", doesn't match targets
        adapter.Debug(1, "filtered out")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_target_allows_exact_match(self):
        adapter, handler = self._make_adapter(log_debug_target="test_process")
        adapter.Debug(1, "should pass")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_matches_id_root(self):
        """Perl matches idRoot (part before first _)."""
        adapter, handler = self._make_adapter(log_debug_target="test")
        # process_name is "test_process", idRoot is "test"
        adapter.Debug(1, "root match")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_rejects_partial_prefix(self):
        """Perl does exact match, not startswith: 'tes' should NOT match 'test_process'."""
        adapter, handler = self._make_adapter(log_debug_target="tes")
        adapter.Debug(1, "should not match")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_target_matches_underscore_prefix(self):
        """Perl also matches _id and _idRoot forms."""
        adapter, handler = self._make_adapter(log_debug_target="_test_process")
        adapter.Debug(1, "underscore match")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_empty_matches_all(self):
        """Empty target in Perl means match all processes."""
        adapter, handler = self._make_adapter(log_debug_target="")
        # Empty target → target check is skipped → matches all
        adapter.Debug(1, "should pass")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_bypass_with_dump_console(self):
        adapter, handler = self._make_adapter(
            log_debug_target="other", dump_console=True,
        )
        adapter.Debug(1, "console bypass")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_info(self):
        adapter, handler = self._make_adapter()
        adapter.Info("info msg")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].levelno == logging.INFO
        adapter._logger.handlers.clear()

    def test_warning(self):
        adapter, handler = self._make_adapter()
        adapter.Warning("warn msg")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].levelno == logging.WARNING
        adapter._logger.handlers.clear()

    def test_error(self):
        adapter, handler = self._make_adapter()
        adapter.Error("err msg")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].levelno == logging.ERROR
        adapter._logger.handlers.clear()

    def test_fatal_exits(self):
        adapter, handler = self._make_adapter()
        with pytest.raises(SystemExit):
            adapter.Fatal("fatal msg")

    def test_get_config(self):
        adapter, _ = self._make_adapter()
        cfg = adapter.get_config()
        assert cfg["log_debug"] == 1
        adapter._logger.handlers.clear()

    def test_close_removes_handlers(self):
        adapter, handler = self._make_adapter()
        assert len(adapter._logger.handlers) > 0
        adapter.close()
        assert len(adapter._logger.handlers) == 0


# ===================================================================
# TestSetupZmLogging
# ===================================================================

class TestSetupZmLogging:
    """Tests for setup_zm_logging."""

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_returns_adapter(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_app", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert isinstance(adapter, ZMLogAdapter)
        assert adapter._process_name == "test_app"
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_dump_console_adds_stream_handler(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_console", override={
            "dump_console": True,
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        stream_handlers = [
            h for h in adapter._logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, (logging.handlers.SysLogHandler,))
        ]
        assert len(stream_handlers) >= 1
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_overrides_take_priority(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_override", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "log_debug": True,
            "log_level_debug": 9,
        })
        assert adapter._config["log_debug"] == True
        assert adapter._config["log_level_debug"] == 9
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={
        "ZM_LOG_DEBUG": "1",
        "ZM_LOG_DEBUG_LEVEL": "4",
    })
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_db_config_applied(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_db_cfg", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert adapter._config["log_debug"] == 1
        assert adapter._config["log_level_debug"] == 4
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": None, "dbpassword": None, "dbhost": None,
        "dbname": None, "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_works_without_db(self, mock_sig, mock_conf, mock_db):
        """setup_zm_logging works even when no DB creds are available."""
        adapter = setup_zm_logging(name="test_nodb", override={
            "log_level_file": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert isinstance(adapter, ZMLogAdapter)
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_file_handler_created(self, mock_sig, mock_conf, mock_db, tmp_path):
        adapter = setup_zm_logging(name="test_file", override={
            "log_level_file": 0,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "logpath": str(tmp_path),
        })
        file_handlers = [
            h for h in adapter._logger.handlers
            if isinstance(h, logging.handlers.WatchedFileHandler)
        ]
        assert len(file_handlers) == 1
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_syslog_uses_local1_facility(self, mock_sig, mock_conf, mock_db):
        """Perl uses facility=local1 for syslog."""
        with patch("pyzm.log.logging.handlers.SysLogHandler") as mock_sh_cls:
            mock_sh = MagicMock()
            mock_sh_cls.return_value = mock_sh
            mock_sh_cls.LOG_LOCAL1 = logging.handlers.SysLogHandler.LOG_LOCAL1

            adapter = setup_zm_logging(name="test_syslog", override={
                "log_level_file": _ZM_OFF,
                "log_level_db": _ZM_OFF,
                "log_level_syslog": 0,
            })
            mock_sh_cls.assert_called_once_with(
                address="/dev/log",
                facility=logging.handlers.SysLogHandler.LOG_LOCAL1,
            )
            # Verify ident is set for syslog
            assert "test_syslog[" in mock_sh.ident
            adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_debug_file_overrides_log_path(self, mock_sig, mock_conf, mock_db, tmp_path):
        """ZM_LOG_DEBUG_FILE overrides log file path and raises file level."""
        debug_file = str(tmp_path / "debug_override.log")
        adapter = setup_zm_logging(name="test_dbgfile", override={
            "log_level_file": 0,  # INFO level
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "log_debug": 1,
            "log_level_debug": 5,
            "log_debug_file": debug_file,
            "logpath": str(tmp_path),
        })
        file_handlers = [
            h for h in adapter._logger.handlers
            if isinstance(h, logging.handlers.WatchedFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == debug_file
        # File level should be raised to debug level
        assert file_handlers[0].level == logging.DEBUG
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={
        "ZM_LOG_LEVEL_SYSLOG": "-1",
        "ZM_LOG_LEVEL_FILE": "1",
        "ZM_LOG_LEVEL_DATABASE": "-2",
        "ZM_LOG_DEBUG": "1",
        "ZM_LOG_DEBUG_LEVEL": "7",
        "ZM_LOG_DEBUG_TARGET": "zmc_m1",
    })
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_round2_override_beats_db(self, mock_sig, mock_conf, mock_db, tmp_path):
        """Override dict (round 2) takes priority over DB config values."""
        adapter = setup_zm_logging(name="test_r2", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "log_debug": True,
            "log_level_debug": 3,  # override DB's 7
        })
        assert adapter._config["log_level_debug"] == 3  # override wins
        assert adapter._config["log_debug_target"] == "zmc_m1"  # DB value kept (no override)
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "zmuser", "webgroup": "zmgroup",
        "logpath": "/var/log/zoneminder",
    })
    @patch("pyzm.log._signal.signal")
    def test_conf_logpath_honored(self, mock_sig, mock_conf, mock_db, monkeypatch):
        """ZM_PATH_LOGS from zm.conf must flow through when no env/override is set.

        Regression test for #45 -- the first defaults loop used to populate
        config['logpath'] with '/var/log/zm' before the conf was read, causing
        the conf value to be silently dropped.
        """
        monkeypatch.delenv("PYZM_LOGPATH", raising=False)
        monkeypatch.delenv("PYZM_WEBUSER", raising=False)
        monkeypatch.delenv("PYZM_WEBGROUP", raising=False)
        adapter = setup_zm_logging(name="test_conf_logpath", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert adapter._config["logpath"] == "/var/log/zoneminder"
        assert adapter._config["webuser"] == "zmuser"
        assert adapter._config["webgroup"] == "zmgroup"
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "zmuser", "webgroup": "zmgroup",
        "logpath": "/var/log/zoneminder",
    })
    @patch("pyzm.log._signal.signal")
    def test_env_overrides_conf_logpath(self, mock_sig, mock_conf, mock_db, monkeypatch):
        """PYZM_LOGPATH env var still wins over the conf-file value."""
        monkeypatch.setenv("PYZM_LOGPATH", "/env/logs")
        adapter = setup_zm_logging(name="test_env_logpath", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert adapter._config["logpath"] == "/env/logs"
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": None, "webgroup": None,
        "logpath": None,
    })
    @patch("pyzm.log._signal.signal")
    def test_final_defaults_when_no_source(self, mock_sig, mock_conf, mock_db, monkeypatch):
        """Hard-coded defaults apply only when env, conf, and override are all empty."""
        monkeypatch.delenv("PYZM_LOGPATH", raising=False)
        monkeypatch.delenv("PYZM_WEBUSER", raising=False)
        monkeypatch.delenv("PYZM_WEBGROUP", raising=False)
        adapter = setup_zm_logging(name="test_default_logpath", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert adapter._config["logpath"] == "/var/log/zm"
        assert adapter._config["webuser"] == "www-data"
        assert adapter._config["webgroup"] == "www-data"
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_signals_registered(self, mock_sig, mock_conf, mock_db):
        """SIGHUP, SIGUSR1, SIGUSR2 handlers are registered."""
        import signal as real_signal
        adapter = setup_zm_logging(name="test_signals", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        sig_calls = {c[0][0] for c in mock_sig.call_args_list}
        assert real_signal.SIGHUP in sig_calls
        assert real_signal.SIGUSR1 in sig_calls
        assert real_signal.SIGUSR2 in sig_calls
        adapter.close()


# ===================================================================
# TestZMDBHandlerPing
# ===================================================================

class TestZMDBHandlerPing:
    """Tests for _ZMDBHandler ping and reconnect behavior."""

    @patch("mysql.connector.connect")
    def test_ping_called_before_write(self, mock_connect):
        """Perl calls $dbh->ping() before every write."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        mock_conn.ping.assert_called_once_with(
            reconnect=False, attempts=1, delay=0,
        )
        handler.close()

    @patch("mysql.connector.connect")
    def test_ping_failure_triggers_reconnect(self, mock_connect):
        """If ping fails, handler reconnects before writing."""
        mock_conn_dead = MagicMock()
        mock_conn_dead.ping.side_effect = Exception("gone")
        mock_conn_new = MagicMock()
        mock_cursor = MagicMock()
        mock_conn_new.cursor.return_value = mock_cursor
        # First connect returns dead conn, second returns fresh conn
        mock_connect.side_effect = [mock_conn_dead, mock_conn_new]

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        # Should have reconnected and written via new connection
        assert mock_connect.call_count == 2
        mock_cursor.execute.assert_called_once()
        handler.close()

    @patch("mysql.connector.connect")
    def test_recursive_logging_guard(self, mock_connect):
        """Emit returns early when _reconnecting=True (prevents infinite loops)."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        handler._reconnecting = True
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        # Should have returned immediately without writing
        mock_conn.cursor.assert_not_called()
        handler.close()


# ===================================================================
# TestZMDBHandlerLevelColumn
# ===================================================================

class TestZMDBHandlerLevelColumn:
    """Verify DB Level column matches Perl for all severities."""

    @patch("mysql.connector.connect")
    def _emit_and_get_level(self, py_level, mock_connect, **extra):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        handler = _ZMDBHandler(
            component="t", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord("t", py_level, "t.py", 1, "m", (), None)
        for k, v in extra.items():
            setattr(record, k, v)
        handler.emit(record)
        vals = mock_cursor.execute.call_args[0][1]
        handler.close()
        return vals[4], vals[5]  # Level, Code

    def test_info_level_0(self):
        level, code = self._emit_and_get_level(logging.INFO)
        assert level == 0
        assert code == "INF"

    def test_warning_level_neg1(self):
        level, code = self._emit_and_get_level(logging.WARNING)
        assert level == -1
        assert code == "WAR"

    def test_error_level_neg2(self):
        level, code = self._emit_and_get_level(logging.ERROR)
        assert level == -2
        assert code == "ERR"

    def test_fatal_level_neg3(self):
        level, code = self._emit_and_get_level(logging.CRITICAL)
        assert level == -3
        assert code == "FAT"

    def test_debug_level_stores_sublevel(self):
        """Debug sub-level 5 -> Level=5, Code=DB5."""
        level, code = self._emit_and_get_level(
            logging.DEBUG, zm_debug_level=5,
        )
        assert level == 5
        assert code == "DB5"

    def test_debug_default_sublevel_1(self):
        """Default debug sub-level is 1 -> Level=1, Code=DB1."""
        level, code = self._emit_and_get_level(logging.DEBUG)
        assert level == 1
        assert code == "DB1"


# ===================================================================
# TestSignalHandlers
# ===================================================================

class TestSignalHandlers:
    """Tests for SIGHUP/SIGUSR1/SIGUSR2 behavior."""

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    def test_sigusr1_increases_verbosity(self, mock_conf, mock_db, tmp_path):
        """SIGUSR1 decreases handler level (more verbose)."""
        import signal
        adapter = setup_zm_logging(name="test_usr1", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "dump_console": True,
        })
        handler = adapter._logger.handlers[0]
        # Start at WARNING so there's room to go down
        handler.setLevel(logging.WARNING)
        # Simulate SIGUSR1
        signal.raise_signal(signal.SIGUSR1)
        assert handler.level == logging.WARNING - 10  # INFO
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    def test_sigusr2_decreases_verbosity(self, mock_conf, mock_db, tmp_path):
        """SIGUSR2 increases handler level (less verbose)."""
        import signal
        adapter = setup_zm_logging(name="test_usr2", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "dump_console": True,
        })
        handler = adapter._logger.handlers[0]
        original_level = handler.level
        signal.raise_signal(signal.SIGUSR2)
        assert handler.level == original_level + 10
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    def test_sighup_reopens_file_handler(self, mock_conf, mock_db, tmp_path):
        """SIGHUP closes and reopens the file handler for log rotation."""
        import signal
        adapter = setup_zm_logging(name="test_hup", override={
            "log_level_file": 0,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "logpath": str(tmp_path),
        })
        old_handlers = adapter._logger.handlers[:]
        old_fh = [h for h in old_handlers
                  if isinstance(h, logging.handlers.WatchedFileHandler)]
        assert len(old_fh) == 1

        signal.raise_signal(signal.SIGHUP)

        new_fh = [h for h in adapter._logger.handlers
                  if isinstance(h, logging.handlers.WatchedFileHandler)]
        assert len(new_fh) == 1
        # Should be a NEW handler instance (old one was closed)
        assert new_fh[0] is not old_fh[0]
        adapter.close()


# ===================================================================
# TestFileFormatAllLevels
# ===================================================================

class TestFileFormatAllLevels:
    """Verify _ZMFileFormatter produces correct codes for all levels."""

    def _format(self, level, **extra):
        fmt = _ZMFileFormatter("test_proc")
        record = logging.LogRecord("t", level, "mod.py", 10, "msg", (), None)
        for k, v in extra.items():
            setattr(record, k, v)
        return fmt.format(record)

    def test_info_code(self):
        assert "].INF [" in self._format(logging.INFO)

    def test_warning_code(self):
        assert "].WAR [" in self._format(logging.WARNING)

    def test_error_code(self):
        assert "].ERR [" in self._format(logging.ERROR)

    def test_fatal_code(self):
        assert "].FAT [" in self._format(logging.CRITICAL)

    def test_debug_levels_1_through_9(self):
        for lvl in range(1, 10):
            result = self._format(logging.DEBUG, zm_debug_level=lvl)
            assert f"].DB{lvl} [" in result


# ===================================================================
# TestGetLogpath
# ===================================================================

class TestGetLogpath:
    """Tests for the public get_logpath() helper (issue #46)."""

    def test_env_var_wins(self, tmp_path, monkeypatch):
        """PYZM_LOGPATH beats conf-file value."""
        (tmp_path / "zm.conf").write_text("ZM_PATH_LOGS=/conf/path\n")
        monkeypatch.setenv("PYZM_LOGPATH", "/env/path")
        assert get_logpath(str(tmp_path)) == "/env/path"

    def test_conf_file_wins_over_default(self, tmp_path, monkeypatch):
        """ZM_PATH_LOGS in zm.conf is used when env is unset."""
        (tmp_path / "zm.conf").write_text("ZM_PATH_LOGS=/conf/path\n")
        monkeypatch.delenv("PYZM_LOGPATH", raising=False)
        assert get_logpath(str(tmp_path)) == "/conf/path"

    def test_default_when_nothing_set(self, tmp_path, monkeypatch):
        """Falls back to /var/log/zm when env and conf are silent."""
        (tmp_path / "zm.conf").write_text("")
        monkeypatch.delenv("PYZM_LOGPATH", raising=False)
        assert get_logpath(str(tmp_path)) == "/var/log/zm"

    def test_does_not_require_setup(self, tmp_path, monkeypatch):
        """get_logpath() works standalone -- no setup_zm_logging needed."""
        (tmp_path / "zm.conf").write_text("ZM_PATH_LOGS=/standalone/path\n")
        monkeypatch.delenv("PYZM_LOGPATH", raising=False)
        # Wipe any pyzm logger handlers to prove we don't peek at them
        logging.getLogger("pyzm").handlers.clear()
        assert get_logpath(str(tmp_path)) == "/standalone/path"


# ===================================================================
# TestGetLogFile
# ===================================================================

class TestGetLogFile:
    """Tests for the public get_log_file() helper (issue #46)."""

    def setup_method(self):
        # Each test starts from a clean pyzm logger
        logging.getLogger("pyzm").handlers.clear()

    def teardown_method(self):
        logging.getLogger("pyzm").handlers.clear()

    def test_returns_none_when_no_handler(self):
        """Before setup_zm_logging -- no file handler attached -- returns None."""
        assert get_log_file() is None

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": None,  # forces final-default fallback
    })
    @patch("pyzm.log._signal.signal")
    def test_returns_path_after_setup(self, mock_sig, mock_conf, mock_db, tmp_path):
        """Returns the WatchedFileHandler's open file path after setup."""
        adapter = setup_zm_logging(name="zmesdetect_m1", override={
            "logpath": str(tmp_path),
            "log_level_file": 0,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        try:
            assert get_log_file() == str(tmp_path / "zmesdetect_m1.log")
        finally:
            adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_returns_none_when_file_logging_disabled(
        self, mock_sig, mock_conf, mock_db,
    ):
        """File handler isn't attached when log_level_file == _ZM_OFF."""
        adapter = setup_zm_logging(name="zm_nofile", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        try:
            assert get_log_file() is None
        finally:
            adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/should/be/ignored",
    })
    @patch("pyzm.log._signal.signal")
    def test_reflects_debug_file_override(
        self, mock_sig, mock_conf, mock_db, tmp_path,
    ):
        """ZM_LOG_DEBUG_FILE override is visible in get_log_file()."""
        debug_file = tmp_path / "custom-debug.log"
        adapter = setup_zm_logging(name="zm_dbg", override={
            "log_debug": 1,
            "log_level_debug": 5,
            "log_debug_file": str(debug_file),
            "log_level_file": 0,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        try:
            assert get_log_file() == str(debug_file)
        finally:
            adapter.close()


# ===================================================================
# TestExceptionText
# ===================================================================

class TestExceptionText:
    """logger.exception() must reach every ZM sink with its traceback.

    The sinks build their output from record.getMessage(), which drops
    exc_info, so an unguarded implementation logs 'Error loading model X'
    with no cause at all.
    """

    @staticmethod
    def _record_with_exc():
        try:
            raise ValueError("plate key missing")
        except ValueError:
            import sys
            return logging.LogRecord(
                "test", logging.ERROR, "/path/to/pipeline.py", 153,
                "Error loading model %s", ("Platerecognizer cloud",),
                sys.exc_info(),
            )

    def test_file_formatter_includes_traceback(self):
        result = _ZMFileFormatter("zmesdetect_m1").format(self._record_with_exc())
        assert "Error loading model Platerecognizer cloud" in result
        assert "ValueError: plate key missing" in result
        assert "Traceback (most recent call last)" in result

    def test_syslog_formatter_includes_traceback(self):
        result = _ZMSyslogFormatter().format(self._record_with_exc())
        assert "ValueError: plate key missing" in result

    @patch("mysql.connector.connect")
    def test_db_handler_includes_traceback(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        handler.emit(self._record_with_exc())

        message = mock_cursor.execute.call_args[0][1][6]
        assert message.startswith("Error loading model Platerecognizer cloud")
        assert "ValueError: plate key missing" in message
        handler.close()

    def test_plain_record_is_unchanged(self):
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "hello %s", ("world",), None,
        )
        assert _ZMSyslogFormatter().format(record) == "INF [hello world]"
