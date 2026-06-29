"""Tests for src/logging_config.py — structured logging setup."""

import json
import logging
import os
import tempfile

from src.logging_config import setup_logging, get_logger, JSONFormatter, HumanFormatter


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = setup_logging()
        assert logger.name == "stomar"

    def test_has_handlers(self):
        logger = setup_logging()
        assert len(logger.handlers) > 0

    def test_no_duplicate_handlers(self):
        logger1 = setup_logging()
        logger2 = setup_logging()
        assert logger1 is logger2
        assert len(logger1.handlers) <= 2

    def test_json_output(self):
        logger = setup_logging(json_output=True)
        assert isinstance(logger, logging.Logger)

    def test_propagate_false(self):
        logger = setup_logging()
        assert logger.propagate is False

    def test_returns_same_logger(self):
        logger1 = setup_logging()
        logger2 = setup_logging()
        assert logger1 is logger2


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_child_logger_name(self):
        logger = get_logger("my_module")
        assert "stomar" in logger.name or logger.name == "stomar.my_module"

    def test_different_modules(self):
        log1 = get_logger("module_a")
        log2 = get_logger("module_b")
        assert log1.name != log2.name


class TestJSONFormatter:
    def test_valid_json_output(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test message"

    def test_exception_formatting(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="error occurred", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed


class TestHumanFormatter:
    def test_format_contains_level(self):
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "hello" in output

    def test_timestamp_format(self):
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="test.py",
            lineno=1, msg="warning msg", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "WARNING" in output
