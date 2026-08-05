"""Tests for pyrcf's logging setup.

Covers three things that silently broke logging before:
    - pyrcf must not reconfigure the root logger of the importing application;
    - throttled log calls must be throttled per call site, not globally per level;
    - a throttled record must report the caller's location, not this module's.
"""

import logging as std_logging
import time

import pytest

from pyrcf.core.logging import (
    PYRCF_LOGGER_NAME,
    ThrottledLogger,
    _ThrottledLogging,
    handler as pyrcf_handler,
    logger,
    throttled_logging,
)

# these tests deliberately inspect the throttled logger registry, since "one logger per call site"
# is the behaviour under test
# pylint: disable=protected-access


class RecordCollector(std_logging.Handler):
    """Handler that keeps every record it is given, so tests can assert on them directly
    instead of fighting pytest's stream capturing."""

    def __init__(self):
        super().__init__(level=std_logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def messages(self):
        return [record.getMessage() for record in self.records]


@pytest.fixture(autouse=True)
def clear_throttled_logger_registry():
    _ThrottledLogging._ACTIVE_THROTTLED_LOGGERS.clear()
    yield
    _ThrottledLogging._ACTIVE_THROTTLED_LOGGERS.clear()


@pytest.fixture(name="collector")
def collector_fixture(monkeypatch):
    """Collect records from every `ThrottledLogger` created during the test."""
    collector = RecordCollector()
    original_init = ThrottledLogger.__init__

    def init_with_collector(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.addHandler(collector)

    monkeypatch.setattr(ThrottledLogger, "__init__", init_with_collector)
    return collector


class TestLoggerScoping:

    def test_logger_is_named_and_not_the_root_logger(self):
        assert logger.name == PYRCF_LOGGER_NAME
        assert logger is not std_logging.getLogger()

    def test_does_not_propagate_to_root(self):
        """Otherwise applications that configure their own root handler get duplicate records."""
        assert logger.propagate is False

    def test_pyrcf_handler_is_not_attached_to_root(self):
        """pyrcf must attach its handler to its own logger and never to the root logger.

        NOTE: this checks pyrcf's own handler specifically rather than asserting that pyrcf's and
        root's handler lists are disjoint. pytest >= 9 injects its capture handlers into *every*
        logger (pyrcf's included), so a disjointness assertion would fail on pytest's handlers
        rather than on anything pyrcf did.
        """
        assert pyrcf_handler in logger.handlers
        assert pyrcf_handler not in std_logging.getLogger().handlers

    def test_logger_emits_records(self):
        collector = RecordCollector()
        logger.addHandler(collector)
        try:
            logger.info("hello from pyrcf")
        finally:
            logger.removeHandler(collector)

        assert "hello from pyrcf" in collector.messages()


class TestThrottledLogging:

    def test_distinct_call_sites_get_distinct_loggers(self):
        """Regression test: every call site used to share one logger per level, so only the
        first site to fire in each period was ever emitted."""
        throttled_logging.warning("site a", 10.0)
        throttled_logging.warning("site b", 10.0)

        assert len(_ThrottledLogging._ACTIVE_THROTTLED_LOGGERS) == 2

    def test_same_call_site_reuses_one_logger(self):
        for _ in range(5):
            throttled_logging.warning("same site", 10.0)

        assert len(_ThrottledLogging._ACTIVE_THROTTLED_LOGGERS) == 1

    def test_logger_name_identifies_the_caller(self):
        throttled_logging.info("locate me", 10.0)

        names = list(_ThrottledLogging._ACTIVE_THROTTLED_LOGGERS)
        assert len(names) == 1
        assert "test_logging.py" in names[0], "logger should be named after the calling file"
        assert "core/logging.py" not in names[0].replace("\\", "/")

    def test_both_call_sites_are_emitted(self, collector: RecordCollector):
        throttled_logging.warning("message from site a", 10.0)
        throttled_logging.warning("message from site b", 10.0)

        assert "message from site a" in collector.messages()
        assert "message from site b" in collector.messages()

    def test_record_points_at_the_real_call_site(self, collector: RecordCollector):
        throttled_logging.warning("check my origin", 10.0)

        (record,) = collector.records
        assert (
            record.filename == "test_logging.py"
        ), f"record should report the caller, but reported {record.filename}:{record.lineno}"

    def test_first_call_is_emitted_immediately(self, collector: RecordCollector):
        """A condition that occurs once must not be swallowed by the throttle window."""
        throttled_logging.warning("transient condition", 3600.0)

        assert collector.messages() == ["transient condition"]

    def test_subsequent_calls_are_throttled(self, collector: RecordCollector):
        for _ in range(20):
            throttled_logging.warning("spam", 10.0)

        assert collector.messages().count("spam") == 1

    def test_messages_resume_after_the_period_elapses(self, collector: RecordCollector):
        deadline = time.perf_counter() + 0.3
        while time.perf_counter() < deadline:
            throttled_logging.warning("periodic", 0.05)
            time.sleep(0.01)

        assert collector.messages().count("periodic") >= 3

    def test_levels_are_recorded_correctly(self, collector: RecordCollector):
        throttled_logging.debug("a debug message", 10.0)
        throttled_logging.warning("a warning message", 10.0)
        throttled_logging.error("an error message", 10.0)

        levels = {record.getMessage(): record.levelno for record in collector.records}
        assert levels["a warning message"] == std_logging.WARNING
        assert levels["an error message"] == std_logging.ERROR


class TestThrottledLoggerRate:

    def test_changed_delay_is_honoured(self, collector: RecordCollector):
        """A call site that later asks for a shorter delay should not keep the original one."""
        throttled_logging.warning("first with a long delay", 3600.0)
        assert len(collector.records) == 1

        time.sleep(0.05)
        throttled_logging.warning("now with a short delay", 0.01)

        assert len(collector.records) == 2, "shortened delay was ignored"

    def test_set_rate_is_a_noop_for_the_same_rate(self):
        throttled_logger = ThrottledLogger(name="rate_test_logger", rate=10.0)
        before = throttled_logger.next_tick if hasattr(throttled_logger, "next_tick") else None
        throttled_logger.set_rate(10.0)
        after = throttled_logger.next_tick if hasattr(throttled_logger, "next_tick") else None
        assert before == after
