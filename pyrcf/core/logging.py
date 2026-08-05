import logging
from typing import Any, Dict
from time import perf_counter
from inspect import currentframe, getframeinfo

# pylint:disable = W0221


class LoggingFormatter(logging.Formatter):
    """
    Custom logging formatter visually consistent with spdlog.
    """

    BOLD_RED: str = "\033[31;1m"
    BOLD_WHITE: str = "\033[37;1m"
    BOLD_YELLOW: str = "\033[33;1m"
    GREEN: str = "\033[32m"
    ON_RED: str = "\033[41m"
    RESET: str = "\033[0m"

    LEVEL_FORMAT: Dict[Any, str] = {
        logging.CRITICAL: f"[{ON_RED}{BOLD_WHITE}critical{RESET}]",
        logging.DEBUG: "[debug]",
        logging.ERROR: f"[{BOLD_RED}error{RESET}]",
        logging.INFO: f"[{GREEN}info{RESET}]",
        logging.WARNING: f"[{BOLD_YELLOW}warning{RESET}]",
    }

    def format(self, record):
        custom_format = (
            "[%(asctime)s] "
            + self.LEVEL_FORMAT.get(record.levelno, "[???]")
            + " %(message)s (%(filename)s:%(lineno)d)"
        )
        formatter = logging.Formatter(custom_format)
        return formatter.format(record)


PYRCF_LOGGER_NAME: str = "pyrcf"
"""Name of the logger used by every pyrcf component. Use
`logging.getLogger(PYRCF_LOGGER_NAME)` from an application to reconfigure pyrcf's logging."""

logger = logging.getLogger(PYRCF_LOGGER_NAME)
"""The logger used by all pyrcf components. Use this instead of the module-level functions of the
standard `logging` module (which would write to the root logger)."""

logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LoggingFormatter())
logger.addHandler(handler)
# NOTE: pyrcf is a library, so it must not touch the root logger or the logging configuration of
# the application importing it. Records are handled by pyrcf's own handler and are deliberately not
# propagated to the root logger (which would duplicate them in applications that configure one).
logger.propagate = False


class ThrottledLogger(logging.Logger):
    """Log data intermittently at specified frequency."""

    def __init__(self, name: str, rate: float, level: int | str = logging.INFO) -> None:
        super().__init__(name, level)
        _handler = logging.StreamHandler()
        _handler.setLevel(logging.DEBUG)
        _handler.setFormatter(LoggingFormatter())
        self.addHandler(_handler)
        self.__period = 1.0 / rate
        # NOTE: the first call is emitted immediately and throttling applies from then on.
        # Waiting out a full period first would silently swallow warnings for conditions that
        # occur only briefly -- exactly the conditions worth reporting.
        self.__next_tick = perf_counter()

    def set_rate(self, rate: float):
        """Change the frequency at which this logger emits records.

        Args:
            rate (float): The new frequency in hz.
        """
        period = 1.0 / rate
        if period == self.__period:
            return
        # re-base the next tick so that changing the rate takes effect immediately rather than
        # after the remainder of the previous (possibly much longer) period
        self.__next_tick = min(self.__next_tick, perf_counter() + period)
        self.__period = period

    def log(self, level, msg, *args, **kwargs):
        if self.__next_tick - perf_counter() <= 0.0:
            self.__next_tick = perf_counter() + self.__period
            super().log(level, msg, *args, **kwargs)


class _ThrottledLogging(logging.Logger):
    # automatically creates a logger and uses existing one for each line in a code.
    # Allows use directly as function call using
    # `throttled_logging.<logtype>("message", delay_time_in_sec)` where `throttled_logging`
    # is an alias of this class defined below.

    _ACTIVE_THROTTLED_LOGGERS = {}

    _CALLER_STACKLEVEL: int = 3
    """Number of frames between the user's call site and `logging.Logger._log`:
    user code -> `_ThrottledLogging.<level>` -> `ThrottledLogger.log`."""

    @staticmethod
    def _get_triggered_logger(suffix: str, rate: float) -> ThrottledLogger:
        # NOTE: two frames up is the caller of `throttled_logging.<level>(...)`; inspecting
        # `currentframe()` here would name this function instead, collapsing every call site in
        # the codebase into a single shared logger (so only one of them would ever be emitted).
        caller_frame = currentframe().f_back.f_back
        info = getframeinfo(caller_frame)
        logger_name = f"{info.filename}_{info.lineno}_{suffix}_logger"
        throttled_logger = _ThrottledLogging._ACTIVE_THROTTLED_LOGGERS.get(logger_name)
        if throttled_logger is None:
            throttled_logger = ThrottledLogger(name=logger_name, rate=rate)
            _ThrottledLogging._ACTIVE_THROTTLED_LOGGERS[logger_name] = throttled_logger
        else:
            # honour a changed delay for an existing call site instead of silently keeping the
            # rate that was requested the first time it was hit
            throttled_logger.set_rate(rate)
        return throttled_logger

    @staticmethod
    def info(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("info", rate=1.0 / delay_sec)
        _logger.log(
            logging.INFO, msg, *args, stacklevel=_ThrottledLogging._CALLER_STACKLEVEL, **kwargs
        )

    @staticmethod
    def debug(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("debug", rate=1.0 / delay_sec)
        _logger.log(
            logging.DEBUG, msg, *args, stacklevel=_ThrottledLogging._CALLER_STACKLEVEL, **kwargs
        )

    @staticmethod
    def warning(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("warning", rate=1.0 / delay_sec)
        _logger.log(
            logging.WARNING, msg, *args, stacklevel=_ThrottledLogging._CALLER_STACKLEVEL, **kwargs
        )

    @staticmethod
    def error(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("error", rate=1.0 / delay_sec)
        _logger.log(
            logging.ERROR, msg, *args, stacklevel=_ThrottledLogging._CALLER_STACKLEVEL, **kwargs
        )

    @staticmethod
    def critical(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("critical", rate=1.0 / delay_sec)
        _logger.log(
            logging.CRITICAL, msg, *args, stacklevel=_ThrottledLogging._CALLER_STACKLEVEL, **kwargs
        )

    @staticmethod
    def fatal(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("fatal", rate=1.0 / delay_sec)
        _logger.log(
            logging.FATAL, msg, *args, stacklevel=_ThrottledLogging._CALLER_STACKLEVEL, **kwargs
        )

    @staticmethod
    def log(level, delay_sec, msg, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("log", rate=1.0 / delay_sec)
        _logger.log(level, msg, *args, stacklevel=_ThrottledLogging._CALLER_STACKLEVEL, **kwargs)


throttled_logging = _ThrottledLogging  # pylint:disable=C0103
"""Log data intermittently once every t seconds. Allows use directly as function call using
`throttled_logging.<logtype>("message", delay_time_in_sec)`"""

__all__ = [
    "PYRCF_LOGGER_NAME",
    "ThrottledLogger",
    "logger",
    "logging",
    "throttled_logging",
]
