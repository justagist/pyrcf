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


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(LoggingFormatter())
logger.addHandler(handler)
logging.basicConfig(level=logging.INFO)


class ThrottledLogger(logging.Logger):
    """Log data intermittently at specified frequency."""

    def __init__(self, name: str, rate: float, level: int | str = logging.INFO) -> None:
        super().__init__(name, level)
        _handler = logging.StreamHandler()
        _handler.setLevel(logging.DEBUG)
        _handler.setFormatter(LoggingFormatter())
        self.addHandler(_handler)
        self.__period = 1.0 / rate
        self.__next_tick = perf_counter() + self.__period

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

    @staticmethod
    def _get_triggered_logger(suffix: str, rate: float) -> ThrottledLogger:
        info = getframeinfo(currentframe())
        logger_name = f"{info.filename}_{info.lineno}_{suffix}_logger"
        if logger_name not in _ThrottledLogging._ACTIVE_THROTTLED_LOGGERS:
            _ThrottledLogging._ACTIVE_THROTTLED_LOGGERS[logger_name] = ThrottledLogger(
                name=logger_name, rate=rate
            )
        return _ThrottledLogging._ACTIVE_THROTTLED_LOGGERS[logger_name]

    @staticmethod
    def info(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("info", rate=1.0 / delay_sec)
        _logger.log(logging.INFO, msg, *args, **kwargs)

    @staticmethod
    def debug(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("debug", rate=1.0 / delay_sec)
        _logger.log(logging.DEBUG, msg, *args, **kwargs)

    @staticmethod
    def warning(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("warning", rate=1.0 / delay_sec)
        _logger.log(logging.WARNING, msg, *args, **kwargs)

    @staticmethod
    def error(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("error", rate=1.0 / delay_sec)
        _logger.log(logging.ERROR, msg, *args, **kwargs)

    @staticmethod
    def critical(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("critical", rate=1.0 / delay_sec)
        _logger.log(logging.CRITICAL, msg, *args, **kwargs)

    @staticmethod
    def fatal(msg, delay_sec, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("fatal", rate=1.0 / delay_sec)
        _logger.log(logging.FATAL, msg, *args, **kwargs)

    @staticmethod
    def log(level, delay_sec, msg, *args, **kwargs):
        _logger = _ThrottledLogging._get_triggered_logger("log", rate=1.0 / delay_sec)
        _logger.log(level, msg, *args, **kwargs)


throttled_logging = _ThrottledLogging  # pylint:disable=C0103
"""Log data intermittently once every t seconds. Allows use directly as function call using
`throttled_logging.<logtype>("message", delay_time_in_sec)`"""

__all__ = ["logging", "ThrottledLogger", "throttled_logging"]
