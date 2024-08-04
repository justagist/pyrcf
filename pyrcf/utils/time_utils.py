"""Utilities for timer and rate-related functionalities."""

from time import perf_counter, sleep, time
from abc import ABC, abstractmethod
from ..core.logging import logging


class ClockBase(ABC):
    """Base class for any Clock implementation for pyrcf control loop and components."""

    @abstractmethod
    def get_time(self) -> float:
        """Get the latest time from this clock.

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.

        Returns:
            float: the current time in seconds.
        """
        raise NotImplementedError("This method has to be implemented in the child class")


class PythonEpochClock(ClockBase):
    """ClockBase implementation to query system time using `time.time()`."""

    def get_time(self) -> float:
        return time()


class PythonPerfClock(ClockBase):
    """ClockBase implementation to query counter using `time.perf_counter()`. Only
    meaningful when comparing values from other `time.perf_counter()` calls."""

    def get_time(self) -> float:
        return perf_counter()


class RateLimiter:
    """
    Modified from https://github.com/upkie/loop-rate-limiters/blob/main/loop_rate_limiters/rate_limiter.py.

    Original License terms below:

    Regulate the frequency between calls to the same instruction.

    This rate limniter is meant to be used in e.g. a loop or callback function.
    It is, in essence, the same as rospy.Rate_. It assumes Python's performance
    counter never jumps backward nor forward, so that it does not handle such
    cases contrary to rospy.Rate_.

    .. _rospy.Rate:
        https://github.com/ros/ros_comm/blob/noetic-devel/clients/rospy/src/rospy/timer.py

    Attributes:
        name: Human-readable name used for logging.
        warn: If set (default), warn when the time between two calls
            exceeded the rate clock.

    #
    # Copyright 2022 Stéphane Caron
    # Copyright 2023 Inria
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """

    __period: float
    __slack: float
    __next_tick: float
    __freq: float
    name: str
    warn: bool

    def __init__(
        self,
        frequency: float,
        name: str = "rate limiter",
        warn: bool = False,
        clock: ClockBase = PythonPerfClock(),
    ):
        """Initialize rate limiter.

        Args:
            frequency: Desired frequency in hertz.
            name: Human-readable name used for logging.
            warn: If set (default), warn when the time between two calls
                exceeded the rate clock.
        """
        self.set_frequency(frequency=frequency)
        self.change_clock(clock=clock)
        self.name = name
        self.warn = warn

    def set_frequency(self, frequency: float):
        self.__period = 1.0 / frequency
        self.__freq = frequency

    def change_clock(self, clock: ClockBase):
        self.__clock = clock
        self.__next_tick = self.__clock.get_time() + self.__period
        self.__slack = 0.0

    @property
    def clock(self) -> ClockBase:
        """The clock used by this class.

        Use `get_time()` on this attribute to get current time in seconds.

        Returns:
            ClockBase: The clock object used by this class.
        """
        return self.__clock

    @property
    def dt(self) -> float:
        """Desired period between two calls to :func:`sleep`, in seconds."""
        return self.__period

    @property
    def next_tick(self) -> float:
        """Time of next clock tick."""
        return self.__next_tick

    @property
    def period(self) -> float:
        """Desired period between two calls to :func:`sleep`, in seconds."""
        return self.__period

    @property
    def frequency(self) -> float:
        """Desired frequency between two calls to :func:`sleep`, in seconds."""
        return self.__freq

    @property
    def slack(self) -> float:
        """Slack duration computed at the last call to :func:`sleep`.

        This duration is in seconds.
        """
        return self.__slack

    def remaining(self) -> float:
        """Get the time remaining until the next expected clock tick.

        Returns:
            Time remaining, in seconds, until the next expected clock tick.
        """
        return self.__next_tick - self.__clock.get_time()

    def sleep(self):
        """Sleep for the duration required to regulate inter-call frequency."""
        self.__slack = self.__next_tick - self.__clock.get_time()
        if self.__slack > 0.0:
            sleep(self.__slack)
        elif self.warn and self.__slack < -0.1 * self.period:
            logging.warning(
                "%s is late by %f [ms]",
                self.name,
                round(1e3 * self.__slack, 1),
            )
        self.__next_tick = self.__clock.get_time() + self.__period


class RateTrigger:
    """
    Can be used to execute something at a specific rate. Checking
    the `.triggered()` method of this class will return True or
    False depending on whether the specified rate has reached.

    NOTE: Warning: When using this with RateLimiter in the same
    loop, try to keep the rate of this class a factor of the rate
    of the RateLimiter object. Otherwise, the RateTrigger will
    not be able to keep the required rate (because RateLimiter
    enforces a sleep). Make sure to use same clock as well.
    """

    def __init__(self, rate: float, clock: ClockBase = PythonPerfClock()):
        """RateTrigger Constructor.

        Can be used to execute something at a specific rate. Checking
        the `.triggered()` method of this class will return True or
        False depending on whether the specified rate has reached.

        NOTE: Warning: When using this with RateLimiter in the same
        loop, try to keep the rate of this class a factor of the rate
        of the RateLimiter object. Otherwise, the RateTrigger will
        not be able to keep the required rate (because RateLimiter
        enforces a sleep).

        Args:
            rate (float): The desired frequency at which this should
                trigger True.
        """
        self.__period = 1.0 / rate
        self.__next_tick = None
        self.change_clock(clock=clock)

    def change_clock(self, clock: ClockBase):
        self.__clock = clock
        self.__next_tick = self.__clock.get_time() + self.__period

    @property
    def clock(self) -> ClockBase:
        """The clock used by this class.

        Use `get_time()` on this attribute to get current time in seconds.

        Returns:
            ClockBase: The clock object used by this class.
        """
        return self.__clock

    @property
    def dt(self) -> float:
        """Desired period between two calls to :func:`sleep`, in seconds."""
        return self.__period

    @property
    def next_tick(self) -> float:
        """Time of next clock tick."""
        return self.__next_tick

    @property
    def period(self) -> float:
        """Desired period between two calls to :func:`sleep`, in seconds."""
        return self.__period

    def triggered(self) -> bool:
        """Check if the specified rate has been triggered."""
        if self.__next_tick - self.__clock.get_time() <= 0.0:
            self.__next_tick = self.__clock.get_time() + self.__period
            return True

        return False

    @property
    def has_triggered(self) -> bool:
        """Attribute to check if the specified rate has been triggered."""
        return self.triggered()

    def __call__(self) -> bool:
        """Calling this class object directly will check if the specified rate has been
        triggered."""
        return self.triggered()
