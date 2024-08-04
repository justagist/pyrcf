from abc import ABC, abstractmethod
from typing import List

from ...utils.time_utils import RateTrigger, PythonPerfClock, ClockBase


class CustomCallbackBase(ABC):
    """Abstract base class for defining custom callbacks to be executed in the control loop."""

    @abstractmethod
    def run_once(self) -> None:
        """To be called in loop."""
        raise NotImplementedError("Method should be implemented in child class")

    def cleanup(self) -> None:
        """Override if custom cleaning up/shutting down is required."""
        return


class RateTriggeredMultiCallbacks(CustomCallbackBase):
    """Execute multiple `CustomCallbackBase` callbacks at the same specified (max) frequency."""

    def __init__(
        self,
        gui_callbacks: List[CustomCallbackBase],
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
    ):
        """Execute multiple `CustomCallbackBase` callbacks at the same specified (max) frequency.

        Args:
            gui_callbacks (List[CustomCallbackBase]): The CustomCallbackBase objects with their corresponding
                callbacks defined.
            rate (float, optional): The rate at which the callbacks for all the SliderCallback
                objects should be called. Defaults to None (call every time `run_once` is
                called).
            clock (ClockBase, optional): The ClockBase object to use for getting time. Defaults to
                PythonPerfClock().
        """
        self._callbacks: List[CustomCallbackBase] = []
        for sc in gui_callbacks:
            assert not isinstance(
                sc, RateTriggeredMultiCallbacks
            ), "Value cannot be of type `RateTriggeredMultiCallbacks`."
            self.add_callback(sc)
        if rate is not None:
            self._trigger = RateTrigger(rate=rate, clock=clock)
        else:
            self._trigger = lambda: True

    def add_callback(self, gui_callback: CustomCallbackBase):
        """Add an instance of PbGUISliderCallback or PbMultiGUISliderSingleCallback to the same rate
        trigger.

        Args:
            gui_callback (CustomCallbackBase): The GUI callback instance
                to add.
        """
        assert isinstance(
            gui_callback, CustomCallbackBase
        ), f"{gui_callback} is not of type `CustomCallbackBase."
        self._callbacks.append(gui_callback)

    def run_once(self):
        """This method has to be called for the slider callbacks to be executed. Will only
        execute if the trigger rate is met."""
        if self._trigger():
            for sc in self._callbacks:
                sc.run_once()

    def cleanup(self):
        for sc in self._callbacks:
            sc.cleanup()
        self._callbacks.clear()
