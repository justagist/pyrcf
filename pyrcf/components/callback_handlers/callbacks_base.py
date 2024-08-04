from abc import ABC, abstractmethod


class CustomCallback(ABC):
    """Abstract base class for defining custom callbacks to be executed in the control loop."""

    @abstractmethod
    def run_once(self) -> None:
        """To be called in loop."""
        raise NotImplementedError("Method should be implemented in child class")

    def cleanup(self) -> None:
        """Override if custom cleaning up/shutting down is required."""
        return
