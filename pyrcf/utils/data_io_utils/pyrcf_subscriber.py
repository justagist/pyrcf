from abc import ABC, abstractmethod
from typing import Mapping, Any
import json
import zmq

from .pyrcf_publisher import DEFAULT_ZMQ_PUBLISH_PORT


class PyRCFSubscriberBase(ABC):
    """Base subscriber class."""

    @abstractmethod
    def read_raw(self) -> str:
        """Return raw string/bytes."""
        raise NotImplementedError("Should be implemented in child class")

    def read_json(self) -> Mapping[str, Any]:
        """Return as python dict (json)."""
        return json.loads(self.read_raw())

    def close(self):
        """Close subscriber cleanly."""
        return


class PyRCFSubscriberZMQ(PyRCFSubscriberBase):
    """A zmq subscriber to read data published using `PyRCFPublisherZMQ`."""

    def __init__(self, port: int = DEFAULT_ZMQ_PUBLISH_PORT, topic: str = ""):
        """A zmq subscriber to read data published using `PyRCFPublisherZMQ`.

        Args:
            port (int, optional): port to publish in. Defaults to 5001.
            topic (str, optional): Starting string value to match against for filtering messages.
                Defaults to "" (all topics are subscribed).
        """

        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)

        # Connects to a bound socket
        self.socket.connect(f"tcp://localhost:{port}")

        # Subscribes to specified topic (all if "")
        self.socket.subscribe(topic)

    def read_raw(self) -> str:  # NOTE: blocking
        return self.socket.recv_string()

    def close(self):
        self.socket.close()

    def __del__(self):
        self.close()
