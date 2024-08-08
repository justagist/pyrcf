from typing import Any, Callable, List
from ...utils.time_utils import ClockBase, PythonPerfClock
from .ctrl_loop_data_publisher_base import CtrlLoopDataPublisherBase
from ...utils.data_io_utils.pyrcf_publisher import (
    DEFAULT_ZMQ_PUBLISH_PORT,
    DEFAULT_PLOTJUGGLER_PUBLISH_PORT,
    PyRCFPublisherZMQ,
)


class ComponentDataPublisherDebugger(CtrlLoopDataPublisherBase):
    """This control loop debugger will stream the data from all the components in the control loop
    so that it can be subscribed to by a zmq subscriber (see `PyRCFSubscriberZMQ`)."""

    def __init__(
        self,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        debug_publish_callables: Callable[[], Any] | List[Callable[[], Any]] = None,
        port: int = DEFAULT_ZMQ_PUBLISH_PORT,
    ):
        """
        This control loop debugger will stream the data from all the components in the control loop
        so that it can be subscribed to by a zmq subscriber (see `PyRCFSubscriberZMQ`).

        NOTE: Because the publisher uses a custom encoder for encoding PyRCF data types to
        serialisable data, this debugger can affect the speed of the control loop (because lossless
        data is prioritised over loop rate). It is better to use the `ComponentDataRecorderDebugger`
        as it is faster if your computer is powerful enough.

        Args:
            rate (float, optional): Rate at which this should be triggered. Defaults to None
                (i.e. use control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
            debug_publish_callables (Callable[[], Any] | List[Callable[[], Any]]): Function handle
                (or list of) that return publishable data (json encodable) for additional streaming
                to plotjuggler. Use this for debugging components. These method(s)/functions(s)
                will be called in the control loop and their return value added to the data being
                published (if publishing is enabled in self.data_streamer_config). Defaults to None.
            port (int, optional): port to publish in. Defaults to 5001.
        """
        super().__init__(
            publisher=PyRCFPublisherZMQ(port=port),
            rate=rate,
            clock=clock,
            debug_publish_callables=debug_publish_callables,
        )


class PlotjugglerLoopDebugger(ComponentDataPublisherDebugger):
    """This control loop debugger will stream the data from all the components in the control loop
    so that it can be visualised in plotjuggler."""

    def __init__(
        self,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        debug_publish_callables: Callable[[], Any] | List[Callable[[], Any]] = None,
        port: int = DEFAULT_PLOTJUGGLER_PUBLISH_PORT,
    ):
        """This control loop debugger will stream the data from all the components in the control
        loop so that it can be visualised in plotjuggler.

        NOTE: publishes to port 9872 (default for plotjuggler)

        NOTE: Because the publisher uses a custom encoder for encoding PyRCF data types to
        serialisable data, this debugger can affect the speed of the control loop.

        Args:
            rate (float, optional): Rate at which this should be triggered. Defaults to None
                (i.e. use control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
            debug_publish_callables (Callable[[], Any] | List[Callable[[], Any]]): Function handle
                (or list of) that return publishable data (json encodable) for additional streaming
                to plotjuggler. Use this for debugging components. These method(s)/functions(s)
                will be called in the control loop and their return value added to the data being
                published (if publishing is enabled in self.data_streamer_config). Defaults to
                None.
            port (int, optional): port to publish in. Defaults to 9872 (plotjuggler zmq subscriber
                default).
        """
        super().__init__(
            rate=rate,
            clock=clock,
            debug_publish_callables=debug_publish_callables,
            port=port,
        )
