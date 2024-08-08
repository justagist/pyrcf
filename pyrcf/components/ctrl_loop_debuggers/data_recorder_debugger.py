from typing import List, Tuple, Callable, Any
import pickle

from ...core.types import LocalMotionPlan, GlobalMotionPlan, RobotCmd, RobotState
from ...utils.time_utils import ClockBase, PythonPerfClock
from .ctrl_loop_debugger_base import CtrlLoopDebuggerBase
from ...core.logging import logging


class ComponentDataRecorderDebugger(CtrlLoopDebuggerBase):
    """A data recording debugger that records all the control loop data directly to a file. Use
    `ComponentDataRecorderDataParser` from `pyrcf.utils.data_io_utils`
    to parse the recorded data file.

    NOTE: if your computer has enough RAM and compute power, this debugger is better (in terms
    of keeping desired control loop rate) than `ComponentDataPublisherDebugger`.
    """

    def __init__(
        self,
        file_name: str,
        rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        extra_data_callables: Callable[[], Any] | List[Callable[[], Any]] = None,
        buffer_size: int = 50,
    ):
        """A data recording debugger that records all the control loop data directly to a file. Use
        `ComponentDataRecorderDataParser` from `pyrcf.utils.data_io_utils`
        to parse the recorded data file.

        NOTE: if your computer has enough RAM and compute power, this debugger is better (in terms
        of keeping desired control loop rate) than `ComponentDataPublisherDebugger`.

        Args:
            file_name (str): Path to the file to write to.
            rate (float, optional): Rate at which data recording should happen. Defaults to None
                (i.e. use control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
            extra_data_callables (Callable[[], Any] | List[Callable[[], Any]], optional):
                Additional function handles that return data that needs to be recorded. Defaults to
                None.
            buffer_size (int, optional): Buffer size to be used to write to file in batches.
                Defaults to 50.

        Raises:
            ValueError: if invalid value in `extra_data_callables`.
        """
        super().__init__(rate, clock)

        self._file = open(file_name, "wb")

        self._additional_handles = []
        if extra_data_callables is not None:
            if callable(extra_data_callables):
                extra_data_callables = [extra_data_callables]
            if not isinstance(extra_data_callables, list):
                raise ValueError(
                    f"{self.__class__.__name__}: Argument to this function should"
                    " be a callable or a list of callables."
                )
            assert all(
                callable(ele) for ele in extra_data_callables
            ), "All elements in `extra_data_callables` should be a function handle."
            for handle in extra_data_callables:
                if handle not in self._additional_handles:
                    self._additional_handles.append(handle)
        self._buffer = []
        self._buffer_size = buffer_size

    def _run_once_impl(
        self,
        t: float,
        dt: float,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        agent_outputs: List[Tuple[LocalMotionPlan, RobotCmd]],
        robot_cmd: RobotCmd,
    ):
        self._buffer.append(
            {
                "t": t,
                "dt": dt,
                "robot_state": robot_state,
                "global_plan": global_plan,
                "agent_outputs": agent_outputs,
                "robot_cmd": robot_cmd,
                "debug_data": [_get_data() for _get_data in self._additional_handles],
            }
        )
        if len(self._buffer) >= self._buffer_size:
            pickle.dump(
                self._buffer,
                self._file,
            )
            self._buffer.clear()

    def shutdown(self):
        if len(self._buffer) > 0:
            pickle.dump(
                self._buffer,
                self._file,
            )
        logging.info(f"Saved data to file: {self._file.name}")
        self._file.close()
