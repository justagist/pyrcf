from abc import ABC, abstractmethod
from typing import List, Tuple
import copy

from ...core.types import GlobalMotionPlan, LocalMotionPlan, RobotState, RobotCmd
from ...utils.time_utils import RateTrigger, ClockBase, PythonPerfClock
from ...core.logging import logging


class CtrlLoopDebuggerBase(ABC):
    """Base class for control loop debuggers. Can only be used with control loops."""

    def __init__(self, rate: float = None, clock: ClockBase = PythonPerfClock()):
        """Base class for control loop debuggers. Can only be used with control loops.

        Args:
            rate (float, optional): Rate at which this should be triggered. Defaults to None
                (i.e. use control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
        """
        self._rate = rate
        if self._rate is not None and self._rate > 0.0:
            self._rate_trigger = RateTrigger(rate=self._rate, clock=clock)
            logging.debug(f"{self.__class__.__name__}: Setting trigger rate to {self._rate}Hz.")

    def _should_run(self):
        if self._rate is None:
            return True
        if self._rate <= 0.0:
            return False
        return self._rate_trigger.triggered()

    def run_once(
        self,
        t: float,
        dt: float,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        agent_outputs: List[Tuple[LocalMotionPlan, RobotCmd]],
        robot_cmd: RobotCmd,
    ):
        """The main method that will be run in every control loop.

        NOTE: SHOULD NOT BE OVERRIDDEN. Override the `_run_once_impl` method.

        This method automatically checks if it is ok to trigger the implemented
        child method, and calls it if appropriate.

        Args:
            t (float): Current time in the control loop.
            dt (float): The current dt in the control loop.
            robot_state (RobotState): robot state after update from the state estimator
                in the control loop.
            global_plan (GlobalMotionPlan): The global plan output produced by the global
                planner.
            agent_outputs (List[Tuple[LocalMotionPlan, RobotCmd]]): The list of outputs
                from all agents in the control loop.
            robot_cmd (RobotCmd): The final command written to the robot.
        """
        if self._should_run():
            self._run_once_impl(
                t=t,
                dt=dt,
                robot_state=copy.deepcopy(robot_state),
                global_plan=copy.deepcopy(global_plan),
                agent_outputs=copy.deepcopy(agent_outputs),
                robot_cmd=copy.deepcopy(robot_cmd),
            )

    @abstractmethod
    def _run_once_impl(
        self,
        t: float,
        dt: float,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        agent_outputs: List[Tuple[LocalMotionPlan, RobotCmd]],
        robot_cmd: RobotCmd,
    ):
        """The main method that will be run in every control loop. Should be overridden
        in the implemented child of `CtrlLoopDebuggerBase`.

        This method will be called automatically if the timer is triggered, as long
        as the main `run_once()` method is called continuously in the control loop.

        Args:
            t (float): Current time in the control loop.
            dt (float): The current dt in the control loop.
            robot_state (RobotState): robot state after update from the state estimator
                in the control loop.
            global_plan (GlobalMotionPlan): The global plan output produced by the global
                planner.
            agent_outputs (List[Tuple[LocalMotionPlan, RobotCmd]]): The list of outputs
                from all agents in the control loop.
            robot_cmd (RobotCmd): The final command written to the robot.
        """
        raise NotImplementedError("This method has to be implemented in the child class")

    def shutdown(self):
        """Cleanly shutdown the debugger. Override in child class if required. The
        base class implements an empty function."""
        return


class DummyDebugger(CtrlLoopDebuggerBase):
    """A dummy debugger to test in pipeline."""

    def __init__(
        self,
        squawk: bool = True,
        rate: float = 10,
        clock: ClockBase = PythonPerfClock(),
    ):
        super().__init__(rate, clock)
        self._squawk = squawk

    def _run_once_impl(
        self,
        t: float,
        dt: float,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        agent_outputs: List[Tuple[LocalMotionPlan | RobotCmd]],
        robot_cmd: RobotCmd,
    ):
        if self._squawk:
            print("Debugger in loop running...")
