from abc import abstractmethod

from ..pyrcf_component import PyRCFComponent
from ...core.types import RobotState, RobotCmd, LocalMotionPlan


class ControllerBase(PyRCFComponent):
    """Base class defining the interface for all controllers that can be used in the control loop."""

    @abstractmethod
    def update(
        self,
        robot_state: RobotState,
        local_plan: LocalMotionPlan,
        t: float = None,
        dt: float = None,
    ) -> RobotCmd:
        """The main control update method to be called in the loop.

        This method will be called after the global plan and local plan are generated in the loop.

        This method should be implemented by any controller implementation.

        Args:
            robot_state (RobotState): The current state information from the robot.
            local_plan (LocalMotionPlan): the latest local plan generated by the local planner used
                in the loop.
            t (float, optional): the current time signature of the control loop. Defaults to None
                (controllers may or may not need this).
            dt (float, optional): the time since the last control loop. Defaults to None
                (controllers may or may not need this).

        Raises:
            NotImplementedError: Raised if this method is not implemented by the child class.

        Returns:
            RobotCmd: The output control command to be sent to the robot.
        """
        raise NotImplementedError("This method has to be implemented in the child class")


class DummyController(ControllerBase):
    """Dummy controller for testing pipeline."""

    def __init__(self, squawk: bool = True):
        """Dummy controller for testing pipeline.

        Args:
            squawk (bool, optional): Verbosity. Defaults to True.
        """
        self._squawk = squawk

    def update(
        self,
        robot_state: RobotState,
        local_plan: LocalMotionPlan,
        t: float = None,
        dt: float = None,
    ) -> RobotCmd:
        """This is a dummy method for sanity checking the control loop."""
        if self._squawk:
            print(f"{__class__.__name__}: Performing dummy control update. time: {t}, dt: {dt}")
        return RobotCmd()